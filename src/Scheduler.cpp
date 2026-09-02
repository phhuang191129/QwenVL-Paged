#include "qwenvl_paged/Scheduler.h"

#include <algorithm>
#include <utility>

namespace qwenvl_paged {

namespace {

std::deque<Request>::iterator find_request(std::deque<Request>& queue, RequestId request_id) {
    for (auto it = queue.begin(); it != queue.end(); ++it) {
        if (it->request_id == request_id) {
            return it;
        }
    }
    return queue.end();
}

SequenceMetadata metadata_from(const Request& request) {
    SequenceMetadata metadata;
    metadata.sequence_id = request.root_sequence_id;
    metadata.request_id = request.request_id;
    metadata.multimodal_spans = request.multimodal_spans;
    metadata.positional_encoding = request.positional_encoding;
    return metadata;
}

} // namespace

Scheduler::Scheduler(SchedulerConfig config, KVCacheManager& cache_manager, MemoryAllocator& allocator)
    : config_(config), cache_manager_(&cache_manager), allocator_(&allocator) {}

void Scheduler::enqueue(Request request) {
    request.state = RequestState::Pending;
    request.remaining_prefill_tokens = request.prompt_tokens;
    request.admission_skips = 0;
    request.lifetime_blocks = 0;
    pending_.push_back(std::move(request));
}

std::uint32_t Scheduler::lifetime_blocks_for(const Request& request) const noexcept {
    const std::uint32_t cached =
        request.prefix_key.empty() ? 0 : cache_manager_->prefix_token_count(request.prefix_key);
    const std::uint32_t usable = cached <= request.prompt_tokens ? cached : 0;
    const std::uint32_t unique =
        request.prompt_tokens > usable ? request.prompt_tokens - usable : 0;
    return blocks_for_tokens(unique + request.sampling.max_decode_tokens);
}

void Scheduler::schedule_prefill_chunk(Request& request, BatchPlan& plan) {
    if (request.remaining_prefill_tokens == 0) {
        return;
    }
    if (plan.scheduled_tokens >= config_.max_batch_tokens) {
        return;
    }
    const std::uint32_t chunk = std::min(
        request.remaining_prefill_tokens, config_.max_batch_tokens - plan.scheduled_tokens);
    if (chunk == 0) {
        return;
    }
    plan.prefill_requests.push_back(request.request_id);
    plan.prefill_tokens.push_back(chunk);
    plan.scheduled_tokens += chunk;
}

void Scheduler::schedule_decode_tokens(BatchPlan& plan) {
    for (const Request& request : active_) {
        if (request.state != RequestState::Decode) {
            continue;
        }
        if (plan.scheduled_tokens >= config_.max_batch_tokens) {
            break;
        }
        plan.decode_requests.push_back(request.request_id);
        plan.scheduled_tokens += 1;
    }
}

bool Scheduler::try_admit(Request& candidate, BatchPlan& plan, std::size_t admitted_this_step) {
    if (plan.scheduled_tokens >= config_.max_batch_tokens) {
        return false;
    }

    const std::uint32_t needed = lifetime_blocks_for(candidate);
    const AllocatorStats stats = allocator_->stats();
    if (needed > stats.total_blocks) {
        return false;
    }

    reclaim_for_admission(candidate.prompt_tokens, candidate.sampling.max_decode_tokens, admitted_this_step);

    if (lifetime_held_ + needed > allocator_->stats().total_blocks) {
        return false;
    }

    if (!cache_manager_->create_sequence(metadata_from(candidate))) {
        return false;
    }

    std::uint32_t attached = 0;
    const std::uint32_t cached =
        candidate.prefix_key.empty() ? 0 : cache_manager_->prefix_token_count(candidate.prefix_key);
    if (cached > 0 && cached <= candidate.prompt_tokens) {
        attached = cache_manager_->attach_prefix(candidate.root_sequence_id, candidate.prefix_key);
    }
    const std::uint32_t still_need =
        candidate.prompt_tokens > attached ? candidate.prompt_tokens - attached : 0;
    if (still_need > 0 &&
        !cache_manager_->reserve_tokens(candidate.root_sequence_id, still_need)) {
        cache_manager_->release_request(candidate.request_id);
        return false;
    }
    if (candidate.prompt_tokens > 0 && attached == 0 && still_need == 0) {
        cache_manager_->release_request(candidate.request_id);
        return false;
    }

    candidate.lifetime_blocks = needed;
    lifetime_held_ += needed;
    candidate.state = RequestState::Prefill;
    if (candidate.remaining_prefill_tokens == 0) {
        candidate.remaining_prefill_tokens = candidate.prompt_tokens;
    }
    schedule_prefill_chunk(candidate, plan);
    return true;
}

BatchPlan Scheduler::schedule_next() {
    BatchPlan plan;

    const auto admit_from_pending = [&]() {
        while (!pending_.empty() && active_.size() < config_.max_active_requests) {
            const bool must_take_head = !config_.size_aware_admission ||
                pending_.front().admission_skips >= config_.admission_skip_limit;

            if (must_take_head) {
                if (!try_admit(pending_.front(), plan, plan.prefill_requests.size())) {
                    break;
                }
                active_.push_back(std::move(pending_.front()));
                pending_.pop_front();
                continue;
            }

            bool admitted = false;
            for (auto it = pending_.begin(); it != pending_.end(); ++it) {
                if (!try_admit(*it, plan, plan.prefill_requests.size())) {
                    continue;
                }
                for (auto skipped = pending_.begin(); skipped != it; ++skipped) {
                    ++skipped->admission_skips;
                }
                active_.push_back(std::move(*it));
                pending_.erase(it);
                admitted = true;
                break;
            }
            if (!admitted) {
                break;
            }
        }
    };

    admit_from_pending();

    for (Request& request : active_) {
        if (request.state == RequestState::Prefill) {
            const bool already =
                std::find(plan.prefill_requests.begin(), plan.prefill_requests.end(), request.request_id) !=
                plan.prefill_requests.end();
            if (!already) {
                schedule_prefill_chunk(request, plan);
            }
        }
    }

    schedule_decode_tokens(plan);
    return plan;
}

void Scheduler::complete_step(RequestId request_id, std::uint32_t produced_tokens) {
    auto it = find_request(active_, request_id);
    if (it == active_.end()) {
        return;
    }

    if (it->state == RequestState::Prefill) {
        if (produced_tokens >= it->remaining_prefill_tokens) {
            it->remaining_prefill_tokens = 0;
            it->state = RequestState::Decode;
        } else {
            it->remaining_prefill_tokens -= produced_tokens;
        }
    }
}

void Scheduler::cancel(RequestId request_id) {
    for (std::deque<Request>* queue : {&pending_, &active_, &preempted_}) {
        auto it = find_request(*queue, request_id);
        if (it != queue->end()) {
            if (it->lifetime_blocks > 0 && lifetime_held_ >= it->lifetime_blocks) {
                lifetime_held_ -= it->lifetime_blocks;
            }
            cache_manager_->release_request(it->request_id);
            queue->erase(it);
            preemption_info_.erase(request_id);
            return;
        }
    }
}

bool Scheduler::preempt(RequestId request_id, std::string reason) {
    auto it = find_request(active_, request_id);
    if (it == active_.end()) {
        return false;
    }

    Request request = std::move(*it);
    active_.erase(it);
    request.state = RequestState::Preempted;

    if (request.lifetime_blocks > 0 && lifetime_held_ >= request.lifetime_blocks) {
        lifetime_held_ -= request.lifetime_blocks;
    }

    PreemptionInfo info;
    info.reason = std::move(reason);
    info.swapped_blocks = cache_manager_->swap_out_request(request.request_id);
    preemption_info_[request.request_id] = std::move(info);

    preempted_.push_back(std::move(request));
    return true;
}

bool Scheduler::resume(RequestId request_id) {
    auto it = find_request(preempted_, request_id);
    if (it == preempted_.end()) {
        return false;
    }

    if (!cache_manager_->swap_in_request(it->request_id)) {
        return false;
    }

    Request request = std::move(*it);
    preempted_.erase(it);
    request.state = request.remaining_prefill_tokens > 0 ? RequestState::Prefill : RequestState::Decode;
    lifetime_held_ += request.lifetime_blocks;
    preemption_info_.erase(request_id);
    active_.push_back(std::move(request));
    return true;
}

std::optional<PreemptionInfo> Scheduler::preemption_info(RequestId request_id) const {
    auto it = preemption_info_.find(request_id);
    if (it == preemption_info_.end()) {
        return std::nullopt;
    }
    return it->second;
}

std::optional<RequestState> Scheduler::state(RequestId request_id) const {
    for (const std::deque<Request>* queue : {&pending_, &active_, &preempted_}) {
        for (const Request& request : *queue) {
            if (request.request_id == request_id) {
                return request.state;
            }
        }
    }
    return std::nullopt;
}

std::size_t Scheduler::active_size() const noexcept {
    return active_.size();
}

std::uint32_t Scheduler::blocks_for_tokens(std::uint32_t token_count) const noexcept {
    const PhysicalBlock* block = allocator_->block(0);
    if (block == nullptr) {
        return 0;
    }

    const BlockShape& shape = block->shape();
    const std::uint32_t tokens_per_block = shape.tokens_per_block;
    if (tokens_per_block == 0) {
        return 0;
    }
    const std::uint32_t token_blocks = (token_count + tokens_per_block - 1) / tokens_per_block;
    const std::uint32_t tables = shape.per_layer_frames() ? shape.num_layers : 1;
    return token_blocks * tables;
}

void Scheduler::reclaim_for_admission(
    std::uint32_t prompt_tokens, std::uint32_t decode_tokens, std::size_t admitted_this_step) {
    if (config_.preemption_watermark_blocks == 0) {
        return;
    }

    const AllocatorStats stats = allocator_->stats();
    const std::uint32_t needed = blocks_for_tokens(prompt_tokens + decode_tokens);
    if (needed == 0 || needed > stats.total_blocks) {
        return;
    }

    const std::uint32_t required =
        std::min(needed + config_.preemption_watermark_blocks, stats.total_blocks);

    while (active_.size() > admitted_this_step &&
           (allocator_->stats().free_blocks < required || lifetime_held_ + needed > stats.total_blocks)) {
        const std::uint32_t before = allocator_->stats().free_blocks;

        const RequestId victim = active_[active_.size() - admitted_this_step - 1].request_id;
        if (!preempt(victim, "cache pressure watermark")) {
            break;
        }

        if (allocator_->stats().free_blocks <= before) {
            break;
        }
    }
}

} // namespace qwenvl_paged
