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
    pending_.push_back(std::move(request));
}

BatchPlan Scheduler::schedule_next() {
    BatchPlan plan;

    while (!pending_.empty() && active_.size() < config_.max_active_requests) {
        Request& candidate = pending_.front();
        if (plan.scheduled_tokens + candidate.prompt_tokens > config_.max_batch_tokens) {
            break;
        }

        reclaim_for_admission(candidate.prompt_tokens, plan.prefill_requests.size());

        if (!cache_manager_->create_sequence(metadata_from(candidate))) {
            break;
        }
        if (!cache_manager_->reserve_tokens(candidate.root_sequence_id, candidate.prompt_tokens)) {
            // Leave nothing behind, so a later step can admit this request once
            // blocks are reclaimed.
            cache_manager_->release_sequence(candidate.root_sequence_id);
            break;
        }

        candidate.state = RequestState::Prefill;
        plan.prefill_requests.push_back(candidate.request_id);
        plan.scheduled_tokens += candidate.prompt_tokens;

        active_.push_back(std::move(candidate));
        pending_.pop_front();
    }

    for (const Request& request : active_) {
        if (request.state == RequestState::Decode) {
            plan.decode_requests.push_back(request.request_id);
            plan.scheduled_tokens += 1;
        }
    }

    return plan;
}

void Scheduler::complete_step(RequestId request_id, std::uint32_t /*produced_tokens*/) {
    auto it = find_request(active_, request_id);
    if (it == active_.end()) {
        return;
    }

    if (it->state == RequestState::Prefill) {
        it->state = RequestState::Decode;
    }
}

void Scheduler::cancel(RequestId request_id) {
    for (std::deque<Request>* queue : {&pending_, &active_, &preempted_}) {
        auto it = find_request(*queue, request_id);
        if (it != queue->end()) {
            cache_manager_->release_sequence(it->root_sequence_id);
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

    PreemptionInfo info;
    info.reason = std::move(reason);
    info.swapped_blocks = cache_manager_->swap_out_sequence(request.root_sequence_id);
    preemption_info_[request.request_id] = std::move(info);

    preempted_.push_back(std::move(request));
    return true;
}

bool Scheduler::resume(RequestId request_id) {
    auto it = find_request(preempted_, request_id);
    if (it == preempted_.end()) {
        return false;
    }

    if (!cache_manager_->swap_in_sequence(it->root_sequence_id)) {
        return false;
    }

    Request request = std::move(*it);
    preempted_.erase(it);
    request.state = RequestState::Decode;
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

    const std::uint32_t tokens_per_block = block->shape().tokens_per_block;
    if (tokens_per_block == 0) {
        return 0;
    }
    return (token_count + tokens_per_block - 1) / tokens_per_block;
}

void Scheduler::reclaim_for_admission(std::uint32_t prompt_tokens, std::size_t admitted_this_step) {
    if (config_.preemption_watermark_blocks == 0) {
        return;
    }

    const AllocatorStats stats = allocator_->stats();
    const std::uint32_t needed = blocks_for_tokens(prompt_tokens);
    if (needed == 0 || needed > stats.total_blocks) {
        // A prompt larger than the whole pool can never be admitted. Evicting
        // healthy requests for it would only thrash the batch.
        return;
    }

    // The reserve is best-effort: a prompt that needs nearly the whole pool must
    // stay admissible rather than starve behind a watermark it can never clear.
    const std::uint32_t required =
        std::min(needed + config_.preemption_watermark_blocks, stats.total_blocks);

    while (active_.size() > admitted_this_step && allocator_->stats().free_blocks < required) {
        const std::uint32_t before = allocator_->stats().free_blocks;

        // Newest first among requests that predate this call: the oldest active
        // request is closest to finishing, so it is the most expensive one to
        // roll back, and anything admitted moments ago is already in the plan.
        const RequestId victim = active_[active_.size() - admitted_this_step - 1].request_id;
        if (!preempt(victim, "cache pressure watermark")) {
            break;
        }

        if (allocator_->stats().free_blocks <= before) {
            // Preemption reclaimed nothing, which happens when no swap backend
            // is installed. Parking more requests cannot help.
            break;
        }
    }
}

} // namespace qwenvl_paged
