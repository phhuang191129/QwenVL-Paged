#include "qwenvl_paged/Scheduler.h"

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

        cache_manager_->create_sequence(metadata_from(candidate));
        cache_manager_->reserve_tokens(candidate.root_sequence_id, candidate.prompt_tokens);

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
            return;
        }
    }
}

bool Scheduler::preempt(RequestId request_id, std::string /*reason*/) {
    auto it = find_request(active_, request_id);
    if (it == active_.end()) {
        return false;
    }

    Request request = std::move(*it);
    active_.erase(it);
    request.state = RequestState::Preempted;
    preempted_.push_back(std::move(request));
    return true;
}

bool Scheduler::resume(RequestId request_id) {
    auto it = find_request(preempted_, request_id);
    if (it == preempted_.end()) {
        return false;
    }

    Request request = std::move(*it);
    preempted_.erase(it);
    request.state = RequestState::Decode;
    active_.push_back(std::move(request));
    return true;
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

} // namespace qwenvl_paged
