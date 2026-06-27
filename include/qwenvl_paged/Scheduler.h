#pragma once

#include "qwenvl_paged/KVCacheManager.h"
#include "qwenvl_paged/MemoryAllocator.h"

#include <cstdint>
#include <deque>
#include <optional>
#include <string>
#include <vector>

namespace qwenvl_paged {

/**
 * @brief High-level lifecycle state for an inference request.
 */
enum class RequestState : std::uint8_t {
    Pending,
    Prefill,
    Decode,
    Preempted,
    Finished,
    Cancelled
};

/**
 * @brief Sampling metadata relevant to scheduler branching decisions.
 */
struct SamplingConfig {
    std::uint32_t num_parallel_samples{1};
    std::uint32_t max_decode_tokens{0};
};

/**
 * @brief User-facing inference request tracked by the continuous batcher.
 */
struct Request {
    RequestId request_id{0};
    SequenceId root_sequence_id{0};
    std::uint32_t prompt_tokens{0};
    SamplingConfig sampling{};
    std::vector<MultimodalSpan> multimodal_spans;
    PositionalEncodingMetadata positional_encoding;
    RequestState state{RequestState::Pending};
};

/**
 * @brief Scheduler tuning knobs that do not affect allocator correctness.
 */
struct SchedulerConfig {
    std::uint32_t max_active_requests{0};
    std::uint32_t max_batch_tokens{0};
    std::uint32_t preemption_watermark_blocks{0};
};

/**
 * @brief One executable continuous-batching step.
 */
struct BatchPlan {
    std::vector<RequestId> prefill_requests;
    std::vector<RequestId> decode_requests;
    std::uint32_t scheduled_tokens{0};
};

/**
 * @brief Handles admission, preemption, and decode-step scheduling.
 *
 * The scheduler never owns physical memory directly. It uses KVCacheManager for
 * sequence cache lifecycle and reads allocator statistics to make continuous
 * batching decisions under cache pressure.
 *
 * This phase-1 scheduler is not thread-safe. It should run on the engine event
 * loop, with external network/API threads handing requests over through an
 * ingress queue before `enqueue` is called.
 */
class Scheduler {
public:
    /**
     * @brief Creates a scheduler over an existing cache manager and allocator.
     *
     * The referenced cache manager and allocator are non-owning dependencies and
     * must outlive the scheduler, typically as members of a higher-level Engine.
     */
    Scheduler(SchedulerConfig config, KVCacheManager& cache_manager, MemoryAllocator& allocator);

    ~Scheduler() = default;

    Scheduler(const Scheduler&) = delete;
    Scheduler& operator=(const Scheduler&) = delete;
    Scheduler(Scheduler&&) noexcept = default;
    Scheduler& operator=(Scheduler&&) noexcept = default;

    /**
     * @brief Enqueues a new request for future admission.
     */
    void enqueue(Request request);

    /**
     * @brief Builds the next batch plan and updates request states.
     */
    [[nodiscard]] BatchPlan schedule_next();

    /**
     * @brief Marks progress for a request after backend execution.
     */
    void complete_step(RequestId request_id, std::uint32_t produced_tokens);

    /**
     * @brief Cancels a request and releases its cache state.
     */
    void cancel(RequestId request_id);

    /**
     * @brief Preempts a request when cache pressure requires it.
     */
    bool preempt(RequestId request_id, std::string reason);

    /**
     * @brief Attempts to resume a previously preempted request.
     */
    bool resume(RequestId request_id);

    /**
     * @brief Returns the current state of a request if it is tracked.
     */
    [[nodiscard]] std::optional<RequestState> state(RequestId request_id) const;

    /**
     * @brief Returns the number of requests currently eligible for execution.
     */
    [[nodiscard]] std::size_t active_size() const noexcept;

private:
    SchedulerConfig config_{};
    KVCacheManager* cache_manager_{nullptr};
    MemoryAllocator* allocator_{nullptr};
    std::deque<Request> pending_;
    std::deque<Request> active_;
    std::deque<Request> preempted_;
};

} // namespace qwenvl_paged
