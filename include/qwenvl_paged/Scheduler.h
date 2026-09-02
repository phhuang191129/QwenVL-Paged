#pragma once

#include "qwenvl_paged/KVCacheManager.h"
#include "qwenvl_paged/MemoryAllocator.h"

#include <cstdint>
#include <deque>
#include <optional>
#include <string>
#include <unordered_map>
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
    /**
     * @brief Content identity of the shareable prompt prefix, empty if none.
     */
    std::string prefix_key;
    RequestState state{RequestState::Pending};
    /**
     * @brief Prompt tokens not yet scheduled as prefill work.
     *
     * Set to `prompt_tokens` on enqueue. `complete_step` subtracts produced
     * tokens; the request moves to Decode when this hits zero.
     */
    std::uint32_t remaining_prefill_tokens{0};
    /**
     * @brief Successful admissions that skipped this request while it was head.
     *
     * Used by size-aware admission to bound starvation.
     */
    std::uint32_t admission_skips{0};
    /**
     * @brief Pool blocks reserved against this request's prompt plus decode.
     */
    std::uint32_t lifetime_blocks{0};
};

/**
 * @brief Scheduler tuning knobs that do not affect allocator correctness.
 */
struct SchedulerConfig {
    std::uint32_t max_active_requests{0};
    std::uint32_t max_batch_tokens{0};
    /**
     * @brief Free blocks the scheduler tries to keep in reserve after admitting.
     *
     * Zero disables automatic preemption entirely, leaving `preempt` purely
     * caller-driven. Any positive value turns on the policy described on
     * `schedule_next`.
     */
    std::uint32_t preemption_watermark_blocks{0};
    /**
     * @brief When true, a request that does not fit may be skipped for a smaller
     *        later one, up to `admission_skip_limit` successful skips.
     *
     * The default is strict FIFO so week-9 traces stay reproducible.
     */
    bool size_aware_admission{false};
    /**
     * @brief How many times a waiting head may be skipped before it blocks again.
     */
    std::uint32_t admission_skip_limit{8};
};

/**
 * @brief Why a request was preempted and how much cache that reclaimed.
 */
struct PreemptionInfo {
    std::string reason;
    std::uint32_t swapped_blocks{0};
};

/**
 * @brief One executable continuous-batching step.
 */
struct BatchPlan {
    std::vector<RequestId> prefill_requests;
    /**
     * @brief Prefill tokens to run this step, parallel to `prefill_requests`.
     */
    std::vector<std::uint32_t> prefill_tokens;
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
     *
     * A pending request is only admitted when the cache manager can reserve its
     * prompt and the pool can still cover that prompt plus the request's decode
     * budget. Under cache pressure the request stays pending and no sequence
     * state is left behind, so a later step can admit it once blocks are
     * reclaimed.
     *
     * Prefill longer than the remaining `max_batch_tokens` is split across
     * steps. Decode tokens count against the same budget.
     *
     * When `SchedulerConfig::preemption_watermark_blocks` is positive, the
     * scheduler also reclaims cache on its own before admitting: it preempts
     * active requests, newest first, until the candidate's prompt-plus-decode
     * plus the configured reserve fits. The newest request is chosen because
     * the oldest is closest to finishing. The policy is demand-driven, so
     * nothing is preempted while no request is waiting.
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
     *
     * Every sequence of the request is swapped out, so private sampling-branch
     * frames reclaim. Shared prompt frames stay, by the existing swap_out
     * refcount rule.
     */
    bool preempt(RequestId request_id, std::string reason);

    /**
     * @brief Attempts to resume a previously preempted request.
     *
     * Fails and leaves the request preempted when the pool cannot swap its cache
     * back in.
     */
    bool resume(RequestId request_id);

    /**
     * @brief Returns preemption metadata while a request is preempted.
     */
    [[nodiscard]] std::optional<PreemptionInfo> preemption_info(RequestId request_id) const;

    /**
     * @brief Returns the current state of a request if it is tracked.
     */
    [[nodiscard]] std::optional<RequestState> state(RequestId request_id) const;

    /**
     * @brief Returns the number of requests currently eligible for execution.
     */
    [[nodiscard]] std::size_t active_size() const noexcept;

private:
    /**
     * @brief Returns how many physical frames a prompt of this length occupies.
     *
     * Equals token-blocks when every layer shares one frame. Multiplies by
     * `num_layers` when `BlockShape::layers_per_frame == 1`.
     */
    [[nodiscard]] std::uint32_t blocks_for_tokens(std::uint32_t token_count) const noexcept;

    /**
     * @brief Preempts active requests until a prompt-plus-decode plus reserve fits.
     *
     * `admitted_this_step` is the number of requests already admitted by the
     * current `schedule_next` call. Those sit at the back of the active queue
     * and are excluded from victim selection, so a batch plan can never contain
     * a request that the same call went on to preempt.
     *
     * Does nothing when the policy is disabled, when the request already fits,
     * or when no amount of preemption could help.
     */
    void reclaim_for_admission(
        std::uint32_t prompt_tokens,
        std::uint32_t decode_tokens,
        std::size_t admitted_this_step);

    /**
     * @brief Blocks reserved against prompt plus decode, minus a published prefix.
     */
    [[nodiscard]] std::uint32_t lifetime_blocks_for(const Request& request) const noexcept;

    /**
     * @brief Tries to admit `candidate` into `plan`. On success the caller pops it.
     */
    bool try_admit(Request& candidate, BatchPlan& plan, std::size_t admitted_this_step);

    void schedule_prefill_chunk(Request& request, BatchPlan& plan);
    void schedule_decode_tokens(BatchPlan& plan);

    SchedulerConfig config_{};
    KVCacheManager* cache_manager_{nullptr};
    MemoryAllocator* allocator_{nullptr};
    std::deque<Request> pending_;
    std::deque<Request> active_;
    std::deque<Request> preempted_;
    std::unordered_map<RequestId, PreemptionInfo> preemption_info_;
    std::uint32_t lifetime_held_{0};
};

} // namespace qwenvl_paged
