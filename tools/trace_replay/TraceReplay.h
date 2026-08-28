#pragma once

#include "qwenvl_paged/KVCacheManager.h"
#include "qwenvl_paged/Scheduler.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace qwenvl_paged::replay {

/**
 * @brief Model and workload geometry recorded in a trace's header record.
 *
 * The visual fields are authoritative: they come from the published Qwen3-VL
 * preprocessor and text configs by way of `tools/trace_gen/qwen3vl_trace.py`.
 */
struct TraceHeader {
    std::string model;
    std::string mix;
    std::string dimension_source;
    std::string pixel_budget;
    std::uint64_t seed{0};
    std::uint32_t num_requests{0};
    std::uint32_t num_layers{0};
    std::uint32_t num_kv_heads{0};
    std::uint32_t head_dim{0};
    std::uint32_t bytes_per_element{0};
    std::uint32_t max_prompt_tokens{0};
    std::uint32_t max_decode_budget{0};
    std::uint32_t visual_tokens_max{0};
    std::uint32_t images_total{0};
};

/**
 * @brief One image's exact Qwen3-VL patch grid and resulting LLM token count.
 */
struct TraceImage {
    std::uint32_t grid_t{0};
    std::uint32_t grid_h{0};
    std::uint32_t grid_w{0};
    std::uint32_t visual_tokens{0};
};

/**
 * @brief One replayable inference request.
 */
struct TraceRequest {
    RequestId request_id{0};
    std::uint64_t arrival_us{0};
    std::uint32_t text_tokens{0};
    std::uint32_t prompt_tokens{0};
    /** @brief The request's `max_new_tokens`: what a reservation must cover. */
    std::uint32_t decode_budget{0};
    /** @brief Tokens actually produced before stopping, which is usually fewer. */
    std::uint32_t decode_actual{0};
    std::uint32_t num_parallel_samples{1};
    std::string prefix_key;
    std::vector<TraceImage> images;
    std::vector<MultimodalSpan> multimodal_spans;
    PositionalEncodingMetadata positional_encoding;
};

struct Trace {
    TraceHeader header;
    std::vector<TraceRequest> requests;
};

/**
 * @brief Parses a JSONL trace produced by `tools/trace_gen/qwen3vl_trace.py`.
 *
 * Returns nullopt and fills `error` on a malformed file or unknown schema
 * version, so a caller never silently replays a partially understood trace.
 */
[[nodiscard]] std::optional<Trace> load_trace(const std::string& path, std::string& error);

/**
 * @brief Pool sizing and policy for one replay run.
 *
 * `arrival_us` in the trace is deliberately ignored: week 9 has no cost model
 * for a step, so the replay is closed-loop. Every request is enqueued up front
 * and the block pool is the only thing throttling admission, which is the
 * maximum-pressure case and the one a memory subsystem should be judged on.
 */
struct ReplayConfig {
    std::uint32_t tokens_per_block{16};
    std::uint32_t max_blocks{0};
    std::uint32_t max_active_requests{16};
    std::uint32_t max_batch_tokens{32768};
    std::uint32_t preemption_watermark_blocks{0};
    std::uint32_t swap_slots{0};
    /**
     * @brief Upper bound on what the contiguous baseline reserves per request.
     *
     * The baseline reserves one contiguous run of `prompt + max_new_tokens`,
     * capped here. Reserving the request's own declared budget rather than a
     * flat model maximum makes it the strongest honest baseline: it still cannot
     * reclaim the tokens a request never produces, and its run lengths vary, so
     * external fragmentation is measured rather than assumed away.
     */
    std::uint32_t baseline_max_context{8192};
    /**
     * @brief Hard iteration bound so a head-of-line stall reports instead of hanging.
     */
    std::uint64_t max_steps{2000000};
};

/**
 * @brief Aggregate outcome of one replay run.
 *
 * Fields that do not apply to a mode stay zero: the contiguous baseline has no
 * copy-on-write or swap, and reports fragmentation instead.
 */
struct ReplayMetrics {
    std::string mode;
    std::uint64_t steps{0};
    std::uint32_t requests_completed{0};
    std::uint32_t requests_admitted{0};
    std::uint64_t tokens_decoded{0};

    std::uint32_t peak_blocks_in_use{0};
    std::size_t peak_bytes_in_use{0};
    /** @brief Largest batch either mode sustained, which cache pressure caps. */
    std::uint32_t peak_active_requests{0};
    /** @brief Token slots the pool handed out, summed over every admission. */
    std::uint64_t reserved_token_slots{0};
    /** @brief Token slots that ever held a committed token. */
    std::uint64_t committed_tokens{0};

    std::uint64_t preemptions{0};
    std::uint64_t resumes{0};
    std::uint64_t failed_resumes{0};
    std::uint64_t swapped_blocks_out{0};
    std::uint64_t forks{0};
    std::uint64_t cow_events{0};

    /** @brief Steps where a request was waiting and none could be admitted. */
    std::uint64_t head_of_line_stall_steps{0};
    /**
     * @brief Admissions refused although total free blocks would have sufficed.
     *
     * External fragmentation, which paging is supposed to eliminate entirely.
     */
    std::uint64_t fragmentation_failures{0};
    /** @brief Requests whose prompt cannot fit the pool at any point. */
    std::uint32_t requests_never_admissible{0};

    /** @brief Largest per-sequence tail slack observed, in token slots. */
    std::uint32_t max_partial_block_slack{0};
    /** @brief True when the trace ran to completion rather than stalling out. */
    bool completed{false};
    /** @brief True when every block returned to the pool at the end. */
    bool leak_free{false};

    /**
     * @brief Committed token positions per token slot the pool had handed out.
     *
     * A step-weighted ratio of sums rather than a mean of ratios. Values above
     * 1.0 are expected once sequences share blocks: the same slot is serving
     * more than one sequence, which is the whole point of reference counting.
     */
    [[nodiscard]] double utilization() const noexcept;
};

/**
 * @brief Smallest pool that lets `max_active_requests` run without exhaustion.
 *
 * The scheduler admits on whether a *prompt* fits, not whether the prompt plus
 * the decode budget fits, so a pool smaller than this can admit a batch it
 * cannot carry to completion. With automatic preemption disabled that is a
 * permanent stall rather than a slowdown, which `run_paged` reports instead of
 * hiding. Sizing the pool from the workload is the week 9 answer; making
 * admission itself decode-aware is a week 14 change.
 */
[[nodiscard]] std::uint32_t suggested_pool_blocks(const Trace& trace, const ReplayConfig& config);

/**
 * @brief Replays a trace through the paged cache subsystem.
 */
[[nodiscard]] ReplayMetrics run_paged(const Trace& trace, const ReplayConfig& config);

/**
 * @brief Replays a trace through a contiguous worst-case-reservation allocator.
 *
 * The baseline is a real allocator over the same number of frames: it must find
 * a contiguous run for each request and can genuinely fail an admission, so its
 * fragmentation figure is measured rather than assumed.
 */
[[nodiscard]] ReplayMetrics run_contiguous(const Trace& trace, const ReplayConfig& config);

/**
 * @brief Returns the CSV header matching `to_csv_row`.
 */
[[nodiscard]] std::string csv_header();

/**
 * @brief Formats one metrics row, prefixed with the trace and pool identity.
 */
[[nodiscard]] std::string to_csv_row(
    const Trace& trace,
    const ReplayConfig& config,
    const ReplayMetrics& metrics);

} // namespace qwenvl_paged::replay
