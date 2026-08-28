#include "TraceReplay.h"

#include "qwenvl_paged/MemoryAllocator.h"
#include "qwenvl_paged/SwapBackend.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <unordered_map>
#include <unordered_set>

namespace qwenvl_paged::replay {

namespace {

constexpr int kSupportedSchemaVersion = 1;

std::uint32_t ceil_div(std::uint32_t value, std::uint32_t divisor) {
    return divisor == 0 ? 0 : (value + divisor - 1) / divisor;
}

PositionEncodingKind kind_from_string(const std::string& name) {
    if (name == "image") {
        return PositionEncodingKind::Image2D;
    }
    if (name == "video") {
        return PositionEncodingKind::Video3D;
    }
    return PositionEncodingKind::Text1D;
}

Position3D position_from_json(const nlohmann::json& array) {
    Position3D position;
    position.temporal = array.at(0).get<std::int64_t>();
    position.height = array.at(1).get<std::int64_t>();
    position.width = array.at(2).get<std::int64_t>();
    return position;
}

TraceHeader header_from_json(const nlohmann::json& record) {
    TraceHeader header;
    header.model = record.at("model").get<std::string>();
    header.mix = record.at("mix").get<std::string>();
    header.dimension_source = record.at("dimension_source").get<std::string>();
    header.pixel_budget = record.at("pixel_budget").get<std::string>();
    header.seed = record.at("seed").get<std::uint64_t>();
    header.num_requests = record.at("num_requests").get<std::uint32_t>();
    header.num_layers = record.at("num_layers").get<std::uint32_t>();
    header.num_kv_heads = record.at("num_kv_heads").get<std::uint32_t>();
    header.head_dim = record.at("head_dim").get<std::uint32_t>();
    header.bytes_per_element = record.at("bytes_per_element").get<std::uint32_t>();
    header.max_prompt_tokens = record.at("max_prompt_tokens").get<std::uint32_t>();
    header.max_decode_budget = record.at("max_decode_budget").get<std::uint32_t>();
    header.visual_tokens_max = record.at("visual_tokens_max").get<std::uint32_t>();
    header.images_total = record.at("images_total").get<std::uint32_t>();
    return header;
}

TraceRequest request_from_json(const nlohmann::json& record) {
    TraceRequest request;
    request.request_id = record.at("request_id").get<RequestId>();
    request.arrival_us = record.at("arrival_us").get<std::uint64_t>();
    request.text_tokens = record.at("text_tokens").get<std::uint32_t>();
    request.prompt_tokens = record.at("prompt_tokens").get<std::uint32_t>();
    request.decode_budget = record.at("decode_budget").get<std::uint32_t>();
    request.decode_actual = record.at("decode_actual").get<std::uint32_t>();
    request.num_parallel_samples = record.at("num_parallel_samples").get<std::uint32_t>();
    request.prefix_key = record.at("prefix_key").get<std::string>();

    for (const nlohmann::json& image : record.at("images")) {
        TraceImage parsed;
        parsed.grid_t = image.at("grid_t").get<std::uint32_t>();
        parsed.grid_h = image.at("grid_h").get<std::uint32_t>();
        parsed.grid_w = image.at("grid_w").get<std::uint32_t>();
        parsed.visual_tokens = image.at("visual_tokens").get<std::uint32_t>();
        request.images.push_back(parsed);
    }

    std::uint32_t feature_index = 0;
    for (const nlohmann::json& span : record.at("mrope_spans")) {
        PositionalEncodingSpan parsed;
        parsed.start_token = span.at("start_token").get<TokenPosition>();
        parsed.token_count = span.at("token_count").get<std::uint32_t>();
        parsed.kind = kind_from_string(span.at("kind").get<std::string>());
        parsed.origin = position_from_json(span.at("origin"));
        parsed.stride = position_from_json(span.at("stride"));
        request.positional_encoding.spans.push_back(parsed);

        if (parsed.kind != PositionEncodingKind::Text1D) {
            MultimodalSpan feature;
            feature.start_token = parsed.start_token;
            feature.token_count = parsed.token_count;
            feature.feature_index = feature_index++;
            request.multimodal_spans.push_back(feature);
        }
    }

    return request;
}

/**
 * @brief Per-request replay bookkeeping the Scheduler does not track itself.
 *
 * `complete_step` only performs the Prefill to Decode transition and never
 * enforces `max_decode_tokens`, and nothing auto-resumes a preempted request, so
 * the driver owns decode accounting, capacity growth, completion, and resume.
 */
struct RequestProgress {
    const TraceRequest* request{nullptr};
    std::vector<SequenceId> sequences;
    std::uint32_t committed{0};
    std::uint32_t decoded{0};
    bool admitted{false};
    bool finished{false};
};

/**
 * @brief Makes the logical block holding `position` writable, counting copies.
 *
 * A real engine writes token by token, but copy-on-write is decided per block,
 * so touching one position per block is the same number of decisions.
 */
void touch_position(
    KVCacheManager& cache,
    SequenceId sequence_id,
    TokenPosition position,
    std::uint32_t tokens_per_block,
    ReplayMetrics& metrics) {
    const std::optional<CacheView> view = cache.cache_view(sequence_id);
    if (!view.has_value() || view->block_table == nullptr) {
        return;
    }
    const auto logical = static_cast<LogicalBlockIndex>(position / tokens_per_block);
    const std::optional<PhysicalBlockId> before = view->block_table->lookup(logical);
    const std::optional<PhysicalBlockId> after = cache.ensure_token_writable(sequence_id, position);
    if (before.has_value() && after.has_value() && *before != *after) {
        ++metrics.cow_events;
    }
}

/**
 * @brief Grows a sequence by one block when `position` falls past the table.
 *
 * `reserve_tokens` rounds up and always appends, so it cannot fill a partially
 * used tail block. This is the guard the end-to-end test uses.
 */
bool ensure_capacity(
    KVCacheManager& cache,
    SequenceId sequence_id,
    TokenPosition position,
    std::uint32_t tokens_per_block) {
    const std::optional<CacheView> view = cache.cache_view(sequence_id);
    if (!view.has_value() || view->block_table == nullptr) {
        return false;
    }
    if (position / tokens_per_block < view->block_table->size()) {
        return true;
    }
    return cache.reserve_tokens(sequence_id, 1);
}

/** @brief Token slots a sequence's mapped blocks currently back. */
std::uint32_t reserved_slots(
    const KVCacheManager& cache,
    SequenceId sequence_id,
    std::uint32_t tokens_per_block) {
    const std::optional<CacheView> view = cache.cache_view(sequence_id);
    if (!view.has_value() || view->block_table == nullptr) {
        return 0;
    }
    return static_cast<std::uint32_t>(view->block_table->size()) * tokens_per_block;
}

} // namespace

double ReplayMetrics::utilization() const noexcept {
    return reserved_token_slots == 0
        ? 0.0
        : static_cast<double>(committed_tokens) / static_cast<double>(reserved_token_slots);
}

std::uint32_t suggested_pool_blocks(const Trace& trace, const ReplayConfig& config) {
    std::uint32_t worst_request_blocks = 0;
    for (const TraceRequest& request : trace.requests) {
        const std::uint32_t branches = std::max<std::uint32_t>(1, request.num_parallel_samples);
        const std::uint32_t prompt_blocks = ceil_div(request.prompt_tokens, config.tokens_per_block);
        // Each branch grows its own decode blocks and copies at most the shared
        // tail block of the prompt on its first write.
        const std::uint32_t branch_blocks =
            ceil_div(request.decode_budget, config.tokens_per_block) + 2;
        worst_request_blocks =
            std::max(worst_request_blocks, prompt_blocks + branches * branch_blocks);
    }
    return worst_request_blocks * std::max<std::uint32_t>(1, config.max_active_requests);
}

std::optional<Trace> load_trace(const std::string& path, std::string& error) {
    std::ifstream input(path);
    if (!input) {
        error = "cannot open " + path;
        return std::nullopt;
    }

    Trace trace;
    bool saw_header = false;
    std::string line;
    std::size_t line_number = 0;

    while (std::getline(input, line)) {
        ++line_number;
        if (line.empty()) {
            continue;
        }

        nlohmann::json record;
        try {
            record = nlohmann::json::parse(line);
        } catch (const nlohmann::json::exception& parse_error) {
            error = path + ":" + std::to_string(line_number) + ": " + parse_error.what();
            return std::nullopt;
        }

        try {
            const std::string kind = record.value("record", "");
            if (kind == "header") {
                const int version = record.at("schema_version").get<int>();
                if (version != kSupportedSchemaVersion) {
                    error = path + ": unsupported schema_version " + std::to_string(version);
                    return std::nullopt;
                }
                trace.header = header_from_json(record);
                saw_header = true;
            } else if (kind == "request") {
                trace.requests.push_back(request_from_json(record));
            } else {
                error = path + ":" + std::to_string(line_number) + ": unknown record '" + kind + "'";
                return std::nullopt;
            }
        } catch (const nlohmann::json::exception& field_error) {
            error = path + ":" + std::to_string(line_number) + ": " + field_error.what();
            return std::nullopt;
        }
    }

    if (!saw_header) {
        error = path + ": missing header record";
        return std::nullopt;
    }
    return trace;
}

ReplayMetrics run_paged(const Trace& trace, const ReplayConfig& config) {
    ReplayMetrics metrics;
    metrics.mode = "paged";

    const std::uint32_t tokens_per_block = config.tokens_per_block;

    AllocatorConfig allocator_config;
    allocator_config.block_shape = BlockShape{
        tokens_per_block,
        trace.header.num_layers,
        trace.header.num_kv_heads,
        trace.header.head_dim,
        trace.header.bytes_per_element};
    allocator_config.max_blocks = config.max_blocks;

    HostSwapBackend swap(config.swap_slots);
    MemoryAllocator allocator(allocator_config);
    if (config.swap_slots > 0) {
        allocator.set_swap_backend(&swap);
    }
    KVCacheManager cache(allocator);
    Scheduler scheduler(
        SchedulerConfig{
            config.max_active_requests,
            config.max_batch_tokens,
            config.preemption_watermark_blocks},
        cache,
        allocator);

    const std::size_t block_bytes = allocator_config.block_shape.byte_size();

    std::vector<RequestProgress> progress(trace.requests.size());
    std::unordered_map<RequestId, std::size_t> index_of;
    SequenceId next_fork_sequence = 1;

    for (std::size_t i = 0; i < trace.requests.size(); ++i) {
        const TraceRequest& request = trace.requests[i];
        progress[i].request = &request;
        index_of[request.request_id] = i;
        next_fork_sequence = std::max<SequenceId>(next_fork_sequence, request.request_id + 1);

        if (ceil_div(request.prompt_tokens, tokens_per_block) > config.max_blocks ||
            request.prompt_tokens > config.max_batch_tokens) {
            ++metrics.requests_never_admissible;
        }

        Request scheduled;
        scheduled.request_id = request.request_id;
        scheduled.root_sequence_id = request.request_id;
        scheduled.prompt_tokens = request.prompt_tokens;
        scheduled.sampling.num_parallel_samples = request.num_parallel_samples;
        scheduled.sampling.max_decode_tokens = request.decode_budget;
        scheduled.multimodal_spans = request.multimodal_spans;
        scheduled.positional_encoding = request.positional_encoding;
        scheduler.enqueue(std::move(scheduled));
    }

    std::size_t waiting = trace.requests.size();
    std::size_t completed = 0;
    std::vector<RequestId> live;                  // admitted and unfinished, in admission order
    std::unordered_set<RequestId> preempted_ids;  // parked as of the previous step
    std::uint32_t no_progress_steps = 0;

    while (completed < trace.requests.size() && metrics.steps < config.max_steps) {
        const BatchPlan plan = scheduler.schedule_next();
        ++metrics.steps;
        bool progress_made = false;

        std::unordered_set<RequestId> in_plan;
        in_plan.insert(plan.prefill_requests.begin(), plan.prefill_requests.end());
        in_plan.insert(plan.decode_requests.begin(), plan.decode_requests.end());

        // Admission checks that a prompt fits, not that the prompt plus its
        // decode budget fits, so a batch can outgrow the pool mid-decode. The
        // standard recovery is to park the newest active request, matching the
        // scheduler's own watermark policy: the oldest is closest to finishing,
        // so rolling it back throws away the most work.
        //
        // Returns whether frames actually came back, not merely whether a request
        // changed state. Without a swap backend a preemption reclaims nothing, and
        // calling that progress would spin instead of reporting the stall.
        const auto reclaim_for_decode = [&]() {
            const std::uint32_t free_before = allocator.stats().free_blocks;
            for (auto it = live.rbegin(); it != live.rend(); ++it) {
                if (progress[index_of.at(*it)].finished || in_plan.count(*it) == 0) {
                    continue;
                }
                if (scheduler.preempt(*it, "decode growth failed")) {
                    return allocator.stats().free_blocks > free_before;
                }
            }
            return false;
        };

        for (const RequestId id : plan.prefill_requests) {
            RequestProgress& state = progress[index_of.at(id)];
            const TraceRequest& request = *state.request;

            state.admitted = true;
            state.sequences.push_back(request.request_id);
            ++metrics.requests_admitted;
            --waiting;
            live.push_back(id);

            for (std::uint32_t block = 0; block * tokens_per_block < request.prompt_tokens; ++block) {
                touch_position(cache, request.request_id, block * tokens_per_block, tokens_per_block, metrics);
            }

            // Parallel sampling branches share the prompt until one of them writes.
            for (std::uint32_t branch = 1; branch < request.num_parallel_samples; ++branch) {
                SequenceMetadata child;
                child.sequence_id = next_fork_sequence++;
                child.request_id = request.request_id;
                child.multimodal_spans = request.multimodal_spans;
                child.positional_encoding = request.positional_encoding;
                if (cache.fork_sequence(request.request_id, child)) {
                    state.sequences.push_back(child.sequence_id);
                    ++metrics.forks;
                }
            }

            state.committed = request.prompt_tokens;
            scheduler.complete_step(id, request.prompt_tokens);
            progress_made = true;
        }

        bool growth_failed = false;
        for (const RequestId id : plan.decode_requests) {
            RequestProgress& state = progress[index_of.at(id)];
            if (scheduler.preemption_info(id).has_value()) {
                // Parked by a reclaim earlier in this same loop. Its blocks are in
                // swap, so it cannot be stepped until it is resumed.
                continue;
            }
            const TokenPosition position = state.committed;

            bool grew = true;
            for (const SequenceId sequence : state.sequences) {
                if (!ensure_capacity(cache, sequence, position, tokens_per_block)) {
                    grew = false;
                    break;
                }
            }
            if (!grew) {
                // The pool is exhausted. Park a request to reclaim frames and
                // retry this one on a later step.
                growth_failed = true;
                if (reclaim_for_decode()) {
                    progress_made = true;
                }
                continue;
            }

            for (const SequenceId sequence : state.sequences) {
                touch_position(cache, sequence, position, tokens_per_block, metrics);
            }

            ++state.committed;
            ++state.decoded;
            metrics.tokens_decoded += state.sequences.size();
            scheduler.complete_step(id, 1);
            progress_made = true;

            if (state.decoded >= state.request->decode_actual) {
                // The scheduler releases the root sequence; forks are the driver's.
                for (std::size_t branch = 1; branch < state.sequences.size(); ++branch) {
                    cache.release_sequence(state.sequences[branch]);
                }
                scheduler.cancel(id);
                state.finished = true;
                state.sequences.clear();
                ++completed;
            }
        }

        // `preemption_info` exists exactly while a request is parked, so a
        // presence diff counts entries into that state whatever caused them:
        // the watermark policy, or the decode-failure path above.
        for (const RequestId id : live) {
            if (progress[index_of.at(id)].finished) {
                continue;
            }
            const std::optional<PreemptionInfo> info = scheduler.preemption_info(id);
            if (!info.has_value()) {
                preempted_ids.erase(id);
            } else if (preempted_ids.insert(id).second) {
                ++metrics.preemptions;
                metrics.swapped_blocks_out += info->swapped_blocks;
            }
        }

        // Resuming in the same step that reclaimed frames would immediately undo
        // the reclaim, so only try while the pool is not under acute pressure.
        // `live` is in admission order, which keeps resume order deterministic.
        if (!growth_failed) {
            for (const RequestId id : live) {
                if (preempted_ids.count(id) == 0) {
                    continue;
                }
                // A resume is not progress: it is the reverse of reclaiming. The
                // decode it enables on the next step is what counts.
                if (scheduler.resume(id)) {
                    preempted_ids.erase(id);
                    ++metrics.resumes;
                } else {
                    ++metrics.failed_resumes;
                }
            }
        }

        live.erase(
            std::remove_if(
                live.begin(),
                live.end(),
                [&](RequestId id) { return progress[index_of.at(id)].finished; }),
            live.end());

        if (waiting > 0 && plan.prefill_requests.empty()) {
            ++metrics.head_of_line_stall_steps;
        }

        metrics.peak_active_requests =
            std::max(metrics.peak_active_requests, static_cast<std::uint32_t>(plan.decode_requests.size()));

        const AllocatorStats stats = allocator.stats();
        const std::uint32_t blocks_in_use = stats.total_blocks - stats.free_blocks;
        metrics.peak_blocks_in_use = std::max(metrics.peak_blocks_in_use, blocks_in_use);
        metrics.peak_bytes_in_use =
            std::max(metrics.peak_bytes_in_use, static_cast<std::size_t>(blocks_in_use) * block_bytes);
        metrics.reserved_token_slots += static_cast<std::uint64_t>(blocks_in_use) * tokens_per_block;

        for (const RequestId id : live) {
            const RequestProgress& state = progress[index_of.at(id)];
            if (preempted_ids.count(id) != 0) {
                // Parked: its tokens live in swap, so they back none of the
                // blocks counted in the denominator above.
                continue;
            }
            for (const SequenceId sequence : state.sequences) {
                metrics.committed_tokens += state.committed;
                const std::uint32_t reserved = reserved_slots(cache, sequence, tokens_per_block);
                if (reserved > state.committed) {
                    metrics.max_partial_block_slack =
                        std::max(metrics.max_partial_block_slack, reserved - state.committed);
                }
            }
        }

        if (progress_made) {
            no_progress_steps = 0;
        } else if (++no_progress_steps >= 2) {
            // Nothing changed and nothing can: the trace is permanently stalled.
            break;
        }
    }

    metrics.requests_completed = static_cast<std::uint32_t>(completed);
    metrics.completed = completed == trace.requests.size();

    const AllocatorStats final_stats = allocator.stats();
    metrics.leak_free = final_stats.free_blocks == final_stats.total_blocks &&
        final_stats.active_blocks == 0 && final_stats.shared_blocks == 0 &&
        final_stats.swapped_blocks == 0;
    return metrics;
}

ReplayMetrics run_contiguous(const Trace& trace, const ReplayConfig& config) {
    ReplayMetrics metrics;
    metrics.mode = "contiguous";

    const std::uint32_t tokens_per_block = config.tokens_per_block;
    const std::size_t block_bytes = BlockShape{
        tokens_per_block,
        trace.header.num_layers,
        trace.header.num_kv_heads,
        trace.header.head_dim,
        trace.header.bytes_per_element}
                                        .byte_size();

    struct Reservation {
        std::size_t index{0};
        std::uint32_t start_block{0};
        std::uint32_t block_count{0};
        std::uint32_t committed{0};
        std::uint32_t decoded{0};
    };

    // One contiguous run per request, sized to prompt plus the request's own
    // declared max_new_tokens. Run lengths therefore vary, which is what makes
    // external fragmentation possible at all.
    std::vector<std::uint32_t> request_blocks(trace.requests.size(), 0);
    std::vector<bool> servable(trace.requests.size(), true);
    for (std::size_t i = 0; i < trace.requests.size(); ++i) {
        const TraceRequest& request = trace.requests[i];
        const std::uint32_t reserved_tokens = std::min(
            config.baseline_max_context,
            request.prompt_tokens + request.decode_budget);
        request_blocks[i] = ceil_div(reserved_tokens, tokens_per_block);

        // A request that cannot fit even an empty pool can never be served.
        // Skipping it is charity to the baseline: the alternative is a permanent
        // head-of-line stall that would end the run early.
        if (request.prompt_tokens + request.decode_budget > config.baseline_max_context ||
            request_blocks[i] > config.max_blocks) {
            servable[i] = false;
            ++metrics.requests_never_admissible;
        }
    }

    std::vector<bool> used(config.max_blocks, false);
    std::uint32_t blocks_in_use = 0;
    std::vector<Reservation> active;
    std::size_t next_pending = 0;
    std::size_t completed = 0;

    const auto first_fit = [&](std::uint32_t needed, std::uint32_t& start) {
        std::uint32_t run = 0;
        for (std::uint32_t block = 0; block < config.max_blocks; ++block) {
            run = used[block] ? 0 : run + 1;
            if (run == needed) {
                start = block + 1 - needed;
                return true;
            }
        }
        return false;
    };

    while (completed < trace.requests.size() && metrics.steps < config.max_steps) {
        ++metrics.steps;
        bool progress_made = false;
        const std::size_t admitted_before = next_pending;

        while (next_pending < trace.requests.size() && active.size() < config.max_active_requests) {
            if (!servable[next_pending]) {
                ++next_pending;
                ++completed;
                progress_made = true;
                continue;
            }

            const std::uint32_t needed = request_blocks[next_pending];
            std::uint32_t start = 0;
            if (!first_fit(needed, start)) {
                if (config.max_blocks - blocks_in_use >= needed) {
                    // Enough free frames exist, just not as one run.
                    ++metrics.fragmentation_failures;
                }
                break;
            }

            for (std::uint32_t block = 0; block < needed; ++block) {
                used[start + block] = true;
            }
            blocks_in_use += needed;
            active.push_back(
                Reservation{next_pending, start, needed, trace.requests[next_pending].prompt_tokens, 0});
            ++metrics.requests_admitted;
            ++next_pending;
            progress_made = true;
        }

        for (std::size_t i = 0; i < active.size();) {
            Reservation& reservation = active[i];
            const TraceRequest& request = trace.requests[reservation.index];
            ++reservation.committed;
            ++reservation.decoded;
            ++metrics.tokens_decoded;
            progress_made = true;

            if (reservation.decoded >= request.decode_actual) {
                for (std::uint32_t block = 0; block < reservation.block_count; ++block) {
                    used[reservation.start_block + block] = false;
                }
                blocks_in_use -= reservation.block_count;
                active[i] = active.back();
                active.pop_back();
                ++completed;
            } else {
                ++i;
            }
        }

        if (next_pending < trace.requests.size() && admitted_before == next_pending) {
            ++metrics.head_of_line_stall_steps;
        }

        metrics.peak_active_requests =
            std::max(metrics.peak_active_requests, static_cast<std::uint32_t>(active.size()));
        metrics.peak_blocks_in_use = std::max(metrics.peak_blocks_in_use, blocks_in_use);
        metrics.peak_bytes_in_use =
            std::max(metrics.peak_bytes_in_use, static_cast<std::size_t>(blocks_in_use) * block_bytes);
        metrics.reserved_token_slots += static_cast<std::uint64_t>(blocks_in_use) * tokens_per_block;
        for (const Reservation& reservation : active) {
            metrics.committed_tokens += reservation.committed;
            const std::uint32_t reserved = reservation.block_count * tokens_per_block;
            if (reserved > reservation.committed) {
                metrics.max_partial_block_slack =
                    std::max(metrics.max_partial_block_slack, reserved - reservation.committed);
            }
        }

        if (!progress_made) {
            break;
        }
    }

    metrics.requests_completed = static_cast<std::uint32_t>(completed);
    metrics.completed = completed == trace.requests.size();
    metrics.leak_free = blocks_in_use == 0;
    return metrics;
}

std::string csv_header() {
    return "trace_mix,pixel_budget,dimension_source,mode,pool_blocks,tokens_per_block,"
           "baseline_max_context,preemption_watermark,requests,requests_admitted,"
           "requests_completed,requests_never_admissible,steps,tokens_decoded,"
           "peak_active_requests,peak_blocks_in_use,peak_bytes_in_use,utilization,"
           "max_partial_block_slack,"
           "preemptions,resumes,failed_resumes,swapped_blocks_out,forks,cow_events,"
           "head_of_line_stall_steps,fragmentation_failures,completed,leak_free";
}

std::string to_csv_row(const Trace& trace, const ReplayConfig& config, const ReplayMetrics& metrics) {
    std::ostringstream row;
    row << trace.header.mix << ',' << trace.header.pixel_budget << ','
        << trace.header.dimension_source << ',' << metrics.mode << ',' << config.max_blocks << ','
        << config.tokens_per_block << ',' << config.baseline_max_context << ','
        << config.preemption_watermark_blocks << ',' << trace.requests.size() << ','
        << metrics.requests_admitted << ',' << metrics.requests_completed << ','
        << metrics.requests_never_admissible << ',' << metrics.steps << ','
        << metrics.tokens_decoded << ',' << metrics.peak_active_requests << ','
        << metrics.peak_blocks_in_use << ','
        << metrics.peak_bytes_in_use << ',' << std::fixed << std::setprecision(6)
        << metrics.utilization() << ',' << metrics.max_partial_block_slack << ','
        << metrics.preemptions << ',' << metrics.resumes << ',' << metrics.failed_resumes << ','
        << metrics.swapped_blocks_out << ',' << metrics.forks << ',' << metrics.cow_events << ','
        << metrics.head_of_line_stall_steps << ',' << metrics.fragmentation_failures << ','
        << (metrics.completed ? 1 : 0) << ',' << (metrics.leak_free ? 1 : 0);
    return row.str();
}

} // namespace qwenvl_paged::replay
