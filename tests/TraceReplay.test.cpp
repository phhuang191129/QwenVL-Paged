#include "TraceReplay.h"

#include <gtest/gtest.h>

#include <cstdio>
#include <fstream>
#include <string>

namespace qwenvl_paged::replay {
namespace {

std::string trace_path(const std::string& name) {
    return std::string(QWENVL_TRACE_DIR) + "/" + name;
}

Trace load_or_fail(const std::string& name) {
    std::string error;
    std::optional<Trace> trace = load_trace(trace_path(name), error);
    if (!trace.has_value()) {
        ADD_FAILURE() << "failed to load " << name << ": " << error
                      << " (run tools/trace_gen/generate_all.sh)";
        return Trace{};
    }
    return *trace;
}

/**
 * @brief Pool sized from the trace so the paged batch never exhausts mid-decode.
 *
 * The same pool leaves the contiguous baseline heavily constrained, which is the
 * comparison: identical memory, different addressing.
 */
ReplayConfig config_for(const Trace& trace) {
    ReplayConfig config;
    config.max_active_requests = 4;
    config.max_batch_tokens = 65536;
    config.baseline_max_context = 8192;
    config.max_blocks = suggested_pool_blocks(trace, config);
    return config;
}

TEST(TraceReplay, CarriesRealQwen3VlGeometry) {
    const Trace trace = load_or_fail("bimodal.jsonl");
    ASSERT_FALSE(trace.requests.empty());

    EXPECT_EQ(trace.header.num_layers, 28U);
    EXPECT_EQ(trace.header.num_kv_heads, 8U);
    EXPECT_EQ(trace.header.head_dim, 128U);
    EXPECT_EQ(trace.header.bytes_per_element, 2U);
    EXPECT_EQ(trace.requests.size(), trace.header.num_requests);

    // Every image request must carry both the M-RoPE spans and the multimodal
    // feature spans the cache manager is supposed to pass through untouched.
    std::size_t image_requests = 0;
    for (const TraceRequest& request : trace.requests) {
        if (request.images.empty()) {
            continue;
        }
        ++image_requests;
        EXPECT_EQ(request.multimodal_spans.size(), request.images.size());

        std::uint32_t spanned = 0;
        for (const PositionalEncodingSpan& span : request.positional_encoding.spans) {
            EXPECT_EQ(span.start_token, spanned);
            spanned += span.token_count;
        }
        EXPECT_EQ(spanned, request.prompt_tokens);

        for (std::size_t i = 0; i < request.images.size(); ++i) {
            EXPECT_EQ(request.multimodal_spans[i].token_count, request.images[i].visual_tokens);
        }
    }
    EXPECT_GT(image_requests, 0U);
}

TEST(TraceReplay, ReplayIsDeterministic) {
    const Trace trace = load_or_fail("bimodal.jsonl");
    ASSERT_FALSE(trace.requests.empty());
    const ReplayConfig config = config_for(trace);

    const std::string first = to_csv_row(trace, config, run_paged(trace, config));
    const std::string second = to_csv_row(trace, config, run_paged(trace, config));
    EXPECT_EQ(first, second);

    const std::string baseline_first = to_csv_row(trace, config, run_contiguous(trace, config));
    const std::string baseline_second = to_csv_row(trace, config, run_contiguous(trace, config));
    EXPECT_EQ(baseline_first, baseline_second);
}

TEST(TraceReplay, LeavesNoBlocksBehindAfterEveryRequestFinishes) {
    const Trace trace = load_or_fail("bimodal.jsonl");
    ASSERT_FALSE(trace.requests.empty());
    const ReplayConfig config = config_for(trace);

    const ReplayMetrics metrics = run_paged(trace, config);
    EXPECT_TRUE(metrics.completed) << "trace stalled after " << metrics.requests_completed
                                   << " of " << trace.requests.size() << " requests";
    EXPECT_EQ(metrics.requests_completed, trace.requests.size());
    EXPECT_TRUE(metrics.leak_free);
}

TEST(TraceReplay, InternalWasteIsBoundedByOnePartialBlockPerSequence) {
    const Trace trace = load_or_fail("image-heavy.jsonl");
    ASSERT_FALSE(trace.requests.empty());
    const ReplayConfig config = config_for(trace);

    const ReplayMetrics metrics = run_paged(trace, config);
    ASSERT_TRUE(metrics.completed);
    // Reserved capacity may exceed committed length by at most the tail block.
    EXPECT_LT(metrics.max_partial_block_slack, config.tokens_per_block);
}

TEST(TraceReplay, PagedUtilizationBeatsContiguousBaseline) {
    for (const char* name : {"bimodal.jsonl", "image-heavy.jsonl", "text-only.jsonl"}) {
        const Trace trace = load_or_fail(name);
        ASSERT_FALSE(trace.requests.empty()) << name;
        const ReplayConfig config = config_for(trace);

        const ReplayMetrics paged = run_paged(trace, config);
        const ReplayMetrics contiguous = run_contiguous(trace, config);

        EXPECT_GT(paged.utilization(), contiguous.utilization())
            << name << ": paged " << paged.utilization() << " vs contiguous "
            << contiguous.utilization();
    }
}

TEST(TraceReplay, PagedHoldsTheSameBatchInFewerBlocks) {
    const Trace trace = load_or_fail("image-heavy.jsonl");
    ASSERT_FALSE(trace.requests.empty());
    const ReplayConfig config = config_for(trace);

    const ReplayMetrics paged = run_paged(trace, config);
    const ReplayMetrics contiguous = run_contiguous(trace, config);
    ASSERT_TRUE(paged.completed);
    ASSERT_TRUE(contiguous.completed);

    // Both sustain the same batch here, because the pool is sized for the paged
    // worst case and so leaves the baseline room too. The difference is what the
    // batch costs: the baseline reserves for max_new_tokens and cannot reclaim
    // the tokens a request never produces.
    ASSERT_EQ(paged.peak_active_requests, contiguous.peak_active_requests);
    EXPECT_LT(paged.peak_blocks_in_use, contiguous.peak_blocks_in_use);
}

TEST(TraceReplay, ForkingSharesPromptBlocksAcrossSamplingBranches) {
    const Trace trace = load_or_fail("image-heavy-parallel4.jsonl");
    ASSERT_FALSE(trace.requests.empty());

    ReplayConfig config;
    config.max_active_requests = 2;
    config.max_batch_tokens = 65536;
    config.max_blocks = suggested_pool_blocks(trace, config);

    const ReplayMetrics metrics = run_paged(trace, config);
    ASSERT_TRUE(metrics.completed);
    EXPECT_GT(metrics.forks, 0U);
    // Branches share every prompt block, so utilization exceeds one committed
    // token per allocated slot.
    EXPECT_GT(metrics.utilization(), 1.0);
    EXPECT_TRUE(metrics.leak_free);
}

TEST(TraceReplay, ReportsStallInsteadOfHangingWhenAPromptCannotFit) {
    const Trace trace = load_or_fail("bimodal-default-budget.jsonl");
    ASSERT_FALSE(trace.requests.empty());

    ReplayConfig config = config_for(trace);
    config.max_blocks = 64;  // far too small for a default-budget image prompt

    const ReplayMetrics metrics = run_paged(trace, config);
    EXPECT_FALSE(metrics.completed);
    EXPECT_GT(metrics.requests_never_admissible, 0U);
    EXPECT_LT(metrics.steps, config.max_steps);
}

TEST(TraceReplay, RejectsUnknownSchemaVersion) {
    const std::string path = std::string(QWENVL_TRACE_DIR) + "/.bad-schema.jsonl";
    {
        std::ofstream out(path);
        out << R"({"record":"header","schema_version":99})" << "\n";
    }

    std::string error;
    EXPECT_FALSE(load_trace(path, error).has_value());
    EXPECT_NE(error.find("schema_version"), std::string::npos);
    std::remove(path.c_str());
}

TEST(TraceReplay, RejectsTraceWithoutHeader) {
    const std::string path = std::string(QWENVL_TRACE_DIR) + "/.no-header.jsonl";
    {
        std::ofstream out(path);
        out << R"({"record":"request","request_id":1})" << "\n";
    }

    std::string error;
    EXPECT_FALSE(load_trace(path, error).has_value());
    std::remove(path.c_str());
}

} // namespace
} // namespace qwenvl_paged::replay
