/**
 * @brief End-to-end specification test for prefill plus decode.
 *
 * This exercises the whole stack the way an engine event loop would: the
 * scheduler admits several requests, the cache manager reserves and grows their
 * block tables, KV is written through the copy-on-write path, and the reference
 * PagedAttention kernel reads it back through a CacheView. Every attention
 * result is checked against a flat contiguous reference, so any mistake in the
 * logical-to-physical translation shows up as a numeric mismatch.
 *
 * Until the Week 8 members are implemented this binary is expected to fail at
 * the LINK stage, which is the intended red state for this phase.
 */

#include "qwenvl_paged/CacheLayout.h"
#include "qwenvl_paged/KVCacheManager.h"
#include "qwenvl_paged/MemoryAllocator.h"
#include "qwenvl_paged/PagedAttention.h"
#include "qwenvl_paged/Scheduler.h"
#include "qwenvl_paged/SwapBackend.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <optional>
#include <vector>

namespace qwenvl_paged {
namespace {

constexpr std::uint32_t kTokensPerBlock = 4;
constexpr std::uint32_t kNumLayers = 2;
constexpr std::uint32_t kNumKvHeads = 2;
constexpr std::uint32_t kHeadDim = 4;
constexpr std::uint32_t kNumQueryHeads = 4; // grouped-query: two query heads per kv head
constexpr std::uint32_t kPromptTokens = 6;  // spans two blocks, second one partial
constexpr std::uint32_t kDecodeTokens = 3;  // the last one forces a new block
constexpr std::uint32_t kMaxBlocks = 16;
constexpr float kScale = 0.5F;
constexpr float kTolerance = 1e-5F;

BlockShape make_shape() {
    BlockShape shape;
    shape.tokens_per_block = kTokensPerBlock;
    shape.num_layers = kNumLayers;
    shape.num_kv_heads = kNumKvHeads;
    shape.head_dim = kHeadDim;
    shape.bytes_per_element = static_cast<std::uint32_t>(sizeof(float));
    return shape;
}

/**
 * @brief Per-sequence deterministic cache contents in [-1, 1).
 *
 * Mixing the sequence id in means a kernel that strayed into another request's
 * blocks would produce a visibly wrong result rather than a plausible one.
 */
float pattern(
    SequenceId sequence_id,
    std::uint32_t layer,
    KVStream stream,
    std::uint32_t token,
    std::uint32_t head,
    std::uint32_t dim) {
    const std::uint32_t stream_id = (stream == KVStream::Key) ? 0U : 1U;
    std::uint32_t mixed = 2166136261U;
    for (const std::uint32_t part :
         {static_cast<std::uint32_t>(sequence_id), layer, stream_id, token, head, dim}) {
        mixed = (mixed ^ part) * 16777619U;
    }
    return static_cast<float>(mixed % 2000U) / 1000.0F - 1.0F;
}

std::vector<float> make_query(SequenceId sequence_id, TokenPosition position) {
    std::vector<float> query(static_cast<std::size_t>(kNumQueryHeads) * kHeadDim);
    for (std::size_t i = 0; i < query.size(); ++i) {
        query[i] = pattern(sequence_id, 0, KVStream::Key, position, 0, static_cast<std::uint32_t>(i));
    }
    return query;
}

/**
 * @brief Textbook attention over the same values, read from a flat generator.
 */
std::vector<float> reference_attention(
    SequenceId sequence_id,
    const std::vector<float>& query,
    std::uint32_t layer,
    std::uint32_t context_len) {
    constexpr std::uint32_t kGroupSize = kNumQueryHeads / kNumKvHeads;

    std::vector<float> out(static_cast<std::size_t>(kNumQueryHeads) * kHeadDim, 0.0F);
    for (std::uint32_t q_head = 0; q_head < kNumQueryHeads; ++q_head) {
        const std::uint32_t kv_head = q_head / kGroupSize;
        const float* q = query.data() + static_cast<std::size_t>(q_head) * kHeadDim;

        std::vector<float> scores(context_len, 0.0F);
        float max_score = -std::numeric_limits<float>::infinity();
        for (std::uint32_t token = 0; token < context_len; ++token) {
            float dot = 0.0F;
            for (std::uint32_t dim = 0; dim < kHeadDim; ++dim) {
                dot += q[dim] * pattern(sequence_id, layer, KVStream::Key, token, kv_head, dim);
            }
            scores[token] = dot * kScale;
            max_score = std::max(max_score, scores[token]);
        }

        float denominator = 0.0F;
        float* accumulator = out.data() + static_cast<std::size_t>(q_head) * kHeadDim;
        for (std::uint32_t token = 0; token < context_len; ++token) {
            const float weight = std::exp(scores[token] - max_score);
            denominator += weight;
            for (std::uint32_t dim = 0; dim < kHeadDim; ++dim) {
                accumulator[dim] +=
                    weight * pattern(sequence_id, layer, KVStream::Value, token, kv_head, dim);
            }
        }

        for (std::uint32_t dim = 0; dim < kHeadDim; ++dim) {
            accumulator[dim] /= denominator;
        }
    }
    return out;
}

class EndToEndTest : public ::testing::Test {
protected:
    EndToEndTest()
        : swap_(kMaxBlocks), allocator_(make_allocator_config()), cache_(allocator_),
          scheduler_(make_scheduler_config(), cache_, allocator_) {
        allocator_.set_swap_backend(&swap_);
        layout_.shape = make_shape();
    }

    static AllocatorConfig make_allocator_config() {
        AllocatorConfig config;
        config.block_shape = make_shape();
        config.max_blocks = kMaxBlocks;
        return config;
    }

    static SchedulerConfig make_scheduler_config() {
        SchedulerConfig config;
        config.max_active_requests = 4;
        config.max_batch_tokens = 64;
        return config;
    }

    /**
     * @brief Writes every layer, stream, and head of one token position.
     */
    void write_token(SequenceId sequence_id, TokenPosition position) {
        const std::optional<PhysicalBlockId> physical =
            cache_.ensure_token_writable(sequence_id, position);
        ASSERT_TRUE(physical.has_value()) << "sequence " << sequence_id << " token " << position;

        PhysicalBlock* block = allocator_.block(*physical);
        ASSERT_NE(block, nullptr);
        float* base = reinterpret_cast<float*>(block->data());

        for (std::uint32_t layer = 0; layer < kNumLayers; ++layer) {
            for (const KVStream stream : {KVStream::Key, KVStream::Value}) {
                for (std::uint32_t head = 0; head < kNumKvHeads; ++head) {
                    const std::optional<std::size_t> offset = layout_.element_offset(
                        layer, stream, position % kTokensPerBlock, head);
                    ASSERT_TRUE(offset.has_value());
                    for (std::uint32_t dim = 0; dim < kHeadDim; ++dim) {
                        base[*offset + dim] =
                            pattern(sequence_id, layer, stream, position, head, dim);
                    }
                }
            }
        }
    }

    /**
     * @brief Runs attention at one position and checks it against the reference.
     */
    void check_attention(SequenceId sequence_id, TokenPosition position) {
        const std::optional<CacheView> view = cache_.cache_view(sequence_id);
        ASSERT_TRUE(view.has_value());

        const std::vector<float> query = make_query(sequence_id, position);
        const std::uint32_t context_len = position + 1;

        for (std::uint32_t layer = 0; layer < kNumLayers; ++layer) {
            PagedAttentionParams params;
            params.layer = layer;
            params.num_query_heads = kNumQueryHeads;
            params.context_len = context_len;
            params.scale = kScale;

            std::vector<float> out(query.size(), 0.0F);
            ASSERT_TRUE(paged_attention_decode<float>(*view, query.data(), params, out.data()))
                << "sequence " << sequence_id << " position " << position << " layer " << layer;

            const std::vector<float> expected =
                reference_attention(sequence_id, query, layer, context_len);
            for (std::size_t i = 0; i < out.size(); ++i) {
                EXPECT_NEAR(out[i], expected[i], kTolerance)
                    << "sequence " << sequence_id << " position " << position << " layer " << layer
                    << " element " << i;
            }
        }
    }

    /**
     * @brief Grows the block table when the next token falls past the last block.
     */
    void ensure_capacity(SequenceId sequence_id, TokenPosition position) {
        const std::optional<CacheView> view = cache_.cache_view(sequence_id);
        ASSERT_TRUE(view.has_value());
        if (position / kTokensPerBlock >= view->block_table->size()) {
            ASSERT_TRUE(cache_.reserve_tokens(sequence_id, 1));
        }
    }

    HostSwapBackend swap_;
    MemoryAllocator allocator_;
    KVCacheManager cache_;
    Scheduler scheduler_;
    KVBlockLayout layout_{};
};

TEST_F(EndToEndTest, PrefillThenDecodeAcrossMultipleActiveRequests) {
    const std::vector<RequestId> request_ids{1, 2, 3};

    for (const RequestId id : request_ids) {
        Request request;
        request.request_id = id;
        request.root_sequence_id = id;
        request.prompt_tokens = kPromptTokens;
        request.sampling.max_decode_tokens = kDecodeTokens;
        scheduler_.enqueue(request);
    }

    // --- Prefill --------------------------------------------------------
    const BatchPlan prefill_plan = scheduler_.schedule_next();
    ASSERT_EQ(prefill_plan.prefill_requests.size(), request_ids.size());
    EXPECT_TRUE(prefill_plan.decode_requests.empty());
    EXPECT_EQ(prefill_plan.scheduled_tokens, kPromptTokens * request_ids.size());

    for (const RequestId id : prefill_plan.prefill_requests) {
        for (TokenPosition position = 0; position < kPromptTokens; ++position) {
            ASSERT_NO_FATAL_FAILURE(write_token(id, position));
        }
        // Causal prefill is the decode kernel applied per prompt position with a
        // growing context, so position p only ever sees tokens 0..p.
        for (TokenPosition position = 0; position < kPromptTokens; ++position) {
            ASSERT_NO_FATAL_FAILURE(check_attention(id, position));
        }
        scheduler_.complete_step(id, 1);
        EXPECT_EQ(scheduler_.state(id), RequestState::Decode);
    }

    // --- Decode ---------------------------------------------------------
    for (std::uint32_t step = 0; step < kDecodeTokens; ++step) {
        const BatchPlan plan = scheduler_.schedule_next();
        ASSERT_EQ(plan.decode_requests.size(), request_ids.size());
        EXPECT_TRUE(plan.prefill_requests.empty());

        const TokenPosition position = kPromptTokens + step;
        for (const RequestId id : plan.decode_requests) {
            ASSERT_NO_FATAL_FAILURE(ensure_capacity(id, position));
            ASSERT_NO_FATAL_FAILURE(write_token(id, position));
            ASSERT_NO_FATAL_FAILURE(check_attention(id, position));
            scheduler_.complete_step(id, 1);
        }
    }

    // Interleaved admission plus mid-decode growth means at least one request
    // ends up holding frames that are not one contiguous run, which is the case
    // the paged kernel exists to handle.
    bool any_scattered = false;
    for (const RequestId id : request_ids) {
        const std::optional<CacheView> view = cache_.cache_view(id);
        ASSERT_TRUE(view.has_value());
        const std::vector<BlockTableEntry>& entries = view->block_table->entries();
        ASSERT_GE(entries.size(), 2u);

        PhysicalBlockId lowest = entries.front().physical_id;
        PhysicalBlockId highest = entries.front().physical_id;
        for (const BlockTableEntry& entry : entries) {
            lowest = std::min(lowest, entry.physical_id);
            highest = std::max(highest, entry.physical_id);
        }
        if (highest - lowest + 1 != entries.size()) {
            any_scattered = true;
        }
    }
    EXPECT_TRUE(any_scattered);

    // --- Teardown -------------------------------------------------------
    for (const RequestId id : request_ids) {
        scheduler_.cancel(id);
        EXPECT_FALSE(cache_.contains(id));
    }

    const AllocatorStats stats = allocator_.stats();
    EXPECT_EQ(stats.free_blocks, stats.total_blocks);
    EXPECT_EQ(stats.active_blocks, 0u);
    EXPECT_EQ(stats.shared_blocks, 0u);
    EXPECT_EQ(stats.swapped_blocks, 0u);
}

TEST_F(EndToEndTest, PreemptedRequestResumesWithIntactCache) {
    Request request;
    request.request_id = 1;
    request.root_sequence_id = 1;
    request.prompt_tokens = kPromptTokens;
    request.sampling.max_decode_tokens = kDecodeTokens;
    scheduler_.enqueue(request);

    ASSERT_EQ(scheduler_.schedule_next().prefill_requests.size(), 1u);
    for (TokenPosition position = 0; position < kPromptTokens; ++position) {
        ASSERT_NO_FATAL_FAILURE(write_token(1, position));
    }
    scheduler_.complete_step(1, 1);

    const TokenPosition last = kPromptTokens - 1;
    ASSERT_NO_FATAL_FAILURE(check_attention(1, last));

    ASSERT_TRUE(scheduler_.preempt(1, "cache pressure"));
    {
        // While swapped out the cache holds no frame, so the kernel must refuse
        // rather than read whatever now owns those frames.
        const std::optional<CacheView> view = cache_.cache_view(1);
        ASSERT_TRUE(view.has_value());
        const std::vector<float> query = make_query(1, last);
        std::vector<float> out(query.size(), 0.0F);
        PagedAttentionParams params;
        params.layer = 0;
        params.num_query_heads = kNumQueryHeads;
        params.context_len = last + 1;
        params.scale = kScale;
        EXPECT_FALSE(paged_attention_decode<float>(*view, query.data(), params, out.data()));
    }

    ASSERT_TRUE(scheduler_.resume(1));
    ASSERT_NO_FATAL_FAILURE(check_attention(1, last));

    scheduler_.cancel(1);
    const AllocatorStats stats = allocator_.stats();
    EXPECT_EQ(stats.free_blocks, stats.total_blocks);
    EXPECT_EQ(stats.swapped_blocks, 0u);
}

} // namespace
} // namespace qwenvl_paged
