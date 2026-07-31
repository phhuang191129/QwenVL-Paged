/**
 * @brief Specification tests for the reference CPU PagedAttention path.
 *
 * These tests are written against the documented header contract in
 * include/qwenvl_paged/PagedAttention.h. Until the kernel is implemented this
 * binary is expected to fail at the LINK stage with undefined references, which
 * is the intended red state for this phase.
 *
 * The tests are layered on purpose. CacheView.test.cpp already proves the
 * element layout is injective and that logical-to-physical translation is
 * correct, so the tests here focus on what only the kernel can get wrong:
 * reading cached tokens in logical order while their blocks are physically
 * scattered, honoring the grouped-query head mapping, and refusing to read a
 * cache it cannot fully resolve.
 */

#include "qwenvl_paged/CacheLayout.h"
#include "qwenvl_paged/KVCacheManager.h"
#include "qwenvl_paged/MemoryAllocator.h"
#include "qwenvl_paged/PagedAttention.h"
#include "qwenvl_paged/SwapBackend.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <optional>
#include <utility>
#include <vector>

namespace qwenvl_paged {
namespace {

constexpr float kTolerance = 1e-5F;

BlockShape make_shape(
    std::uint32_t tokens_per_block,
    std::uint32_t num_layers,
    std::uint32_t num_kv_heads,
    std::uint32_t head_dim) {
    BlockShape shape;
    shape.tokens_per_block = tokens_per_block;
    shape.num_layers = num_layers;
    shape.num_kv_heads = num_kv_heads;
    shape.head_dim = head_dim;
    shape.bytes_per_element = static_cast<std::uint32_t>(sizeof(float));
    return shape;
}

/**
 * @brief Allocator, swap space, and cache manager wired together for one test.
 */
struct Harness {
    Harness(BlockShape shape, std::uint32_t max_blocks)
        : swap(64), allocator(make_config(shape, max_blocks)), cache(allocator) {
        allocator.set_swap_backend(&swap);
        layout.shape = shape;
    }

    static AllocatorConfig make_config(BlockShape shape, std::uint32_t max_blocks) {
        AllocatorConfig config;
        config.block_shape = shape;
        config.max_blocks = max_blocks;
        return config;
    }

    HostSwapBackend swap;
    MemoryAllocator allocator;
    KVCacheManager cache;
    KVBlockLayout layout{};
};

/**
 * @brief Deterministic pseudo-random cache contents in [-1, 1).
 */
float pattern(
    std::uint32_t layer,
    KVStream stream,
    std::uint32_t token,
    std::uint32_t head,
    std::uint32_t dim) {
    const std::uint32_t stream_id = (stream == KVStream::Key) ? 0U : 1U;
    std::uint32_t mixed = 2166136261U;
    for (const std::uint32_t part : {layer, stream_id, token, head, dim}) {
        mixed = (mixed ^ part) * 16777619U;
    }
    return static_cast<float>(mixed % 2000U) / 1000.0F - 1.0F;
}

/**
 * @brief Writes one head vector into the paged cache via the copy-on-write path.
 */
bool write_head_vector(
    Harness& harness,
    SequenceId sequence_id,
    KVStream stream,
    std::uint32_t layer,
    TokenPosition token,
    std::uint32_t kv_head,
    const std::vector<float>& vector) {
    const std::optional<PhysicalBlockId> physical =
        harness.cache.ensure_token_writable(sequence_id, token);
    if (!physical.has_value()) {
        return false;
    }

    PhysicalBlock* block = harness.allocator.block(*physical);
    if (block == nullptr) {
        return false;
    }

    const std::optional<std::size_t> offset = harness.layout.element_offset(
        layer, stream, token % harness.layout.shape.tokens_per_block, kv_head);
    if (!offset.has_value()) {
        return false;
    }

    float* base = reinterpret_cast<float*>(block->data());
    for (std::size_t i = 0; i < vector.size(); ++i) {
        base[*offset + i] = vector[i];
    }
    return true;
}

/**
 * @brief Fills every layer, stream, token, and head of a sequence with `pattern`.
 */
bool populate_pattern(Harness& harness, SequenceId sequence_id, std::uint32_t context_len) {
    const BlockShape& shape = harness.layout.shape;
    for (std::uint32_t layer = 0; layer < shape.num_layers; ++layer) {
        for (const KVStream stream : {KVStream::Key, KVStream::Value}) {
            for (std::uint32_t token = 0; token < context_len; ++token) {
                for (std::uint32_t head = 0; head < shape.num_kv_heads; ++head) {
                    std::vector<float> vector(shape.head_dim);
                    for (std::uint32_t dim = 0; dim < shape.head_dim; ++dim) {
                        vector[dim] = pattern(layer, stream, token, head, dim);
                    }
                    if (!write_head_vector(harness, sequence_id, stream, layer, token, head, vector)) {
                        return false;
                    }
                }
            }
        }
    }
    return true;
}

/**
 * @brief Textbook attention over a flat, contiguous mirror of the cache.
 *
 * The paged kernel must agree with this even though it reads the same logical
 * tokens out of scattered physical blocks.
 */
std::vector<float> reference_attention(
    const BlockShape& shape,
    const std::vector<float>& query,
    std::uint32_t layer,
    std::uint32_t num_query_heads,
    std::uint32_t context_len,
    float scale) {
    const std::uint32_t head_dim = shape.head_dim;
    const std::uint32_t group_size = num_query_heads / shape.num_kv_heads;

    std::vector<float> out(static_cast<std::size_t>(num_query_heads) * head_dim, 0.0F);
    for (std::uint32_t q_head = 0; q_head < num_query_heads; ++q_head) {
        const std::uint32_t kv_head = q_head / group_size;
        const float* q = query.data() + static_cast<std::size_t>(q_head) * head_dim;

        std::vector<float> scores(context_len, 0.0F);
        float max_score = -std::numeric_limits<float>::infinity();
        for (std::uint32_t token = 0; token < context_len; ++token) {
            float dot = 0.0F;
            for (std::uint32_t dim = 0; dim < head_dim; ++dim) {
                dot += q[dim] * pattern(layer, KVStream::Key, token, kv_head, dim);
            }
            scores[token] = dot * scale;
            max_score = std::max(max_score, scores[token]);
        }

        float denominator = 0.0F;
        float* accumulator = out.data() + static_cast<std::size_t>(q_head) * head_dim;
        for (std::uint32_t token = 0; token < context_len; ++token) {
            const float weight = std::exp(scores[token] - max_score);
            denominator += weight;
            for (std::uint32_t dim = 0; dim < head_dim; ++dim) {
                accumulator[dim] += weight * pattern(layer, KVStream::Value, token, kv_head, dim);
            }
        }

        for (std::uint32_t dim = 0; dim < head_dim; ++dim) {
            accumulator[dim] /= denominator;
        }
    }
    return out;
}

/**
 * @brief Reserves `blocks` logical blocks whose frames are not adjacent.
 *
 * A spacer sequence takes every other frame and is then released, so the
 * sequence under test ends up with a physically scattered block table. A kernel
 * that walked frames linearly instead of following the page table would read
 * the spacer's recycled memory and fail the comparison tests.
 */
void reserve_scattered(
    Harness& harness, SequenceId sequence_id, SequenceId spacer_id, std::uint32_t blocks) {
    const std::uint32_t tokens_per_block = harness.layout.shape.tokens_per_block;
    ASSERT_TRUE(harness.cache.create_sequence(SequenceMetadata{spacer_id, spacer_id, {}, {}}));
    for (std::uint32_t i = 0; i < blocks; ++i) {
        ASSERT_TRUE(harness.cache.reserve_tokens(sequence_id, tokens_per_block));
        ASSERT_TRUE(harness.cache.reserve_tokens(spacer_id, tokens_per_block));
    }
    harness.cache.release_sequence(spacer_id);
}

/**
 * @brief Returns true when a table's frames do not form one contiguous run.
 *
 * The allocator hands out frames in descending order, so merely checking that
 * frames are not ascending would pass for any sequence. Comparing the frame
 * span against the frame count instead only holds when the spacer really did
 * punch holes into the sequence's frames.
 */
bool is_physically_scattered(const BlockTable& table) {
    const std::vector<BlockTableEntry>& entries = table.entries();
    if (entries.size() < 2) {
        return false;
    }

    PhysicalBlockId lowest = entries.front().physical_id;
    PhysicalBlockId highest = entries.front().physical_id;
    for (const BlockTableEntry& entry : entries) {
        lowest = std::min(lowest, entry.physical_id);
        highest = std::max(highest, entry.physical_id);
    }
    return highest - lowest + 1 != entries.size();
}

PagedAttentionParams make_params(
    std::uint32_t layer, std::uint32_t num_query_heads, std::uint32_t context_len, float scale) {
    PagedAttentionParams params;
    params.layer = layer;
    params.num_query_heads = num_query_heads;
    params.context_len = context_len;
    params.scale = scale;
    return params;
}

// --- Argument validation -------------------------------------------------

TEST(PagedAttentionTest, RejectsAnInvalidView) {
    const CacheView view;
    const std::vector<float> query(4, 1.0F);
    std::vector<float> out(4, 0.0F);

    EXPECT_FALSE(paged_attention_decode<float>(view, query.data(), make_params(0, 1, 1, 1.0F), out.data()));
}

TEST(PagedAttentionTest, RejectsElementSizeMismatch) {
    Harness harness(make_shape(4, 1, 1, 4), 4);
    ASSERT_TRUE(harness.cache.create_sequence(SequenceMetadata{1, 1, {}, {}}));
    ASSERT_TRUE(harness.cache.reserve_tokens(1, 4));

    const std::optional<CacheView> view = harness.cache.cache_view(1);
    ASSERT_TRUE(view.has_value());

    const std::vector<double> query(4, 1.0);
    std::vector<double> out(4, 0.0);

    EXPECT_FALSE(
        paged_attention_decode<double>(*view, query.data(), make_params(0, 1, 4, 1.0F), out.data()));
}

TEST(PagedAttentionTest, RejectsNullQueryOrOutput) {
    Harness harness(make_shape(4, 1, 1, 4), 4);
    ASSERT_TRUE(harness.cache.create_sequence(SequenceMetadata{1, 1, {}, {}}));
    ASSERT_TRUE(harness.cache.reserve_tokens(1, 4));

    const std::optional<CacheView> view = harness.cache.cache_view(1);
    ASSERT_TRUE(view.has_value());

    const std::vector<float> query(4, 1.0F);
    std::vector<float> out(4, 0.0F);
    const PagedAttentionParams params = make_params(0, 1, 4, 1.0F);

    EXPECT_FALSE(paged_attention_decode<float>(*view, nullptr, params, out.data()));
    EXPECT_FALSE(paged_attention_decode<float>(*view, query.data(), params, nullptr));
}

TEST(PagedAttentionTest, RejectsZeroContextLength) {
    Harness harness(make_shape(4, 1, 1, 4), 4);
    ASSERT_TRUE(harness.cache.create_sequence(SequenceMetadata{1, 1, {}, {}}));
    ASSERT_TRUE(harness.cache.reserve_tokens(1, 4));

    const std::optional<CacheView> view = harness.cache.cache_view(1);
    ASSERT_TRUE(view.has_value());

    const std::vector<float> query(4, 1.0F);
    std::vector<float> out(4, 0.0F);

    EXPECT_FALSE(
        paged_attention_decode<float>(*view, query.data(), make_params(0, 1, 0, 1.0F), out.data()));
}

TEST(PagedAttentionTest, RejectsLayerBeyondBlockShape) {
    Harness harness(make_shape(4, 2, 1, 4), 4);
    ASSERT_TRUE(harness.cache.create_sequence(SequenceMetadata{1, 1, {}, {}}));
    ASSERT_TRUE(harness.cache.reserve_tokens(1, 4));

    const std::optional<CacheView> view = harness.cache.cache_view(1);
    ASSERT_TRUE(view.has_value());

    const std::vector<float> query(4, 1.0F);
    std::vector<float> out(4, 0.0F);

    EXPECT_FALSE(
        paged_attention_decode<float>(*view, query.data(), make_params(2, 1, 4, 1.0F), out.data()));
}

TEST(PagedAttentionTest, RejectsQueryHeadsThatDoNotGroupOntoKvHeads) {
    Harness harness(make_shape(4, 1, 2, 4), 4);
    ASSERT_TRUE(harness.cache.create_sequence(SequenceMetadata{1, 1, {}, {}}));
    ASSERT_TRUE(harness.cache.reserve_tokens(1, 4));

    const std::optional<CacheView> view = harness.cache.cache_view(1);
    ASSERT_TRUE(view.has_value());

    const std::vector<float> query(3 * 4, 1.0F);
    std::vector<float> out(3 * 4, 0.0F);

    // 3 query heads cannot be split evenly across 2 kv heads.
    EXPECT_FALSE(
        paged_attention_decode<float>(*view, query.data(), make_params(0, 3, 4, 1.0F), out.data()));
    // Neither can zero query heads.
    EXPECT_FALSE(
        paged_attention_decode<float>(*view, query.data(), make_params(0, 0, 4, 1.0F), out.data()));
}

TEST(PagedAttentionTest, RejectsContextLongerThanTheMappedBlocks) {
    Harness harness(make_shape(4, 1, 1, 4), 4);
    ASSERT_TRUE(harness.cache.create_sequence(SequenceMetadata{1, 1, {}, {}}));
    ASSERT_TRUE(harness.cache.reserve_tokens(1, 4)); // one block, four token slots

    const std::optional<CacheView> view = harness.cache.cache_view(1);
    ASSERT_TRUE(view.has_value());

    const std::vector<float> query(4, 1.0F);
    std::vector<float> out(4, 0.0F);

    EXPECT_FALSE(
        paged_attention_decode<float>(*view, query.data(), make_params(0, 1, 5, 1.0F), out.data()));
}

TEST(PagedAttentionTest, RejectsSwappedOutContextWithoutWritingOutput) {
    Harness harness(make_shape(4, 1, 1, 4), 4);
    ASSERT_TRUE(harness.cache.create_sequence(SequenceMetadata{1, 1, {}, {}}));
    ASSERT_TRUE(harness.cache.reserve_tokens(1, 8));
    ASSERT_TRUE(populate_pattern(harness, 1, 8));
    ASSERT_EQ(harness.cache.swap_out_sequence(1), 2u);

    const std::optional<CacheView> view = harness.cache.cache_view(1);
    ASSERT_TRUE(view.has_value());

    const std::vector<float> query(4, 1.0F);
    std::vector<float> out(4, -1.0F);

    EXPECT_FALSE(
        paged_attention_decode<float>(*view, query.data(), make_params(0, 1, 8, 1.0F), out.data()));
    for (const float value : out) {
        EXPECT_FLOAT_EQ(value, -1.0F) << "output must be untouched on failure";
    }
}

// --- Correctness ---------------------------------------------------------

TEST(PagedAttentionTest, UniformKeysAverageTheCachedValues) {
    constexpr std::uint32_t kHeadDim = 4;
    constexpr std::uint32_t kContext = 8;
    Harness harness(make_shape(4, 1, 1, kHeadDim), 8);

    ASSERT_TRUE(harness.cache.create_sequence(SequenceMetadata{1, 1, {}, {}}));
    ASSERT_TRUE(harness.cache.reserve_tokens(1, kContext));

    // Identical keys make every score equal, so softmax is exactly uniform and
    // the expected output is the plain mean of the cached values.
    float expected = 0.0F;
    for (std::uint32_t token = 0; token < kContext; ++token) {
        const std::vector<float> key(kHeadDim, 1.0F);
        const std::vector<float> value(kHeadDim, static_cast<float>(token + 1));
        ASSERT_TRUE(write_head_vector(harness, 1, KVStream::Key, 0, token, 0, key));
        ASSERT_TRUE(write_head_vector(harness, 1, KVStream::Value, 0, token, 0, value));
        expected += static_cast<float>(token + 1);
    }
    expected /= static_cast<float>(kContext);

    const std::optional<CacheView> view = harness.cache.cache_view(1);
    ASSERT_TRUE(view.has_value());

    const std::vector<float> query(kHeadDim, 0.5F);
    std::vector<float> out(kHeadDim, 0.0F);
    ASSERT_TRUE(paged_attention_decode<float>(
        *view, query.data(), make_params(0, 1, kContext, 1.0F), out.data()));

    for (const float value : out) {
        EXPECT_NEAR(value, expected, kTolerance);
    }
}

TEST(PagedAttentionTest, SelectsEachLogicalTokenInOrderAcrossScatteredBlocks) {
    // head_dim >= context lets every token carry a distinct one-hot key, so a
    // sharply scaled query addresses exactly one logical position. Sweeping the
    // target across block boundaries proves the kernel resolves logical token N
    // to the physical slot that actually holds token N, and not to whatever
    // frame happens to sit N slots into the pool.
    constexpr std::uint32_t kTokensPerBlock = 4;
    constexpr std::uint32_t kHeadDim = 8;
    constexpr std::uint32_t kContext = 8;
    constexpr float kSharpScale = 50.0F;

    Harness harness(make_shape(kTokensPerBlock, 1, 1, kHeadDim), 8);
    ASSERT_TRUE(harness.cache.create_sequence(SequenceMetadata{1, 1, {}, {}}));
    reserve_scattered(harness, 1, 99, kContext / kTokensPerBlock);

    const std::optional<CacheView> setup_view = harness.cache.cache_view(1);
    ASSERT_TRUE(setup_view.has_value());
    ASSERT_TRUE(is_physically_scattered(*setup_view->block_table))
        << "test setup failed to scatter the block table";

    for (std::uint32_t token = 0; token < kContext; ++token) {
        std::vector<float> key(kHeadDim, 0.0F);
        key[token] = 1.0F;
        const std::vector<float> value(kHeadDim, static_cast<float>(token + 1) * 10.0F);
        ASSERT_TRUE(write_head_vector(harness, 1, KVStream::Key, 0, token, 0, key));
        ASSERT_TRUE(write_head_vector(harness, 1, KVStream::Value, 0, token, 0, value));
    }

    const std::optional<CacheView> view = harness.cache.cache_view(1);
    ASSERT_TRUE(view.has_value());

    for (std::uint32_t target = 0; target < kContext; ++target) {
        std::vector<float> query(kHeadDim, 0.0F);
        query[target] = 1.0F;

        std::vector<float> out(kHeadDim, 0.0F);
        ASSERT_TRUE(paged_attention_decode<float>(
            *view, query.data(), make_params(0, 1, kContext, kSharpScale), out.data()))
            << "target token " << target;

        const float expected = static_cast<float>(target + 1) * 10.0F;
        for (const float value : out) {
            EXPECT_NEAR(value, expected, 1e-3F) << "target token " << target;
        }
    }
}

TEST(PagedAttentionTest, MatchesContiguousReferenceOnScatteredBlocks) {
    constexpr std::uint32_t kTokensPerBlock = 4;
    constexpr std::uint32_t kNumLayers = 2;
    constexpr std::uint32_t kNumKvHeads = 2;
    constexpr std::uint32_t kHeadDim = 4;
    constexpr std::uint32_t kContext = 10; // spans three blocks, last one partial

    Harness harness(make_shape(kTokensPerBlock, kNumLayers, kNumKvHeads, kHeadDim), 12);
    ASSERT_TRUE(harness.cache.create_sequence(SequenceMetadata{1, 1, {}, {}}));
    reserve_scattered(harness, 1, 99, 3);
    ASSERT_TRUE(populate_pattern(harness, 1, kContext));

    const std::optional<CacheView> view = harness.cache.cache_view(1);
    ASSERT_TRUE(view.has_value());
    ASSERT_TRUE(is_physically_scattered(*view->block_table));

    std::vector<float> query(static_cast<std::size_t>(kNumKvHeads) * kHeadDim);
    for (std::size_t i = 0; i < query.size(); ++i) {
        query[i] = 0.25F * static_cast<float>(i) - 0.5F;
    }

    // Running both layers proves the kernel stays inside the layer it was asked
    // for, since each layer holds a different pattern.
    for (std::uint32_t layer = 0; layer < kNumLayers; ++layer) {
        std::vector<float> out(query.size(), 0.0F);
        ASSERT_TRUE(paged_attention_decode<float>(
            *view, query.data(), make_params(layer, kNumKvHeads, kContext, 0.5F), out.data()));

        const std::vector<float> expected = reference_attention(
            harness.layout.shape, query, layer, kNumKvHeads, kContext, 0.5F);
        ASSERT_EQ(out.size(), expected.size());
        for (std::size_t i = 0; i < out.size(); ++i) {
            EXPECT_NEAR(out[i], expected[i], kTolerance) << "layer " << layer << " element " << i;
        }
    }
}

TEST(PagedAttentionTest, GroupedQueryHeadsReadTheirOwnKvHead) {
    constexpr std::uint32_t kHeadDim = 4;
    constexpr std::uint32_t kNumKvHeads = 2;
    constexpr std::uint32_t kNumQueryHeads = 4; // group size 2
    constexpr std::uint32_t kContext = 4;

    Harness harness(make_shape(4, 1, kNumKvHeads, kHeadDim), 4);
    ASSERT_TRUE(harness.cache.create_sequence(SequenceMetadata{1, 1, {}, {}}));
    ASSERT_TRUE(harness.cache.reserve_tokens(1, kContext));

    // Uniform keys per kv head again reduce the expected output to the mean of
    // that head's values, so each query head's answer names the kv head it read.
    const std::vector<float> kv_head_value{100.0F, 200.0F};
    for (std::uint32_t kv_head = 0; kv_head < kNumKvHeads; ++kv_head) {
        for (std::uint32_t token = 0; token < kContext; ++token) {
            const std::vector<float> key(kHeadDim, 1.0F);
            const std::vector<float> value(kHeadDim, kv_head_value[kv_head]);
            ASSERT_TRUE(write_head_vector(harness, 1, KVStream::Key, 0, token, kv_head, key));
            ASSERT_TRUE(write_head_vector(harness, 1, KVStream::Value, 0, token, kv_head, value));
        }
    }

    const std::optional<CacheView> view = harness.cache.cache_view(1);
    ASSERT_TRUE(view.has_value());

    const std::vector<float> query(static_cast<std::size_t>(kNumQueryHeads) * kHeadDim, 1.0F);
    std::vector<float> out(query.size(), 0.0F);
    ASSERT_TRUE(paged_attention_decode<float>(
        *view, query.data(), make_params(0, kNumQueryHeads, kContext, 1.0F), out.data()));

    for (std::uint32_t q_head = 0; q_head < kNumQueryHeads; ++q_head) {
        const float expected = kv_head_value[q_head / (kNumQueryHeads / kNumKvHeads)];
        for (std::uint32_t dim = 0; dim < kHeadDim; ++dim) {
            EXPECT_NEAR(out[q_head * kHeadDim + dim], expected, kTolerance) << "query head " << q_head;
        }
    }
}

TEST(PagedAttentionTest, ForkedBranchesSeeTheirOwnCacheAfterCopyOnWrite) {
    constexpr std::uint32_t kHeadDim = 4;
    constexpr std::uint32_t kContext = 4;
    Harness harness(make_shape(4, 1, 1, kHeadDim), 8);

    ASSERT_TRUE(harness.cache.create_sequence(SequenceMetadata{1, 1, {}, {}}));
    ASSERT_TRUE(harness.cache.reserve_tokens(1, kContext));
    for (std::uint32_t token = 0; token < kContext; ++token) {
        ASSERT_TRUE(write_head_vector(harness, 1, KVStream::Key, 0, token, 0, std::vector<float>(kHeadDim, 1.0F)));
        ASSERT_TRUE(write_head_vector(harness, 1, KVStream::Value, 0, token, 0, std::vector<float>(kHeadDim, 5.0F)));
    }

    ASSERT_TRUE(harness.cache.fork_sequence(1, SequenceMetadata{2, 1, {}, {}}));
    for (std::uint32_t token = 0; token < kContext; ++token) {
        ASSERT_TRUE(write_head_vector(harness, 2, KVStream::Value, 0, token, 0, std::vector<float>(kHeadDim, 9.0F)));
    }

    const std::vector<float> query(kHeadDim, 1.0F);
    const PagedAttentionParams params = make_params(0, 1, kContext, 1.0F);

    const std::pair<SequenceId, float> expectations[] = {{1, 5.0F}, {2, 9.0F}};
    for (const auto& [sequence_id, expected] : expectations) {
        const std::optional<CacheView> view = harness.cache.cache_view(sequence_id);
        ASSERT_TRUE(view.has_value());

        std::vector<float> out(kHeadDim, 0.0F);
        ASSERT_TRUE(paged_attention_decode<float>(*view, query.data(), params, out.data()));
        for (const float value : out) {
            EXPECT_NEAR(value, expected, kTolerance) << "sequence " << sequence_id;
        }
    }
}

} // namespace
} // namespace qwenvl_paged
