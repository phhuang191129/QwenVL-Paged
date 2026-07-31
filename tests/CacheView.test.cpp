/**
 * @brief Specification tests for KVBlockLayout and the kernel-facing CacheView.
 *
 * These tests are written against the documented header contracts in
 * include/qwenvl_paged/CacheLayout.h and the CacheView struct in
 * include/qwenvl_paged/KVCacheManager.h. Until those members are implemented
 * this binary is expected to fail at the LINK stage with undefined references,
 * which is the intended red state for this phase.
 */

#include "qwenvl_paged/CacheLayout.h"
#include "qwenvl_paged/KVCacheManager.h"
#include "qwenvl_paged/MemoryAllocator.h"
#include "qwenvl_paged/SwapBackend.h"

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <set>
#include <vector>

namespace qwenvl_paged {
namespace {

constexpr std::uint32_t kTokensPerBlock = 4;
constexpr std::uint32_t kNumLayers = 2;
constexpr std::uint32_t kNumKvHeads = 2;
constexpr std::uint32_t kHeadDim = 4;

BlockShape make_shape() {
    BlockShape shape;
    shape.tokens_per_block = kTokensPerBlock;
    shape.num_layers = kNumLayers;
    shape.num_kv_heads = kNumKvHeads;
    shape.head_dim = kHeadDim;
    shape.bytes_per_element = static_cast<std::uint32_t>(sizeof(float));
    return shape;
}

KVBlockLayout make_layout() {
    KVBlockLayout layout;
    layout.shape = make_shape();
    return layout;
}

AllocatorConfig make_allocator_config(std::uint32_t max_blocks) {
    AllocatorConfig config;
    config.block_shape = make_shape();
    config.max_blocks = max_blocks;
    return config;
}

/**
 * @brief Writes one head vector into the cache through the copy-on-write path.
 */
bool write_head_vector(
    KVCacheManager& cache,
    MemoryAllocator& allocator,
    SequenceId sequence_id,
    KVStream stream,
    std::uint32_t layer,
    TokenPosition token,
    std::uint32_t kv_head,
    float value) {
    const std::optional<PhysicalBlockId> physical = cache.ensure_token_writable(sequence_id, token);
    if (!physical.has_value()) {
        return false;
    }

    PhysicalBlock* block = allocator.block(*physical);
    if (block == nullptr) {
        return false;
    }

    const KVBlockLayout layout = make_layout();
    const std::optional<std::size_t> offset =
        layout.element_offset(layer, stream, token % kTokensPerBlock, kv_head);
    if (!offset.has_value()) {
        return false;
    }

    float* base = reinterpret_cast<float*>(block->data());
    for (std::uint32_t dim = 0; dim < kHeadDim; ++dim) {
        base[*offset + dim] = value;
    }
    return true;
}

class CacheViewTest : public ::testing::Test {
protected:
    CacheViewTest() : swap_(16), allocator_(make_allocator_config(8)), cache_(allocator_) {
        allocator_.set_swap_backend(&swap_);
    }

    HostSwapBackend swap_;
    MemoryAllocator allocator_;
    KVCacheManager cache_;
};

// --- KVBlockLayout -------------------------------------------------------

TEST(KVBlockLayoutTest, StridesFollowLayerStreamTokenHeadOrder) {
    const KVBlockLayout layout = make_layout();

    EXPECT_EQ(layout.head_stride(), kHeadDim);
    EXPECT_EQ(layout.token_stride(), kNumKvHeads * kHeadDim);
    EXPECT_EQ(layout.stream_stride(), kTokensPerBlock * kNumKvHeads * kHeadDim);
    EXPECT_EQ(layout.layer_stride(), kKVStreamCount * kTokensPerBlock * kNumKvHeads * kHeadDim);
}

TEST(KVBlockLayoutTest, ElementCountMatchesBlockByteSize) {
    const KVBlockLayout layout = make_layout();

    EXPECT_EQ(layout.element_count() * layout.shape.bytes_per_element, layout.shape.byte_size());
}

TEST(KVBlockLayoutTest, HeadVectorsAreContiguousInMemory) {
    const KVBlockLayout layout = make_layout();

    const std::optional<std::size_t> first = layout.element_offset(0, KVStream::Key, 0, 0);
    const std::optional<std::size_t> second = layout.element_offset(0, KVStream::Key, 0, 1);
    ASSERT_TRUE(first.has_value());
    ASSERT_TRUE(second.has_value());

    // head_dim is innermost, so consecutive heads sit exactly head_dim apart.
    EXPECT_EQ(*second - *first, kHeadDim);
}

TEST(KVBlockLayoutTest, KeyAndValueHalvesDoNotOverlap) {
    const KVBlockLayout layout = make_layout();

    const std::optional<std::size_t> key = layout.element_offset(1, KVStream::Key, 2, 1);
    const std::optional<std::size_t> value = layout.element_offset(1, KVStream::Value, 2, 1);
    ASSERT_TRUE(key.has_value());
    ASSERT_TRUE(value.has_value());

    EXPECT_EQ(*value - *key, layout.stream_stride());
}

TEST(KVBlockLayoutTest, EveryCoordinateMapsToADistinctInBoundsSlot) {
    const KVBlockLayout layout = make_layout();

    // A layout bug that aliases two coordinates silently corrupts one token's
    // cache, so walk the entire index space and prove the mapping is injective
    // and stays inside the block.
    std::set<std::size_t> seen;
    for (std::uint32_t layer = 0; layer < kNumLayers; ++layer) {
        for (const KVStream stream : {KVStream::Key, KVStream::Value}) {
            for (std::uint32_t token = 0; token < kTokensPerBlock; ++token) {
                for (std::uint32_t head = 0; head < kNumKvHeads; ++head) {
                    const std::optional<std::size_t> offset =
                        layout.element_offset(layer, stream, token, head);
                    ASSERT_TRUE(offset.has_value());
                    for (std::uint32_t dim = 0; dim < kHeadDim; ++dim) {
                        const std::size_t slot = *offset + dim;
                        ASSERT_LT(slot, layout.element_count());
                        EXPECT_TRUE(seen.insert(slot).second) << "aliased element " << slot;
                    }
                }
            }
        }
    }

    EXPECT_EQ(seen.size(), layout.element_count());
}

TEST(KVBlockLayoutTest, RejectsOutOfRangeCoordinates) {
    const KVBlockLayout layout = make_layout();

    EXPECT_FALSE(layout.element_offset(kNumLayers, KVStream::Key, 0, 0).has_value());
    EXPECT_FALSE(layout.element_offset(0, KVStream::Key, kTokensPerBlock, 0).has_value());
    EXPECT_FALSE(layout.element_offset(0, KVStream::Key, 0, kNumKvHeads).has_value());
}

// --- CacheView -----------------------------------------------------------

TEST(CacheViewTest_Static, DefaultConstructedViewIsInvalid) {
    const CacheView view;

    EXPECT_FALSE(view.valid());
    EXPECT_EQ(view.block_bytes(0), nullptr);
    EXPECT_EQ(view.slot<float>(KVStream::Key, 0, 0, 0), nullptr);
}

TEST_F(CacheViewTest, ViewCarriesBlockTableLayoutAndAllocator) {
    ASSERT_TRUE(cache_.create_sequence(SequenceMetadata{1, 1, {}, {}}));
    ASSERT_TRUE(cache_.reserve_tokens(1, kTokensPerBlock));

    const std::optional<CacheView> view = cache_.cache_view(1);
    ASSERT_TRUE(view.has_value());

    EXPECT_TRUE(view->valid());
    EXPECT_EQ(view->sequence_id, 1u);
    EXPECT_NE(view->block_table, nullptr);
    EXPECT_EQ(view->allocator, &allocator_);
    EXPECT_EQ(view->layout.shape.tokens_per_block, kTokensPerBlock);
    EXPECT_EQ(view->layout.shape.head_dim, kHeadDim);
}

TEST_F(CacheViewTest, BlockBytesResolveMappedLogicalBlocks) {
    ASSERT_TRUE(cache_.create_sequence(SequenceMetadata{1, 1, {}, {}}));
    ASSERT_TRUE(cache_.reserve_tokens(1, 2 * kTokensPerBlock));

    const std::optional<CacheView> view = cache_.cache_view(1);
    ASSERT_TRUE(view.has_value());

    for (LogicalBlockIndex index = 0; index < 2; ++index) {
        const std::optional<PhysicalBlockId> physical = view->block_table->lookup(index);
        ASSERT_TRUE(physical.has_value());
        EXPECT_EQ(view->block_bytes(index), allocator_.block(*physical)->data());
    }
}

TEST_F(CacheViewTest, BlockBytesReturnsNullForUnmappedIndex) {
    ASSERT_TRUE(cache_.create_sequence(SequenceMetadata{1, 1, {}, {}}));
    ASSERT_TRUE(cache_.reserve_tokens(1, kTokensPerBlock));

    const std::optional<CacheView> view = cache_.cache_view(1);
    ASSERT_TRUE(view.has_value());

    EXPECT_EQ(view->block_bytes(1), nullptr);
}

TEST_F(CacheViewTest, BlockBytesFaultsOnSwappedOutBlock) {
    ASSERT_TRUE(cache_.create_sequence(SequenceMetadata{1, 1, {}, {}}));
    ASSERT_TRUE(cache_.reserve_tokens(1, kTokensPerBlock));
    ASSERT_EQ(cache_.swap_out_sequence(1), 1u);

    const std::optional<CacheView> view = cache_.cache_view(1);
    ASSERT_TRUE(view.has_value());

    // The frame went back to the free list, so reading it would serve another
    // sequence's bytes. The view must fault instead.
    EXPECT_EQ(view->block_bytes(0), nullptr);
    EXPECT_EQ(view->slot<float>(KVStream::Key, 0, 0, 0), nullptr);
}

TEST_F(CacheViewTest, SlotResolvesLogicalTokensAcrossBlockBoundaries) {
    constexpr std::uint32_t kTokens = 2 * kTokensPerBlock;
    ASSERT_TRUE(cache_.create_sequence(SequenceMetadata{1, 1, {}, {}}));
    ASSERT_TRUE(cache_.reserve_tokens(1, kTokens));

    for (TokenPosition token = 0; token < kTokens; ++token) {
        ASSERT_TRUE(write_head_vector(
            cache_, allocator_, 1, KVStream::Key, 0, token, 0, static_cast<float>(token)));
    }

    const std::optional<CacheView> view = cache_.cache_view(1);
    ASSERT_TRUE(view.has_value());

    for (TokenPosition token = 0; token < kTokens; ++token) {
        const float* slot = view->slot<float>(KVStream::Key, 0, token, 0);
        ASSERT_NE(slot, nullptr) << "token " << token;
        EXPECT_FLOAT_EQ(slot[0], static_cast<float>(token)) << "token " << token;
    }
}

TEST_F(CacheViewTest, SlotSeparatesStreamsLayersAndHeads) {
    ASSERT_TRUE(cache_.create_sequence(SequenceMetadata{1, 1, {}, {}}));
    ASSERT_TRUE(cache_.reserve_tokens(1, kTokensPerBlock));

    ASSERT_TRUE(write_head_vector(cache_, allocator_, 1, KVStream::Key, 0, 1, 0, 11.0F));
    ASSERT_TRUE(write_head_vector(cache_, allocator_, 1, KVStream::Value, 0, 1, 0, 22.0F));
    ASSERT_TRUE(write_head_vector(cache_, allocator_, 1, KVStream::Key, 1, 1, 0, 33.0F));
    ASSERT_TRUE(write_head_vector(cache_, allocator_, 1, KVStream::Key, 0, 1, 1, 44.0F));

    const std::optional<CacheView> view = cache_.cache_view(1);
    ASSERT_TRUE(view.has_value());

    EXPECT_FLOAT_EQ(view->slot<float>(KVStream::Key, 0, 1, 0)[0], 11.0F);
    EXPECT_FLOAT_EQ(view->slot<float>(KVStream::Value, 0, 1, 0)[0], 22.0F);
    EXPECT_FLOAT_EQ(view->slot<float>(KVStream::Key, 1, 1, 0)[0], 33.0F);
    EXPECT_FLOAT_EQ(view->slot<float>(KVStream::Key, 0, 1, 1)[0], 44.0F);
}

TEST_F(CacheViewTest, SlotRejectsElementSizeMismatch) {
    ASSERT_TRUE(cache_.create_sequence(SequenceMetadata{1, 1, {}, {}}));
    ASSERT_TRUE(cache_.reserve_tokens(1, kTokensPerBlock));

    const std::optional<CacheView> view = cache_.cache_view(1);
    ASSERT_TRUE(view.has_value());

    // The block was sized for 4-byte elements; handing out a double* would read
    // past the end of the last head vector.
    EXPECT_EQ(view->slot<double>(KVStream::Key, 0, 0, 0), nullptr);
    EXPECT_NE(view->slot<float>(KVStream::Key, 0, 0, 0), nullptr);
}

TEST_F(CacheViewTest, SlotRejectsOutOfRangeCoordinates) {
    ASSERT_TRUE(cache_.create_sequence(SequenceMetadata{1, 1, {}, {}}));
    ASSERT_TRUE(cache_.reserve_tokens(1, kTokensPerBlock));

    const std::optional<CacheView> view = cache_.cache_view(1);
    ASSERT_TRUE(view.has_value());

    EXPECT_EQ(view->slot<float>(KVStream::Key, kNumLayers, 0, 0), nullptr);
    EXPECT_EQ(view->slot<float>(KVStream::Key, 0, 0, kNumKvHeads), nullptr);
    EXPECT_EQ(view->slot<float>(KVStream::Key, 0, kTokensPerBlock, 0), nullptr);
}

TEST_F(CacheViewTest, SlotFollowsCopyOnWriteRemapping) {
    ASSERT_TRUE(cache_.create_sequence(SequenceMetadata{1, 1, {}, {}}));
    ASSERT_TRUE(cache_.reserve_tokens(1, kTokensPerBlock));
    ASSERT_TRUE(write_head_vector(cache_, allocator_, 1, KVStream::Key, 0, 0, 0, 7.0F));
    ASSERT_TRUE(cache_.fork_sequence(1, SequenceMetadata{2, 1, {}, {}}));

    {
        const std::optional<CacheView> parent = cache_.cache_view(1);
        const std::optional<CacheView> child = cache_.cache_view(2);
        ASSERT_TRUE(parent.has_value());
        ASSERT_TRUE(child.has_value());
        EXPECT_EQ(
            parent->slot<float>(KVStream::Key, 0, 0, 0), child->slot<float>(KVStream::Key, 0, 0, 0));
    }

    ASSERT_TRUE(write_head_vector(cache_, allocator_, 2, KVStream::Key, 0, 0, 0, 9.0F));

    const std::optional<CacheView> parent = cache_.cache_view(1);
    const std::optional<CacheView> child = cache_.cache_view(2);
    ASSERT_TRUE(parent.has_value());
    ASSERT_TRUE(child.has_value());

    const float* parent_slot = parent->slot<float>(KVStream::Key, 0, 0, 0);
    const float* child_slot = child->slot<float>(KVStream::Key, 0, 0, 0);
    ASSERT_NE(parent_slot, nullptr);
    ASSERT_NE(child_slot, nullptr);

    EXPECT_NE(parent_slot, child_slot);
    EXPECT_FLOAT_EQ(parent_slot[0], 7.0F);
    EXPECT_FLOAT_EQ(child_slot[0], 9.0F);
}

} // namespace
} // namespace qwenvl_paged
