/**
 * @brief Specification tests for the core types in qwenvl_paged/Block.h.
 *
 * These tests are written against the documented header contract in
 * include/qwenvl_paged/Block.h. BlockShape::byte_size, LogicalBlock::empty, and
 * PhysicalBlock have no .cpp implementation yet, so this test binary is expected
 * to fail at the LINK stage (undefined references) until those are implemented.
 * That failure is intentional for this phase of the project.
 */

#include "qwenvl_paged/Block.h"

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <vector>

namespace qwenvl_paged {
namespace {

/**
 * @brief Builds a small but valid BlockShape shared across tests.
 */
BlockShape make_shape() {
    BlockShape shape;
    shape.tokens_per_block = 4;
    shape.num_layers = 2;
    shape.num_kv_heads = 2;
    shape.head_dim = 8;
    shape.bytes_per_element = 2;
    return shape;
}

TEST(BlockShapeTest, ByteSizeIsNonZeroForValidShape) {
    EXPECT_GT(make_shape().byte_size(), 0u);
}

TEST(BlockShapeTest, ByteSizeScalesLinearlyWithTokensPerBlock) {
    const std::size_t base = make_shape().byte_size();

    BlockShape doubled = make_shape();
    doubled.tokens_per_block *= 2;

    EXPECT_EQ(doubled.byte_size(), base * 2);
}

TEST(BlockShapeTest, ByteSizeScalesLinearlyWithLayerCount) {
    const std::size_t base = make_shape().byte_size();

    BlockShape doubled = make_shape();
    doubled.num_layers *= 2;

    EXPECT_EQ(doubled.byte_size(), base * 2);
}

TEST(LogicalBlockTest, DefaultBlockIsEmpty) {
    LogicalBlock block;
    EXPECT_TRUE(block.empty());
}

TEST(LogicalBlockTest, BlockWithCommittedTokensIsNotEmpty) {
    LogicalBlock block;
    block.token_count = 3;
    EXPECT_FALSE(block.empty());
}

TEST(PhysicalBlockTest, ExposesIdAndShape) {
    const BlockShape shape = make_shape();
    PhysicalBlock block(7, shape);

    EXPECT_EQ(block.id(), 7u);
    EXPECT_EQ(block.shape().byte_size(), shape.byte_size());
}

TEST(PhysicalBlockTest, AllocationSatisfiesRequestedAlignment) {
    HostMemoryOptions options;
    options.alignment_bytes = kDefaultBlockAlignmentBytes;

    // Allocate many live blocks at once. A single block can be aligned by luck
    // even from an allocator that ignores alignment, so checking a whole batch
    // that stays alive simultaneously makes an unaligned implementation fail
    // with near-certainty rather than by chance.
    constexpr std::size_t kBlockCount = 64;
    std::vector<PhysicalBlock> blocks;
    blocks.reserve(kBlockCount);
    for (std::size_t i = 0; i < kBlockCount; ++i) {
        blocks.emplace_back(static_cast<PhysicalBlockId>(i), make_shape(), options);
    }

    for (const PhysicalBlock& block : blocks) {
        ASSERT_NE(block.data(), nullptr);
        EXPECT_EQ(block.alignment_bytes(), kDefaultBlockAlignmentBytes);
        EXPECT_EQ(reinterpret_cast<std::uintptr_t>(block.data()) % kDefaultBlockAlignmentBytes, 0u);
    }
}

TEST(PhysicalBlockTest, SizeCoversShapeAndIsAlignmentRounded) {
    HostMemoryOptions options;
    options.alignment_bytes = kDefaultBlockAlignmentBytes;

    const BlockShape shape = make_shape();
    PhysicalBlock block(1, shape, options);

    EXPECT_GE(block.size_bytes(), shape.byte_size());
    EXPECT_EQ(block.size_bytes() % kDefaultBlockAlignmentBytes, 0u);
}

TEST(PhysicalBlockTest, PinnedMemoryRequestIsReported) {
    HostMemoryOptions options;
    options.prefer_pinned_memory = true;

    PhysicalBlock block(1, make_shape(), options);

    EXPECT_TRUE(block.pinned_memory_requested());
}

TEST(PhysicalBlockTest, SwapExchangesStorageContents) {
    const BlockShape shape = make_shape();
    PhysicalBlock first(1, shape);
    PhysicalBlock second(2, shape);

    ASSERT_GE(first.size_bytes(), 4u);
    ASSERT_GE(second.size_bytes(), 4u);

    for (std::size_t i = 0; i < 4; ++i) {
        first.data()[i] = static_cast<std::byte>(0xA0 + i);
        second.data()[i] = static_cast<std::byte>(0xB0 + i);
    }

    first.swap(second);

    for (std::size_t i = 0; i < 4; ++i) {
        EXPECT_EQ(first.data()[i], static_cast<std::byte>(0xB0 + i));
        EXPECT_EQ(second.data()[i], static_cast<std::byte>(0xA0 + i));
    }
}

} // namespace
} // namespace qwenvl_paged
