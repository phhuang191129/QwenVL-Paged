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

TEST(BlockShapeTest, OneLayerPerFrameShrinksByteSizeByLayerCount) {
    const BlockShape all_layers = make_shape();
    BlockShape packed = make_shape();
    packed.layers_per_frame = 1;

    EXPECT_EQ(packed.byte_size(), all_layers.byte_size() / all_layers.num_layers);
    EXPECT_TRUE(packed.per_layer_frames());
    EXPECT_EQ(packed.frame_layers(), 1u);
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
    std::vector<std::byte> storage(shape.byte_size());
    PhysicalBlock block(7, shape, storage.data(), storage.size());

    EXPECT_EQ(block.id(), 7u);
    EXPECT_EQ(block.shape().byte_size(), shape.byte_size());
    EXPECT_EQ(block.size_bytes(), storage.size());
}

TEST(PhysicalBlockTest, FrameIsAWindowOntoPoolStorageRatherThanACopy) {
    const BlockShape shape = make_shape();
    std::vector<std::byte> storage(shape.byte_size());
    PhysicalBlock block(0, shape, storage.data(), storage.size());

    ASSERT_EQ(block.data(), storage.data());

    // The pool owns the bytes, so a write through the frame has to land in the
    // pool. Anything else would mean a device mirror of the pool could not be
    // produced by copying the pool.
    block.data()[0] = std::byte{0xAB};
    EXPECT_EQ(storage[0], std::byte{0xAB});
}

// Alignment, stride rounding, and the pinned-memory request are properties of
// the pool that owns the slab, not of an individual frame, so they are asserted
// in MemoryAllocator.test.cpp.

} // namespace
} // namespace qwenvl_paged
