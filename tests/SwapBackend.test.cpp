/**
 * @brief Specification tests for qwenvl_paged::SwapBackend and HostSwapBackend.
 *
 * These tests are written against the documented header contract in
 * include/qwenvl_paged/SwapBackend.h. HostSwapBackend has no .cpp
 * implementation yet, so this test binary is expected to fail at the LINK stage
 * (undefined references) until it is implemented. That failure is intentional
 * for this phase of the project.
 *
 * The backend is deliberately tested without an allocator: it only moves bytes
 * in and out of blocks, and knows nothing about frames, mappings, or sequences.
 */

#include "qwenvl_paged/SwapBackend.h"

#include <gtest/gtest.h>

#include <cstddef>
#include <optional>

namespace qwenvl_paged {
namespace {

BlockShape make_block_shape() {
    BlockShape shape;
    shape.tokens_per_block = 16;
    shape.num_layers = 2;
    shape.num_kv_heads = 2;
    shape.head_dim = 8;
    shape.bytes_per_element = 4;
    return shape;
}

/**
 * @brief Fills a block with a deterministic, seed-dependent byte pattern.
 */
void fill_block(PhysicalBlock& block, std::size_t seed) {
    for (std::size_t i = 0; i < block.size_bytes(); ++i) {
        block.data()[i] = static_cast<std::byte>((i + seed) % 251);
    }
}

/**
 * @brief Returns true when both blocks hold the same bytes.
 */
bool have_equal_bytes(const PhysicalBlock& lhs, const PhysicalBlock& rhs) {
    if (lhs.size_bytes() != rhs.size_bytes()) {
        return false;
    }
    for (std::size_t i = 0; i < lhs.size_bytes(); ++i) {
        if (lhs.data()[i] != rhs.data()[i]) {
            return false;
        }
    }
    return true;
}

TEST(HostSwapBackendTest, StartsEmptyWithRequestedCapacity) {
    HostSwapBackend backend(4);

    EXPECT_EQ(backend.resident_slots(), 0u);
    EXPECT_EQ(backend.capacity_slots(), 4u);
}

TEST(HostSwapBackendTest, StoreReturnsSlotAndTracksResidency) {
    HostSwapBackend backend(4);
    PhysicalBlock block(0, make_block_shape());
    fill_block(block, 1);

    const std::optional<SwapSlotId> slot = backend.store(block);

    ASSERT_TRUE(slot.has_value());
    EXPECT_EQ(backend.resident_slots(), 1u);
}

TEST(HostSwapBackendTest, StoreGivesDistinctSlotsToResidentBlocks) {
    HostSwapBackend backend(4);
    PhysicalBlock first(0, make_block_shape());
    PhysicalBlock second(1, make_block_shape());

    const std::optional<SwapSlotId> first_slot = backend.store(first);
    const std::optional<SwapSlotId> second_slot = backend.store(second);

    ASSERT_TRUE(first_slot.has_value());
    ASSERT_TRUE(second_slot.has_value());
    EXPECT_NE(*first_slot, *second_slot);
    EXPECT_EQ(backend.resident_slots(), 2u);
}

TEST(HostSwapBackendTest, LoadRestoresStoredBytesIntoAnotherBlock) {
    HostSwapBackend backend(4);
    PhysicalBlock source(0, make_block_shape());
    PhysicalBlock destination(1, make_block_shape());
    fill_block(source, 7);
    fill_block(destination, 200);
    ASSERT_FALSE(have_equal_bytes(source, destination));

    const std::optional<SwapSlotId> slot = backend.store(source);
    ASSERT_TRUE(slot.has_value());

    EXPECT_TRUE(backend.load(*slot, destination));
    EXPECT_TRUE(have_equal_bytes(source, destination));
}

TEST(HostSwapBackendTest, LoadKeepsSlotResidentSoItCanBeRetried) {
    HostSwapBackend backend(4);
    PhysicalBlock source(0, make_block_shape());
    PhysicalBlock destination(1, make_block_shape());
    fill_block(source, 3);

    const std::optional<SwapSlotId> slot = backend.store(source);
    ASSERT_TRUE(slot.has_value());
    ASSERT_TRUE(backend.load(*slot, destination));

    EXPECT_EQ(backend.resident_slots(), 1u);
    EXPECT_TRUE(backend.load(*slot, destination));
}

TEST(HostSwapBackendTest, LoadUnknownSlotFails) {
    HostSwapBackend backend(4);
    PhysicalBlock block(0, make_block_shape());

    EXPECT_FALSE(backend.load(9999, block));
}

TEST(HostSwapBackendTest, DiscardDropsSlotAndPreventsLoad) {
    HostSwapBackend backend(4);
    PhysicalBlock block(0, make_block_shape());

    const std::optional<SwapSlotId> slot = backend.store(block);
    ASSERT_TRUE(slot.has_value());

    backend.discard(*slot);

    EXPECT_EQ(backend.resident_slots(), 0u);
    EXPECT_FALSE(backend.load(*slot, block));
}

TEST(HostSwapBackendTest, StoreFailsWhenSwapSpaceIsFull) {
    HostSwapBackend backend(1);
    PhysicalBlock first(0, make_block_shape());
    PhysicalBlock second(1, make_block_shape());

    ASSERT_TRUE(backend.store(first).has_value());

    EXPECT_FALSE(backend.store(second).has_value());
    EXPECT_EQ(backend.resident_slots(), 1u);
}

TEST(HostSwapBackendTest, DiscardFreesCapacityForAnotherStore) {
    HostSwapBackend backend(1);
    PhysicalBlock first(0, make_block_shape());
    PhysicalBlock second(1, make_block_shape());

    const std::optional<SwapSlotId> slot = backend.store(first);
    ASSERT_TRUE(slot.has_value());
    backend.discard(*slot);

    EXPECT_TRUE(backend.store(second).has_value());
    EXPECT_EQ(backend.resident_slots(), 1u);
}

TEST(HostSwapBackendTest, StoredCopyIsIndependentOfTheSourceBlock) {
    HostSwapBackend backend(4);
    PhysicalBlock source(0, make_block_shape());
    PhysicalBlock restored(1, make_block_shape());
    fill_block(source, 11);

    const std::optional<SwapSlotId> slot = backend.store(source);
    ASSERT_TRUE(slot.has_value());

    // Overwriting the source after the store must not change what was swapped.
    PhysicalBlock expected(2, make_block_shape());
    fill_block(expected, 11);
    fill_block(source, 99);

    ASSERT_TRUE(backend.load(*slot, restored));
    EXPECT_TRUE(have_equal_bytes(restored, expected));
}

} // namespace
} // namespace qwenvl_paged
