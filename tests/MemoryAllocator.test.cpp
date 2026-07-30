/**
 * @brief Specification tests for qwenvl_paged::MemoryAllocator.
 *
 * These tests are written against the documented header contract in
 * include/qwenvl_paged/MemoryAllocator.h and include/qwenvl_paged/Block.h.
 * MemoryAllocator and PhysicalBlock currently have no .cpp implementation, so
 * this test binary is expected to fail at the LINK stage (undefined
 * references) until those classes are implemented. That failure is
 * intentional for this phase of the project.
 */

#include "qwenvl_paged/MemoryAllocator.h"
#include "qwenvl_paged/SwapBackend.h"

#include <gtest/gtest.h>

#include <cstddef>
#include <optional>
#include <vector>

namespace qwenvl_paged {
namespace {

/**
 * @brief Builds a small but valid AllocatorConfig shared across tests.
 */
AllocatorConfig make_test_config(std::uint32_t max_blocks = 4) {
    AllocatorConfig config;
    config.block_shape.tokens_per_block = 16;
    config.block_shape.num_layers = 2;
    config.block_shape.num_kv_heads = 2;
    config.block_shape.head_dim = 8;
    config.block_shape.bytes_per_element = 4;
    config.max_blocks = max_blocks;
    return config;
}

TEST(MemoryAllocatorTest, ConstructsWithValidConfig) {
    MemoryAllocator allocator(make_test_config());

    const AllocatorStats stats = allocator.stats();
    EXPECT_EQ(stats.total_blocks, 4u);
    EXPECT_EQ(stats.free_blocks, 4u);
    EXPECT_EQ(stats.active_blocks, 0u);
}

TEST(MemoryAllocatorTest, AllocateReturnsDistinctIdsUntilExhausted) {
    MemoryAllocator allocator(make_test_config(4));

    std::vector<PhysicalBlockId> ids;
    for (int i = 0; i < 4; ++i) {
        std::optional<PhysicalBlockId> id = allocator.allocate();
        ASSERT_TRUE(id.has_value());
        ids.push_back(*id);
    }

    for (std::size_t i = 0; i < ids.size(); ++i) {
        for (std::size_t j = i + 1; j < ids.size(); ++j) {
            EXPECT_NE(ids[i], ids[j]);
        }
    }

    EXPECT_FALSE(allocator.allocate().has_value());
}

TEST(MemoryAllocatorTest, ReleaseReturnsBlockToFreeListForReuse) {
    MemoryAllocator allocator(make_test_config(1));

    const std::optional<PhysicalBlockId> first = allocator.allocate();
    ASSERT_TRUE(first.has_value());
    EXPECT_FALSE(allocator.allocate().has_value());

    allocator.release(*first);
    const std::optional<PhysicalBlockId> second = allocator.allocate();
    ASSERT_TRUE(second.has_value());
    EXPECT_EQ(*second, *first);
}

TEST(MemoryAllocatorTest, RetainRequiresMatchingReleaseCountBeforeFree) {
    MemoryAllocator allocator(make_test_config(1));

    const std::optional<PhysicalBlockId> id = allocator.allocate();
    ASSERT_TRUE(id.has_value());

    allocator.retain(*id);
    ASSERT_NE(allocator.info(*id), nullptr);
    EXPECT_EQ(allocator.info(*id)->ref_count, 2u);

    allocator.release(*id);
    EXPECT_FALSE(allocator.allocate().has_value());

    allocator.release(*id);
    EXPECT_TRUE(allocator.allocate().has_value());
}

// TEST(MemoryAllocatorTest, CopyBlockCopiesBytesFromSourceToDestination) {
//     MemoryAllocator allocator(make_test_config(2));

//     const std::optional<PhysicalBlockId> source = allocator.allocate();
//     const std::optional<PhysicalBlockId> destination = allocator.allocate();
//     ASSERT_TRUE(source.has_value());
//     ASSERT_TRUE(destination.has_value());

//     PhysicalBlock* source_block = allocator.block(*source);
//     ASSERT_NE(source_block, nullptr);
//     const std::size_t size = source_block->size_bytes();
//     for (std::size_t i = 0; i < size; ++i) {
//         source_block->data()[i] = static_cast<std::byte>(i % 256);
//     }

//     allocator.copy_block(*source, *destination);

//     const PhysicalBlock* destination_block = allocator.block(*destination);
//     ASSERT_NE(destination_block, nullptr);
//     ASSERT_EQ(destination_block->size_bytes(), size);
//     for (std::size_t i = 0; i < size; ++i) {
//         EXPECT_EQ(destination_block->data()[i], source_block->data()[i]);
//     }
// }

TEST(MemoryAllocatorTest, BlockAndInfoReturnNullptrForInvalidId) {
    MemoryAllocator allocator(make_test_config(1));

    constexpr PhysicalBlockId kInvalidId = 999;
    EXPECT_EQ(allocator.block(kInvalidId), nullptr);
    EXPECT_EQ(allocator.info(kInvalidId), nullptr);
}

TEST(MemoryAllocatorTest, StatsReflectAllocateAndReleaseCalls) {
    MemoryAllocator allocator(make_test_config(4));

    const std::optional<PhysicalBlockId> first = allocator.allocate();
    const std::optional<PhysicalBlockId> second = allocator.allocate();
    ASSERT_TRUE(first.has_value());
    ASSERT_TRUE(second.has_value());

    AllocatorStats stats = allocator.stats();
    EXPECT_EQ(stats.total_blocks, 4u);
    EXPECT_EQ(stats.free_blocks, 2u);
    EXPECT_EQ(stats.active_blocks, 2u);

    allocator.release(*first);
    stats = allocator.stats();
    EXPECT_EQ(stats.free_blocks, 3u);
    EXPECT_EQ(stats.active_blocks, 1u);
}

TEST(MemoryAllocatorTest, CanAllocateReflectsFreeBlockCount) {
    MemoryAllocator allocator(make_test_config(2));

    EXPECT_TRUE(allocator.can_allocate(2));
    EXPECT_FALSE(allocator.can_allocate(3));

    ASSERT_TRUE(allocator.allocate().has_value());
    EXPECT_TRUE(allocator.can_allocate(1));
    EXPECT_FALSE(allocator.can_allocate(2));
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
 * @brief Returns true when a block still holds the pattern written by fill_block.
 */
bool holds_pattern(const PhysicalBlock& block, std::size_t seed) {
    for (std::size_t i = 0; i < block.size_bytes(); ++i) {
        if (block.data()[i] != static_cast<std::byte>((i + seed) % 251)) {
            return false;
        }
    }
    return true;
}

TEST(MemoryAllocatorTest, SwapOutFailsWithoutBackend) {
    MemoryAllocator allocator(make_test_config(1));

    const std::optional<PhysicalBlockId> id = allocator.allocate();
    ASSERT_TRUE(id.has_value());

    EXPECT_FALSE(allocator.swap_out(*id).has_value());
    EXPECT_EQ(allocator.stats().free_blocks, 0u);
}

TEST(MemoryAllocatorTest, SwapOutReclaimsFrameForReuse) {
    MemoryAllocator allocator(make_test_config(1));
    HostSwapBackend backend(4);
    allocator.set_swap_backend(&backend);

    const std::optional<PhysicalBlockId> id = allocator.allocate();
    ASSERT_TRUE(id.has_value());
    ASSERT_FALSE(allocator.can_allocate(1));

    const std::optional<SwapSlotId> slot = allocator.swap_out(*id);

    ASSERT_TRUE(slot.has_value());
    EXPECT_EQ(backend.resident_slots(), 1u);
    EXPECT_TRUE(allocator.can_allocate(1));
    EXPECT_EQ(allocator.stats().free_blocks, 1u);
    EXPECT_TRUE(allocator.allocate().has_value());
}

TEST(MemoryAllocatorTest, SwapOutAdvancesGenerationOfReclaimedFrame) {
    MemoryAllocator allocator(make_test_config(1));
    HostSwapBackend backend(4);
    allocator.set_swap_backend(&backend);

    const std::optional<PhysicalBlockId> id = allocator.allocate();
    ASSERT_TRUE(id.has_value());
    ASSERT_NE(allocator.info(*id), nullptr);
    const std::uint64_t generation_before = allocator.info(*id)->generation;

    ASSERT_TRUE(allocator.swap_out(*id).has_value());

    const PhysicalBlockInfo* info = allocator.info(*id);
    ASSERT_NE(info, nullptr);
    EXPECT_GT(info->generation, generation_before);
    EXPECT_EQ(info->ref_count, 0u);
}

TEST(MemoryAllocatorTest, SwapOutRejectsSharedBlock) {
    MemoryAllocator allocator(make_test_config(2));
    HostSwapBackend backend(4);
    allocator.set_swap_backend(&backend);

    const std::optional<PhysicalBlockId> id = allocator.allocate();
    ASSERT_TRUE(id.has_value());
    allocator.retain(*id);

    EXPECT_FALSE(allocator.swap_out(*id).has_value());
    EXPECT_EQ(backend.resident_slots(), 0u);
    ASSERT_NE(allocator.info(*id), nullptr);
    EXPECT_EQ(allocator.info(*id)->ref_count, 2u);
}

TEST(MemoryAllocatorTest, SwapOutRejectsFreeAndInvalidBlocks) {
    MemoryAllocator allocator(make_test_config(1));
    HostSwapBackend backend(4);
    allocator.set_swap_backend(&backend);

    const std::optional<PhysicalBlockId> id = allocator.allocate();
    ASSERT_TRUE(id.has_value());
    allocator.release(*id);

    EXPECT_FALSE(allocator.swap_out(*id).has_value());
    EXPECT_FALSE(allocator.swap_out(999).has_value());
    EXPECT_EQ(backend.resident_slots(), 0u);
}

TEST(MemoryAllocatorTest, SwapOutFailsWhenSwapSpaceIsFull) {
    MemoryAllocator allocator(make_test_config(2));
    HostSwapBackend backend(1);
    allocator.set_swap_backend(&backend);

    const std::optional<PhysicalBlockId> first = allocator.allocate();
    const std::optional<PhysicalBlockId> second = allocator.allocate();
    ASSERT_TRUE(first.has_value());
    ASSERT_TRUE(second.has_value());
    ASSERT_TRUE(allocator.swap_out(*first).has_value());

    EXPECT_FALSE(allocator.swap_out(*second).has_value());
    ASSERT_NE(allocator.info(*second), nullptr);
    EXPECT_EQ(allocator.info(*second)->ref_count, 1u);
}

TEST(MemoryAllocatorTest, SwapInRestoresContentsIntoAFrame) {
    // A single-frame pool forces swap-in to reuse the same frame after it has
    // been scribbled over, so the restored bytes can only come from the backend.
    MemoryAllocator allocator(make_test_config(1));
    HostSwapBackend backend(4);
    allocator.set_swap_backend(&backend);

    const std::optional<PhysicalBlockId> id = allocator.allocate();
    ASSERT_TRUE(id.has_value());
    ASSERT_NE(allocator.block(*id), nullptr);
    fill_block(*allocator.block(*id), 5);

    const std::optional<SwapSlotId> slot = allocator.swap_out(*id);
    ASSERT_TRUE(slot.has_value());

    const std::optional<PhysicalBlockId> scratch = allocator.allocate();
    ASSERT_TRUE(scratch.has_value());
    fill_block(*allocator.block(*scratch), 90);
    allocator.release(*scratch);

    const std::optional<PhysicalBlockId> restored = allocator.swap_in(*slot);

    ASSERT_TRUE(restored.has_value());
    ASSERT_NE(allocator.block(*restored), nullptr);
    EXPECT_TRUE(holds_pattern(*allocator.block(*restored), 5));

    const PhysicalBlockInfo* info = allocator.info(*restored);
    ASSERT_NE(info, nullptr);
    EXPECT_EQ(info->ref_count, 1u);
    EXPECT_EQ(info->state, BlockState::Active);
}

TEST(MemoryAllocatorTest, SwapInConsumesTheSlot) {
    MemoryAllocator allocator(make_test_config(2));
    HostSwapBackend backend(4);
    allocator.set_swap_backend(&backend);

    const std::optional<PhysicalBlockId> id = allocator.allocate();
    ASSERT_TRUE(id.has_value());
    const std::optional<SwapSlotId> slot = allocator.swap_out(*id);
    ASSERT_TRUE(slot.has_value());

    ASSERT_TRUE(allocator.swap_in(*slot).has_value());

    EXPECT_EQ(backend.resident_slots(), 0u);
    EXPECT_FALSE(allocator.swap_in(*slot).has_value());
}

TEST(MemoryAllocatorTest, SwapInFailsWithoutConsumingSlotWhenPoolIsExhausted) {
    MemoryAllocator allocator(make_test_config(1));
    HostSwapBackend backend(4);
    allocator.set_swap_backend(&backend);

    const std::optional<PhysicalBlockId> id = allocator.allocate();
    ASSERT_TRUE(id.has_value());
    const std::optional<SwapSlotId> slot = allocator.swap_out(*id);
    ASSERT_TRUE(slot.has_value());
    ASSERT_TRUE(allocator.allocate().has_value());

    EXPECT_FALSE(allocator.swap_in(*slot).has_value());
    EXPECT_EQ(backend.resident_slots(), 1u);
}

TEST(MemoryAllocatorTest, SwapInUnknownSlotFails) {
    MemoryAllocator allocator(make_test_config(2));
    HostSwapBackend backend(4);
    allocator.set_swap_backend(&backend);

    EXPECT_FALSE(allocator.swap_in(9999).has_value());
    EXPECT_EQ(allocator.stats().free_blocks, 2u);
}

TEST(MemoryAllocatorTest, StatsCountSwappedBlocksSeparatelyFromFrames) {
    MemoryAllocator allocator(make_test_config(2));
    HostSwapBackend backend(4);
    allocator.set_swap_backend(&backend);

    const std::optional<PhysicalBlockId> first = allocator.allocate();
    const std::optional<PhysicalBlockId> second = allocator.allocate();
    ASSERT_TRUE(first.has_value());
    ASSERT_TRUE(second.has_value());
    ASSERT_EQ(allocator.stats().swapped_blocks, 0u);

    const std::optional<SwapSlotId> first_slot = allocator.swap_out(*first);
    ASSERT_TRUE(first_slot.has_value());
    ASSERT_TRUE(allocator.swap_out(*second).has_value());

    AllocatorStats stats = allocator.stats();
    EXPECT_EQ(stats.swapped_blocks, 2u);
    EXPECT_EQ(stats.free_blocks, 2u);
    EXPECT_EQ(stats.active_blocks, 0u);

    ASSERT_TRUE(allocator.swap_in(*first_slot).has_value());

    stats = allocator.stats();
    EXPECT_EQ(stats.swapped_blocks, 1u);
    EXPECT_EQ(stats.active_blocks, 1u);
}

TEST(MemoryAllocatorTest, SelectEvictionCandidateIsNulloptWithoutSelector) {
    MemoryAllocator allocator(make_test_config(2));
    ASSERT_TRUE(allocator.allocate().has_value());

    EXPECT_FALSE(allocator.select_eviction_candidate().has_value());
}

TEST(MemoryAllocatorTest, SelectEvictionCandidateDelegatesToRegisteredPolicy) {
    MemoryAllocator allocator(make_test_config(2));

    const std::optional<PhysicalBlockId> victim = allocator.allocate();
    ASSERT_TRUE(victim.has_value());

    std::uint32_t observed_total_blocks = 0;
    allocator.set_eviction_selector(
        [&observed_total_blocks, victim](const MemoryAllocator& self) -> std::optional<PhysicalBlockId> {
            observed_total_blocks = self.stats().total_blocks;
            return victim;
        });

    const std::optional<PhysicalBlockId> candidate = allocator.select_eviction_candidate();

    ASSERT_TRUE(candidate.has_value());
    EXPECT_EQ(*candidate, *victim);
    EXPECT_EQ(observed_total_blocks, 2u);
}

TEST(MemoryAllocatorTest, SelectEvictionCandidateHonoursAPolicyThatDeclines) {
    MemoryAllocator allocator(make_test_config(2));
    allocator.set_eviction_selector(
        [](const MemoryAllocator&) { return std::optional<PhysicalBlockId>{}; });

    EXPECT_FALSE(allocator.select_eviction_candidate().has_value());
}

TEST(MemoryAllocatorTest, AllocateNeverEvictsOnItsOwn) {
    // The eviction hook is advisory: only whoever owns the logical mappings may
    // act on it, so an exhausted pool must still fail the allocation.
    MemoryAllocator allocator(make_test_config(1));
    HostSwapBackend backend(4);
    allocator.set_swap_backend(&backend);

    const std::optional<PhysicalBlockId> occupied = allocator.allocate();
    ASSERT_TRUE(occupied.has_value());

    bool selector_invoked = false;
    allocator.set_eviction_selector(
        [&selector_invoked, occupied](const MemoryAllocator&) {
            selector_invoked = true;
            return occupied;
        });

    EXPECT_FALSE(allocator.allocate().has_value());
    EXPECT_FALSE(selector_invoked);
    EXPECT_EQ(backend.resident_slots(), 0u);
}

} // namespace
} // namespace qwenvl_paged
