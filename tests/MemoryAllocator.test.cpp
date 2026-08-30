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

#include "qwenvl_paged/KVCacheManager.h"
#include "qwenvl_paged/MemoryAllocator.h"
#include "qwenvl_paged/SwapBackend.h"

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <optional>
#include <utility>
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

TEST(MemoryAllocatorTest, PoolIsOneSlabAddressableByBlockId) {
    // The invariant an execution backend depends on: a PhysicalBlockId is an
    // offset into one allocation, so a device mirror can be produced by copying
    // the slab and indexed by the same ids. The Triton decode kernel resolves a
    // block as physical_id * elements_per_block with no pointer table, which is
    // only correct if this holds.
    constexpr std::uint32_t kBlockCount = 64;
    MemoryAllocator allocator(make_test_config(kBlockCount));

    ASSERT_NE(allocator.pool_base(), nullptr);
    EXPECT_EQ(allocator.pool_bytes(), allocator.block_stride_bytes() * kBlockCount);

    for (std::uint32_t id = 0; id < kBlockCount; ++id) {
        const PhysicalBlock* frame = allocator.block(id);
        ASSERT_NE(frame, nullptr);
        EXPECT_EQ(frame->data(), allocator.pool_base() + id * allocator.block_stride_bytes());
        EXPECT_EQ(frame->size_bytes(), allocator.block_stride_bytes());
    }
}

TEST(MemoryAllocatorTest, AdoptsASlabTheCallerOwns) {
    // How a device pool arrives: torch allocates and keeps the tensor alive, the
    // allocator only does the bookkeeping. Sizing it from the free helpers has to
    // agree with what the allocator then computes for itself, or the last frame
    // hangs off the end of the caller's buffer.
    constexpr std::uint32_t kBlockCount = 16;
    const AllocatorConfig config = make_test_config(kBlockCount);

    std::vector<std::byte> storage(pool_bytes_for(config));
    MemoryAllocator allocator(config, storage.data());

    EXPECT_FALSE(allocator.owns_pool());
    EXPECT_EQ(allocator.pool_base(), storage.data());
    EXPECT_EQ(allocator.block_stride_bytes(), block_stride_for(config));
    EXPECT_EQ(allocator.pool_bytes(), storage.size());

    for (std::uint32_t id = 0; id < kBlockCount; ++id) {
        const PhysicalBlock* frame = allocator.block(id);
        ASSERT_NE(frame, nullptr);
        EXPECT_EQ(frame->data(), storage.data() + id * allocator.block_stride_bytes());
    }

    // The last frame must end exactly at the end of what the caller allocated.
    const PhysicalBlock* last = allocator.block(kBlockCount - 1);
    ASSERT_NE(last, nullptr);
    EXPECT_EQ(last->data() + last->size_bytes(), storage.data() + storage.size());
}

TEST(MemoryAllocatorTest, AdoptedSlabIsWrittenThroughRatherThanCopied) {
    const AllocatorConfig config = make_test_config(4);
    std::vector<std::byte> storage(pool_bytes_for(config), std::byte{0});
    MemoryAllocator allocator(config, storage.data());

    const std::optional<PhysicalBlockId> id = allocator.allocate();
    ASSERT_TRUE(id.has_value());
    allocator.block(*id)->data()[0] = std::byte{0xCD};

    EXPECT_EQ(storage[*id * allocator.block_stride_bytes()], std::byte{0xCD});
}

TEST(MemoryAllocatorTest, OwnedPoolReportsOwnership) {
    MemoryAllocator allocator(make_test_config());

    EXPECT_TRUE(allocator.owns_pool());
    EXPECT_NE(allocator.pool_base(), nullptr);
}

TEST(MemoryAllocatorTest, FramesDoNotOverlap) {
    constexpr std::uint32_t kBlockCount = 8;
    MemoryAllocator allocator(make_test_config(kBlockCount));

    // Stamp each frame with its own id, then read every frame back. Any stride
    // smaller than a frame would have let a later write bleed into an earlier
    // frame's bytes.
    for (std::uint32_t id = 0; id < kBlockCount; ++id) {
        PhysicalBlock* frame = allocator.block(id);
        ASSERT_NE(frame, nullptr);
        std::memset(frame->data(), static_cast<int>(id), frame->size_bytes());
    }

    for (std::uint32_t id = 0; id < kBlockCount; ++id) {
        const PhysicalBlock* frame = allocator.block(id);
        ASSERT_NE(frame, nullptr);
        for (std::size_t i = 0; i < frame->size_bytes(); ++i) {
            ASSERT_EQ(frame->data()[i], static_cast<std::byte>(id)) << "frame " << id << " byte " << i;
        }
    }
}

TEST(MemoryAllocatorTest, EveryFrameSatisfiesRequestedAlignment) {
    // A single frame can be aligned by luck even from an allocator that ignores
    // alignment, so checking the whole pool makes an unaligned implementation
    // fail with near-certainty rather than by chance.
    constexpr std::uint32_t kBlockCount = 64;
    AllocatorConfig config = make_test_config(kBlockCount);
    config.memory_options.alignment_bytes = kDefaultBlockAlignmentBytes;
    MemoryAllocator allocator(config);

    EXPECT_EQ(allocator.alignment_bytes(), kDefaultBlockAlignmentBytes);

    for (std::uint32_t id = 0; id < kBlockCount; ++id) {
        const PhysicalBlock* frame = allocator.block(id);
        ASSERT_NE(frame, nullptr);
        EXPECT_EQ(reinterpret_cast<std::uintptr_t>(frame->data()) % kDefaultBlockAlignmentBytes, 0u);
    }
}

TEST(MemoryAllocatorTest, BlockStrideCoversShapeAndIsAlignmentRounded) {
    AllocatorConfig config = make_test_config();
    config.memory_options.alignment_bytes = kDefaultBlockAlignmentBytes;
    MemoryAllocator allocator(config);

    EXPECT_GE(allocator.block_stride_bytes(), config.block_shape.byte_size());
    EXPECT_EQ(allocator.block_stride_bytes() % kDefaultBlockAlignmentBytes, 0u);
}

TEST(MemoryAllocatorTest, PinnedMemoryRequestIsReported) {
    AllocatorConfig config = make_test_config();
    config.memory_options.prefer_pinned_memory = true;
    MemoryAllocator allocator(config);

    EXPECT_TRUE(allocator.pinned_memory_requested());
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

// The seam a pool the CPU cannot address depends on. copy_block is the only
// site copy-on-write needs to move bytes through, so redirecting it is what
// lets an owner of device storage keep forking.
TEST(MemoryAllocatorTest, CopyHookReplacesTheDefaultMemcpy) {
    MemoryAllocator allocator(make_test_config(2));

    const std::optional<PhysicalBlockId> source = allocator.allocate();
    const std::optional<PhysicalBlockId> destination = allocator.allocate();
    ASSERT_TRUE(source.has_value());
    ASSERT_TRUE(destination.has_value());

    PhysicalBlock* source_block = allocator.block(*source);
    PhysicalBlock* destination_block = allocator.block(*destination);
    ASSERT_NE(source_block, nullptr);
    ASSERT_NE(destination_block, nullptr);
    source_block->data()[0] = std::byte{0xAB};
    destination_block->data()[0] = std::byte{0x00};

    std::optional<std::pair<PhysicalBlockId, PhysicalBlockId>> observed;
    allocator.set_copy_hook([&observed](PhysicalBlockId from, PhysicalBlockId to) {
        observed = std::make_pair(from, to);
    });

    allocator.copy_block(*source, *destination);

    ASSERT_TRUE(observed.has_value());
    EXPECT_EQ(observed->first, *source);
    EXPECT_EQ(observed->second, *destination);
    EXPECT_EQ(destination_block->data()[0], std::byte{0x00})
        << "the default memcpy still ran, so the hook did not replace it";
}

TEST(MemoryAllocatorTest, CopyOnWriteRoutesThroughTheCopyHook) {
    MemoryAllocator allocator(make_test_config(4));
    KVCacheManager cache(allocator);

    constexpr SequenceId kParent = 1;
    constexpr SequenceId kChild = 2;
    ASSERT_TRUE(cache.create_sequence(SequenceMetadata{kParent, kParent, {}, {}}));
    ASSERT_TRUE(cache.reserve_tokens(kParent, 1));
    ASSERT_TRUE(cache.fork_sequence(kParent, SequenceMetadata{kChild, kChild, {}, {}}));

    std::uint32_t calls = 0;
    allocator.set_copy_hook([&calls](PhysicalBlockId, PhysicalBlockId) { ++calls; });

    EXPECT_TRUE(cache.ensure_token_writable(kChild, 0).has_value());
    EXPECT_EQ(calls, 1u) << "materializing a shared block did not go through the hook";
}

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
