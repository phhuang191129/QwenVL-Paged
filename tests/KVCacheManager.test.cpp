/**
 * @brief Specification tests for qwenvl_paged::KVCacheManager.
 *
 * These tests are written against the documented header contract in
 * include/qwenvl_paged/KVCacheManager.h. KVCacheManager (and the MemoryAllocator
 * it depends on) have no .cpp implementation yet, so this test binary is
 * expected to fail at the LINK stage (undefined references) until those classes
 * are implemented. That failure is intentional for this phase of the project.
 *
 * The fixture uses a block shape with tokens_per_block == 16, so a reservation
 * of N * 16 tokens is expected to map exactly N logical blocks.
 */

#include "qwenvl_paged/KVCacheManager.h"
#include "qwenvl_paged/MemoryAllocator.h"
#include "qwenvl_paged/SwapBackend.h"

#include <gtest/gtest.h>

#include <cstddef>
#include <optional>

namespace qwenvl_paged {
namespace {

constexpr std::uint32_t kTokensPerBlock = 16;

/**
 * @brief Builds an allocator config with enough blocks for the tests.
 */
AllocatorConfig make_allocator_config(std::uint32_t max_blocks = 32) {
    AllocatorConfig config;
    config.block_shape.tokens_per_block = kTokensPerBlock;
    config.block_shape.num_layers = 2;
    config.block_shape.num_kv_heads = 2;
    config.block_shape.head_dim = 8;
    config.block_shape.bytes_per_element = 2;
    config.max_blocks = max_blocks;
    return config;
}

/**
 * @brief Builds minimal metadata for a text-only sequence.
 */
SequenceMetadata make_metadata(SequenceId sequence_id, RequestId request_id = 1) {
    SequenceMetadata metadata;
    metadata.sequence_id = sequence_id;
    metadata.request_id = request_id;
    return metadata;
}

TEST(KVCacheManagerTest, CreateSequenceIsTracked) {
    MemoryAllocator allocator(make_allocator_config());
    KVCacheManager manager(allocator);

    EXPECT_FALSE(manager.contains(1));
    EXPECT_TRUE(manager.create_sequence(make_metadata(1)));
    EXPECT_TRUE(manager.contains(1));
}

TEST(KVCacheManagerTest, NewSequenceHasEmptyCacheView) {
    MemoryAllocator allocator(make_allocator_config());
    KVCacheManager manager(allocator);
    ASSERT_TRUE(manager.create_sequence(make_metadata(1)));

    const std::optional<CacheView> view = manager.cache_view(1);
    ASSERT_TRUE(view.has_value());
    EXPECT_EQ(view->sequence_id, 1u);
    EXPECT_EQ(view->cache_kind, CacheKind::TextKV);
    ASSERT_NE(view->block_table, nullptr);
    EXPECT_TRUE(view->block_table->empty());
}

TEST(KVCacheManagerTest, CacheViewForUnknownSequenceIsNullopt) {
    MemoryAllocator allocator(make_allocator_config());
    KVCacheManager manager(allocator);

    EXPECT_FALSE(manager.cache_view(99).has_value());
}

TEST(KVCacheManagerTest, ReserveTokensMapsWholeLogicalBlocks) {
    MemoryAllocator allocator(make_allocator_config());
    KVCacheManager manager(allocator);
    ASSERT_TRUE(manager.create_sequence(make_metadata(1)));

    ASSERT_TRUE(manager.reserve_tokens(1, kTokensPerBlock * 2));

    const std::optional<CacheView> view = manager.cache_view(1);
    ASSERT_TRUE(view.has_value());
    ASSERT_NE(view->block_table, nullptr);
    EXPECT_EQ(view->block_table->size(), 2u);
}

TEST(KVCacheManagerTest, ReserveTokensOnUnknownSequenceFails) {
    MemoryAllocator allocator(make_allocator_config());
    KVCacheManager manager(allocator);

    EXPECT_FALSE(manager.reserve_tokens(99, kTokensPerBlock));
}

TEST(KVCacheManagerTest, ReleaseSequenceReturnsBlocksToAllocator) {
    MemoryAllocator allocator(make_allocator_config());
    KVCacheManager manager(allocator);
    ASSERT_TRUE(manager.create_sequence(make_metadata(1)));
    ASSERT_TRUE(manager.reserve_tokens(1, kTokensPerBlock * 2));

    const std::uint32_t free_before_release = allocator.stats().free_blocks;
    manager.release_sequence(1);

    EXPECT_FALSE(manager.contains(1));
    EXPECT_GT(allocator.stats().free_blocks, free_before_release);
}

TEST(KVCacheManagerTest, ForkSharesPhysicalBlocksAndBumpsRefCount) {
    MemoryAllocator allocator(make_allocator_config());
    KVCacheManager manager(allocator);
    ASSERT_TRUE(manager.create_sequence(make_metadata(1)));
    ASSERT_TRUE(manager.reserve_tokens(1, kTokensPerBlock));

    const std::optional<CacheView> parent_view = manager.cache_view(1);
    ASSERT_TRUE(parent_view.has_value());
    const std::optional<PhysicalBlockId> shared = parent_view->block_table->lookup(0);
    ASSERT_TRUE(shared.has_value());

    ASSERT_TRUE(manager.fork_sequence(1, make_metadata(2)));
    EXPECT_TRUE(manager.contains(2));

    const std::optional<CacheView> child_view = manager.cache_view(2);
    ASSERT_TRUE(child_view.has_value());
    EXPECT_EQ(child_view->block_table->lookup(0), shared);

    const PhysicalBlockInfo* info = allocator.info(*shared);
    ASSERT_NE(info, nullptr);
    EXPECT_EQ(info->ref_count, 2u);
}

TEST(KVCacheManagerTest, ForkUnknownParentFails) {
    MemoryAllocator allocator(make_allocator_config());
    KVCacheManager manager(allocator);

    EXPECT_FALSE(manager.fork_sequence(99, make_metadata(2)));
}

TEST(KVCacheManagerTest, EnsureTokenWritableCopiesSharedBlock) {
    MemoryAllocator allocator(make_allocator_config());
    KVCacheManager manager(allocator);
    ASSERT_TRUE(manager.create_sequence(make_metadata(1)));
    ASSERT_TRUE(manager.reserve_tokens(1, kTokensPerBlock));
    ASSERT_TRUE(manager.fork_sequence(1, make_metadata(2)));

    const std::optional<PhysicalBlockId> shared =
        manager.cache_view(1)->block_table->lookup(0);
    ASSERT_TRUE(shared.has_value());

    const std::optional<PhysicalBlockId> writable =
        manager.ensure_token_writable(2, 0);
    ASSERT_TRUE(writable.has_value());
    EXPECT_NE(*writable, *shared);

    EXPECT_EQ(manager.cache_view(2)->block_table->lookup(0), writable);

    const PhysicalBlockInfo* old_info = allocator.info(*shared);
    ASSERT_NE(old_info, nullptr);
    EXPECT_EQ(old_info->ref_count, 1u);
}

TEST(KVCacheManagerTest, EnsureTokenWritableIsInPlaceForSoleOwner) {
    MemoryAllocator allocator(make_allocator_config());
    KVCacheManager manager(allocator);
    ASSERT_TRUE(manager.create_sequence(make_metadata(1)));
    ASSERT_TRUE(manager.reserve_tokens(1, kTokensPerBlock));

    const std::optional<PhysicalBlockId> owned =
        manager.cache_view(1)->block_table->lookup(0);
    ASSERT_TRUE(owned.has_value());
    const PhysicalBlockInfo* info = allocator.info(*owned);
    ASSERT_NE(info, nullptr);
    ASSERT_EQ(info->ref_count, 1u);

    const std::uint32_t free_before = allocator.stats().free_blocks;

    const std::optional<PhysicalBlockId> writable =
        manager.ensure_token_writable(1, 0);

    ASSERT_TRUE(writable.has_value());
    EXPECT_EQ(*writable, *owned);
    EXPECT_EQ(manager.cache_view(1)->block_table->lookup(0), owned);
    EXPECT_EQ(allocator.stats().free_blocks, free_before);
}

TEST(KVCacheManagerTest, EnsureTokenWritableOnUnknownSequenceIsNullopt) {
    MemoryAllocator allocator(make_allocator_config());
    KVCacheManager manager(allocator);

    EXPECT_FALSE(manager.ensure_token_writable(99, 0).has_value());
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

TEST(KVCacheManagerTest, SwapOutSequenceReclaimsFramesAndMarksEntries) {
    MemoryAllocator allocator(make_allocator_config(4));
    HostSwapBackend backend(8);
    allocator.set_swap_backend(&backend);
    KVCacheManager manager(allocator);
    ASSERT_TRUE(manager.create_sequence(make_metadata(1)));
    ASSERT_TRUE(manager.reserve_tokens(1, kTokensPerBlock * 2));
    ASSERT_EQ(allocator.stats().free_blocks, 2u);

    EXPECT_EQ(manager.swap_out_sequence(1), 2u);

    EXPECT_EQ(allocator.stats().free_blocks, 4u);
    EXPECT_EQ(backend.resident_slots(), 2u);

    const std::optional<CacheView> view = manager.cache_view(1);
    ASSERT_TRUE(view.has_value());
    ASSERT_NE(view->block_table, nullptr);
    // The logical mapping survives eviction; only the frame is gone.
    EXPECT_EQ(view->block_table->size(), 2u);
    for (const BlockTableEntry& entry : view->block_table->entries()) {
        EXPECT_TRUE(entry.swap_slot.has_value());
        EXPECT_FALSE(view->block_table->lookup(entry.logical.index).has_value());
    }
}

TEST(KVCacheManagerTest, SwapOutSequenceWithoutBackendSwapsNothing) {
    MemoryAllocator allocator(make_allocator_config(4));
    KVCacheManager manager(allocator);
    ASSERT_TRUE(manager.create_sequence(make_metadata(1)));
    ASSERT_TRUE(manager.reserve_tokens(1, kTokensPerBlock));

    EXPECT_EQ(manager.swap_out_sequence(1), 0u);
    EXPECT_TRUE(manager.cache_view(1)->block_table->lookup(0).has_value());
}

TEST(KVCacheManagerTest, SwapOutSequenceSkipsBlocksSharedWithAFork) {
    MemoryAllocator allocator(make_allocator_config(4));
    HostSwapBackend backend(8);
    allocator.set_swap_backend(&backend);
    KVCacheManager manager(allocator);
    ASSERT_TRUE(manager.create_sequence(make_metadata(1)));
    ASSERT_TRUE(manager.reserve_tokens(1, kTokensPerBlock));
    ASSERT_TRUE(manager.fork_sequence(1, make_metadata(2)));

    // The sibling branch still maps this frame, so evicting it would strand it.
    EXPECT_EQ(manager.swap_out_sequence(1), 0u);
    EXPECT_EQ(backend.resident_slots(), 0u);
    EXPECT_TRUE(manager.cache_view(1)->block_table->lookup(0).has_value());
    EXPECT_TRUE(manager.cache_view(2)->block_table->lookup(0).has_value());
}

TEST(KVCacheManagerTest, SwapOutUnknownSequenceSwapsNothing) {
    MemoryAllocator allocator(make_allocator_config(4));
    HostSwapBackend backend(8);
    allocator.set_swap_backend(&backend);
    KVCacheManager manager(allocator);

    EXPECT_EQ(manager.swap_out_sequence(99), 0u);
}

TEST(KVCacheManagerTest, SwapInSequenceRestoresMappingsAndContents) {
    // A single-frame pool forces the swap-in to reuse the frame after it has
    // been scribbled over, so restored bytes can only come from the backend.
    MemoryAllocator allocator(make_allocator_config(1));
    HostSwapBackend backend(8);
    allocator.set_swap_backend(&backend);
    KVCacheManager manager(allocator);
    ASSERT_TRUE(manager.create_sequence(make_metadata(1)));
    ASSERT_TRUE(manager.reserve_tokens(1, kTokensPerBlock));

    const std::optional<PhysicalBlockId> original = manager.cache_view(1)->block_table->lookup(0);
    ASSERT_TRUE(original.has_value());
    ASSERT_NE(allocator.block(*original), nullptr);
    fill_block(*allocator.block(*original), 21);

    ASSERT_EQ(manager.swap_out_sequence(1), 1u);

    const std::optional<PhysicalBlockId> scratch = allocator.allocate();
    ASSERT_TRUE(scratch.has_value());
    fill_block(*allocator.block(*scratch), 130);
    allocator.release(*scratch);

    EXPECT_TRUE(manager.swap_in_sequence(1));

    const std::optional<PhysicalBlockId> restored = manager.cache_view(1)->block_table->lookup(0);
    ASSERT_TRUE(restored.has_value());
    EXPECT_FALSE(manager.cache_view(1)->block_table->entry(0)->swap_slot.has_value());
    EXPECT_EQ(backend.resident_slots(), 0u);
    ASSERT_NE(allocator.block(*restored), nullptr);
    EXPECT_TRUE(holds_pattern(*allocator.block(*restored), 21));
}

TEST(KVCacheManagerTest, SwapInSequenceFailsAtomicallyWhenPoolCannotBackIt) {
    MemoryAllocator allocator(make_allocator_config(2));
    HostSwapBackend backend(8);
    allocator.set_swap_backend(&backend);
    KVCacheManager manager(allocator);
    ASSERT_TRUE(manager.create_sequence(make_metadata(1)));
    ASSERT_TRUE(manager.reserve_tokens(1, kTokensPerBlock * 2));
    ASSERT_EQ(manager.swap_out_sequence(1), 2u);

    // Leave room for only one of the two swapped blocks.
    ASSERT_TRUE(allocator.allocate().has_value());
    ASSERT_EQ(allocator.stats().free_blocks, 1u);

    EXPECT_FALSE(manager.swap_in_sequence(1));

    EXPECT_EQ(allocator.stats().free_blocks, 1u);
    EXPECT_EQ(backend.resident_slots(), 2u);
    for (const BlockTableEntry& entry : manager.cache_view(1)->block_table->entries()) {
        EXPECT_TRUE(entry.swap_slot.has_value());
    }
}

TEST(KVCacheManagerTest, SwapInSequenceIsANoOpForAResidentSequence) {
    MemoryAllocator allocator(make_allocator_config(4));
    HostSwapBackend backend(8);
    allocator.set_swap_backend(&backend);
    KVCacheManager manager(allocator);
    ASSERT_TRUE(manager.create_sequence(make_metadata(1)));
    ASSERT_TRUE(manager.reserve_tokens(1, kTokensPerBlock));

    const std::uint32_t free_before = allocator.stats().free_blocks;

    EXPECT_TRUE(manager.swap_in_sequence(1));
    EXPECT_EQ(allocator.stats().free_blocks, free_before);
}

TEST(KVCacheManagerTest, ReleaseSequenceDiscardsSwappedSlots) {
    MemoryAllocator allocator(make_allocator_config(4));
    HostSwapBackend backend(8);
    allocator.set_swap_backend(&backend);
    KVCacheManager manager(allocator);
    ASSERT_TRUE(manager.create_sequence(make_metadata(1)));
    ASSERT_TRUE(manager.reserve_tokens(1, kTokensPerBlock * 2));
    ASSERT_EQ(manager.swap_out_sequence(1), 2u);

    manager.release_sequence(1);

    EXPECT_FALSE(manager.contains(1));
    EXPECT_EQ(backend.resident_slots(), 0u);
    const AllocatorStats stats = allocator.stats();
    EXPECT_EQ(stats.free_blocks, 4u);
    EXPECT_EQ(stats.swapped_blocks, 0u);
}

TEST(KVCacheManagerTest, ForkRefusesASwappedOutParent) {
    MemoryAllocator allocator(make_allocator_config(4));
    HostSwapBackend backend(8);
    allocator.set_swap_backend(&backend);
    KVCacheManager manager(allocator);
    ASSERT_TRUE(manager.create_sequence(make_metadata(1)));
    ASSERT_TRUE(manager.reserve_tokens(1, kTokensPerBlock));
    ASSERT_EQ(manager.swap_out_sequence(1), 1u);

    EXPECT_FALSE(manager.fork_sequence(1, make_metadata(2)));
    EXPECT_FALSE(manager.contains(2));

    ASSERT_TRUE(manager.swap_in_sequence(1));
    EXPECT_TRUE(manager.fork_sequence(1, make_metadata(2)));
}

} // namespace
} // namespace qwenvl_paged
