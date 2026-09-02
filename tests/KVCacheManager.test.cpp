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
#include "qwenvl_paged/PagedAttention.h"
#include "qwenvl_paged/SwapBackend.h"

#include <gtest/gtest.h>

#include <cstddef>
#include <optional>
#include <string>
#include <vector>

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

TEST(KVCacheManagerTest, SequencesForReturnsForksOfTheSameRequest) {
    MemoryAllocator allocator(make_allocator_config());
    KVCacheManager manager(allocator);
    ASSERT_TRUE(manager.create_sequence(make_metadata(1, 7)));
    ASSERT_TRUE(manager.fork_sequence(1, make_metadata(2, 7)));

    const std::vector<SequenceId> ids = manager.sequences_for(7);
    ASSERT_EQ(ids.size(), 2u);
    EXPECT_EQ(ids[0], 1u);
    EXPECT_EQ(ids[1], 2u);
    EXPECT_TRUE(manager.sequences_for(99).empty());
}

TEST(KVCacheManagerTest, ReleaseRequestDropsEverySequence) {
    MemoryAllocator allocator(make_allocator_config());
    KVCacheManager manager(allocator);
    ASSERT_TRUE(manager.create_sequence(make_metadata(1, 7)));
    ASSERT_TRUE(manager.reserve_tokens(1, kTokensPerBlock));
    ASSERT_TRUE(manager.fork_sequence(1, make_metadata(2, 7)));

    manager.release_request(7);

    EXPECT_FALSE(manager.contains(1));
    EXPECT_FALSE(manager.contains(2));
    EXPECT_TRUE(manager.sequences_for(7).empty());
    EXPECT_EQ(allocator.stats().free_blocks, allocator.stats().total_blocks);
}

TEST(KVCacheManagerTest, PrefixPublishAndAttachSharesBlocks) {
    MemoryAllocator allocator(make_allocator_config());
    KVCacheManager manager(allocator);
    ASSERT_TRUE(manager.create_sequence(make_metadata(1, 1)));
    ASSERT_TRUE(manager.reserve_tokens(1, kTokensPerBlock * 2));
    fill_block(*allocator.block(*manager.cache_view(1)->block_table->lookup(0)), 11);
    fill_block(*allocator.block(*manager.cache_view(1)->block_table->lookup(1)), 12);

    ASSERT_TRUE(manager.publish_prefix(1, "sys-img"));
    EXPECT_EQ(manager.prefix_token_count("sys-img"), kTokensPerBlock * 2);

    ASSERT_TRUE(manager.create_sequence(make_metadata(2, 2)));
    EXPECT_EQ(manager.attach_prefix(2, "sys-img"), kTokensPerBlock * 2);
    EXPECT_EQ(manager.cache_view(2)->block_table->lookup(0), manager.cache_view(1)->block_table->lookup(0));
    EXPECT_EQ(allocator.info(*manager.cache_view(1)->block_table->lookup(0))->ref_count, 3u);

    const std::optional<PhysicalBlockId> child = manager.ensure_token_writable(2, 0);
    ASSERT_TRUE(child.has_value());
    EXPECT_NE(*child, *manager.cache_view(1)->block_table->lookup(0));
}

TEST(KVCacheManagerTest, PrefixCollisionDoesNotShare) {
    MemoryAllocator allocator(make_allocator_config());
    KVCacheManager manager(allocator);
    ASSERT_TRUE(manager.create_sequence(make_metadata(1, 1)));
    ASSERT_TRUE(manager.create_sequence(make_metadata(2, 2)));
    ASSERT_TRUE(manager.reserve_tokens(1, kTokensPerBlock));
    ASSERT_TRUE(manager.reserve_tokens(2, kTokensPerBlock));
    fill_block(*allocator.block(*manager.cache_view(1)->block_table->lookup(0)), 1);
    fill_block(*allocator.block(*manager.cache_view(2)->block_table->lookup(0)), 99);

    ASSERT_TRUE(manager.publish_prefix(1, "same-key"));
    ASSERT_TRUE(manager.publish_prefix(2, "same-key"));
    EXPECT_EQ(manager.prefix_token_count("same-key"), 0u);

    ASSERT_TRUE(manager.create_sequence(make_metadata(3, 3)));
    EXPECT_EQ(manager.attach_prefix(3, "same-key"), 0u);
    EXPECT_TRUE(manager.cache_view(3)->block_table->empty());
}

TEST(KVCacheManagerTest, PrefixReleaseOfOneSharerLeavesTheOther) {
    MemoryAllocator allocator(make_allocator_config());
    KVCacheManager manager(allocator);
    ASSERT_TRUE(manager.create_sequence(make_metadata(1, 1)));
    ASSERT_TRUE(manager.reserve_tokens(1, kTokensPerBlock));
    fill_block(*allocator.block(*manager.cache_view(1)->block_table->lookup(0)), 3);
    ASSERT_TRUE(manager.publish_prefix(1, "shared"));

    ASSERT_TRUE(manager.create_sequence(make_metadata(2, 2)));
    ASSERT_EQ(manager.attach_prefix(2, "shared"), kTokensPerBlock);

    manager.release_sequence(1);
    EXPECT_TRUE(manager.contains(2));
    EXPECT_TRUE(manager.cache_view(2)->block_table->lookup(0).has_value());
    EXPECT_EQ(manager.prefix_token_count("shared"), kTokensPerBlock);

    manager.release_sequence(2);
    EXPECT_EQ(allocator.stats().free_blocks, allocator.stats().total_blocks);
    EXPECT_EQ(allocator.stats().shared_blocks, 0u);
    EXPECT_EQ(manager.prefix_token_count("shared"), 0u);
}

TEST(KVCacheManagerTest, PerLayerReserveAllocatesOneFramePerLayerAndPacksLayerMajor) {
    AllocatorConfig config = make_allocator_config(16);
    config.block_shape.layers_per_frame = 1;
    MemoryAllocator allocator(config);
    KVCacheManager manager(allocator);
    ASSERT_TRUE(manager.create_sequence(make_metadata(1)));
    ASSERT_TRUE(manager.reserve_tokens(1, kTokensPerBlock * 2));

    EXPECT_EQ(allocator.stats().free_blocks, 12u);

    const std::optional<CacheView> view = manager.cache_view(1);
    ASSERT_TRUE(view.has_value());
    ASSERT_NE(view->layer_tables, nullptr);
    ASSERT_EQ(view->layer_tables->size(), 2u);
    EXPECT_EQ(view->block_table->size(), 2u);
    EXPECT_EQ((*view->layer_tables)[1].size(), 2u);

    const PhysicalBlockId l0_0 = *(*view->layer_tables)[0].lookup(0);
    const PhysicalBlockId l0_1 = *(*view->layer_tables)[0].lookup(1);
    const PhysicalBlockId l1_0 = *(*view->layer_tables)[1].lookup(0);
    // The free list pops descending ids, so a bulk reserve packs as n, n-1, ...
    EXPECT_EQ(l0_0, l0_1 + 1);
    EXPECT_EQ(l0_1, l1_0 + 1);
}

TEST(KVCacheManagerTest, PerLayerPrefixAttachMapsEveryLayer) {
    AllocatorConfig config = make_allocator_config(16);
    config.block_shape.layers_per_frame = 1;
    MemoryAllocator allocator(config);
    KVCacheManager manager(allocator);
    ASSERT_TRUE(manager.create_sequence(make_metadata(1, 1)));
    ASSERT_TRUE(manager.reserve_tokens(1, kTokensPerBlock));
    fill_block(*allocator.block(*manager.cache_view(1)->layer_tables->front().lookup(0)), 11);
    fill_block(*allocator.block(*(*manager.cache_view(1)->layer_tables)[1].lookup(0)), 12);

    ASSERT_TRUE(manager.publish_prefix(1, "per-layer"));
    ASSERT_TRUE(manager.create_sequence(make_metadata(2, 2)));
    EXPECT_EQ(manager.attach_prefix(2, "per-layer"), kTokensPerBlock);

    const std::optional<CacheView> parent = manager.cache_view(1);
    const std::optional<CacheView> child = manager.cache_view(2);
    ASSERT_TRUE(parent.has_value());
    ASSERT_TRUE(child.has_value());
    EXPECT_EQ((*child->layer_tables)[0].lookup(0), (*parent->layer_tables)[0].lookup(0));
    EXPECT_EQ((*child->layer_tables)[1].lookup(0), (*parent->layer_tables)[1].lookup(0));
}

TEST(KVCacheManagerTest, SharedPrefixProducesIdenticalAttention) {
    AllocatorConfig config = make_allocator_config();
    config.block_shape.bytes_per_element = static_cast<std::uint32_t>(sizeof(float));
    MemoryAllocator allocator(config);
    KVCacheManager manager(allocator);

    ASSERT_TRUE(manager.create_sequence(make_metadata(1, 1)));
    ASSERT_TRUE(manager.create_sequence(make_metadata(2, 2)));
    ASSERT_TRUE(manager.reserve_tokens(1, kTokensPerBlock));
    ASSERT_TRUE(manager.reserve_tokens(2, kTokensPerBlock));

    const KVBlockLayout layout{allocator.block(0)->shape()};
    auto write_pattern = [&](SequenceId seq, std::uint32_t seed) {
        const PhysicalBlockId id = *manager.ensure_token_writable(seq, 0);
        float* base = reinterpret_cast<float*>(allocator.block(id)->data());
        for (std::uint32_t i = 0; i < layout.element_count(); ++i) {
            base[i] = static_cast<float>((i + seed) % 17) * 0.1F;
        }
    };
    write_pattern(1, 4);
    write_pattern(2, 4);
    ASSERT_TRUE(manager.publish_prefix(1, "attn"));
    ASSERT_TRUE(manager.create_sequence(make_metadata(3, 3)));
    ASSERT_EQ(manager.attach_prefix(3, "attn"), kTokensPerBlock);

    const std::vector<float> query(16, 0.25F);
    std::vector<float> independent(16, 0.0F);
    std::vector<float> shared(16, 0.0F);
    PagedAttentionParams params;
    params.layer = 0;
    params.num_query_heads = 2;
    params.context_len = 1;
    params.scale = 1.0F;
    ASSERT_TRUE(paged_attention_decode<float>(*manager.cache_view(2), query.data(), params, independent.data()));
    ASSERT_TRUE(paged_attention_decode<float>(*manager.cache_view(3), query.data(), params, shared.data()));
    for (std::size_t i = 0; i < independent.size(); ++i) {
        EXPECT_FLOAT_EQ(independent[i], shared[i]);
    }
}

} // namespace
} // namespace qwenvl_paged
