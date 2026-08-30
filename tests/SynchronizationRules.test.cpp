/**
 * @brief Executable form of the synchronization rules in docs/architecture.md.
 *
 * The rules were written as prose for a future asynchronous backend, and the
 * verification tables in docs/performance.md have carried "every rule
 * exercised" as their one unchecked box since week 15. Each rule that can be
 * observed from a single thread gets a test here, named for the rule it pins
 * down, so the box can be checked against something that runs.
 *
 * Two cannot be tested as written and say so in place rather than being faked:
 * rule 3 needs completion events that no engine produces yet, and rule 4 is a
 * threading discipline with no single-threaded observable. Rule 3 still gets a
 * test of the hazard it exists to prevent, because that hazard is real today.
 */

#include "qwenvl_paged/KVCacheManager.h"
#include "qwenvl_paged/MemoryAllocator.h"
#include "qwenvl_paged/PagedAttention.h"
#include "qwenvl_paged/SwapBackend.h"

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

namespace qwenvl_paged {
namespace {

constexpr std::uint32_t kTokensPerBlock = 4;
constexpr std::uint32_t kHeadDim = 4;
constexpr std::uint32_t kContextLen = 8;
constexpr SequenceId kParent = 1;
constexpr SequenceId kChild = 2;

AllocatorConfig make_config(std::uint32_t max_blocks = 8) {
    AllocatorConfig config;
    config.block_shape.tokens_per_block = kTokensPerBlock;
    config.block_shape.num_layers = 1;
    config.block_shape.num_kv_heads = 1;
    config.block_shape.head_dim = kHeadDim;
    config.block_shape.bytes_per_element = static_cast<std::uint32_t>(sizeof(float));
    config.max_blocks = max_blocks;
    return config;
}

/**
 * @brief Creates a sequence with kContextLen tokens of reserved, written cache.
 */
void make_sequence(KVCacheManager& cache, MemoryAllocator& allocator, SequenceId sequence) {
    ASSERT_TRUE(cache.create_sequence(SequenceMetadata{sequence, sequence, {}, {}}));
    ASSERT_TRUE(cache.reserve_tokens(sequence, kContextLen));

    for (std::uint32_t token = 0; token < kContextLen; ++token) {
        const std::optional<PhysicalBlockId> physical = cache.ensure_token_writable(sequence, token);
        ASSERT_TRUE(physical.has_value());
        PhysicalBlock* block = allocator.block(*physical);
        ASSERT_NE(block, nullptr);
        std::fill_n(reinterpret_cast<float*>(block->data()), block->size_bytes() / sizeof(float), 1.0F);
    }
}

PhysicalBlockId first_frame(KVCacheManager& cache, SequenceId sequence) {
    const std::optional<CacheView> view = cache.cache_view(sequence);
    EXPECT_TRUE(view.has_value());
    return view->block_table->entries().front().physical_id;
}

// --- Rule 1: views are borrowed, not owned -------------------------------

TEST(SynchronizationRuleTest, ResolvedBlockPointerDoesNotSurviveACacheMutation) {
    MemoryAllocator allocator(make_config());
    KVCacheManager cache(allocator);
    make_sequence(cache, allocator, kParent);
    ASSERT_TRUE(cache.fork_sequence(kParent, SequenceMetadata{kChild, kChild, {}, {}}));

    const std::optional<CacheView> view = cache.cache_view(kChild);
    ASSERT_TRUE(view.has_value());
    const std::byte* before = view->block_bytes(0);
    ASSERT_NE(before, nullptr);

    // Copy-on-write remaps the child's logical block 0 to a fresh frame. A
    // backend that cached the pointer above and enqueued against it next step
    // would now be reading the frame the parent kept.
    ASSERT_TRUE(cache.ensure_token_writable(kChild, 0).has_value());

    EXPECT_NE(view->block_bytes(0), before);
}

// --- Rule 2: copy-on-write happens before the write is enqueued ----------

TEST(SynchronizationRuleTest, ForkedBranchesShareAFrameUntilEnsureTokenWritable) {
    MemoryAllocator allocator(make_config());
    KVCacheManager cache(allocator);
    make_sequence(cache, allocator, kParent);
    ASSERT_TRUE(cache.fork_sequence(kParent, SequenceMetadata{kChild, kChild, {}, {}}));

    const PhysicalBlockId shared = first_frame(cache, kParent);
    ASSERT_EQ(first_frame(cache, kChild), shared);

    const PhysicalBlockInfo* info = allocator.info(shared);
    ASSERT_NE(info, nullptr);
    EXPECT_EQ(info->ref_count, 2u) << "the fork did not actually share the frame";

    // This is the call the rule says must happen on the engine thread before a
    // kernel is enqueued. A kernel handed the view's frame instead would write
    // straight through into the parent's cache.
    const std::optional<PhysicalBlockId> writable = cache.ensure_token_writable(kChild, 0);
    ASSERT_TRUE(writable.has_value());
    EXPECT_NE(*writable, shared);

    constexpr float kSentinel = 42.0F;
    reinterpret_cast<float*>(allocator.block(*writable)->data())[0] = kSentinel;

    EXPECT_NE(reinterpret_cast<const float*>(allocator.block(shared)->data())[0], kSentinel)
        << "writing through the materialized frame reached the parent";
    EXPECT_EQ(first_frame(cache, kParent), shared) << "the parent was remapped by the child's write";
}

// --- Rule 3: a frame may not be recycled while a kernel still reads it ---

/**
 * Rule 3 cannot be tested as written: it obliges an engine to wait on a step's
 * completion event before releasing a frame, and nothing in the tree produces
 * completion events yet. What is testable is the hazard that makes the rule
 * necessary, and it is not hypothetical -- release publishes the frame for
 * reuse with no grace period at all.
 */
TEST(SynchronizationRuleTest, ReleasedFrameIsHandedToTheNextAllocateImmediately) {
    MemoryAllocator allocator(make_config());

    const std::optional<PhysicalBlockId> first = allocator.allocate();
    ASSERT_TRUE(first.has_value());
    allocator.release(*first);

    const std::optional<PhysicalBlockId> second = allocator.allocate();
    ASSERT_TRUE(second.has_value());
    EXPECT_EQ(*second, *first) << "an in-flight kernel reading the first frame would now be racing "
                                  "whoever was handed the second";
}

// --- Rule 4: reference counts stay on the engine thread ------------------

// No test. The rule forbids calling retain, release, or the free list from a
// stream callback or worker thread, and a single-threaded process cannot
// observe the difference. Enforcing it needs either a thread-sanitized test
// with a real backend thread or an annotation the compiler checks; both wait on
// there being an asynchronous backend to violate it.

// --- Rule 5: block identity is stable, block placement is not ------------

TEST(SynchronizationRuleTest, RecycledFrameKeepsItsStorageButAdvancesItsGeneration) {
    MemoryAllocator allocator(make_config());

    const std::optional<PhysicalBlockId> id = allocator.allocate();
    ASSERT_TRUE(id.has_value());
    const PhysicalBlockInfo* info = allocator.info(*id);
    ASSERT_NE(info, nullptr);

    const std::uint64_t generation_before = info->generation;
    const std::byte* storage_before = allocator.block(*id)->data();

    allocator.release(*id);
    const std::optional<PhysicalBlockId> reused = allocator.allocate();
    ASSERT_TRUE(reused.has_value());
    ASSERT_EQ(*reused, *id);

    // Placement is fixed by construction now that the pool is one slab, so an id
    // always addresses the same bytes. Identity is therefore not enough to tell
    // a live observation from a stale one, which is exactly what generation is
    // for: the id and the address both match, and only the counter reveals that
    // the frame has been through the free list.
    EXPECT_EQ(allocator.block(*id)->data(), storage_before);
    EXPECT_GT(allocator.info(*id)->generation, generation_before);
}

TEST(SynchronizationRuleTest, CopyOnWriteRemapsTheLogicalBlockWithoutMovingAnyFrame) {
    MemoryAllocator allocator(make_config());
    KVCacheManager cache(allocator);
    make_sequence(cache, allocator, kParent);
    ASSERT_TRUE(cache.fork_sequence(kParent, SequenceMetadata{kChild, kChild, {}, {}}));

    const PhysicalBlockId shared = first_frame(cache, kChild);
    const std::byte* shared_storage = allocator.block(shared)->data();

    ASSERT_TRUE(cache.ensure_token_writable(kChild, 0).has_value());

    EXPECT_NE(first_frame(cache, kChild), shared) << "the logical block was not remapped";
    EXPECT_EQ(allocator.block(shared)->data(), shared_storage) << "an allocated frame moved";
}

// --- Rule 6: swapped-out blocks have no frame at all ---------------------

TEST(SynchronizationRuleTest, SwappedOutSequenceHasNoFrameAndIsRefusedWhole) {
    MemoryAllocator allocator(make_config());
    HostSwapBackend backend(8);
    allocator.set_swap_backend(&backend);
    KVCacheManager cache(allocator);
    make_sequence(cache, allocator, kParent);

    ASSERT_GT(cache.swap_out_sequence(kParent), 0u);

    const std::optional<CacheView> view = cache.cache_view(kParent);
    ASSERT_TRUE(view.has_value());
    EXPECT_EQ(view->block_bytes(0), nullptr);
    EXPECT_EQ(view->slot<float>(KVStream::Key, 0, 0, 0), nullptr);

    const std::vector<float> query(kHeadDim, 1.0F);
    constexpr float kUntouched = -7.0F;
    std::vector<float> out(kHeadDim, kUntouched);

    PagedAttentionParams params;
    params.layer = 0;
    params.num_query_heads = 1;
    params.context_len = kContextLen;
    params.scale = 1.0F;

    EXPECT_FALSE(paged_attention_decode<float>(*view, query.data(), params, out.data()));
    for (const float value : out) {
        EXPECT_EQ(value, kUntouched) << "the kernel wrote a partial result before faulting";
    }
}

TEST(SynchronizationRuleTest, SwapInRestoresTheFrameAndTheKernelAcceptsTheCallAgain) {
    MemoryAllocator allocator(make_config());
    HostSwapBackend backend(8);
    allocator.set_swap_backend(&backend);
    KVCacheManager cache(allocator);
    make_sequence(cache, allocator, kParent);

    ASSERT_GT(cache.swap_out_sequence(kParent), 0u);
    ASSERT_TRUE(cache.swap_in_sequence(kParent));

    const std::optional<CacheView> view = cache.cache_view(kParent);
    ASSERT_TRUE(view.has_value());
    EXPECT_NE(view->block_bytes(0), nullptr);

    const std::vector<float> query(kHeadDim, 1.0F);
    std::vector<float> out(kHeadDim, 0.0F);

    PagedAttentionParams params;
    params.layer = 0;
    params.num_query_heads = 1;
    params.context_len = kContextLen;
    params.scale = 1.0F;

    EXPECT_TRUE(paged_attention_decode<float>(*view, query.data(), params, out.data()));
}

} // namespace
} // namespace qwenvl_paged
