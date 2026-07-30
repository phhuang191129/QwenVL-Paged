/**
 * @brief Specification tests for qwenvl_paged::BlockTable.
 *
 * These tests are written against the documented header contract in
 * include/qwenvl_paged/BlockTable.h. BlockTable has no .cpp implementation yet,
 * so this test binary is expected to fail at the LINK stage (undefined
 * references) until the class is implemented. That failure is intentional for
 * this phase of the project.
 */

#include "qwenvl_paged/BlockTable.h"

#include <gtest/gtest.h>

namespace qwenvl_paged {
namespace {

/**
 * @brief Builds a logical block descriptor for a given logical index.
 */
LogicalBlock make_logical(SequenceId sequence_id, LogicalBlockIndex index) {
    LogicalBlock logical;
    logical.sequence_id = sequence_id;
    logical.index = index;
    logical.cache_kind = CacheKind::TextKV;
    logical.start_token = index * 16;
    logical.token_count = 16;
    return logical;
}

TEST(BlockTableTest, NewTableExposesIdentityAndIsEmpty) {
    BlockTable table(42, CacheKind::VisionKV);

    EXPECT_EQ(table.sequence_id(), 42u);
    EXPECT_EQ(table.cache_kind(), CacheKind::VisionKV);
    EXPECT_TRUE(table.empty());
    EXPECT_EQ(table.size(), 0u);
}

TEST(BlockTableTest, MapThenLookupResolvesPhysicalId) {
    BlockTable table(1);
    table.map(make_logical(1, 0), 100, true);

    const std::optional<PhysicalBlockId> resolved = table.lookup(0);
    ASSERT_TRUE(resolved.has_value());
    EXPECT_EQ(*resolved, 100u);
    EXPECT_EQ(table.size(), 1u);
    EXPECT_FALSE(table.empty());
}

TEST(BlockTableTest, MapReplacesExistingMapping) {
    BlockTable table(1);
    table.map(make_logical(1, 0), 100, true);
    table.map(make_logical(1, 0), 200, true);

    const std::optional<PhysicalBlockId> resolved = table.lookup(0);
    ASSERT_TRUE(resolved.has_value());
    EXPECT_EQ(*resolved, 200u);
    EXPECT_EQ(table.size(), 1u);
}

TEST(BlockTableTest, LookupMissReturnsNullopt) {
    BlockTable table(1);
    EXPECT_FALSE(table.lookup(5).has_value());
}

TEST(BlockTableTest, UnmapRemovesMappingAndReturnsPhysicalId) {
    BlockTable table(1);
    table.map(make_logical(1, 0), 100, true);

    const std::optional<PhysicalBlockId> removed = table.unmap(0);
    ASSERT_TRUE(removed.has_value());
    EXPECT_EQ(*removed, 100u);
    EXPECT_FALSE(table.lookup(0).has_value());
}

TEST(BlockTableTest, UnmapMissingReturnsNullopt) {
    BlockTable table(1);
    EXPECT_FALSE(table.unmap(0).has_value());
}

TEST(BlockTableTest, EntryReportsWritableFlagAndMissReturnsNull) {
    BlockTable table(1);
    table.map(make_logical(1, 0), 100, false);

    const BlockTableEntry* present = table.entry(0);
    ASSERT_NE(present, nullptr);
    EXPECT_EQ(present->physical_id, 100u);
    EXPECT_FALSE(present->writable);

    EXPECT_EQ(table.entry(1), nullptr);
}

TEST(BlockTableTest, SetWritableUpdatesEntry) {
    BlockTable table(1);
    table.map(make_logical(1, 0), 100, false);

    table.set_writable(0, true);

    const BlockTableEntry* entry = table.entry(0);
    ASSERT_NE(entry, nullptr);
    EXPECT_TRUE(entry->writable);
}

TEST(BlockTableTest, MutableEntryAllowsInPlaceUpdate) {
    BlockTable table(1);
    table.map(make_logical(1, 0), 100, true);

    BlockTableEntry* entry = table.mutable_entry(0);
    ASSERT_NE(entry, nullptr);
    entry->physical_id = 555;

    const std::optional<PhysicalBlockId> resolved = table.lookup(0);
    ASSERT_TRUE(resolved.has_value());
    EXPECT_EQ(*resolved, 555u);
}

TEST(BlockTableTest, EntriesReturnedInLogicalOrder) {
    BlockTable table(1);
    // Insert out of logical order to prove entries() sorts by logical index
    // rather than merely echoing insertion order.
    table.map(make_logical(1, 2), 102, true);
    table.map(make_logical(1, 0), 100, true);
    table.map(make_logical(1, 1), 101, true);

    const std::vector<BlockTableEntry>& entries = table.entries();
    ASSERT_EQ(entries.size(), 3u);
    EXPECT_EQ(entries[0].logical.index, 0u);
    EXPECT_EQ(entries[0].physical_id, 100u);
    EXPECT_EQ(entries[1].logical.index, 1u);
    EXPECT_EQ(entries[1].physical_id, 101u);
    EXPECT_EQ(entries[2].logical.index, 2u);
    EXPECT_EQ(entries[2].physical_id, 102u);
}

TEST(BlockTableTest, MappedEntryStartsWithoutASwapSlot) {
    BlockTable table(1);
    table.map(make_logical(1, 0), 100, true);

    const BlockTableEntry* entry = table.entry(0);
    ASSERT_NE(entry, nullptr);
    EXPECT_FALSE(entry->swap_slot.has_value());
}

TEST(BlockTableTest, LookupFaultsOnASwappedOutEntry) {
    BlockTable table(1);
    table.map(make_logical(1, 0), 100, true);

    BlockTableEntry* entry = table.mutable_entry(0);
    ASSERT_NE(entry, nullptr);
    entry->swap_slot = SwapSlotId{7};

    // The frame behind physical_id has been reclaimed, so the mapping must not
    // resolve until the block is swapped back in, but the entry stays in the
    // table so the swap slot can be found again.
    EXPECT_FALSE(table.lookup(0).has_value());
    EXPECT_EQ(table.size(), 1u);
    ASSERT_NE(table.entry(0), nullptr);
    EXPECT_EQ(table.entry(0)->swap_slot, SwapSlotId{7});
}

TEST(BlockTableTest, LookupResolvesAgainAfterSwapSlotIsCleared) {
    BlockTable table(1);
    table.map(make_logical(1, 0), 100, true);

    BlockTableEntry* entry = table.mutable_entry(0);
    ASSERT_NE(entry, nullptr);
    entry->swap_slot = SwapSlotId{7};
    ASSERT_FALSE(table.lookup(0).has_value());

    entry->physical_id = 200;
    entry->swap_slot.reset();

    const std::optional<PhysicalBlockId> resolved = table.lookup(0);
    ASSERT_TRUE(resolved.has_value());
    EXPECT_EQ(*resolved, 200u);
}

TEST(BlockTableTest, ForkSharesPhysicalBlocksWithWritableCleared) {
    BlockTable parent(1);
    parent.map(make_logical(1, 0), 100, true);
    parent.map(make_logical(1, 1), 101, true);

    const BlockTable child = parent.fork(2);

    EXPECT_EQ(child.sequence_id(), 2u);
    ASSERT_EQ(child.size(), parent.size());

    EXPECT_EQ(child.lookup(0), parent.lookup(0));
    EXPECT_EQ(child.lookup(1), parent.lookup(1));

    for (const BlockTableEntry& entry : child.entries()) {
        EXPECT_FALSE(entry.writable);
    }
}

} // namespace
} // namespace qwenvl_paged
