#pragma once

#include "qwenvl_paged/Block.h"

#include <cstddef>
#include <optional>
#include <vector>

namespace qwenvl_paged {

/**
 * @brief One virtual-to-physical mapping entry in a sequence block table.
 */
struct BlockTableEntry {
    LogicalBlock logical{};
    PhysicalBlockId physical_id{0};
    bool writable{true};
    /**
     * @brief Engaged while this block's contents live in the swap backend.
     *
     * The physical frame has been reclaimed, so `physical_id` is stale until the
     * block is swapped back in and the entry is remapped.
     */
    std::optional<SwapSlotId> swap_slot{};
};

/**
 * @brief OS-like page table for one request sequence and one cache stream.
 *
 * A block table maps logical token positions to physical KV cache blocks. The
 * table can be forked for parallel sampling, where children initially share
 * physical blocks with their parent and later materialize private blocks via
 * copy-on-write.
 *
 * This phase-1 table is not thread-safe and should be mutated only by the
 * engine event loop through KVCacheManager or MemoryAllocator operations.
 */
class BlockTable {
public:
    /**
     * @brief Creates an empty block table for a sequence.
     */
    explicit BlockTable(SequenceId sequence_id, CacheKind cache_kind = CacheKind::TextKV);

    ~BlockTable() = default;

    BlockTable(const BlockTable&) = default;
    BlockTable& operator=(const BlockTable&) = default;
    BlockTable(BlockTable&&) noexcept = default;
    BlockTable& operator=(BlockTable&&) noexcept = default;

    /**
     * @brief Returns the sequence that owns this virtual address space.
     */
    [[nodiscard]] SequenceId sequence_id() const noexcept;

    /**
     * @brief Returns the cache stream represented by this table.
     */
    [[nodiscard]] CacheKind cache_kind() const noexcept;

    /**
     * @brief Returns the number of mapped logical blocks.
     */
    [[nodiscard]] std::size_t size() const noexcept;

    /**
     * @brief Returns true when no logical block is mapped.
     */
    [[nodiscard]] bool empty() const noexcept;

    /**
     * @brief Adds or replaces one logical-to-physical mapping.
     */
    void map(LogicalBlock logical, PhysicalBlockId physical_id, bool writable);

    /**
     * @brief Removes a mapping and returns the physical block if present.
     */
    [[nodiscard]] std::optional<PhysicalBlockId> unmap(LogicalBlockIndex index);

    /**
     * @brief Resolves a logical block index into a physical block identifier.
     *
     * Returns nullopt for an unmapped index and for a swapped-out block, whose
     * frame has been reclaimed and must be swapped back in before any access.
     */
    [[nodiscard]] std::optional<PhysicalBlockId> lookup(LogicalBlockIndex index) const;

    /**
     * @brief Returns the complete mapping entry for a logical block if present.
     */
    [[nodiscard]] const BlockTableEntry* entry(LogicalBlockIndex index) const noexcept;

    /**
     * @brief Returns a mutable mapping entry for allocator-controlled updates.
     */
    [[nodiscard]] BlockTableEntry* mutable_entry(LogicalBlockIndex index) noexcept;

    /**
     * @brief Marks a mapped logical block writable or shared.
     */
    void set_writable(LogicalBlockIndex index, bool writable);

    /**
     * @brief Returns all mapping entries in logical-block order.
     */
    [[nodiscard]] const std::vector<BlockTableEntry>& entries() const noexcept;

    /**
     * @brief Creates a child table that shares every current physical block.
     *
     * The returned table has writable flags cleared so the allocator must call
     * copy-on-write before either branch mutates a shared block.
     */
    [[nodiscard]] BlockTable fork(SequenceId child_sequence_id) const;

private:
    SequenceId sequence_id_{0};
    CacheKind cache_kind_{CacheKind::TextKV};
    std::vector<BlockTableEntry> entries_;
};

} // namespace qwenvl_paged
