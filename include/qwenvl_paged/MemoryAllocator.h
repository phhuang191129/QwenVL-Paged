#pragma once

#include "qwenvl_paged/Block.h"
#include "qwenvl_paged/BlockTable.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

namespace qwenvl_paged {

/**
 * @brief Immutable configuration for the physical block allocator.
 */
struct AllocatorConfig {
    BlockShape block_shape{};
    HostMemoryOptions memory_options{};
    std::uint32_t max_blocks{0};
};

/**
 * @brief Observable allocator counters used by scheduling policy.
 */
struct AllocatorStats {
    std::uint32_t total_blocks{0};
    std::uint32_t free_blocks{0};
    std::uint32_t active_blocks{0};
    std::uint32_t shared_blocks{0};
    std::size_t bytes_reserved{0};
};

/**
 * @brief Internal metadata tracked for each physical block.
 */
struct PhysicalBlockInfo {
    PhysicalBlockId id{0};
    BlockState state{BlockState::Free};
    /**
     * @brief Non-atomic reference count owned by the engine event loop.
     *
     * This must become atomic or allocator-mutex protected if blocks can be
     * retained or released from multiple threads in a future design.
     */
    std::uint32_t ref_count{0};
    std::uint64_t generation{0};
};

/**
 * @brief Owns physical KV cache blocks and implements copy-on-write.
 *
 * MemoryAllocator is the only component allowed to create, recycle, share, or
 * duplicate physical blocks. It exposes small page-allocation primitives to
 * BlockTable and KVCacheManager while hiding free-list and refcount details.
 *
 * This phase-1 allocator is not thread-safe. All mutation must occur on the
 * engine event loop so `ensure_writable`, free-list updates, and refcount
 * transitions are observed in a deterministic order.
 */
class MemoryAllocator {
public:
    /**
     * @brief Creates a fixed-capacity CPU block pool.
     */
    explicit MemoryAllocator(AllocatorConfig config);

    ~MemoryAllocator() = default;

    MemoryAllocator(const MemoryAllocator&) = delete;
    MemoryAllocator& operator=(const MemoryAllocator&) = delete;
    MemoryAllocator(MemoryAllocator&&) noexcept = default;
    MemoryAllocator& operator=(MemoryAllocator&&) noexcept = default;

    /**
     * @brief Allocates one free physical block if capacity is available.
     */
    [[nodiscard]] std::optional<PhysicalBlockId> allocate();

    /**
     * @brief Releases one reference to a physical block.
     *
     * When the reference count reaches zero, the block is returned to the free
     * list and its generation is advanced to invalidate stale observations.
     */
    void release(PhysicalBlockId id);

    /**
     * @brief Adds one shared reference to a physical block.
     */
    void retain(PhysicalBlockId id);

    /**
     * @brief Ensures that a logical block table entry is privately writable.
     *
     * If the entry already has a single reference, it is marked writable and
     * returned. If the physical block is shared, a new block is allocated, bytes
     * are copied, the old reference is released, and the table is remapped.
     */
    [[nodiscard]] std::optional<PhysicalBlockId> ensure_writable(
        BlockTable& table,
        LogicalBlockIndex index);

    /**
     * @brief Copies the bytes from one physical block to another.
     */
    void copy_block(PhysicalBlockId source, PhysicalBlockId destination);

    /**
     * @brief Returns a mutable physical block by id, or nullptr if invalid.
     */
    [[nodiscard]] PhysicalBlock* block(PhysicalBlockId id) noexcept;

    /**
     * @brief Returns an immutable physical block by id, or nullptr if invalid.
     */
    [[nodiscard]] const PhysicalBlock* block(PhysicalBlockId id) const noexcept;

    /**
     * @brief Returns immutable metadata for a physical block, or nullptr if invalid.
     */
    [[nodiscard]] const PhysicalBlockInfo* info(PhysicalBlockId id) const noexcept;

    /**
     * @brief Returns a point-in-time allocator statistics snapshot.
     */
    [[nodiscard]] AllocatorStats stats() const noexcept;

    /**
     * @brief Returns true when at least the requested number of blocks is free.
     */
    [[nodiscard]] bool can_allocate(std::uint32_t block_count) const noexcept;

private:
    AllocatorConfig config_{};
    std::vector<std::unique_ptr<PhysicalBlock>> blocks_;
    std::vector<PhysicalBlockInfo> infos_;
    std::vector<PhysicalBlockId> free_list_;
};

} // namespace qwenvl_paged
