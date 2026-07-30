#pragma once

#include "qwenvl_paged/Block.h"
#include "qwenvl_paged/SwapBackend.h"

#include <cstddef>
#include <cstdint>
#include <functional>
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
    /**
     * @brief Blocks whose contents currently live in the swap backend.
     *
     * These hold no physical frame, so they are counted independently of the
     * free/active/shared frame states above.
     */
    std::uint32_t swapped_blocks{0};
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

class MemoryAllocator;

/**
 * @brief Policy hook naming a block worth evicting under cache pressure.
 *
 * The selector only chooses; it must not mutate allocator state. Executing the
 * eviction stays with whoever owns the logical mappings, because the allocator
 * cannot know which sequence would have to be remapped.
 */
using EvictionCandidateSelector =
    std::function<std::optional<PhysicalBlockId>(const MemoryAllocator&)>;

/**
 * @brief Owns physical KV cache blocks and implements copy-on-write.
 *
 * MemoryAllocator is the only component allowed to create, recycle, share, or
 * duplicate physical blocks. It exposes small page-allocation primitives to
 * BlockTable and KVCacheManager while hiding free-list and refcount details.
 *
 * This phase-1 allocator is not thread-safe. All mutation must occur on the
 * engine event loop so free-list updates and refcount transitions are observed
 * in a deterministic order.
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
     * @brief Installs the backend that holds evicted block contents.
     *
     * The backend is a non-owning dependency and must outlive the allocator.
     * Passing nullptr disables swapping.
     */
    void set_swap_backend(SwapBackend* backend) noexcept;

    /**
     * @brief Evicts a block's contents to the swap backend and reclaims its frame.
     *
     * On success the frame returns to the free list and its generation advances,
     * so any block table still mapping this id must be repointed at the returned
     * slot. Fails when no backend is installed, the block is free, the block is
     * shared by more than one mapping, or the backend is full.
     */
    [[nodiscard]] std::optional<SwapSlotId> swap_out(PhysicalBlockId id);

    /**
     * @brief Restores a swapped slot into a newly allocated physical frame.
     *
     * The slot is discarded once its bytes are restored. Fails without consuming
     * the slot when no backend is installed, the slot is unknown, or the pool has
     * no free frame.
     */
    [[nodiscard]] std::optional<PhysicalBlockId> swap_in(SwapSlotId slot);

    /**
     * @brief Registers the eviction candidate selection policy.
     */
    void set_eviction_selector(EvictionCandidateSelector selector);

    /**
     * @brief Asks the registered policy for a block worth evicting.
     *
     * Advisory only: `allocate` never evicts on its own. Returns nullopt when no
     * selector is registered or the policy names no candidate.
     */
    [[nodiscard]] std::optional<PhysicalBlockId> select_eviction_candidate() const;

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
    SwapBackend* swap_backend_{nullptr};
    EvictionCandidateSelector eviction_selector_{};
};

} // namespace qwenvl_paged
