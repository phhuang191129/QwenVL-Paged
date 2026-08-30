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

/**
 * @brief Returns the byte distance between consecutive frames for a config.
 *
 * The block's byte size rounded up to the configured alignment. Exposed so a
 * caller sizing storage for the adopting constructor arrives at the same stride
 * the allocator will use.
 */
[[nodiscard]] std::size_t block_stride_for(const AllocatorConfig& config) noexcept;

/**
 * @brief Returns the total bytes a pool for this config needs.
 */
[[nodiscard]] std::size_t pool_bytes_for(const AllocatorConfig& config) noexcept;

class MemoryAllocator;

/**
 * @brief Moves one block's bytes to another, in place of the default memcpy.
 *
 * Both blocks belong to the pool, so an implementation only needs to know how
 * to address its own frames. See `MemoryAllocator::set_copy_hook`.
 */
using BlockCopyHook = std::function<void(PhysicalBlockId source, PhysicalBlockId destination)>;

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
     * @brief Creates a fixed-capacity block pool over storage it allocates itself.
     */
    explicit MemoryAllocator(AllocatorConfig config);

    /**
     * @brief Creates a pool over a slab the caller owns.
     *
     * `adopted_pool` must hold at least `pool_bytes_for(config)` bytes, meet the
     * configured alignment, and outlive the allocator. This is the seam a device
     * pool arrives through: torch allocates the tensor and keeps it alive, the
     * allocator does the bookkeeping, and neither has to know about the other's
     * allocator.
     *
     * Note what an adopted slab must still be, though. Three sites dereference
     * frame bytes on the host: `copy_block`, the swap backend, and the CPU
     * reference kernel reached through `CacheView`. Storage the CPU cannot
     * address therefore disables copy-on-write, swapping, and
     * `paged_attention_decode`. `set_copy_hook` lifts the first of those, which
     * is the one copy-on-write needs; the other two remain host-only.
     */
    MemoryAllocator(AllocatorConfig config, std::byte* adopted_pool);

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
     * @brief Replaces the host memcpy `copy_block` performs by default.
     *
     * The one site that has to move bytes for copy-on-write, and therefore the
     * one that decides whether a pool can live somewhere the CPU cannot reach.
     * The hook takes block ids rather than pointers so an owner of device
     * storage can address its own frames however it likes without being handed
     * a pointer it must not dereference.
     */
    void set_copy_hook(BlockCopyHook hook);

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
     * @brief Drops a swapped block's contents without restoring them.
     *
     * Needed when a sequence is released while some of its blocks are evicted,
     * so finished work does not leak swap slots.
     */
    void discard_swapped(SwapSlotId slot);

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

    /**
     * @brief Returns the base of the single slab backing every frame.
     *
     * Frame `id` occupies `[pool_base() + id * block_stride_bytes(),
     * block_stride_bytes())`. An execution backend mirrors the pool by copying
     * this range and indexing it by the same ids, which is what lets a device
     * kernel resolve a block without a host-supplied pointer table.
     */
    [[nodiscard]] std::byte* pool_base() noexcept;

    /**
     * @brief Returns the immutable base of the slab backing every frame.
     */
    [[nodiscard]] const std::byte* pool_base() const noexcept;

    /**
     * @brief Returns the byte distance between consecutive frames.
     *
     * This is `BlockShape::byte_size()` rounded up to the configured alignment,
     * so it can exceed the bytes a block's contents actually occupy.
     */
    [[nodiscard]] std::size_t block_stride_bytes() const noexcept;

    /**
     * @brief Returns the total byte size of the slab.
     */
    [[nodiscard]] std::size_t pool_bytes() const noexcept;

    /**
     * @brief Returns the byte alignment applied to the pool and to each frame.
     */
    [[nodiscard]] std::size_t alignment_bytes() const noexcept;

    /**
     * @brief Returns true when the pool allocation was requested as pinned.
     */
    [[nodiscard]] bool pinned_memory_requested() const noexcept;

    /**
     * @brief Returns true when the slab belongs to the caller rather than the pool.
     */
    [[nodiscard]] bool owns_pool() const noexcept;

private:
    static void release_pool_memory(std::byte* ptr) noexcept;

    void bind_frames(std::byte* base);

    AllocatorConfig config_{};
    std::size_t block_stride_bytes_{0};
    std::size_t pool_bytes_{0};
    std::unique_ptr<std::byte, decltype(&MemoryAllocator::release_pool_memory)> owned_pool_{
        nullptr,
        &MemoryAllocator::release_pool_memory};
    std::byte* pool_base_{nullptr};
    std::vector<PhysicalBlock> blocks_;
    std::vector<PhysicalBlockInfo> infos_;
    std::vector<PhysicalBlockId> free_list_;
    SwapBackend* swap_backend_{nullptr};
    EvictionCandidateSelector eviction_selector_{};
    BlockCopyHook copy_hook_{};
};

} // namespace qwenvl_paged
