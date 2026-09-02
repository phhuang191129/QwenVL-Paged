#pragma once

#include <cstddef>
#include <cstdint>
#include <utility>

namespace qwenvl_paged {

using RequestId = std::uint64_t;
using SequenceId = std::uint64_t;
using TokenPosition = std::uint32_t;
using LogicalBlockIndex = std::uint32_t;
using PhysicalBlockId = std::uint32_t;
using SwapSlotId = std::uint64_t;

inline constexpr std::size_t kDefaultBlockAlignmentBytes = 256;

/**
 * @brief Identifies the type of cache data stored in a physical block.
 *
 * Qwen3-VL may need different cache streams for text tokens, visual features,
 * or future modality-specific state. The allocator treats all streams as block
 * backed memory while higher layers interpret the layout.
 */
enum class CacheKind : std::uint8_t {
    TextKV,
    VisionKV,
    RopeState,
    Auxiliary
};

/**
 * @brief Runtime state of a physical block inside the allocator pool.
 */
enum class BlockState : std::uint8_t {
    Free,
    Active,
    Shared,
    Swapped
};

/**
 * @brief Static shape and capacity metadata for one physical cache block.
 */
struct BlockShape {
    std::uint32_t tokens_per_block{0};
    std::uint32_t num_layers{0};
    std::uint32_t num_kv_heads{0};
    std::uint32_t head_dim{0};
    std::uint32_t bytes_per_element{0};
    /**
     * @brief Layers stored in one physical frame. Zero means `num_layers`
     *        (every layer in one frame). Set to 1 for layer-major packing:
     *        one frame holds one layer's K/V for `tokens_per_block` tokens.
     */
    std::uint32_t layers_per_frame{0};

    /**
     * @brief Layers actually stored in one allocated frame.
     */
    [[nodiscard]] std::uint32_t frame_layers() const noexcept {
        return layers_per_frame == 0 ? num_layers : layers_per_frame;
    }

    /**
     * @brief True when each physical frame holds a single layer.
     */
    [[nodiscard]] bool per_layer_frames() const noexcept {
        return frame_layers() == 1 && num_layers > 1;
    }

    /**
     * @brief Returns the number of bytes needed by a full K/V cache block.
     */
    [[nodiscard]] std::size_t byte_size() const noexcept;
};

/**
 * @brief Host allocation policy for the block pool.
 *
 * The alignment applies to the pool base and to the per-frame stride, so every
 * frame inherits it. The CPU build uses C++17 `std::aligned_alloc` with
 * `std::free`; a CUDA build can switch that single allocation site to
 * `cudaMallocHost`/`cudaFreeHost` or `cudaMalloc` without changing scheduler or
 * block-table ownership semantics.
 */
struct HostMemoryOptions {
    std::size_t alignment_bytes{kDefaultBlockAlignmentBytes};
    bool prefer_pinned_memory{false};
};

/**
 * @brief Logical page descriptor used by a sequence-local block table.
 *
 * Logical blocks are virtual memory slots. They do not own storage and can be
 * remapped to different physical blocks during copy-on-write or swap recovery.
 */
struct LogicalBlock {
    SequenceId sequence_id{0};
    LogicalBlockIndex index{0};
    CacheKind cache_kind{CacheKind::TextKV};
    TokenPosition start_token{0};
    std::uint32_t token_count{0};

    /**
     * @brief Returns true when the logical block has no committed tokens.
     */
    [[nodiscard]] bool empty() const noexcept;
};

/**
 * @brief One frame in the allocator's contiguous block pool.
 *
 * A frame does not own its bytes. MemoryAllocator allocates the whole pool as a
 * single aligned slab and binds frame `i` to the range at
 * `i * MemoryAllocator::block_stride_bytes()`, so a PhysicalBlockId is an offset
 * into that slab rather than a handle to an independent allocation.
 *
 * That is a requirement rather than a tidiness preference. An execution backend
 * has to mirror this pool in device memory, and the GPU kernel in
 * `tools/triton_kernel/paged_attention_decode.py` addresses it arithmetically as
 * `physical_id * elements_per_block` with no pointer table. Per-frame
 * allocations can satisfy neither, because nothing constrains where they land
 * relative to each other. Switching the pool to `cudaMalloc` is then one
 * allocation site, not one per block.
 */
class PhysicalBlock {
public:
    /**
     * @brief Binds a frame identity and shape to storage owned by the pool.
     */
    PhysicalBlock(
        PhysicalBlockId id,
        BlockShape shape,
        std::byte* storage,
        std::size_t size_bytes) noexcept;

    /**
     * @brief Returns the allocator-wide stable block identifier.
     */
    [[nodiscard]] PhysicalBlockId id() const noexcept;

    /**
     * @brief Returns immutable shape metadata for this block.
     */
    [[nodiscard]] const BlockShape& shape() const noexcept;

    /**
     * @brief Returns mutable access to the raw block bytes.
     */
    [[nodiscard]] std::byte* data() noexcept;

    /**
     * @brief Returns immutable access to the raw block bytes.
     */
    [[nodiscard]] const std::byte* data() const noexcept;

    /**
     * @brief Returns the byte capacity of this frame's slice of the pool.
     */
    [[nodiscard]] std::size_t size_bytes() const noexcept;

private:
    PhysicalBlockId id_{0};
    BlockShape shape_{};
    std::byte* storage_{nullptr};
    std::size_t size_bytes_{0};
};

} // namespace qwenvl_paged
