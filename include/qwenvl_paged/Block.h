#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
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
     * @brief Returns the number of bytes needed by a full K/V cache block.
     */
    [[nodiscard]] std::size_t byte_size() const noexcept;
};

/**
 * @brief Host allocation policy for CPU-backed physical blocks.
 *
 * The first implementation should use C++17 `std::aligned_alloc` with
 * `std::free`. Future CUDA builds can switch this allocation boundary to
 * `cudaMallocHost`/`cudaFreeHost` without changing scheduler or block-table
 * ownership semantics.
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
 * @brief RAII-owned CPU memory backing one physical PagedAttention block.
 *
 * The CPU prototype owns aligned host bytes with a smart pointer and custom
 * deleter. Future GPU backends can replace the allocation functions with pinned
 * host memory or device-aware handles while preserving the physical block
 * identity and lifecycle contract.
 */
class PhysicalBlock {
public:
    /**
     * @brief Allocates an aligned host-backed physical block.
     */
    PhysicalBlock(
        PhysicalBlockId id,
        BlockShape shape,
        HostMemoryOptions memory_options = {});

    ~PhysicalBlock() = default;

    PhysicalBlock(const PhysicalBlock&) = delete;
    PhysicalBlock& operator=(const PhysicalBlock&) = delete;
    PhysicalBlock(PhysicalBlock&&) noexcept = default;
    PhysicalBlock& operator=(PhysicalBlock&&) noexcept = default;

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
     * @brief Returns the allocated byte capacity of this block.
     */
    [[nodiscard]] std::size_t size_bytes() const noexcept;

    /**
     * @brief Returns the byte alignment requested for this block allocation.
     */
    [[nodiscard]] std::size_t alignment_bytes() const noexcept;

    /**
     * @brief Returns true when the backing allocation was requested as pinned.
     */
    [[nodiscard]] bool pinned_memory_requested() const noexcept;

    /**
     * @brief Swaps this block's storage with another physical block.
     */
    void swap(PhysicalBlock& other) noexcept;

private:
    static void release_host_memory(std::byte* ptr) noexcept;

    PhysicalBlockId id_{0};
    BlockShape shape_{};
    HostMemoryOptions memory_options_{};
    std::size_t size_bytes_{0};
    std::unique_ptr<std::byte, decltype(&PhysicalBlock::release_host_memory)> storage_{
        nullptr,
        &PhysicalBlock::release_host_memory};
};

} // namespace qwenvl_paged
