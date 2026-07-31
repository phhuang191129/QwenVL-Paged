#pragma once

#include "qwenvl_paged/Block.h"
#include "qwenvl_paged/BlockTable.h"
#include "qwenvl_paged/CacheLayout.h"
#include "qwenvl_paged/MemoryAllocator.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <unordered_map>
#include <vector>

namespace qwenvl_paged {

/**
 * @brief Describes a contiguous multimodal feature span in request token space.
 *
 * The cache manager stores this metadata alongside sequence state so Qwen3-VL
 * image/video features can be aligned with logical KV blocks without changing
 * physical allocator behavior.
 */
struct MultimodalSpan {
    TokenPosition start_token{0};
    std::uint32_t token_count{0};
    std::uint32_t feature_index{0};
};

/**
 * @brief Position encoding layout associated with a token span.
 */
enum class PositionEncodingKind : std::uint8_t {
    Text1D,
    Image2D,
    Video3D
};

/**
 * @brief Multi-axis origin or stride for Qwen-VL M-RoPE positions.
 */
struct Position3D {
    std::int64_t temporal{0};
    std::int64_t height{0};
    std::int64_t width{0};
};

/**
 * @brief Describes positional encoding for a contiguous logical token span.
 *
 * Visual inputs are not treated as a simple flat text-token stream. One image
 * can expand into many patch tokens whose M-RoPE positions require 2D or 3D
 * coordinates. The allocator still stores logical blocks, while this metadata
 * lets model-specific code reconstruct the correct positional IDs.
 */
struct PositionalEncodingSpan {
    TokenPosition start_token{0};
    std::uint32_t token_count{0};
    PositionEncodingKind kind{PositionEncodingKind::Text1D};
    Position3D origin{};
    Position3D stride{};
};

/**
 * @brief Sequence-local positional metadata for text and multimodal tokens.
 */
struct PositionalEncodingMetadata {
    std::vector<PositionalEncodingSpan> spans;
};

/**
 * @brief Metadata required to create a new cache-backed sequence.
 */
struct SequenceMetadata {
    SequenceId sequence_id{0};
    RequestId request_id{0};
    std::vector<MultimodalSpan> multimodal_spans;
    PositionalEncodingMetadata positional_encoding;
};

/**
 * @brief Read-only, backend-neutral view of one sequence cache table.
 *
 * This is the whole contract an execution backend needs in order to read a
 * paged KV cache: the page table, a way to turn a physical block id into bytes,
 * and the element layout inside a block. It deliberately exposes no writable
 * storage. Writes go through KVCacheManager::ensure_token_writable, which is
 * the only path that can honor copy-on-write before a branch mutates a shared
 * block.
 *
 * A view borrows the cache manager's state. It is invalidated by anything that
 * remaps the sequence (copy-on-write, swap in/out, release), so backends must
 * re-acquire it after every cache mutation rather than caching it across steps.
 */
struct CacheView {
    SequenceId sequence_id{0};
    CacheKind cache_kind{CacheKind::TextKV};
    const BlockTable* block_table{nullptr};
    const MemoryAllocator* allocator{nullptr};
    KVBlockLayout layout{};

    /**
     * @brief Returns true when the view can resolve cache storage.
     */
    [[nodiscard]] bool valid() const noexcept;

    /**
     * @brief Returns the bytes backing one logical block.
     *
     * Returns nullptr when the logical block is unmapped or currently swapped
     * out, so a backend faults instead of reading a recycled frame.
     */
    [[nodiscard]] const std::byte* block_bytes(LogicalBlockIndex index) const noexcept;

    /**
     * @brief Resolves one (stream, layer, token, kv head) vector in the cache.
     *
     * The returned pointer addresses `BlockShape::head_dim` contiguous elements.
     * `token` is a sequence-global logical position; this call performs the
     * logical-to-physical translation. Returns nullptr when `sizeof(T)` does not
     * match the configured element size, when any coordinate is out of range, or
     * when the containing block is unmapped or swapped out.
     */
    template <typename T>
    [[nodiscard]] const T* slot(
        KVStream stream,
        std::uint32_t layer,
        TokenPosition token,
        std::uint32_t kv_head) const noexcept;
};

template <typename T>
const T* CacheView::slot(
    KVStream stream,
    std::uint32_t layer,
    TokenPosition token,
    std::uint32_t kv_head) const noexcept {
    if (!valid() || sizeof(T) != layout.shape.bytes_per_element) {
        return nullptr;
    }

    const std::uint32_t tokens_per_block = layout.shape.tokens_per_block;
    const std::optional<std::size_t> offset =
        layout.element_offset(layer, stream, token % tokens_per_block, kv_head);
    if (!offset.has_value()) {
        return nullptr;
    }

    const std::byte* bytes = block_bytes(static_cast<LogicalBlockIndex>(token / tokens_per_block));
    if (bytes == nullptr) {
        return nullptr;
    }

    return reinterpret_cast<const T*>(bytes) + *offset;
}

/**
 * @brief Owns sequence cache lifecycle above the physical allocator.
 *
 * KVCacheManager creates and releases block tables, appends token capacity,
 * forks cache for parallel sampling, and provides backend-neutral cache views
 * consumed by CPU or future CUDA/Triton execution backends.
 *
 * This phase-1 class is not thread-safe. It is intended to be mutated only by
 * the engine event loop after request ownership has crossed the ingress queue.
 */
class KVCacheManager {
public:
    /**
     * @brief Creates a manager that uses the supplied allocator.
     */
    explicit KVCacheManager(MemoryAllocator& allocator);

    ~KVCacheManager() = default;

    KVCacheManager(const KVCacheManager&) = delete;
    KVCacheManager& operator=(const KVCacheManager&) = delete;
    KVCacheManager(KVCacheManager&&) noexcept = default;
    KVCacheManager& operator=(KVCacheManager&&) noexcept = default;

    /**
     * @brief Creates an empty cache table for a sequence.
     */
    bool create_sequence(SequenceMetadata metadata);

    /**
     * @brief Forks an existing sequence for parallel sampling.
     *
     * Physical blocks are shared until either branch writes to a shared block.
     * Fails when the parent holds swapped-out blocks, which have no frame for a
     * child to share; swap the parent back in first.
     */
    bool fork_sequence(SequenceId parent_id, SequenceMetadata child_metadata);

    /**
     * @brief Reserves additional logical token capacity for a sequence.
     */
    bool reserve_tokens(SequenceId sequence_id, std::uint32_t token_count);

    /**
     * @brief Materializes writable storage for the logical block containing a token.
     */
    [[nodiscard]] std::optional<PhysicalBlockId> ensure_token_writable(
        SequenceId sequence_id,
        TokenPosition token_position,
        CacheKind cache_kind = CacheKind::TextKV);

    /**
     * @brief Evicts a sequence's cache blocks to the allocator's swap backend.
     *
     * Blocks shared with another branch are skipped: their frame is still in use
     * by a sibling, so reclaiming it would strand that branch.
     *
     * @return The number of logical blocks moved to swap.
     */
    std::uint32_t swap_out_sequence(SequenceId sequence_id);

    /**
     * @brief Restores a sequence's swapped-out blocks into physical frames.
     *
     * All-or-nothing: when the pool cannot back every swapped block, the
     * sequence is left untouched and still swapped out.
     */
    bool swap_in_sequence(SequenceId sequence_id);

    /**
     * @brief Releases all cache blocks owned or referenced by a sequence.
     *
     * Swapped-out blocks are discarded from the swap backend rather than
     * released to the block pool, since they hold no frame.
     */
    void release_sequence(SequenceId sequence_id);

    /**
     * @brief Returns a backend-facing cache view for a sequence.
     */
    [[nodiscard]] std::optional<CacheView> cache_view(
        SequenceId sequence_id,
        CacheKind cache_kind = CacheKind::TextKV) const;

    /**
     * @brief Returns true when the sequence is currently known.
     */
    [[nodiscard]] bool contains(SequenceId sequence_id) const noexcept;

private:
    struct SequenceState {
        SequenceMetadata metadata{};
        BlockTable text_table;
    };

    MemoryAllocator* allocator_{nullptr};
    std::unordered_map<SequenceId, SequenceState> sequences_;
};

} // namespace qwenvl_paged
