#pragma once

#include "qwenvl_paged/Block.h"
#include "qwenvl_paged/BlockTable.h"
#include "qwenvl_paged/CacheLayout.h"
#include "qwenvl_paged/MemoryAllocator.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
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
    const std::vector<BlockTable>* layer_tables{nullptr};
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
    [[nodiscard]] const std::byte* block_bytes(LogicalBlockIndex index, std::uint32_t layer) const noexcept;

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
    const bool per_layer = layout.shape.per_layer_frames();
    if (per_layer && (layer_tables == nullptr || layer >= layer_tables->size())) {
        return nullptr;
    }
    const std::uint32_t offset_layer = per_layer ? 0 : layer;
    const std::optional<std::size_t> offset =
        layout.element_offset(offset_layer, stream, token % tokens_per_block, kv_head);
    if (!offset.has_value()) {
        return nullptr;
    }

    const LogicalBlockIndex index = static_cast<LogicalBlockIndex>(token / tokens_per_block);
    const std::byte* bytes = per_layer ? block_bytes(index, layer) : block_bytes(index);
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
        CacheKind cache_kind = CacheKind::TextKV,
        std::uint32_t layer = 0);

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
     * @brief Returns every sequence created or forked under this request.
     */
    [[nodiscard]] std::vector<SequenceId> sequences_for(RequestId request_id) const;

    /**
     * @brief Swaps out every sequence of a request.
     *
     * Shared frames (refcount != 1) are skipped by the existing swap_out rule,
     * so a forked prompt stays resident while private branch blocks reclaim.
     */
    std::uint32_t swap_out_request(RequestId request_id);

    /**
     * @brief Restores every swapped sequence of a request, all-or-nothing.
     */
    bool swap_in_request(RequestId request_id);

    /**
     * @brief Releases every sequence of a request.
     */
    void release_request(RequestId request_id);

    /**
     * @brief How many prompt tokens a unique prefix record currently covers.
     *
     * Zero when the key is unknown or when two different contents collided on
     * the same key, so a caller must not share.
     */
    [[nodiscard]] std::uint32_t prefix_token_count(const std::string& key) const;

    /**
     * @brief Indexes this sequence's resident blocks under `key`.
     *
     * Pins the frames so they survive the publisher being released while
     * another request is still attached. A second publish of the same key with
     * different contents is recorded as a collision rather than a replacement.
     */
    bool publish_prefix(SequenceId sequence_id, const std::string& key);

    /**
     * @brief Maps a published prefix onto an empty sequence, retaining each frame.
     *
     * Returns the token count covered, or 0 on miss / collision. Mapped blocks
     * are not writable; the first write copy-on-writes as usual.
     */
    std::uint32_t attach_prefix(SequenceId sequence_id, const std::string& key);

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
        std::vector<BlockTable> tables;
        std::string prefix_key;
        std::uint64_t prefix_hash{0};

        BlockTable& text_table() { return tables.front(); }
        const BlockTable& text_table() const { return tables.front(); }
    };

    [[nodiscard]] BlockShape pool_shape() const noexcept;
    [[nodiscard]] std::uint32_t table_count() const noexcept;

    struct PrefixRecord {
        std::uint64_t content_hash{0};
        std::uint32_t token_count{0};
        std::uint32_t users{0};
        std::vector<PhysicalBlockId> physical_ids;
    };

    void index_sequence(RequestId request_id, SequenceId sequence_id);
    void unindex_sequence(RequestId request_id, SequenceId sequence_id);
    void drop_prefix_user(SequenceState& state);
    [[nodiscard]] std::uint32_t tokens_per_block() const noexcept;
    [[nodiscard]] std::uint64_t content_hash_of(const BlockTable& table) const;
    [[nodiscard]] std::uint64_t content_hash_of(const std::vector<BlockTable>& tables) const;

    MemoryAllocator* allocator_{nullptr};
    std::unordered_map<SequenceId, SequenceState> sequences_;
    std::unordered_map<RequestId, std::vector<SequenceId>> sequences_by_request_;
    std::unordered_map<std::string, std::vector<PrefixRecord>> prefix_index_;
};

} // namespace qwenvl_paged
