#pragma once

#include "qwenvl_paged/Block.h"

#include <cstddef>
#include <cstdint>
#include <optional>

namespace qwenvl_paged {

/**
 * @brief Selects the key or the value half of a cache block.
 */
enum class KVStream : std::uint8_t {
    Key,
    Value
};

/**
 * @brief Number of streams (K and V) stored side by side in every block.
 */
inline constexpr std::uint32_t kKVStreamCount = 2;

/**
 * @brief Element addressing for the bytes inside one physical cache block.
 *
 * `BlockShape` states how large a block is; this struct states how the elements
 * are ordered inside it. The prototype uses
 *
 *     [layer][K|V][token_in_block][kv_head][head_dim]
 *
 * with `head_dim` innermost, so the whole K (or V) vector of one token and one
 * head is contiguous. That is the access pattern of the reference attention
 * loop, and it is also the granularity at which a CUDA or Triton kernel would
 * want to issue a vectorized load.
 *
 * All strides and offsets are counted in *elements*, not bytes. Multiply by
 * `BlockShape::bytes_per_element` for a byte offset.
 */
struct KVBlockLayout {
    BlockShape shape{};

    /**
     * @brief Distance between two consecutive kv heads of the same token.
     */
    [[nodiscard]] std::size_t head_stride() const noexcept;

    /**
     * @brief Distance between two consecutive tokens of the same stream.
     */
    [[nodiscard]] std::size_t token_stride() const noexcept;

    /**
     * @brief Distance from the key half of a layer to its value half.
     */
    [[nodiscard]] std::size_t stream_stride() const noexcept;

    /**
     * @brief Distance between two consecutive layers.
     */
    [[nodiscard]] std::size_t layer_stride() const noexcept;

    /**
     * @brief Total number of elements addressable inside one block.
     */
    [[nodiscard]] std::size_t element_count() const noexcept;

    /**
     * @brief Offset of the first element of one (layer, stream, token, head) vector.
     *
     * Returns nullopt when any coordinate is outside the block shape, so callers
     * never form an out-of-bounds pointer from a bad index.
     */
    [[nodiscard]] std::optional<std::size_t> element_offset(
        std::uint32_t layer,
        KVStream stream,
        std::uint32_t token_in_block,
        std::uint32_t kv_head) const noexcept;
};

} // namespace qwenvl_paged
