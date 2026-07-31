#include "qwenvl_paged/CacheLayout.h"

namespace qwenvl_paged {

std::size_t KVBlockLayout::head_stride() const noexcept {
    return shape.head_dim;
}

std::size_t KVBlockLayout::token_stride() const noexcept {
    return static_cast<std::size_t>(shape.num_kv_heads) * head_stride();
}

std::size_t KVBlockLayout::stream_stride() const noexcept {
    return static_cast<std::size_t>(shape.tokens_per_block) * token_stride();
}

std::size_t KVBlockLayout::layer_stride() const noexcept {
    return kKVStreamCount * stream_stride();
}

std::size_t KVBlockLayout::element_count() const noexcept {
    return static_cast<std::size_t>(shape.num_layers) * layer_stride();
}

std::optional<std::size_t> KVBlockLayout::element_offset(
    std::uint32_t layer,
    KVStream stream,
    std::uint32_t token_in_block,
    std::uint32_t kv_head) const noexcept {
    if (layer >= shape.num_layers || token_in_block >= shape.tokens_per_block ||
        kv_head >= shape.num_kv_heads) {
        return std::nullopt;
    }

    const std::size_t stream_index = (stream == KVStream::Key) ? 0U : 1U;
    return layer * layer_stride() + stream_index * stream_stride() +
           token_in_block * token_stride() + kv_head * head_stride();
}

} // namespace qwenvl_paged
