#include "qwenvl_paged/Block.h"

namespace qwenvl_paged {

std::size_t BlockShape::byte_size() const noexcept {
    // A full block stores both the K and V streams for every layer, head, and
    // token slot, hence the factor of two.
    return static_cast<std::size_t>(tokens_per_block) * frame_layers() *
           num_kv_heads * head_dim * bytes_per_element * 2u;
}

bool LogicalBlock::empty() const noexcept {
    return token_count == 0;
}

PhysicalBlock::PhysicalBlock(
    PhysicalBlockId id,
    BlockShape shape,
    std::byte* storage,
    std::size_t size_bytes) noexcept
    : id_(id), shape_(shape), storage_(storage), size_bytes_(size_bytes) {}

PhysicalBlockId PhysicalBlock::id() const noexcept {
    return id_;
}

const BlockShape& PhysicalBlock::shape() const noexcept {
    return shape_;
}

std::byte* PhysicalBlock::data() noexcept {
    return storage_;
}

const std::byte* PhysicalBlock::data() const noexcept {
    return storage_;
}

std::size_t PhysicalBlock::size_bytes() const noexcept {
    return size_bytes_;
}

} // namespace qwenvl_paged
