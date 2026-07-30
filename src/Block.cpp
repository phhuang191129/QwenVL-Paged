#include "qwenvl_paged/Block.h"

#include <cstdlib>
#include <utility>

namespace qwenvl_paged {

std::size_t BlockShape::byte_size() const noexcept {
    // A full block stores both the K and V streams for every layer, head, and
    // token slot, hence the factor of two.
    return static_cast<std::size_t>(tokens_per_block) * num_layers *
           num_kv_heads * head_dim * bytes_per_element * 2u;
}

bool LogicalBlock::empty() const noexcept {
    return token_count == 0;
}

namespace {

std::size_t round_up(std::size_t value, std::size_t alignment) noexcept {
    if (alignment == 0) {
        return value;
    }
    return ((value + alignment - 1) / alignment) * alignment;
}

} // namespace

PhysicalBlock::PhysicalBlock(PhysicalBlockId id, BlockShape shape, HostMemoryOptions memory_options)
    : id_(id), shape_(shape), memory_options_(memory_options) {
    const std::size_t alignment =
        memory_options_.alignment_bytes == 0 ? kDefaultBlockAlignmentBytes : memory_options_.alignment_bytes;

    // std::aligned_alloc requires the allocation size to be a multiple of the
    // alignment, so round up and never request zero bytes.
    std::size_t bytes = round_up(shape_.byte_size(), alignment);
    if (bytes == 0) {
        bytes = alignment;
    }

    storage_.reset(static_cast<std::byte*>(std::aligned_alloc(alignment, bytes)));
    size_bytes_ = bytes;
}

PhysicalBlockId PhysicalBlock::id() const noexcept {
    return id_;
}

const BlockShape& PhysicalBlock::shape() const noexcept {
    return shape_;
}

std::byte* PhysicalBlock::data() noexcept {
    return storage_.get();
}

const std::byte* PhysicalBlock::data() const noexcept {
    return storage_.get();
}

std::size_t PhysicalBlock::size_bytes() const noexcept {
    return size_bytes_;
}

std::size_t PhysicalBlock::alignment_bytes() const noexcept {
    return memory_options_.alignment_bytes;
}

bool PhysicalBlock::pinned_memory_requested() const noexcept {
    return memory_options_.prefer_pinned_memory;
}

void PhysicalBlock::swap(PhysicalBlock& other) noexcept {
    // Identity (id_) stays with the block; only the backing storage and its
    // descriptive metadata move so the block continues to describe its bytes.
    std::swap(shape_, other.shape_);
    std::swap(memory_options_, other.memory_options_);
    std::swap(size_bytes_, other.size_bytes_);
    storage_.swap(other.storage_);
}

void PhysicalBlock::release_host_memory(std::byte* ptr) noexcept {
    std::free(ptr);
}

} // namespace qwenvl_paged
