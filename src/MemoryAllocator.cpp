#include "qwenvl_paged/MemoryAllocator.h"

#include <cstdlib>
#include <cstring>
#include <utility>

namespace qwenvl_paged {

namespace {

std::size_t round_up(std::size_t value, std::size_t alignment) noexcept {
    if (alignment == 0) {
        return value;
    }
    return ((value + alignment - 1) / alignment) * alignment;
}

std::size_t effective_alignment(const AllocatorConfig& config) noexcept {
    return config.memory_options.alignment_bytes == 0 ? kDefaultBlockAlignmentBytes
                                                      : config.memory_options.alignment_bytes;
}

} // namespace

std::size_t block_stride_for(const AllocatorConfig& config) noexcept {
    // Rounding the stride to the alignment keeps every frame aligned, not just
    // the pool base, so a backend can DMA any single frame on its own.
    const std::size_t alignment = effective_alignment(config);
    const std::size_t stride = round_up(config.block_shape.byte_size(), alignment);
    return stride == 0 ? alignment : stride;
}

std::size_t pool_bytes_for(const AllocatorConfig& config) noexcept {
    return block_stride_for(config) * config.max_blocks;
}

MemoryAllocator::MemoryAllocator(AllocatorConfig config) : config_(config) {
    block_stride_bytes_ = block_stride_for(config_);
    pool_bytes_ = block_stride_bytes_ * config_.max_blocks;

    // std::aligned_alloc requires a size that is a multiple of the alignment,
    // which a whole number of aligned strides already is.
    if (pool_bytes_ > 0) {
        owned_pool_.reset(
            static_cast<std::byte*>(std::aligned_alloc(effective_alignment(config_), pool_bytes_)));
    }

    bind_frames(owned_pool_.get());
}

MemoryAllocator::MemoryAllocator(AllocatorConfig config, std::byte* adopted_pool)
    : config_(config) {
    block_stride_bytes_ = block_stride_for(config_);
    pool_bytes_ = block_stride_bytes_ * config_.max_blocks;

    bind_frames(adopted_pool);
}

void MemoryAllocator::bind_frames(std::byte* base) {
    pool_base_ = base;

    blocks_.reserve(config_.max_blocks);
    infos_.reserve(config_.max_blocks);
    free_list_.reserve(config_.max_blocks);

    for (std::uint32_t i = 0; i < config_.max_blocks; ++i) {
        const PhysicalBlockId id = i;
        std::byte* storage =
            base == nullptr ? nullptr : base + static_cast<std::size_t>(id) * block_stride_bytes_;
        blocks_.emplace_back(id, config_.block_shape, storage, block_stride_bytes_);

        PhysicalBlockInfo info;
        info.id = id;
        info.state = BlockState::Free;
        info.ref_count = 0;
        info.generation = 0;
        infos_.push_back(info);

        free_list_.push_back(id);
    }
}

std::optional<PhysicalBlockId> MemoryAllocator::allocate() {
    if (free_list_.empty()) {
        return std::nullopt;
    }

    const PhysicalBlockId id = free_list_.back();
    free_list_.pop_back();

    PhysicalBlockInfo& info = infos_[id];
    info.state = BlockState::Active;
    info.ref_count = 1;
    return id;
}

void MemoryAllocator::release(PhysicalBlockId id) {
    if (id >= infos_.size()) {
        return;
    }

    PhysicalBlockInfo& info = infos_[id];
    if (info.ref_count == 0) {
        return;
    }

    --info.ref_count;
    if (info.ref_count == 0) {
        info.state = BlockState::Free;
        ++info.generation;
        free_list_.push_back(id);
    } else if (info.ref_count == 1) {
        info.state = BlockState::Active;
    }
}

void MemoryAllocator::retain(PhysicalBlockId id) {
    if (id >= infos_.size()) {
        return;
    }

    PhysicalBlockInfo& info = infos_[id];
    if (info.ref_count == 0) {
        return;
    }

    ++info.ref_count;
    info.state = BlockState::Shared;
}

void MemoryAllocator::copy_block(PhysicalBlockId source, PhysicalBlockId destination) {
    PhysicalBlock* src = block(source);
    PhysicalBlock* dst = block(destination);
    if (src == nullptr || dst == nullptr) {
        return;
    }

    const std::size_t bytes = src->size_bytes() < dst->size_bytes() ? src->size_bytes() : dst->size_bytes();
    std::memcpy(dst->data(), src->data(), bytes);
}

PhysicalBlock* MemoryAllocator::block(PhysicalBlockId id) noexcept {
    if (id >= blocks_.size()) {
        return nullptr;
    }
    return &blocks_[id];
}

const PhysicalBlock* MemoryAllocator::block(PhysicalBlockId id) const noexcept {
    if (id >= blocks_.size()) {
        return nullptr;
    }
    return &blocks_[id];
}

void MemoryAllocator::set_swap_backend(SwapBackend* backend) noexcept {
    swap_backend_ = backend;
}

std::optional<SwapSlotId> MemoryAllocator::swap_out(PhysicalBlockId id) {
    if (swap_backend_ == nullptr || id >= infos_.size()) {
        return std::nullopt;
    }

    PhysicalBlockInfo& info = infos_[id];
    // A free block has nothing to evict, and a shared one is still mapped by a
    // sibling branch that would be stranded without its frame.
    if (info.ref_count != 1) {
        return std::nullopt;
    }

    PhysicalBlock* frame = block(id);
    if (frame == nullptr) {
        return std::nullopt;
    }

    const std::optional<SwapSlotId> slot = swap_backend_->store(*frame);
    if (!slot.has_value()) {
        return std::nullopt;
    }

    info.ref_count = 0;
    info.state = BlockState::Free;
    ++info.generation;
    free_list_.push_back(id);
    return slot;
}

std::optional<PhysicalBlockId> MemoryAllocator::swap_in(SwapSlotId slot) {
    if (swap_backend_ == nullptr) {
        return std::nullopt;
    }

    const std::optional<PhysicalBlockId> id = allocate();
    if (!id.has_value()) {
        return std::nullopt;
    }

    PhysicalBlock* frame = block(*id);
    if (frame == nullptr || !swap_backend_->load(slot, *frame)) {
        release(*id);
        return std::nullopt;
    }

    swap_backend_->discard(slot);
    return id;
}

void MemoryAllocator::discard_swapped(SwapSlotId slot) {
    if (swap_backend_ != nullptr) {
        swap_backend_->discard(slot);
    }
}

void MemoryAllocator::set_eviction_selector(EvictionCandidateSelector selector) {
    eviction_selector_ = std::move(selector);
}

std::optional<PhysicalBlockId> MemoryAllocator::select_eviction_candidate() const {
    if (!eviction_selector_) {
        return std::nullopt;
    }
    return eviction_selector_(*this);
}

const PhysicalBlockInfo* MemoryAllocator::info(PhysicalBlockId id) const noexcept {
    if (id >= infos_.size()) {
        return nullptr;
    }
    return &infos_[id];
}

AllocatorStats MemoryAllocator::stats() const noexcept {
    AllocatorStats stats;
    stats.total_blocks = static_cast<std::uint32_t>(infos_.size());

    for (const PhysicalBlockInfo& info : infos_) {
        switch (info.state) {
            case BlockState::Free:
                ++stats.free_blocks;
                break;
            case BlockState::Active:
                ++stats.active_blocks;
                break;
            case BlockState::Shared:
                ++stats.shared_blocks;
                break;
            case BlockState::Swapped:
                break;
        }
    }

    if (swap_backend_ != nullptr) {
        stats.swapped_blocks = static_cast<std::uint32_t>(swap_backend_->resident_slots());
    }

    stats.bytes_reserved = pool_bytes_;

    return stats;
}

bool MemoryAllocator::can_allocate(std::uint32_t block_count) const noexcept {
    return free_list_.size() >= block_count;
}

std::byte* MemoryAllocator::pool_base() noexcept {
    return pool_base_;
}

const std::byte* MemoryAllocator::pool_base() const noexcept {
    return pool_base_;
}

std::size_t MemoryAllocator::block_stride_bytes() const noexcept {
    return block_stride_bytes_;
}

std::size_t MemoryAllocator::pool_bytes() const noexcept {
    return pool_bytes_;
}

std::size_t MemoryAllocator::alignment_bytes() const noexcept {
    return config_.memory_options.alignment_bytes;
}

bool MemoryAllocator::pinned_memory_requested() const noexcept {
    return config_.memory_options.prefer_pinned_memory;
}

bool MemoryAllocator::owns_pool() const noexcept {
    return owned_pool_ != nullptr;
}

void MemoryAllocator::release_pool_memory(std::byte* ptr) noexcept {
    std::free(ptr);
}

} // namespace qwenvl_paged
