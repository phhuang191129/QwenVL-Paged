#include "qwenvl_paged/MemoryAllocator.h"

#include <cstring>

namespace qwenvl_paged {

MemoryAllocator::MemoryAllocator(AllocatorConfig config) : config_(config) {
    blocks_.reserve(config_.max_blocks);
    infos_.reserve(config_.max_blocks);
    free_list_.reserve(config_.max_blocks);

    for (std::uint32_t i = 0; i < config_.max_blocks; ++i) {
        const PhysicalBlockId id = i;
        blocks_.push_back(std::make_unique<PhysicalBlock>(id, config_.block_shape, config_.memory_options));

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
    return blocks_[id].get();
}

const PhysicalBlock* MemoryAllocator::block(PhysicalBlockId id) const noexcept {
    if (id >= blocks_.size()) {
        return nullptr;
    }
    return blocks_[id].get();
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

    for (const std::unique_ptr<PhysicalBlock>& block : blocks_) {
        stats.bytes_reserved += block->size_bytes();
    }

    return stats;
}

bool MemoryAllocator::can_allocate(std::uint32_t block_count) const noexcept {
    return free_list_.size() >= block_count;
}

} // namespace qwenvl_paged
