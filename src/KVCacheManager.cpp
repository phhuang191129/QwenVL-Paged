#include "qwenvl_paged/KVCacheManager.h"

#include <utility>

namespace qwenvl_paged {

KVCacheManager::KVCacheManager(MemoryAllocator& allocator) : allocator_(&allocator) {}

bool KVCacheManager::create_sequence(SequenceMetadata metadata) {
    const SequenceId sequence_id = metadata.sequence_id;
    if (sequences_.find(sequence_id) != sequences_.end()) {
        return false;
    }

    sequences_.emplace(sequence_id, SequenceState{std::move(metadata), BlockTable(sequence_id)});
    return true;
}

bool KVCacheManager::fork_sequence(SequenceId parent_id, SequenceMetadata child_metadata) {
    auto parent_it = sequences_.find(parent_id);
    if (parent_it == sequences_.end()) {
        return false;
    }

    const SequenceId child_id = child_metadata.sequence_id;
    if (sequences_.find(child_id) != sequences_.end()) {
        return false;
    }

    BlockTable child_table = parent_it->second.text_table.fork(child_id);
    for (const BlockTableEntry& entry : child_table.entries()) {
        allocator_->retain(entry.physical_id);
    }

    sequences_.emplace(child_id, SequenceState{std::move(child_metadata), std::move(child_table)});
    return true;
}

bool KVCacheManager::reserve_tokens(SequenceId sequence_id, std::uint32_t token_count) {
    auto it = sequences_.find(sequence_id);
    if (it == sequences_.end()) {
        return false;
    }

    const std::uint32_t tokens_per_block = allocator_->block(0) != nullptr
                                               ? allocator_->block(0)->shape().tokens_per_block
                                               : 0;
    if (tokens_per_block == 0) {
        return false;
    }

    const std::uint32_t blocks_needed = (token_count + tokens_per_block - 1) / tokens_per_block;
    if (!allocator_->can_allocate(blocks_needed)) {
        return false;
    }

    BlockTable& table = it->second.text_table;
    const LogicalBlockIndex base_index = static_cast<LogicalBlockIndex>(table.size());
    for (std::uint32_t offset = 0; offset < blocks_needed; ++offset) {
        const std::optional<PhysicalBlockId> physical = allocator_->allocate();
        if (!physical.has_value()) {
            return false;
        }

        const LogicalBlockIndex index = base_index + offset;
        LogicalBlock logical;
        logical.sequence_id = sequence_id;
        logical.index = index;
        logical.cache_kind = table.cache_kind();
        logical.start_token = index * tokens_per_block;
        logical.token_count = tokens_per_block;
        table.map(logical, *physical, true);
    }

    return true;
}

std::optional<PhysicalBlockId> KVCacheManager::ensure_token_writable(
    SequenceId sequence_id, TokenPosition token_position, CacheKind /*cache_kind*/) {
    auto it = sequences_.find(sequence_id);
    if (it == sequences_.end()) {
        return std::nullopt;
    }

    BlockTable& table = it->second.text_table;
    if (table.empty()) {
        return std::nullopt;
    }

    const std::uint32_t tokens_per_block = table.entries().front().logical.token_count;
    if (tokens_per_block == 0) {
        return std::nullopt;
    }

    const LogicalBlockIndex index = static_cast<LogicalBlockIndex>(token_position / tokens_per_block);
    BlockTableEntry* entry = table.mutable_entry(index);
    if (entry == nullptr) {
        return std::nullopt;
    }

    const PhysicalBlockInfo* info = allocator_->info(entry->physical_id);
    if (info == nullptr) {
        return std::nullopt;
    }

    if (info->ref_count <= 1) {
        entry->writable = true;
        return entry->physical_id;
    }

    const std::optional<PhysicalBlockId> copy = allocator_->allocate();
    if (!copy.has_value()) {
        return std::nullopt;
    }

    allocator_->copy_block(entry->physical_id, *copy);
    allocator_->release(entry->physical_id);
    entry->physical_id = *copy;
    entry->writable = true;
    return *copy;
}

void KVCacheManager::release_sequence(SequenceId sequence_id) {
    auto it = sequences_.find(sequence_id);
    if (it == sequences_.end()) {
        return;
    }

    for (const BlockTableEntry& entry : it->second.text_table.entries()) {
        allocator_->release(entry.physical_id);
    }
    sequences_.erase(it);
}

std::optional<CacheView> KVCacheManager::cache_view(SequenceId sequence_id, CacheKind cache_kind) const {
    auto it = sequences_.find(sequence_id);
    if (it == sequences_.end()) {
        return std::nullopt;
    }

    CacheView view;
    view.sequence_id = sequence_id;
    view.cache_kind = cache_kind;
    view.block_table = &it->second.text_table;
    return view;
}

bool KVCacheManager::contains(SequenceId sequence_id) const noexcept {
    return sequences_.find(sequence_id) != sequences_.end();
}

} // namespace qwenvl_paged
