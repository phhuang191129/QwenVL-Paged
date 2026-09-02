#include "qwenvl_paged/KVCacheManager.h"

#include <algorithm>
#include <utility>

namespace qwenvl_paged {

bool CacheView::valid() const noexcept {
    return block_table != nullptr && allocator != nullptr && layout.element_count() > 0;
}

const std::byte* CacheView::block_bytes(LogicalBlockIndex index) const noexcept {
    return block_bytes(index, 0);
}

const std::byte* CacheView::block_bytes(LogicalBlockIndex index, std::uint32_t layer) const noexcept {
    if (!valid()) {
        return nullptr;
    }

    const BlockTable* table = block_table;
    if (layer_tables != nullptr && layer < layer_tables->size()) {
        table = &(*layer_tables)[layer];
    } else if (layer != 0) {
        return nullptr;
    }

    const std::optional<PhysicalBlockId> physical = table->lookup(index);
    if (!physical.has_value()) {
        return nullptr;
    }

    const PhysicalBlock* block = allocator->block(*physical);
    return block == nullptr ? nullptr : block->data();
}

KVCacheManager::KVCacheManager(MemoryAllocator& allocator) : allocator_(&allocator) {}

BlockShape KVCacheManager::pool_shape() const noexcept {
    const PhysicalBlock* block = allocator_->block(0);
    return block == nullptr ? BlockShape{} : block->shape();
}

std::uint32_t KVCacheManager::table_count() const noexcept {
    const BlockShape shape = pool_shape();
    return shape.per_layer_frames() ? shape.num_layers : 1;
}

void KVCacheManager::index_sequence(RequestId request_id, SequenceId sequence_id) {
    sequences_by_request_[request_id].push_back(sequence_id);
}

void KVCacheManager::unindex_sequence(RequestId request_id, SequenceId sequence_id) {
    auto it = sequences_by_request_.find(request_id);
    if (it == sequences_by_request_.end()) {
        return;
    }
    auto& ids = it->second;
    ids.erase(std::remove(ids.begin(), ids.end(), sequence_id), ids.end());
    if (ids.empty()) {
        sequences_by_request_.erase(it);
    }
}

std::uint32_t KVCacheManager::tokens_per_block() const noexcept {
    const PhysicalBlock* block = allocator_->block(0);
    return block == nullptr ? 0 : block->shape().tokens_per_block;
}

std::uint64_t KVCacheManager::content_hash_of(const BlockTable& table) const {
    std::uint64_t hash = 14695981039346656037ULL;
    for (const BlockTableEntry& entry : table.entries()) {
        if (entry.swap_slot.has_value()) {
            continue;
        }
        const PhysicalBlock* block = allocator_->block(entry.physical_id);
        if (block == nullptr) {
            continue;
        }
        const auto* bytes = static_cast<const unsigned char*>(static_cast<const void*>(block->data()));
        for (std::size_t i = 0; i < block->size_bytes(); ++i) {
            hash ^= static_cast<std::uint64_t>(bytes[i]);
            hash *= 1099511628211ULL;
        }
    }
    return hash;
}

std::uint64_t KVCacheManager::content_hash_of(const std::vector<BlockTable>& tables) const {
    std::uint64_t hash = 14695981039346656037ULL;
    for (const BlockTable& table : tables) {
        // Mix table hashes in layer-major order so a 1-layer-per-frame prefix
        // is not interchangeable with an all-layers-in-one-frame prefix.
        const std::uint64_t table_hash = content_hash_of(table);
        hash ^= table_hash;
        hash *= 1099511628211ULL;
    }
    return hash;
}

void KVCacheManager::drop_prefix_user(SequenceState& state) {
    if (state.prefix_key.empty()) {
        return;
    }

    auto it = prefix_index_.find(state.prefix_key);
    const std::uint64_t hash = state.prefix_hash;
    state.prefix_key.clear();
    state.prefix_hash = 0;
    if (it == prefix_index_.end()) {
        return;
    }

    auto rec = std::find_if(it->second.begin(), it->second.end(), [hash](const PrefixRecord& record) {
        return record.content_hash == hash;
    });
    if (rec == it->second.end()) {
        return;
    }
    if (rec->users > 0) {
        --rec->users;
    }
    if (rec->users > 0) {
        return;
    }

    for (const PhysicalBlockId id : rec->physical_ids) {
        allocator_->release(id);
    }
    it->second.erase(rec);
    if (it->second.empty()) {
        prefix_index_.erase(it);
    }
}

bool KVCacheManager::create_sequence(SequenceMetadata metadata) {
    const SequenceId sequence_id = metadata.sequence_id;
    if (sequences_.find(sequence_id) != sequences_.end()) {
        return false;
    }

    const RequestId request_id = metadata.request_id;
    SequenceState state;
    state.metadata = std::move(metadata);
    const std::uint32_t tables = table_count();
    state.tables.reserve(tables);
    for (std::uint32_t i = 0; i < tables; ++i) {
        state.tables.emplace_back(sequence_id);
    }
    sequences_.emplace(sequence_id, std::move(state));
    index_sequence(request_id, sequence_id);
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

    for (const BlockTable& table : parent_it->second.tables) {
        for (const BlockTableEntry& entry : table.entries()) {
            if (entry.swap_slot.has_value()) {
                return false;
            }
        }
    }

    SequenceState child;
    child.metadata = std::move(child_metadata);
    child.tables.reserve(parent_it->second.tables.size());
    for (const BlockTable& table : parent_it->second.tables) {
        BlockTable forked = table.fork(child_id);
        for (const BlockTableEntry& entry : forked.entries()) {
            allocator_->retain(entry.physical_id);
        }
        child.tables.push_back(std::move(forked));
    }

    const RequestId request_id = child.metadata.request_id;
    sequences_.emplace(child_id, std::move(child));
    index_sequence(request_id, child_id);
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
    const std::uint32_t frames_needed =
        blocks_needed * static_cast<std::uint32_t>(it->second.tables.size());
    if (!allocator_->can_allocate(frames_needed)) {
        return false;
    }

    // Layer-major: fill each layer's table completely before the next, so a
    // bulk reserve packs one layer's frames next to each other in the pool.
    for (BlockTable& table : it->second.tables) {
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
    }

    return true;
}

std::optional<PhysicalBlockId> KVCacheManager::ensure_token_writable(
    SequenceId sequence_id,
    TokenPosition token_position,
    CacheKind /*cache_kind*/,
    std::uint32_t layer) {
    auto it = sequences_.find(sequence_id);
    if (it == sequences_.end() || it->second.tables.empty()) {
        return std::nullopt;
    }

    if (layer >= it->second.tables.size()) {
        return std::nullopt;
    }
    BlockTable& table = it->second.tables[layer];
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

    if (entry->swap_slot.has_value()) {
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

std::uint32_t KVCacheManager::swap_out_sequence(SequenceId sequence_id) {
    auto it = sequences_.find(sequence_id);
    if (it == sequences_.end()) {
        return 0;
    }

    std::uint32_t swapped = 0;
    for (BlockTable& table : it->second.tables) {
        std::vector<LogicalBlockIndex> resident;
        for (const BlockTableEntry& entry : table.entries()) {
            if (!entry.swap_slot.has_value()) {
                resident.push_back(entry.logical.index);
            }
        }

        for (const LogicalBlockIndex index : resident) {
            BlockTableEntry* entry = table.mutable_entry(index);
            if (entry == nullptr) {
                continue;
            }

            const std::optional<SwapSlotId> slot = allocator_->swap_out(entry->physical_id);
            if (!slot.has_value()) {
                continue;
            }

            entry->swap_slot = slot;
            ++swapped;
        }
    }

    return swapped;
}

bool KVCacheManager::swap_in_sequence(SequenceId sequence_id) {
    auto it = sequences_.find(sequence_id);
    if (it == sequences_.end()) {
        return false;
    }

    std::uint32_t needed = 0;
    for (const BlockTable& table : it->second.tables) {
        for (const BlockTableEntry& entry : table.entries()) {
            if (entry.swap_slot.has_value()) {
                ++needed;
            }
        }
    }
    if (needed == 0) {
        return true;
    }
    if (!allocator_->can_allocate(needed)) {
        return false;
    }

    for (BlockTable& table : it->second.tables) {
        for (const BlockTableEntry& entry : table.entries()) {
            if (!entry.swap_slot.has_value()) {
                continue;
            }
            BlockTableEntry* mutable_entry = table.mutable_entry(entry.logical.index);
            if (mutable_entry == nullptr) {
                return false;
            }
            const std::optional<PhysicalBlockId> restored =
                allocator_->swap_in(*mutable_entry->swap_slot);
            if (!restored.has_value()) {
                return false;
            }
            mutable_entry->physical_id = *restored;
            mutable_entry->swap_slot.reset();
        }
    }

    return true;
}

void KVCacheManager::release_sequence(SequenceId sequence_id) {
    auto it = sequences_.find(sequence_id);
    if (it == sequences_.end()) {
        return;
    }

    drop_prefix_user(it->second);

    for (const BlockTable& table : it->second.tables) {
        for (const BlockTableEntry& entry : table.entries()) {
            if (entry.swap_slot.has_value()) {
                allocator_->discard_swapped(*entry.swap_slot);
            } else {
                allocator_->release(entry.physical_id);
            }
        }
    }
    unindex_sequence(it->second.metadata.request_id, sequence_id);
    sequences_.erase(it);
}

std::vector<SequenceId> KVCacheManager::sequences_for(RequestId request_id) const {
    auto it = sequences_by_request_.find(request_id);
    if (it == sequences_by_request_.end()) {
        return {};
    }
    return it->second;
}

std::uint32_t KVCacheManager::swap_out_request(RequestId request_id) {
    std::uint32_t swapped = 0;
    for (const SequenceId sequence_id : sequences_for(request_id)) {
        swapped += swap_out_sequence(sequence_id);
    }
    return swapped;
}

bool KVCacheManager::swap_in_request(RequestId request_id) {
    const std::vector<SequenceId> sequences = sequences_for(request_id);
    std::uint32_t needed = 0;
    for (const SequenceId sequence_id : sequences) {
        auto it = sequences_.find(sequence_id);
        if (it == sequences_.end()) {
            continue;
        }
        for (const BlockTable& table : it->second.tables) {
            for (const BlockTableEntry& entry : table.entries()) {
                if (entry.swap_slot.has_value()) {
                    ++needed;
                }
            }
        }
    }
    if (needed > 0 && !allocator_->can_allocate(needed)) {
        return false;
    }
    for (const SequenceId sequence_id : sequences) {
        if (!swap_in_sequence(sequence_id)) {
            return false;
        }
    }
    return true;
}

void KVCacheManager::release_request(RequestId request_id) {
    const std::vector<SequenceId> sequences = sequences_for(request_id);
    for (const SequenceId sequence_id : sequences) {
        release_sequence(sequence_id);
    }
}

std::uint32_t KVCacheManager::prefix_token_count(const std::string& key) const {
    if (key.empty()) {
        return 0;
    }
    auto it = prefix_index_.find(key);
    if (it == prefix_index_.end() || it->second.size() != 1) {
        return 0;
    }
    return it->second.front().token_count;
}

bool KVCacheManager::publish_prefix(SequenceId sequence_id, const std::string& key) {
    if (key.empty()) {
        return false;
    }
    auto it = sequences_.find(sequence_id);
    if (it == sequences_.end() || it->second.text_table().empty()) {
        return false;
    }
    if (it->second.prefix_key == key) {
        return true;
    }

    const std::uint32_t tpb = tokens_per_block();
    if (tpb == 0) {
        return false;
    }

    PrefixRecord record;
    record.content_hash = content_hash_of(it->second.tables);
    record.token_count = static_cast<std::uint32_t>(it->second.text_table().size()) * tpb;
    record.users = 1;
    for (const BlockTable& table : it->second.tables) {
        for (const BlockTableEntry& entry : table.entries()) {
            if (entry.swap_slot.has_value()) {
                return false;
            }
            record.physical_ids.push_back(entry.physical_id);
        }
    }

    auto& records = prefix_index_[key];
    for (PrefixRecord& existing : records) {
        if (existing.content_hash == record.content_hash) {
            ++existing.users;
            it->second.prefix_key = key;
            it->second.prefix_hash = existing.content_hash;
            return true;
        }
    }

    for (const PhysicalBlockId id : record.physical_ids) {
        allocator_->retain(id);
    }
    it->second.prefix_key = key;
    it->second.prefix_hash = record.content_hash;
    records.push_back(std::move(record));
    return true;
}

std::uint32_t KVCacheManager::attach_prefix(SequenceId sequence_id, const std::string& key) {
    if (key.empty()) {
        return 0;
    }
    auto it = sequences_.find(sequence_id);
    if (it == sequences_.end() || !it->second.text_table().empty()) {
        return 0;
    }

    auto pit = prefix_index_.find(key);
    if (pit == prefix_index_.end() || pit->second.size() != 1) {
        return 0;
    }

    PrefixRecord& record = pit->second.front();
    const std::uint32_t tpb = tokens_per_block();
    const std::size_t n_tables = it->second.tables.size();
    if (tpb == 0 || record.physical_ids.empty() || n_tables == 0) {
        return 0;
    }
    if (record.physical_ids.size() % n_tables != 0) {
        return 0;
    }

    const std::size_t ids_per_table = record.physical_ids.size() / n_tables;
    std::size_t id_index = 0;
    for (BlockTable& table : it->second.tables) {
        for (std::size_t i = 0; i < ids_per_table; ++i) {
            allocator_->retain(record.physical_ids[id_index]);
            LogicalBlock logical;
            logical.sequence_id = sequence_id;
            logical.index = static_cast<LogicalBlockIndex>(i);
            logical.cache_kind = table.cache_kind();
            logical.start_token = static_cast<TokenPosition>(i * tpb);
            logical.token_count = tpb;
            table.map(logical, record.physical_ids[id_index], false);
            ++id_index;
        }
    }
    ++record.users;
    it->second.prefix_key = key;
    it->second.prefix_hash = record.content_hash;
    return record.token_count;
}

std::optional<CacheView> KVCacheManager::cache_view(SequenceId sequence_id, CacheKind cache_kind) const {
    auto it = sequences_.find(sequence_id);
    if (it == sequences_.end()) {
        return std::nullopt;
    }

    CacheView view;
    view.sequence_id = sequence_id;
    view.cache_kind = cache_kind;
    view.block_table = &it->second.text_table();
    view.layer_tables = &it->second.tables;
    view.allocator = allocator_;
    if (const PhysicalBlock* block = allocator_->block(0); block != nullptr) {
        view.layout.shape = block->shape();
    }
    return view;
}

bool KVCacheManager::contains(SequenceId sequence_id) const noexcept {
    return sequences_.find(sequence_id) != sequences_.end();
}

} // namespace qwenvl_paged
