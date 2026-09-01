#include "qwenvl_paged/KVCacheManager.h"

#include <algorithm>
#include <utility>

namespace qwenvl_paged {

bool CacheView::valid() const noexcept {
    return block_table != nullptr && allocator != nullptr && layout.element_count() > 0;
}

const std::byte* CacheView::block_bytes(LogicalBlockIndex index) const noexcept {
    if (!valid()) {
        return nullptr;
    }

    // lookup already faults on a swapped-out entry, whose frame went back to the
    // free list and may now belong to another sequence.
    const std::optional<PhysicalBlockId> physical = block_table->lookup(index);
    if (!physical.has_value()) {
        return nullptr;
    }

    const PhysicalBlock* block = allocator->block(*physical);
    return block == nullptr ? nullptr : block->data();
}

KVCacheManager::KVCacheManager(MemoryAllocator& allocator) : allocator_(&allocator) {}

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
    sequences_.emplace(sequence_id, SequenceState{std::move(metadata), BlockTable(sequence_id), {}, 0});
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

    for (const BlockTableEntry& entry : parent_it->second.text_table.entries()) {
        if (entry.swap_slot.has_value()) {
            return false;
        }
    }

    BlockTable child_table = parent_it->second.text_table.fork(child_id);
    for (const BlockTableEntry& entry : child_table.entries()) {
        allocator_->retain(entry.physical_id);
    }

    const RequestId request_id = child_metadata.request_id;
    sequences_.emplace(child_id, SequenceState{std::move(child_metadata), std::move(child_table), {}, 0});
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

    BlockTable& table = it->second.text_table;
    std::vector<LogicalBlockIndex> resident;
    for (const BlockTableEntry& entry : table.entries()) {
        if (!entry.swap_slot.has_value()) {
            resident.push_back(entry.logical.index);
        }
    }

    std::uint32_t swapped = 0;
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

    return swapped;
}

bool KVCacheManager::swap_in_sequence(SequenceId sequence_id) {
    auto it = sequences_.find(sequence_id);
    if (it == sequences_.end()) {
        return false;
    }

    BlockTable& table = it->second.text_table;
    std::vector<LogicalBlockIndex> swapped;
    for (const BlockTableEntry& entry : table.entries()) {
        if (entry.swap_slot.has_value()) {
            swapped.push_back(entry.logical.index);
        }
    }

    if (swapped.empty()) {
        return true;
    }
    if (!allocator_->can_allocate(static_cast<std::uint32_t>(swapped.size()))) {
        return false;
    }

    for (const LogicalBlockIndex index : swapped) {
        BlockTableEntry* entry = table.mutable_entry(index);
        if (entry == nullptr) {
            continue;
        }

        const std::optional<PhysicalBlockId> restored = allocator_->swap_in(*entry->swap_slot);
        if (!restored.has_value()) {
            return false;
        }

        entry->physical_id = *restored;
        entry->swap_slot.reset();
    }

    return true;
}

void KVCacheManager::release_sequence(SequenceId sequence_id) {
    auto it = sequences_.find(sequence_id);
    if (it == sequences_.end()) {
        return;
    }

    drop_prefix_user(it->second);

    for (const BlockTableEntry& entry : it->second.text_table.entries()) {
        if (entry.swap_slot.has_value()) {
            allocator_->discard_swapped(*entry.swap_slot);
        } else {
            allocator_->release(entry.physical_id);
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
        for (const BlockTableEntry& entry : it->second.text_table.entries()) {
            if (entry.swap_slot.has_value()) {
                ++needed;
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
    if (it == sequences_.end() || it->second.text_table.empty()) {
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
    record.content_hash = content_hash_of(it->second.text_table);
    record.token_count = static_cast<std::uint32_t>(it->second.text_table.size()) * tpb;
    record.users = 1;
    for (const BlockTableEntry& entry : it->second.text_table.entries()) {
        if (entry.swap_slot.has_value()) {
            return false;
        }
        record.physical_ids.push_back(entry.physical_id);
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
    if (it == sequences_.end() || !it->second.text_table.empty()) {
        return 0;
    }

    auto pit = prefix_index_.find(key);
    if (pit == prefix_index_.end() || pit->second.size() != 1) {
        return 0;
    }

    PrefixRecord& record = pit->second.front();
    const std::uint32_t tpb = tokens_per_block();
    if (tpb == 0 || record.physical_ids.empty()) {
        return 0;
    }

    BlockTable& table = it->second.text_table;
    for (std::size_t i = 0; i < record.physical_ids.size(); ++i) {
        allocator_->retain(record.physical_ids[i]);
        LogicalBlock logical;
        logical.sequence_id = sequence_id;
        logical.index = static_cast<LogicalBlockIndex>(i);
        logical.cache_kind = table.cache_kind();
        logical.start_token = static_cast<TokenPosition>(i * tpb);
        logical.token_count = tpb;
        table.map(logical, record.physical_ids[i], false);
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
    view.block_table = &it->second.text_table;
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
