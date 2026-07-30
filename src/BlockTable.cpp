#include "qwenvl_paged/BlockTable.h"

#include <algorithm>

namespace qwenvl_paged {

namespace {

std::vector<BlockTableEntry>::iterator lower_bound_index(
    std::vector<BlockTableEntry>& entries, LogicalBlockIndex index) {
    return std::lower_bound(
        entries.begin(), entries.end(), index,
        [](const BlockTableEntry& entry, LogicalBlockIndex value) {
            return entry.logical.index < value;
        });
}

std::vector<BlockTableEntry>::const_iterator lower_bound_index(
    const std::vector<BlockTableEntry>& entries, LogicalBlockIndex index) {
    return std::lower_bound(
        entries.begin(), entries.end(), index,
        [](const BlockTableEntry& entry, LogicalBlockIndex value) {
            return entry.logical.index < value;
        });
}

} // namespace

BlockTable::BlockTable(SequenceId sequence_id, CacheKind cache_kind)
    : sequence_id_(sequence_id), cache_kind_(cache_kind) {}

SequenceId BlockTable::sequence_id() const noexcept {
    return sequence_id_;
}

CacheKind BlockTable::cache_kind() const noexcept {
    return cache_kind_;
}

std::size_t BlockTable::size() const noexcept {
    return entries_.size();
}

bool BlockTable::empty() const noexcept {
    return entries_.empty();
}

void BlockTable::map(LogicalBlock logical, PhysicalBlockId physical_id, bool writable) {
    auto it = lower_bound_index(entries_, logical.index);
    if (it != entries_.end() && it->logical.index == logical.index) {
        it->logical = logical;
        it->physical_id = physical_id;
        it->writable = writable;
        return;
    }

    BlockTableEntry entry;
    entry.logical = logical;
    entry.physical_id = physical_id;
    entry.writable = writable;
    entries_.insert(it, entry);
}

std::optional<PhysicalBlockId> BlockTable::unmap(LogicalBlockIndex index) {
    auto it = lower_bound_index(entries_, index);
    if (it == entries_.end() || it->logical.index != index) {
        return std::nullopt;
    }

    const PhysicalBlockId physical_id = it->physical_id;
    entries_.erase(it);
    return physical_id;
}

std::optional<PhysicalBlockId> BlockTable::lookup(LogicalBlockIndex index) const {
    auto it = lower_bound_index(entries_, index);
    if (it == entries_.end() || it->logical.index != index) {
        return std::nullopt;
    }
    return it->physical_id;
}

const BlockTableEntry* BlockTable::entry(LogicalBlockIndex index) const noexcept {
    auto it = lower_bound_index(entries_, index);
    if (it == entries_.end() || it->logical.index != index) {
        return nullptr;
    }
    return &(*it);
}

BlockTableEntry* BlockTable::mutable_entry(LogicalBlockIndex index) noexcept {
    auto it = lower_bound_index(entries_, index);
    if (it == entries_.end() || it->logical.index != index) {
        return nullptr;
    }
    return &(*it);
}

void BlockTable::set_writable(LogicalBlockIndex index, bool writable) {
    if (BlockTableEntry* target = mutable_entry(index)) {
        target->writable = writable;
    }
}

const std::vector<BlockTableEntry>& BlockTable::entries() const noexcept {
    return entries_;
}

BlockTable BlockTable::fork(SequenceId child_sequence_id) const {
    BlockTable child(*this);
    child.sequence_id_ = child_sequence_id;
    for (BlockTableEntry& entry : child.entries_) {
        entry.writable = false;
        entry.logical.sequence_id = child_sequence_id;
    }
    return child;
}

} // namespace qwenvl_paged
