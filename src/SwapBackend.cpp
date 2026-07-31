#include "qwenvl_paged/SwapBackend.h"

#include <cstring>
#include <utility>

namespace qwenvl_paged {

HostSwapBackend::HostSwapBackend(std::size_t max_slots) : max_slots_(max_slots) {}

std::optional<SwapSlotId> HostSwapBackend::store(const PhysicalBlock& block) {
    if (slots_.size() >= max_slots_) {
        return std::nullopt;
    }

    std::vector<std::byte> bytes(block.size_bytes());
    if (!bytes.empty()) {
        std::memcpy(bytes.data(), block.data(), bytes.size());
    }

    const SwapSlotId slot = next_slot_++;
    slots_.emplace(slot, std::move(bytes));
    return slot;
}

bool HostSwapBackend::load(SwapSlotId slot, PhysicalBlock& block) {
    auto it = slots_.find(slot);
    if (it == slots_.end()) {
        return false;
    }

    const std::vector<std::byte>& bytes = it->second;
    if (block.size_bytes() < bytes.size()) {
        return false;
    }
    if (!bytes.empty()) {
        std::memcpy(block.data(), bytes.data(), bytes.size());
    }
    return true;
}

void HostSwapBackend::discard(SwapSlotId slot) {
    slots_.erase(slot);
}

std::size_t HostSwapBackend::resident_slots() const noexcept {
    return slots_.size();
}

std::size_t HostSwapBackend::capacity_slots() const noexcept {
    return max_slots_;
}

} // namespace qwenvl_paged
