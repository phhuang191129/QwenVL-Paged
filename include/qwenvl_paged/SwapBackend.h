#pragma once

#include "qwenvl_paged/Block.h"

#include <cstddef>
#include <optional>
#include <unordered_map>
#include <vector>

namespace qwenvl_paged {

/**
 * @brief Backend-neutral store for the contents of evicted cache blocks.
 *
 * A swap backend holds block bytes while the physical frame that carried them
 * is reclaimed for another sequence. The CPU prototype keeps them in host
 * memory; the same interface is meant to later back onto device-to-host
 * migration or compressed/offloaded block storage without changing allocator
 * or block-table semantics.
 *
 * Slots are opaque handles owned by the backend. A slot id may only be reused
 * after that slot has been discarded.
 *
 * Like the rest of the phase-1 core, implementations are not thread-safe and
 * are mutated only by the engine event loop.
 */
class SwapBackend {
public:
    virtual ~SwapBackend() = default;

    /**
     * @brief Copies a block's bytes into the backend.
     *
     * @return The slot now holding the copy, or nullopt when the backend is full.
     */
    [[nodiscard]] virtual std::optional<SwapSlotId> store(const PhysicalBlock& block) = 0;

    /**
     * @brief Copies a stored slot back into a physical block.
     *
     * The slot stays resident so a failed swap-in can be retried; the caller
     * decides when to discard it.
     *
     * @return False when the slot is unknown or the block is too small.
     */
    [[nodiscard]] virtual bool load(SwapSlotId slot, PhysicalBlock& block) = 0;

    /**
     * @brief Drops a slot without restoring it.
     */
    virtual void discard(SwapSlotId slot) = 0;

    /**
     * @brief Returns how many slots the backend currently holds.
     */
    [[nodiscard]] virtual std::size_t resident_slots() const noexcept = 0;
};

/**
 * @brief Fixed-capacity host-memory swap space used by the CPU prototype.
 */
class HostSwapBackend final : public SwapBackend {
public:
    /**
     * @brief Creates a swap space that can hold at most `max_slots` blocks.
     */
    explicit HostSwapBackend(std::size_t max_slots);

    ~HostSwapBackend() override = default;

    HostSwapBackend(const HostSwapBackend&) = delete;
    HostSwapBackend& operator=(const HostSwapBackend&) = delete;
    HostSwapBackend(HostSwapBackend&&) noexcept = default;
    HostSwapBackend& operator=(HostSwapBackend&&) noexcept = default;

    [[nodiscard]] std::optional<SwapSlotId> store(const PhysicalBlock& block) override;

    [[nodiscard]] bool load(SwapSlotId slot, PhysicalBlock& block) override;

    void discard(SwapSlotId slot) override;

    [[nodiscard]] std::size_t resident_slots() const noexcept override;

    /**
     * @brief Returns the maximum number of slots this swap space can hold.
     */
    [[nodiscard]] std::size_t capacity_slots() const noexcept;

private:
    std::size_t max_slots_{0};
    SwapSlotId next_slot_{1};
    std::unordered_map<SwapSlotId, std::vector<std::byte>> slots_;
};

} // namespace qwenvl_paged
