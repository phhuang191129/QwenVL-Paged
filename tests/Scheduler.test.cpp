/**
 * @brief Specification tests for qwenvl_paged::Scheduler.
 *
 * These tests are written against the documented header contract in
 * include/qwenvl_paged/Scheduler.h. Scheduler (and the KVCacheManager /
 * MemoryAllocator it depends on) have no .cpp implementation yet, so this test
 * binary is expected to fail at the LINK stage (undefined references) until
 * those classes are implemented. That failure is intentional for this phase of
 * the project.
 *
 * SchedulerConfig fields default to 0, whose "unlimited vs. none" meaning is not
 * pinned down by the header. These tests therefore set generous, explicit limits
 * so admission never depends on the 0-default interpretation.
 */

#include "qwenvl_paged/KVCacheManager.h"
#include "qwenvl_paged/MemoryAllocator.h"
#include "qwenvl_paged/Scheduler.h"
#include "qwenvl_paged/SwapBackend.h"

#include <gtest/gtest.h>

#include <cstddef>
#include <optional>
#include <unordered_map>
#include <vector>

namespace qwenvl_paged {
namespace {

AllocatorConfig make_allocator_config(std::uint32_t max_blocks = 64) {
    AllocatorConfig config;
    config.block_shape.tokens_per_block = 16;
    config.block_shape.num_layers = 2;
    config.block_shape.num_kv_heads = 2;
    config.block_shape.head_dim = 8;
    config.block_shape.bytes_per_element = 2;
    config.max_blocks = max_blocks;
    return config;
}

SchedulerConfig make_scheduler_config() {
    SchedulerConfig config;
    config.max_active_requests = 8;
    config.max_batch_tokens = 4096;
    config.preemption_watermark_blocks = 0;
    return config;
}

/**
 * @brief Builds a simple text request with a short prompt.
 */
Request make_request(RequestId request_id, std::uint32_t prompt_tokens = 32) {
    Request request;
    request.request_id = request_id;
    request.root_sequence_id = request_id;
    request.prompt_tokens = prompt_tokens;
    request.sampling.num_parallel_samples = 1;
    request.sampling.max_decode_tokens = 8;
    return request;
}

/**
 * @brief Bundles the allocator + cache manager a scheduler depends on.
 *
 * These are non-owning dependencies of Scheduler and must outlive it, so the
 * fixture keeps them alive alongside the scheduler under test.
 */
class SchedulerTest : public ::testing::Test {
protected:
    SchedulerTest()
        : allocator_(make_allocator_config()),
          cache_manager_(allocator_),
          scheduler_(make_scheduler_config(), cache_manager_, allocator_) {}

    MemoryAllocator allocator_;
    KVCacheManager cache_manager_;
    Scheduler scheduler_;
};

TEST_F(SchedulerTest, StartsEmpty) {
    EXPECT_EQ(scheduler_.active_size(), 0u);
    EXPECT_FALSE(scheduler_.state(1).has_value());
}

TEST_F(SchedulerTest, EnqueueTracksRequestAsPending) {
    scheduler_.enqueue(make_request(1));

    const std::optional<RequestState> state = scheduler_.state(1);
    ASSERT_TRUE(state.has_value());
    EXPECT_EQ(*state, RequestState::Pending);
    EXPECT_EQ(scheduler_.active_size(), 0u);
}

TEST_F(SchedulerTest, ScheduleNextAdmitsPendingAsPrefill) {
    scheduler_.enqueue(make_request(1));

    const BatchPlan plan = scheduler_.schedule_next();

    ASSERT_EQ(plan.prefill_requests.size(), 1u);
    EXPECT_EQ(plan.prefill_requests.front(), 1u);
    EXPECT_EQ(scheduler_.state(1), RequestState::Prefill);
    EXPECT_EQ(scheduler_.active_size(), 1u);
}

TEST_F(SchedulerTest, CompleteStepMovesPrefillToDecode) {
    scheduler_.enqueue(make_request(1));
    static_cast<void>(scheduler_.schedule_next());

    scheduler_.complete_step(1, 1);

    EXPECT_EQ(scheduler_.state(1), RequestState::Decode);
}

TEST_F(SchedulerTest, CancelRemovesRequestFromActiveSet) {
    scheduler_.enqueue(make_request(1));
    static_cast<void>(scheduler_.schedule_next());
    ASSERT_EQ(scheduler_.active_size(), 1u);

    scheduler_.cancel(1);

    EXPECT_EQ(scheduler_.active_size(), 0u);
}

TEST_F(SchedulerTest, PreemptMovesActiveRequestToPreempted) {
    scheduler_.enqueue(make_request(1));
    static_cast<void>(scheduler_.schedule_next());

    EXPECT_TRUE(scheduler_.preempt(1, "cache pressure"));
    EXPECT_EQ(scheduler_.state(1), RequestState::Preempted);
    EXPECT_EQ(scheduler_.active_size(), 0u);
}

TEST_F(SchedulerTest, ResumeReturnsPreemptedRequestToActiveSet) {
    scheduler_.enqueue(make_request(1));
    static_cast<void>(scheduler_.schedule_next());
    ASSERT_TRUE(scheduler_.preempt(1, "cache pressure"));

    EXPECT_TRUE(scheduler_.resume(1));
    EXPECT_NE(scheduler_.state(1), RequestState::Preempted);
    EXPECT_EQ(scheduler_.active_size(), 1u);
}

TEST_F(SchedulerTest, PreemptUnknownRequestReturnsFalse) {
    EXPECT_FALSE(scheduler_.preempt(99, "cache pressure"));
}

TEST_F(SchedulerTest, PreemptionInfoIsAbsentForAnActiveRequest) {
    scheduler_.enqueue(make_request(1));
    static_cast<void>(scheduler_.schedule_next());

    EXPECT_FALSE(scheduler_.preemption_info(1).has_value());
}

/**
 * @brief Scheduler over a pool too small for every request at once.
 *
 * With tokens_per_block == 16, a four-block pool backs 64 prompt tokens in
 * total, so the Week 7 tests can create real cache pressure with short prompts.
 */
class SchedulerMemoryPressureTest : public ::testing::Test {
protected:
    static constexpr std::uint32_t kPoolBlocks = 4;

    SchedulerMemoryPressureTest()
        : allocator_(make_allocator_config(kPoolBlocks)),
          swap_backend_(32),
          cache_manager_(allocator_),
          scheduler_(make_scheduler_config(), cache_manager_, allocator_) {
        allocator_.set_swap_backend(&swap_backend_);
    }

    MemoryAllocator allocator_;
    HostSwapBackend swap_backend_;
    KVCacheManager cache_manager_;
    Scheduler scheduler_;
};

TEST_F(SchedulerMemoryPressureTest, AdmissionIsRefusedWhenCacheCannotBackThePrompt) {
    scheduler_.enqueue(make_request(1, 64));
    scheduler_.enqueue(make_request(2, 32));

    const BatchPlan plan = scheduler_.schedule_next();

    ASSERT_EQ(plan.prefill_requests.size(), 1u);
    EXPECT_EQ(plan.prefill_requests.front(), 1u);
    EXPECT_EQ(scheduler_.state(2), RequestState::Pending);
    EXPECT_EQ(scheduler_.active_size(), 1u);
    // A refused admission must not leave half-created sequence state behind.
    EXPECT_FALSE(cache_manager_.contains(2));
}

TEST_F(SchedulerMemoryPressureTest, PreemptSwapsCacheOutAndRecordsMetadata) {
    scheduler_.enqueue(make_request(1, 32));
    static_cast<void>(scheduler_.schedule_next());
    ASSERT_EQ(allocator_.stats().free_blocks, kPoolBlocks - 2);

    ASSERT_TRUE(scheduler_.preempt(1, "cache pressure"));

    EXPECT_EQ(allocator_.stats().free_blocks, kPoolBlocks);
    EXPECT_EQ(swap_backend_.resident_slots(), 2u);

    const std::optional<PreemptionInfo> info = scheduler_.preemption_info(1);
    ASSERT_TRUE(info.has_value());
    EXPECT_EQ(info->reason, "cache pressure");
    EXPECT_EQ(info->swapped_blocks, 2u);
}

TEST_F(SchedulerMemoryPressureTest, PreemptionUnblocksARefusedAdmission) {
    scheduler_.enqueue(make_request(1, 64));
    scheduler_.enqueue(make_request(2, 32));
    static_cast<void>(scheduler_.schedule_next());
    ASSERT_EQ(scheduler_.state(2), RequestState::Pending);

    ASSERT_TRUE(scheduler_.preempt(1, "cache pressure"));

    const BatchPlan plan = scheduler_.schedule_next();

    ASSERT_EQ(plan.prefill_requests.size(), 1u);
    EXPECT_EQ(plan.prefill_requests.front(), 2u);
    EXPECT_EQ(scheduler_.state(2), RequestState::Prefill);
}

TEST_F(SchedulerMemoryPressureTest, ResumeSwapsCacheBackIn) {
    scheduler_.enqueue(make_request(1, 32));
    static_cast<void>(scheduler_.schedule_next());
    ASSERT_TRUE(scheduler_.preempt(1, "cache pressure"));

    EXPECT_TRUE(scheduler_.resume(1));

    EXPECT_NE(scheduler_.state(1), RequestState::Preempted);
    EXPECT_EQ(swap_backend_.resident_slots(), 0u);
    EXPECT_FALSE(scheduler_.preemption_info(1).has_value());

    const std::optional<CacheView> view = cache_manager_.cache_view(1);
    ASSERT_TRUE(view.has_value());
    ASSERT_NE(view->block_table, nullptr);
    for (const BlockTableEntry& entry : view->block_table->entries()) {
        EXPECT_FALSE(entry.swap_slot.has_value());
        EXPECT_TRUE(view->block_table->lookup(entry.logical.index).has_value());
    }
}

TEST_F(SchedulerMemoryPressureTest, ResumeFailsWhileThePoolIsStillFull) {
    scheduler_.enqueue(make_request(1, 32));
    scheduler_.enqueue(make_request(2, 64));
    static_cast<void>(scheduler_.schedule_next());
    ASSERT_TRUE(scheduler_.preempt(1, "cache pressure"));

    // Request 2 now takes the whole pool, leaving nothing to restore into.
    static_cast<void>(scheduler_.schedule_next());
    ASSERT_EQ(scheduler_.state(2), RequestState::Prefill);
    ASSERT_EQ(allocator_.stats().free_blocks, 0u);

    EXPECT_FALSE(scheduler_.resume(1));
    EXPECT_EQ(scheduler_.state(1), RequestState::Preempted);
    EXPECT_EQ(swap_backend_.resident_slots(), 2u);
    EXPECT_TRUE(scheduler_.preemption_info(1).has_value());
}

TEST_F(SchedulerMemoryPressureTest, CancellingAPreemptedRequestReleasesItsSwapSlots) {
    scheduler_.enqueue(make_request(1, 32));
    static_cast<void>(scheduler_.schedule_next());
    ASSERT_TRUE(scheduler_.preempt(1, "cache pressure"));

    scheduler_.cancel(1);

    EXPECT_FALSE(scheduler_.state(1).has_value());
    EXPECT_FALSE(scheduler_.preemption_info(1).has_value());
    EXPECT_EQ(swap_backend_.resident_slots(), 0u);
    EXPECT_EQ(allocator_.stats().free_blocks, kPoolBlocks);
}

TEST_F(SchedulerMemoryPressureTest, SimulationDrainsMixedRequestsUnderPressure) {
    // Nine blocks of demand against a four-block pool, so the batch can only be
    // drained by preempting, swapping out, and later resuming requests.
    constexpr std::uint32_t kDecodeStepsPerRequest = 2;
    const std::vector<std::uint32_t> prompts = {16, 48, 32, 16, 64};

    std::vector<RequestId> ids;
    for (std::size_t i = 0; i < prompts.size(); ++i) {
        const RequestId id = static_cast<RequestId>(i + 1);
        scheduler_.enqueue(make_request(id, prompts[i]));
        ids.push_back(id);
    }

    const auto first_with_state = [this, &ids](RequestState wanted) -> std::optional<RequestId> {
        for (const RequestId id : ids) {
            if (scheduler_.state(id) == wanted) {
                return id;
            }
        }
        return std::nullopt;
    };

    std::unordered_map<RequestId, std::uint32_t> decode_steps;
    std::size_t finished = 0;
    for (int step = 0; step < 200 && finished < prompts.size(); ++step) {
        // Restore preempted work only once the pressure that caused it is gone,
        // so a resume cannot immediately undo the preemption it depends on.
        if (!first_with_state(RequestState::Pending).has_value()) {
            const std::optional<RequestId> preempted = first_with_state(RequestState::Preempted);
            if (preempted.has_value()) {
                static_cast<void>(scheduler_.resume(*preempted));
            }
        }

        const BatchPlan plan = scheduler_.schedule_next();
        for (const RequestId id : plan.prefill_requests) {
            scheduler_.complete_step(id, 1);
        }
        for (const RequestId id : plan.decode_requests) {
            scheduler_.complete_step(id, 1);
            if (++decode_steps[id] >= kDecodeStepsPerRequest) {
                scheduler_.cancel(id);
                ++finished;
            }
        }

        const bool admission_blocked =
            plan.prefill_requests.empty() && first_with_state(RequestState::Pending).has_value();
        if (admission_blocked) {
            const std::optional<RequestId> victim = first_with_state(RequestState::Decode);
            if (victim.has_value()) {
                EXPECT_TRUE(scheduler_.preempt(*victim, "cache pressure"));
            }
        }
    }

    EXPECT_EQ(finished, prompts.size());
    EXPECT_EQ(allocator_.stats().free_blocks, kPoolBlocks);
    EXPECT_EQ(allocator_.stats().swapped_blocks, 0u);
    EXPECT_EQ(swap_backend_.resident_slots(), 0u);
}

} // namespace
} // namespace qwenvl_paged
