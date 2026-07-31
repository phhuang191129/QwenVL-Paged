/**
 * @brief Latency harness for the allocator core and the scheduler loop.
 *
 * This is a plain std::chrono harness rather than a microbenchmark framework:
 * it exists to give the four Week 8 numbers a stable, dependency-free home so
 * regressions are visible when the allocator changes, not to produce
 * publication-grade statistics.
 *
 * Setup and teardown are kept outside the measured region of every iteration,
 * so each reported figure covers only the operation named in the first column.
 */

#include "qwenvl_paged/KVCacheManager.h"
#include "qwenvl_paged/MemoryAllocator.h"
#include "qwenvl_paged/Scheduler.h"

#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <optional>
#include <string>

namespace {

using qwenvl_paged::AllocatorConfig;
using qwenvl_paged::BatchPlan;
using qwenvl_paged::BlockShape;
using qwenvl_paged::KVCacheManager;
using qwenvl_paged::MemoryAllocator;
using qwenvl_paged::PhysicalBlockId;
using qwenvl_paged::Request;
using qwenvl_paged::Scheduler;
using qwenvl_paged::SchedulerConfig;
using qwenvl_paged::SequenceMetadata;

using Clock = std::chrono::steady_clock;
using Nanoseconds = std::chrono::duration<double, std::nano>;

constexpr std::uint32_t kTokensPerBlock = 16;
constexpr std::uint32_t kNumLayers = 8;
constexpr std::uint32_t kNumKvHeads = 8;
constexpr std::uint32_t kHeadDim = 128;
constexpr std::uint32_t kBytesPerElement = 2; // fp16/bf16
constexpr std::uint32_t kMaxBlocks = 128;

BlockShape make_shape() {
    BlockShape shape;
    shape.tokens_per_block = kTokensPerBlock;
    shape.num_layers = kNumLayers;
    shape.num_kv_heads = kNumKvHeads;
    shape.head_dim = kHeadDim;
    shape.bytes_per_element = kBytesPerElement;
    return shape;
}

AllocatorConfig make_allocator_config() {
    AllocatorConfig config;
    config.block_shape = make_shape();
    config.max_blocks = kMaxBlocks;
    return config;
}

void report(const std::string& name, std::uint64_t operations, double elapsed_ns) {
    const double ns_per_op = elapsed_ns / static_cast<double>(operations);
    std::cout << std::left << std::setw(34) << name << std::right << std::setw(12) << operations
              << std::setw(14) << std::fixed << std::setprecision(1) << ns_per_op << std::setw(16)
              << std::setprecision(0) << (1e9 / ns_per_op) << '\n';
}

/**
 * @brief Time one allocate/release round trip on a warm free list.
 */
void bench_allocate_release() {
    constexpr std::uint64_t kIterations = 200000;
    MemoryAllocator allocator(make_allocator_config());

    const Clock::time_point start = Clock::now();
    for (std::uint64_t i = 0; i < kIterations; ++i) {
        const std::optional<PhysicalBlockId> id = allocator.allocate();
        allocator.release(*id);
    }
    const double elapsed = Nanoseconds(Clock::now() - start).count();

    report("allocate + release", kIterations, elapsed);
}

/**
 * @brief Time forking a prompt of `kBlocksPerSequence` shared blocks.
 */
void bench_fork() {
    constexpr std::uint64_t kIterations = 20000;
    constexpr std::uint32_t kBlocksPerSequence = 8;
    constexpr std::uint32_t kPromptTokens = kBlocksPerSequence * kTokensPerBlock;

    MemoryAllocator allocator(make_allocator_config());
    KVCacheManager cache(allocator);

    double elapsed = 0.0;
    for (std::uint64_t i = 0; i < kIterations; ++i) {
        cache.create_sequence(SequenceMetadata{1, 1, {}, {}});
        cache.reserve_tokens(1, kPromptTokens);

        const Clock::time_point start = Clock::now();
        cache.fork_sequence(1, SequenceMetadata{2, 1, {}, {}});
        elapsed += Nanoseconds(Clock::now() - start).count();

        cache.release_sequence(2);
        cache.release_sequence(1);
    }

    report("fork sequence (8 blocks)", kIterations, elapsed);
}

/**
 * @brief Time the first write to a shared block, which materializes a copy.
 */
void bench_copy_on_write() {
    constexpr std::uint64_t kIterations = 5000;

    MemoryAllocator allocator(make_allocator_config());
    KVCacheManager cache(allocator);

    double elapsed = 0.0;
    for (std::uint64_t i = 0; i < kIterations; ++i) {
        cache.create_sequence(SequenceMetadata{1, 1, {}, {}});
        cache.reserve_tokens(1, kTokensPerBlock);
        cache.fork_sequence(1, SequenceMetadata{2, 1, {}, {}});

        const Clock::time_point start = Clock::now();
        const std::optional<PhysicalBlockId> copy = cache.ensure_token_writable(2, 0);
        elapsed += Nanoseconds(Clock::now() - start).count();
        if (!copy.has_value()) {
            std::cerr << "copy-on-write benchmark failed to materialize a block\n";
            return;
        }

        cache.release_sequence(2);
        cache.release_sequence(1);
    }

    report("copy-on-write (1 block)", kIterations, elapsed);
}

/**
 * @brief Time admitting, stepping, and retiring a full batch of requests.
 */
void bench_scheduler_throughput() {
    constexpr std::uint64_t kCycles = 20000;
    constexpr std::uint32_t kBatchSize = 8;
    constexpr std::uint32_t kPromptTokens = 4 * kTokensPerBlock;

    MemoryAllocator allocator(make_allocator_config());
    KVCacheManager cache(allocator);

    SchedulerConfig config;
    config.max_active_requests = kBatchSize;
    config.max_batch_tokens = kBatchSize * kPromptTokens;
    Scheduler scheduler(config, cache, allocator);

    const Clock::time_point start = Clock::now();
    for (std::uint64_t cycle = 0; cycle < kCycles; ++cycle) {
        for (std::uint32_t i = 0; i < kBatchSize; ++i) {
            Request request;
            request.request_id = i + 1;
            request.root_sequence_id = i + 1;
            request.prompt_tokens = kPromptTokens;
            request.sampling.max_decode_tokens = 1;
            scheduler.enqueue(request);
        }

        const BatchPlan plan = scheduler.schedule_next();
        if (plan.prefill_requests.size() != kBatchSize) {
            std::cerr << "scheduler benchmark failed to admit the full batch\n";
            return;
        }

        for (std::uint32_t i = 0; i < kBatchSize; ++i) {
            scheduler.complete_step(i + 1, 1);
        }
        for (std::uint32_t i = 0; i < kBatchSize; ++i) {
            scheduler.cancel(i + 1);
        }
    }
    const double elapsed = Nanoseconds(Clock::now() - start).count();

    report("scheduler admit + retire request", kCycles * kBatchSize, elapsed);
}

} // namespace

int main() {
    const BlockShape shape = make_shape();
    std::cout << "QwenVL-Paged allocator benchmark\n"
              << "block: " << shape.tokens_per_block << " tokens x " << shape.num_layers
              << " layers x " << shape.num_kv_heads << " kv heads x " << shape.head_dim << " dim, "
              << (shape.byte_size() / 1024) << " KiB/block, pool " << kMaxBlocks << " blocks\n\n"
              << std::left << std::setw(34) << "benchmark" << std::right << std::setw(12)
              << "operations" << std::setw(14) << "ns/op" << std::setw(16) << "ops/sec" << '\n'
              << std::string(76, '-') << '\n';

    bench_allocate_release();
    bench_fork();
    bench_copy_on_write();
    bench_scheduler_throughput();

    return 0;
}
