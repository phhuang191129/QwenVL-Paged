/**
 * @brief Latency harness for the allocator core and the scheduler loop.
 *
 * Week 10: warmup + N timed samples, reporting median / p95 / MAD. Run with
 * `--layers 8` for the week-8 512 KiB block and `--layers 28` for the 2B
 * 1.75 MiB block. Setup stays outside each sample.
 */

#include "Env.h"
#include "Stats.h"

#include "qwenvl_paged/KVCacheManager.h"
#include "qwenvl_paged/MemoryAllocator.h"
#include "qwenvl_paged/Scheduler.h"

#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <optional>
#include <string>
#include <vector>

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

constexpr int kWarmup = 3;
constexpr int kRepeats = 11;
constexpr std::uint32_t kTokensPerBlock = 16;
constexpr std::uint32_t kNumKvHeads = 8;
constexpr std::uint32_t kHeadDim = 128;
constexpr std::uint32_t kBytesPerElement = 2;
constexpr std::uint32_t kMaxBlocks = 128;

struct ShapeConfig {
    std::uint32_t num_layers{8};
};

BlockShape make_shape(const ShapeConfig& cfg) {
    BlockShape shape;
    shape.tokens_per_block = kTokensPerBlock;
    shape.num_layers = cfg.num_layers;
    shape.num_kv_heads = kNumKvHeads;
    shape.head_dim = kHeadDim;
    shape.bytes_per_element = kBytesPerElement;
    return shape;
}

AllocatorConfig make_allocator_config(const ShapeConfig& cfg) {
    AllocatorConfig config;
    config.block_shape = make_shape(cfg);
    config.max_blocks = kMaxBlocks;
    return config;
}

void report_row(
    std::ostream& csv,
    const std::string& name,
    std::uint64_t ops_per_sample,
    const qwenvl_bench::SampleStats& stats,
    std::size_t block_kib) {
    const double ns_per_op = stats.median_ns / static_cast<double>(ops_per_sample);
    const double p95_per_op = stats.p95_ns / static_cast<double>(ops_per_sample);
    const double mad_per_op = stats.mad_ns / static_cast<double>(ops_per_sample);
    std::cout << std::left << std::setw(34) << name << std::right << std::setw(10)
              << ops_per_sample << std::setw(8) << stats.n << std::setw(12) << std::fixed
              << std::setprecision(1) << ns_per_op << std::setw(12) << p95_per_op
              << std::setw(12) << mad_per_op << std::setw(14) << std::setprecision(0)
              << (ns_per_op > 0.0 ? 1e9 / ns_per_op : 0.0) << '\n';
    csv << qwenvl_bench::csv_escape(name) << ',' << block_kib << ',' << ops_per_sample << ','
        << stats.n << ',' << ns_per_op << ',' << p95_per_op << ',' << mad_per_op << ','
        << (ns_per_op > 0.0 ? 1e9 / ns_per_op : 0.0) << '\n';
}

void bench_allocate_release(const ShapeConfig& cfg, std::ostream& csv, std::size_t block_kib) {
    constexpr std::uint64_t kOps = 50000;
    MemoryAllocator allocator(make_allocator_config(cfg));
    const auto stats = qwenvl_bench::measure_body(
        [&]() {
            for (std::uint64_t i = 0; i < kOps; ++i) {
                const std::optional<PhysicalBlockId> id = allocator.allocate();
                allocator.release(*id);
            }
        },
        kWarmup,
        kRepeats);
    report_row(csv, "allocate + release", kOps, stats, block_kib);
}

void bench_fork(const ShapeConfig& cfg, std::ostream& csv, std::size_t block_kib) {
    constexpr std::uint64_t kOps = 2000;
    constexpr std::uint32_t kBlocksPerSequence = 8;
    constexpr std::uint32_t kPromptTokens = kBlocksPerSequence * kTokensPerBlock;

    MemoryAllocator allocator(make_allocator_config(cfg));
    KVCacheManager cache(allocator);

    const auto stats = qwenvl_bench::measure_body(
        [&]() {
            for (std::uint64_t i = 0; i < kOps; ++i) {
                cache.create_sequence(SequenceMetadata{1, 1, {}, {}});
                cache.reserve_tokens(1, kPromptTokens);
                cache.fork_sequence(1, SequenceMetadata{2, 1, {}, {}});
                cache.release_sequence(2);
                cache.release_sequence(1);
            }
        },
        kWarmup,
        kRepeats);
    // Fork is the named operation; create/reserve/release stay in the sample
    // because isolating them needs per-iteration setup that dwarfs 50 ns.
    // Re-time fork alone below so the week-8 column stays comparable.
    report_row(csv, "fork sequence (8 blocks, with setup)", kOps, stats, block_kib);

    std::vector<double> fork_samples;
    fork_samples.reserve(kRepeats);
    for (int warm = 0; warm < kWarmup; ++warm) {
        cache.create_sequence(SequenceMetadata{1, 1, {}, {}});
        cache.reserve_tokens(1, kPromptTokens);
        cache.fork_sequence(1, SequenceMetadata{2, 1, {}, {}});
        cache.release_sequence(2);
        cache.release_sequence(1);
    }
    for (int i = 0; i < kRepeats; ++i) {
        cache.create_sequence(SequenceMetadata{1, 1, {}, {}});
        cache.reserve_tokens(1, kPromptTokens);
        const auto start = qwenvl_bench::Clock::now();
        cache.fork_sequence(1, SequenceMetadata{2, 1, {}, {}});
        fork_samples.push_back(qwenvl_bench::Nanoseconds(qwenvl_bench::Clock::now() - start).count());
        cache.release_sequence(2);
        cache.release_sequence(1);
    }
    report_row(csv, "fork sequence (8 blocks)", 1, qwenvl_bench::summarize(std::move(fork_samples)), block_kib);
}

void bench_copy_on_write(const ShapeConfig& cfg, std::ostream& csv, std::size_t block_kib) {
    MemoryAllocator allocator(make_allocator_config(cfg));
    KVCacheManager cache(allocator);

    std::vector<double> samples;
    samples.reserve(kRepeats);
    for (int i = 0; i < kWarmup + kRepeats; ++i) {
        cache.create_sequence(SequenceMetadata{1, 1, {}, {}});
        cache.reserve_tokens(1, kTokensPerBlock);
        cache.fork_sequence(1, SequenceMetadata{2, 1, {}, {}});

        const auto start = qwenvl_bench::Clock::now();
        const std::optional<PhysicalBlockId> copy = cache.ensure_token_writable(2, 0);
        const double ns = qwenvl_bench::Nanoseconds(qwenvl_bench::Clock::now() - start).count();
        if (!copy.has_value()) {
            std::cerr << "copy-on-write benchmark failed to materialize a block\n";
            return;
        }
        if (i >= kWarmup) {
            samples.push_back(ns);
        }
        cache.release_sequence(2);
        cache.release_sequence(1);
    }
    report_row(csv, "copy-on-write (1 block)", 1, qwenvl_bench::summarize(std::move(samples)), block_kib);
}

void bench_scheduler_throughput(const ShapeConfig& cfg, std::ostream& csv, std::size_t block_kib) {
    constexpr std::uint64_t kCycles = 2000;
    constexpr std::uint32_t kBatchSize = 8;
    constexpr std::uint32_t kPromptTokens = 4 * kTokensPerBlock;

    MemoryAllocator allocator(make_allocator_config(cfg));
    KVCacheManager cache(allocator);

    SchedulerConfig config;
    config.max_active_requests = kBatchSize;
    config.max_batch_tokens = kBatchSize * kPromptTokens;
    Scheduler scheduler(config, cache, allocator);

    const auto stats = qwenvl_bench::measure_body(
        [&]() {
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
                    std::exit(1);
                }

                for (std::uint32_t i = 0; i < kBatchSize; ++i) {
                    scheduler.complete_step(i + 1, 1);
                }
                for (std::uint32_t i = 0; i < kBatchSize; ++i) {
                    scheduler.cancel(i + 1);
                }
            }
        },
        kWarmup,
        kRepeats);
    report_row(csv, "scheduler admit + retire request", kCycles * kBatchSize, stats, block_kib);
}

void run_shape(const ShapeConfig& cfg, std::ostream& csv) {
    const BlockShape shape = make_shape(cfg);
    const std::size_t block_kib = shape.byte_size() / 1024;
    std::cout << "\nblock: " << shape.tokens_per_block << " tokens x " << shape.num_layers
              << " layers x " << shape.num_kv_heads << " kv heads x " << shape.head_dim << " dim, "
              << block_kib << " KiB/block, pool " << kMaxBlocks << " blocks\n"
              << std::left << std::setw(34) << "benchmark" << std::right << std::setw(10)
              << "ops" << std::setw(8) << "n" << std::setw(12) << "med ns/op" << std::setw(12)
              << "p95 ns/op" << std::setw(12) << "MAD ns/op" << std::setw(14) << "ops/sec" << '\n'
              << std::string(102, '-') << '\n';

    bench_allocate_release(cfg, csv, block_kib);
    bench_fork(cfg, csv, block_kib);
    bench_copy_on_write(cfg, csv, block_kib);
    bench_scheduler_throughput(cfg, csv, block_kib);
}

} // namespace

int main(int argc, char** argv) {
    std::vector<std::uint32_t> layers = {8, 28};
    std::string csv_path = "results/week10-allocator.csv";
    bool layers_set = false;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--layers" && i + 1 < argc) {
            if (!layers_set) {
                layers.clear();
                layers_set = true;
            }
            layers.push_back(static_cast<std::uint32_t>(std::stoul(argv[++i])));
        } else if (arg == "--csv" && i + 1 < argc) {
            csv_path = argv[++i];
        } else {
            std::cerr << "usage: qwenvl_paged_bench [--layers 8|28]... [--csv path]\n";
            return 2;
        }
    }

    qwenvl_bench::pin_to_cpu0();
    qwenvl_bench::warmup_clock();
    const qwenvl_bench::EnvSnapshot before = qwenvl_bench::capture_env();

    std::cout << "QwenVL-Paged allocator benchmark\n"
              << "warmup " << kWarmup << ", repeats " << kRepeats << " (median / p95 / MAD)\n";
    if (!before.model_name.empty()) {
        std::cout << before.model_name << ", governor " << before.governor << ", "
                  << before.freq_khz << " kHz\n";
    }

    std::ofstream csv(csv_path);
    if (!csv) {
        std::cerr << "failed to open " << csv_path << '\n';
        return 1;
    }
    csv << "benchmark,block_kib,ops_per_sample,n,median_ns_per_op,p95_ns_per_op,"
           "mad_ns_per_op,ops_per_sec\n";

    for (std::uint32_t num_layers : layers) {
        run_shape(ShapeConfig{num_layers}, csv);
    }

    const qwenvl_bench::EnvSnapshot after = qwenvl_bench::capture_env();
    std::cout << "\nfreq " << before.freq_khz << " -> " << after.freq_khz << " kHz, thermal "
              << after.thermal_mC << " mC\n";
    if (!qwenvl_bench::freq_stable(before, after, 0.10)) {
        std::cerr << "REJECT: clock drifted more than 10%\n";
        return 1;
    }
    return 0;
}
