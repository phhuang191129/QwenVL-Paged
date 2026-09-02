/**
 * @brief Machine roofs and the reference kernel's place on them.
 *
 * STREAM triad, a working-set sweep, AVX-512 FMA peaks, and timed
 * paged_attention_decode / prefill-as-decode. Writes CSV for plot_roofline.py.
 */

#include "Env.h"
#include "Stats.h"

#include "qwenvl_paged/Block.h"
#include "qwenvl_paged/CacheLayout.h"
#include "qwenvl_paged/KVCacheManager.h"
#include "qwenvl_paged/MemoryAllocator.h"
#include "qwenvl_paged/PagedAttention.h"

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <optional>
#include <string>
#include <vector>

#if defined(__AVX512F__)
#include <immintrin.h>
#endif
#if defined(QWENVL_HAVE_OPENMP)
#include <omp.h>
#endif

namespace {

using qwenvl_paged::AllocatorConfig;
using qwenvl_paged::BlockShape;
using qwenvl_paged::CacheKind;
using qwenvl_paged::CacheView;
using qwenvl_paged::KVBlockLayout;
using qwenvl_paged::KVCacheManager;
using qwenvl_paged::KVStream;
using qwenvl_paged::MemoryAllocator;
using qwenvl_paged::PagedAttentionParams;
using qwenvl_paged::PhysicalBlock;
using qwenvl_paged::PhysicalBlockId;
using qwenvl_paged::SequenceMetadata;
using qwenvl_paged::TokenPosition;

constexpr int kWarmup = 3;
constexpr int kRepeats = 11;

struct Csv {
    std::ofstream out;

    void header() {
        out << "kind,name,bytes,flops,median_ns,p95_ns,mad_ns,gbs,gflops,ai,notes\n";
    }

    void row(
        const std::string& kind,
        const std::string& name,
        double bytes,
        double flops,
        const qwenvl_bench::SampleStats& stats,
        const std::string& notes) {
        const double sec = stats.median_ns * 1e-9;
        const double gbs = sec > 0.0 ? (bytes / sec) / 1e9 : 0.0;
        const double gflops = sec > 0.0 ? (flops / sec) / 1e9 : 0.0;
        const double ai = bytes > 0.0 ? flops / bytes : 0.0;
        out << qwenvl_bench::csv_escape(kind) << ',' << qwenvl_bench::csv_escape(name) << ','
            << bytes << ',' << flops << ',' << stats.median_ns << ',' << stats.p95_ns << ','
            << stats.mad_ns << ',' << gbs << ',' << gflops << ',' << ai << ','
            << qwenvl_bench::csv_escape(notes) << '\n';
        std::cout << std::left << std::setw(28) << name << std::right << std::setw(12)
                  << std::fixed << std::setprecision(1) << stats.median_ns << " ns  "
                  << std::setw(8) << std::setprecision(2) << gbs << " GB/s  " << std::setw(8)
                  << gflops << " GFLOP/s  AI " << std::setprecision(3) << ai << '\n';
    }
};

#if defined(__AVX512F__)
void triad_avx512(float* __restrict a, const float* __restrict b, const float* __restrict c, float s, std::size_t n) {
    const __m512 vs = _mm512_set1_ps(s);
    std::size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        const __m512 vb = _mm512_loadu_ps(b + i);
        const __m512 vc = _mm512_loadu_ps(c + i);
        _mm512_storeu_ps(a + i, _mm512_fmadd_ps(vc, vs, vb));
    }
    for (; i < n; ++i) {
        a[i] = b[i] + s * c[i];
    }
}
#endif

void triad(float* a, const float* b, const float* c, float s, std::size_t n) {
#if defined(__AVX512F__)
    triad_avx512(a, b, c, s, n);
#else
    for (std::size_t i = 0; i < n; ++i) {
        a[i] = b[i] + s * c[i];
    }
#endif
}

void bench_stream(Csv& csv) {
    // 16 Mi elements x 3 arrays x 4 B = 192 MiB, well above the 16 MiB L3.
    constexpr std::size_t n = 16 * 1024 * 1024;
    std::vector<float> a(n, 1.0F);
    std::vector<float> b(n, 2.0F);
    std::vector<float> c(n, 3.0F);
    const float scalar = 1.1F;
    const double bytes = 3.0 * static_cast<double>(n) * sizeof(float);

    const auto single = qwenvl_bench::measure_body(
        [&]() {
            triad(a.data(), b.data(), c.data(), scalar, n);
            asm volatile("" : : "g"(a.data()) : "memory");
        },
        kWarmup,
        kRepeats);
    csv.row("stream", "STREAM triad 1 thread", bytes, 2.0 * static_cast<double>(n), single, "AVX-512 a=b+s*c");

#if defined(QWENVL_HAVE_OPENMP)
    const auto multi = qwenvl_bench::measure_body(
        [&]() {
#pragma omp parallel
            {
                const int threads = omp_get_num_threads();
                const int tid = omp_get_thread_num();
                const std::size_t begin = n * static_cast<std::size_t>(tid) / static_cast<std::size_t>(threads);
                const std::size_t end = n * static_cast<std::size_t>(tid + 1) / static_cast<std::size_t>(threads);
                triad(a.data() + begin, b.data() + begin, c.data() + begin, scalar, end - begin);
            }
            asm volatile("" : : "g"(a.data()) : "memory");
        },
        kWarmup,
        kRepeats);
    csv.row("stream", "STREAM triad OpenMP", bytes, 2.0 * static_cast<double>(n), multi, "all cores");
#endif

    constexpr std::size_t kCopy = 64 * 1024 * 1024;
    std::vector<std::uint8_t> src(kCopy, 1);
    std::vector<std::uint8_t> dst(kCopy, 0);
    const auto copy = qwenvl_bench::measure_body(
        [&]() {
            std::memcpy(dst.data(), src.data(), kCopy);
            asm volatile("" : : "g"(dst.data()) : "memory");
        },
        kWarmup,
        kRepeats);
    csv.row("stream", "memcpy 64 MiB", 2.0 * kCopy, 0.0, copy, "read+write bytes");
}

void bench_cache_sweep(Csv& csv) {
    constexpr std::size_t kMax = 32 * 1024 * 1024; // 32 Mi floats = 128 MiB
    std::vector<float> a(kMax, 1.0F);
    std::vector<float> b(kMax, 2.0F);
    std::vector<float> c(kMax, 3.0F);
    const float scalar = 1.1F;

    const std::size_t sizes[] = {
        1024,           // 4 KiB
        8 * 1024,       // 32 KiB  L1d per core
        64 * 1024,      // 256 KiB
        256 * 1024,     // 1 MiB   L2 per core
        1024 * 1024,    // 4 MiB
        4 * 1024 * 1024,  // 16 MiB  L3
        8 * 1024 * 1024,  // 32 MiB
        32 * 1024 * 1024, // 128 MiB DRAM
    };

    for (std::size_t n : sizes) {
        const double bytes = 3.0 * static_cast<double>(n) * sizeof(float);
        const int reps = n < 256 * 1024 ? 64 : (n < 4 * 1024 * 1024 ? 8 : 1);
        const auto stats = qwenvl_bench::measure_body(
            [&]() {
                for (int r = 0; r < reps; ++r) {
                    triad(a.data(), b.data(), c.data(), scalar, n);
                }
                asm volatile("" : : "g"(a.data()) : "memory");
            },
            kWarmup,
            kRepeats);
        qwenvl_bench::SampleStats per = stats;
        per.median_ns /= reps;
        per.p95_ns /= reps;
        per.mad_ns /= reps;
        const std::string name = "triad " + std::to_string((n * sizeof(float)) / 1024) + " KiB";
        csv.row("cache_sweep", name, bytes, 2.0 * static_cast<double>(n), per, "");
    }
}

#if defined(__AVX512F__)
void bench_fma_fp32(Csv& csv) {
    constexpr int kIters = 1 << 22;
    constexpr int kAcc = 16;
    alignas(64) float sink[16];
    const auto stats = qwenvl_bench::measure_body(
        [&]() {
            __m512 acc[kAcc];
            const __m512 mul = _mm512_set1_ps(1.0001F);
            const __m512 add = _mm512_set1_ps(0.0001F);
            for (int i = 0; i < kAcc; ++i) {
                acc[i] = _mm512_set1_ps(static_cast<float>(i + 1));
            }
            for (int n = 0; n < kIters; ++n) {
                for (int i = 0; i < kAcc; ++i) {
                    acc[i] = _mm512_fmadd_ps(acc[i], mul, add);
                }
            }
            _mm512_store_ps(sink, acc[0]);
            asm volatile("" : : "g"(sink[0]) : "memory");
        },
        kWarmup,
        kRepeats);
    const double flops = 16.0 * 2.0 * kAcc * kIters;
    csv.row("fma", "AVX-512 FMA fp32", 0.0, flops, stats, "16 independent zmm accs, 1 core");
}
#else
void bench_fma_fp32(Csv& csv) {
    csv.row("fma", "AVX-512 FMA fp32", 0.0, 0.0, {}, "not compiled with AVX-512");
}
#endif

#if defined(__AVX512BF16__)
void bench_fma_bf16(Csv& csv) {
    constexpr int kIters = 1 << 22;
    constexpr int kAcc = 16;
    alignas(64) float sink[16];
    const auto stats = qwenvl_bench::measure_body(
        [&]() {
            __m512 acc[kAcc];
            const __m512bh mul = (__m512bh)_mm512_set1_epi32(0x3f803f80);
            const __m512bh ones = mul;
            for (int i = 0; i < kAcc; ++i) {
                acc[i] = _mm512_set1_ps(static_cast<float>(i + 1));
            }
            for (int n = 0; n < kIters; ++n) {
                for (int i = 0; i < kAcc; ++i) {
                    acc[i] = _mm512_dpbf16_ps(acc[i], mul, ones);
                }
            }
            _mm512_store_ps(sink, acc[0]);
            asm volatile("" : : "g"(sink[0]) : "memory");
        },
        kWarmup,
        kRepeats);
    // dest += a0*b0 + a1*b1 per lane: 2 mul + 2 add = 64 FLOP / 512-bit insn.
    const double flops = 16.0 * 4.0 * kAcc * kIters;
    csv.row("fma", "AVX-512 VDPBF16PS", 0.0, flops, stats, "64 FLOP/insn, 1 core");
}
#else
void bench_fma_bf16(Csv& csv) {
    csv.row("fma", "AVX-512 VDPBF16PS", 0.0, 0.0, {}, "not compiled with AVX512-BF16");
}
#endif

BlockShape make_2b_shape() {
    BlockShape shape;
    shape.tokens_per_block = 16;
    shape.num_layers = 28;
    shape.num_kv_heads = 8;
    shape.head_dim = 128;
    shape.bytes_per_element = 2;
    return shape;
}

bool fill_sequence(
    KVCacheManager& cache,
    MemoryAllocator& allocator,
    KVBlockLayout layout,
    std::uint64_t sequence_id,
    std::uint32_t context_len,
    std::uint32_t fill_layers = 0) {
    if (!cache.create_sequence(SequenceMetadata{sequence_id, sequence_id, {}, {}})) {
        return false;
    }
    if (!cache.reserve_tokens(sequence_id, context_len)) {
        return false;
    }
    const BlockShape& shape = layout.shape;
    const bool per_layer = shape.per_layer_frames();
    const std::uint32_t layers_to_write =
        fill_layers == 0 || fill_layers > shape.num_layers ? shape.num_layers : fill_layers;
    for (std::uint32_t token = 0; token < context_len; ++token) {
        const std::uint32_t ensure_layers = per_layer ? layers_to_write : 1;
        for (std::uint32_t ensure_layer = 0; ensure_layer < ensure_layers; ++ensure_layer) {
            const std::optional<PhysicalBlockId> physical = cache.ensure_token_writable(
                sequence_id, token, CacheKind::TextKV, ensure_layer);
            if (!physical.has_value()) {
                return false;
            }
            PhysicalBlock* block = allocator.block(*physical);
            if (block == nullptr) {
                return false;
            }
            auto* base = reinterpret_cast<std::uint16_t*>(block->data());
            const std::uint32_t write_from = per_layer ? ensure_layer : 0;
            const std::uint32_t write_to = per_layer ? ensure_layer + 1 : layers_to_write;
            for (std::uint32_t layer = write_from; layer < write_to; ++layer) {
                const std::uint32_t offset_layer = per_layer ? 0 : layer;
                for (KVStream stream : {KVStream::Key, KVStream::Value}) {
                    for (std::uint32_t head = 0; head < shape.num_kv_heads; ++head) {
                        const auto offset = layout.element_offset(
                            offset_layer, stream, token % shape.tokens_per_block, head);
                        if (!offset.has_value()) {
                            return false;
                        }
                        for (std::uint32_t d = 0; d < shape.head_dim; ++d) {
                            base[*offset + d] = static_cast<std::uint16_t>(token + head + d);
                        }
                    }
                }
            }
        }
    }
    return true;
}

void bench_attention(Csv& csv) {
    const BlockShape shape = make_2b_shape();
    const std::uint32_t contexts[] = {128, 512, 1280};
    const std::uint32_t q_heads = 16;
    const std::uint32_t needed_blocks = 1280 / shape.tokens_per_block + 4;

    AllocatorConfig config;
    config.block_shape = shape;
    config.max_blocks = needed_blocks;
    MemoryAllocator allocator(config);
    KVCacheManager cache(allocator);
    KVBlockLayout layout;
    layout.shape = shape;

    if (!fill_sequence(cache, allocator, layout, 1, 1280)) {
        std::cerr << "failed to populate attention cache\n";
        return;
    }
    const std::optional<CacheView> view = cache.cache_view(1);
    if (!view.has_value() || !view->valid()) {
        std::cerr << "failed to open cache view\n";
        return;
    }

    std::vector<std::uint16_t> query(q_heads * shape.head_dim, 1);
    std::vector<std::uint16_t> out(q_heads * shape.head_dim, 0);

    for (std::uint32_t context : contexts) {
        PagedAttentionParams params;
        params.layer = 0;
        params.num_query_heads = q_heads;
        params.context_len = context;
        params.scale = 1.0F / std::sqrt(static_cast<float>(shape.head_dim));

        const double bytes_unfused = static_cast<double>(q_heads) * 2.0 * context * shape.head_dim *
            shape.bytes_per_element;
        const double bytes_fused = static_cast<double>(shape.num_kv_heads) * 2.0 * context *
            shape.head_dim * shape.bytes_per_element;
        const double flops = static_cast<double>(q_heads) * 4.0 * shape.head_dim * context;

        const auto time_kernel = [&](auto kernel, std::string name, double bytes, const char* notes) {
            const auto stats = qwenvl_bench::measure_body(
                [&]() {
                    if (!kernel(*view, query.data(), params, out.data())) {
                        std::cerr << name << " failed\n";
                        std::exit(1);
                    }
                },
                kWarmup,
                kRepeats);
            csv.row("attention_decode", name, bytes, flops, stats, notes);
        };

        time_kernel(
            qwenvl_paged::paged_attention_decode<std::uint16_t>,
            "decode ctx " + std::to_string(context),
            bytes_unfused,
            "reference, unfused");
        time_kernel(
            qwenvl_paged::paged_attention_decode_blocked<std::uint16_t>,
            "blocked ctx " + std::to_string(context),
            bytes_unfused,
            "block-at-a-time");
        time_kernel(
            qwenvl_paged::paged_attention_decode_fused<std::uint16_t>,
            "fused ctx " + std::to_string(context),
            bytes_fused,
            "GQA fusion, two-pass");
        time_kernel(
            qwenvl_paged::paged_attention_decode_fast<std::uint16_t>,
            "fast ctx " + std::to_string(context),
            bytes_fused,
            "fusion + online softmax + AVX-512");
    }

    // Prefill is this kernel at every prefix length. S=256 is long enough to
    // show the quadratic cost without a multi-second sample.
    constexpr std::uint32_t kPrefill = 256;
    PagedAttentionParams params;
    params.layer = 0;
    params.num_query_heads = q_heads;
    params.scale = 1.0F / std::sqrt(static_cast<float>(shape.head_dim));
    const auto prefill_stats = qwenvl_bench::measure_body(
        [&]() {
            for (std::uint32_t p = 0; p < kPrefill; ++p) {
                params.context_len = p + 1;
                if (!qwenvl_paged::paged_attention_decode<std::uint16_t>(*view, query.data(), params, out.data())) {
                    std::cerr << "prefill decode failed\n";
                    std::exit(1);
                }
            }
        },
        1,
        5);

    const double prefill_tokens = static_cast<double>(kPrefill) * (kPrefill + 1) / 2.0;
    const double prefill_bytes =
        static_cast<double>(q_heads) * 2.0 * prefill_tokens * shape.head_dim * shape.bytes_per_element;
    const double prefill_flops =
        static_cast<double>(q_heads) * 4.0 * shape.head_dim * prefill_tokens;
    csv.row(
        "attention_prefill",
        "prefill S=256 as decode",
        prefill_bytes,
        prefill_flops,
        prefill_stats,
        "S independent decode calls; GEMM AI is higher");

    BlockShape packed = shape;
    packed.num_layers = 1;
    AllocatorConfig packed_config;
    packed_config.block_shape = packed;
    packed_config.max_blocks = needed_blocks;
    MemoryAllocator packed_allocator(packed_config);
    KVCacheManager packed_cache(packed_allocator);
    KVBlockLayout packed_layout;
    packed_layout.shape = packed;
    if (!fill_sequence(packed_cache, packed_allocator, packed_layout, 1, 1280)) {
        std::cerr << "failed to populate packed attention cache\n";
        return;
    }
    const std::optional<CacheView> packed_view = packed_cache.cache_view(1);
    if (!packed_view.has_value()) {
        return;
    }
    PagedAttentionParams packed_params;
    packed_params.layer = 0;
    packed_params.num_query_heads = q_heads;
    packed_params.context_len = 1280;
    packed_params.scale = 1.0F / std::sqrt(static_cast<float>(shape.head_dim));
    const auto packed_stats = qwenvl_bench::measure_body(
        [&]() {
            if (!qwenvl_paged::paged_attention_decode_fast<std::uint16_t>(
                    *packed_view, query.data(), packed_params, out.data())) {
                std::cerr << "packed fast decode failed\n";
                std::exit(1);
            }
        },
        kWarmup,
        kRepeats);
    const double packed_bytes = static_cast<double>(shape.num_kv_heads) * 2.0 * 1280 *
        shape.head_dim * shape.bytes_per_element;
    const double packed_flops = static_cast<double>(q_heads) * 4.0 * shape.head_dim * 1280;
    csv.row(
        "attention_decode",
        "fast ctx 1280 packed",
        packed_bytes,
        packed_flops,
        packed_stats,
        "1-layer frames; paging tax vs fast ctx 1280");

    // Same 28-layer model, one layer per frame, allocated layer-major so
    // layer 0's 80 frames sit next to each other instead of 1.75 MiB apart.
    BlockShape layer_major = shape;
    layer_major.layers_per_frame = 1;
    AllocatorConfig layer_config;
    layer_config.block_shape = layer_major;
    layer_config.max_blocks = needed_blocks * shape.num_layers;
    MemoryAllocator layer_allocator(layer_config);
    KVCacheManager layer_cache(layer_allocator);
    KVBlockLayout layer_layout;
    layer_layout.shape = layer_major;
    if (!fill_sequence(layer_cache, layer_allocator, layer_layout, 1, 1280)) {
        std::cerr << "failed to populate layer-major attention cache\n";
        return;
    }
    const std::optional<CacheView> layer_view = layer_cache.cache_view(1);
    if (!layer_view.has_value()) {
        return;
    }
    PagedAttentionParams layer_params = packed_params;
    const auto layer_stats = qwenvl_bench::measure_body(
        [&]() {
            if (!qwenvl_paged::paged_attention_decode_fast<std::uint16_t>(
                    *layer_view, query.data(), layer_params, out.data())) {
                std::cerr << "layer-major fast decode failed\n";
                std::exit(1);
            }
        },
        kWarmup,
        kRepeats);
    csv.row(
        "attention_decode",
        "fast ctx 1280 layer-major",
        packed_bytes,
        packed_flops,
        layer_stats,
        "28 layers, 1 layer/frame, layer-major reserve");

    // Same 140 MiB pool, but only layer 0 is written, so a miss here is the
    // layout rather than the other 27 layers evicting L3 during fill.
    MemoryAllocator layer0_allocator(layer_config);
    KVCacheManager layer0_cache(layer0_allocator);
    if (!fill_sequence(layer0_cache, layer0_allocator, layer_layout, 1, 1280, 1)) {
        std::cerr << "failed to populate layer-major L0-only cache\n";
        return;
    }
    const std::optional<CacheView> layer0_view = layer0_cache.cache_view(1);
    if (!layer0_view.has_value()) {
        return;
    }
    const auto layer0_stats = qwenvl_bench::measure_body(
        [&]() {
            if (!qwenvl_paged::paged_attention_decode_fast<std::uint16_t>(
                    *layer0_view, query.data(), layer_params, out.data())) {
                std::cerr << "layer-major L0-only fast decode failed\n";
                std::exit(1);
            }
        },
        kWarmup,
        kRepeats);
    csv.row(
        "attention_decode",
        "fast ctx 1280 layer-major L0-only",
        packed_bytes,
        packed_flops,
        layer0_stats,
        "28-layer pool, only layer 0 filled");
}

} // namespace

int main(int argc, char** argv) {
    std::string csv_path = "results/week10-roofs.csv";
    bool attention_only = false;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--csv" && i + 1 < argc) {
            csv_path = argv[++i];
        } else if (arg == "--attention") {
            attention_only = true;
        } else {
            std::cerr << "usage: qwenvl_roofline [--csv path] [--attention]\n";
            return 2;
        }
    }

    qwenvl_bench::pin_to_cpu0();
    qwenvl_bench::warmup_clock();
    const qwenvl_bench::EnvSnapshot before = qwenvl_bench::capture_env();
    std::cout << "QwenVL-Paged week 10 roofs\n";
    if (!before.model_name.empty()) {
        std::cout << before.model_name << ", governor " << before.governor << ", "
                  << before.freq_khz << " kHz\n";
    }

    Csv csv;
    csv.out.open(csv_path);
    if (!csv.out) {
        std::cerr << "failed to open " << csv_path << '\n';
        return 1;
    }
    csv.header();

    if (!attention_only) {
        bench_fma_fp32(csv);
        bench_fma_bf16(csv);
        bench_cache_sweep(csv);
    }
    bench_attention(csv);
    const qwenvl_bench::EnvSnapshot mid = qwenvl_bench::capture_env();
    if (!qwenvl_bench::freq_stable(before, mid, 0.10)) {
        std::cerr << "REJECT: clock dropped more than 10% during pinned roofs\n";
        return 1;
    }

    // Unpin so OpenMP STREAM can use every physical core. cpu0 dropping
    // afterward is idle, not a throttle.
#ifdef __linux__
    {
        cpu_set_t set;
        CPU_ZERO(&set);
        for (int cpu = 0; cpu < 16; ++cpu) {
            CPU_SET(cpu, &set);
        }
        sched_setaffinity(0, sizeof(set), &set);
    }
#endif
    bench_stream(csv);

    std::cout << "freq " << before.freq_khz << " -> " << mid.freq_khz
              << " kHz (pinned roofs), thermal " << mid.thermal_mC << " mC\n";
    csv.out << "env,freq_khz," << mid.freq_khz << ",0," << before.freq_khz << ',' << mid.freq_khz
            << ",0,0,0,0,"
            << qwenvl_bench::csv_escape(before.governor + " " + before.model_name) << '\n';
    return 0;
}
