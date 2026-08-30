# QwenVL-Paged

A production-oriented, CPU-first **PagedAttention memory subsystem** for Qwen-VL
style inference, written in modern C++17. The project models KV-cache memory the
same way an operating system models virtual memory: logical token blocks are
mapped through per-sequence page tables onto a pool of physical cache blocks,
with reference counting, copy-on-write sharing for parallel sampling, and a
continuous-batching scheduler on top.

The allocator core is intentionally backend-neutral. Today it manages aligned
host memory; the same interfaces are designed to later back onto pinned host
memory, CUDA allocations, or Triton buffers without changing the scheduler,
cache manager, or block-table semantics.

## Features

- **Physical block pool** with fixed capacity, aligned host allocation, a free
  list, deterministic recycling, and reference counting (`MemoryAllocator`).
- **Virtual block tables** mapping logical token blocks to physical blocks per
  sequence, with fork support for parallel sampling (`BlockTable`).
- **Copy-on-write**: forked sequences share prompt blocks until one branch
  writes, at which point a private block is materialized (`KVCacheManager`).
- **Continuous-batching scheduler** with pending/active/preempted queues,
  admission control, and preempt/resume (`Scheduler`). Set
  `preemption_watermark_blocks` and the scheduler reclaims cache on its own,
  preempting the newest active request until a waiting prompt plus that reserve
  fits; leave it at zero to keep preemption caller-driven.
- **Swap and eviction**: a preempted sequence copies its blocks into a
  backend-neutral swap space and hands the frames back to the pool, then
  restores them on resume (`SwapBackend`, `HostSwapBackend`). The allocator also
  exposes an advisory eviction-candidate hook for custom policies.
- **Multimodal-aware metadata** (M-RoPE 2D/3D positional spans, image/video
  feature spans) carried through the cache manager without changing allocator
  ownership.
- **Reference PagedAttention**: a correctness-only CPU kernel that walks the
  page table token by token, so scattered blocks, grouped-query heads, and
  copy-on-write branches have a defined expected result for a future CUDA or
  Triton kernel to reproduce (`CacheLayout`, `CacheView`, `PagedAttention`).
- **Triton GPU decode kernel**: a device backend that consumes only
  `KVBlockLayout`'s four strides and the flattened block table, validated against
  the same golden vectors as the CPU reference and benchmarked at the real
  Qwen3-VL-2B block shape (`tools/triton_kernel/`).
- **Transformers cache backend**: a `CacheLayerMixin` that stores a Hugging Face
  model's KV in allocator blocks, gated by greedy decodes that must be
  token-identical to `DynamicCache` — on a tiny config and on the real
  Qwen3-VL-2B with an image (`python/paged_cache.py`).

## Architecture

The system separates request scheduling, virtual block mapping, physical memory
ownership, and KV-cache policy so each layer can evolve independently:

| Component          | Responsibility                                                        |
| ------------------ | --------------------------------------------------------------------- |
| `Scheduler`        | Admission control and continuous-batching decisions.                  |
| `KVCacheManager`   | Sequence cache lifecycle: create, reserve, fork, CoW, release.        |
| `BlockTable`       | OS-like page table: logical block -> physical block mapping.          |
| `MemoryAllocator`  | Owns physical blocks, free lists, ref counts, CoW, and swap.          |
| `SwapBackend`      | Holds evicted block bytes while their frame is reused.                |
| `PhysicalBlock`    | Backend-neutral RAII-owned aligned cache block.                       |
| `KVBlockLayout`    | Element ordering inside a block and bounds-checked slot offsets.      |
| `CacheView`        | Read-only kernel contract: page table + storage + layout.             |

For the full design, boundaries, concurrency contract, and phased plan, see
[`docs/architecture.md`](docs/architecture.md).

> **Concurrency:** the phase-1 core is single-threaded by contract. All mutation
> must happen on one engine event loop; reference counts and free-list updates
> are deliberately non-atomic. See the architecture doc before sharing these
> types across threads.

## Project Layout

```
include/qwenvl_paged/   Public headers (the API contract)
  Block.h               Core types, BlockShape, LogicalBlock, PhysicalBlock
  CacheLayout.h         Element ordering and slot offsets inside a block
  SwapBackend.h         Backend-neutral store for evicted block contents
  MemoryAllocator.h     Physical block pool + ref counting + CoW + swap
  BlockTable.h          Logical-to-physical page table
  KVCacheManager.h      Sequence cache lifecycle, multimodal metadata, CacheView
  PagedAttention.h      Reference CPU attention over a paged cache
  Scheduler.h           Continuous-batching scheduler
src/                    Implementations of the headers above
tests/                  GoogleTest specification tests (one per module)
bench/                  std::chrono latency harness for the allocator core
tools/trace_gen/        Qwen3-VL trace generator (Python, processor geometry only)
tools/trace_replay/     Trace replay driver + contiguous-reservation baseline
tools/golden_export/    Golden fixture exporter and its pre-GPU verification gate
tools/triton_kernel/    Triton decode kernel, fixture runner, and GPU benchmark
python/                 pybind11 bindings, the transformers cache, and their checks
traces/                 Generated request traces (JSONL)
fixtures/               Golden attention vectors (.npz) shared by both backends
results/                Measured CSV output backing docs/performance.md
docs/architecture.md    Design document and phased roadmap
docs/roadmap-phase2.md  Weeks 9-16: Qwen3-VL integration, profiling, optimization
docs/performance.md     Measured results, with assumptions and limitations
CMakeLists.txt          Build and test configuration
```

## Requirements

- A C++17 compiler (GCC, Clang, or MSVC).
- [CMake](https://cmake.org/) >= 3.16.
- Network access on the **first** configure: GoogleTest (v1.15.2) and
  nlohmann/json (v3.11.3) are fetched automatically via CMake `FetchContent`.
- For regenerating traces only: Python 3.12 and `transformers==4.57.1`. No model
  weights and no torch are needed. The committed traces under `traces/` mean this
  is not required to build, test, or reproduce the measured results.
- For the GPU kernel only: an NVIDIA GPU plus a Python 3.12 environment with
  torch and Triton, which `tools/triton_kernel/setup_gpu_box.sh` builds. Nothing
  in the C++ build or test suite depends on it.
- For the paged transformers cache only: `transformers==5.16.1`, pinned because
  the cache extension point is version-specific. The Qwen3-VL-2B gate also needs
  torchvision, pillow, accelerate, and about 5 GB for the checkpoint; the tiny
  gate needs none of those.

## Building

```bash
# Configure (downloads GoogleTest on first run) and build everything.
cmake -S . -B build
cmake --build build
```

This produces the `qwenvl_paged_core` static library plus one test executable
per module.

> **Using a system GoogleTest instead of FetchContent?** If your environment
> already provides GoogleTest, you can compile a module directly, e.g.:
>
> ```bash
> g++ -std=c++17 -Iinclude tests/Scheduler.test.cpp src/*.cpp \
>     -lgtest -lgtest_main -lpthread -o scheduler_test
> ```

## Running the Tests

```bash
# Run the full suite through CTest.
ctest --test-dir build --output-on-failure
```

Or run an individual module binary directly:

```bash
./build/Scheduler_test
./build/KVCacheManager_test
./build/BlockTable_test
./build/memory_allocator_test
./build/SwapBackend_test
./build/Block_test
./build/CacheView_test
./build/PagedAttention_test
./build/EndToEnd_test
./build/TraceReplay_test
```

## Benchmarks

`bench/` holds a dependency-free `std::chrono` harness reporting allocator
latency, fork cost, copy-on-write cost, and scheduler throughput. It reports
timings rather than asserting on them, so it is not registered with CTest.

Build with optimizations on, or the numbers are not comparable — CMake does not
set a build type by default:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
./build/qwenvl_paged_bench
```

### Baseline Results

Recorded before any Qwen3-VL inference work, to compare against after
optimization. Setup and teardown are excluded from every measurement, so each
row covers only the named operation.

- **Machine:** AMD Ryzen 7 8845H (8 cores / 16 threads)
- **Compiler:** GCC 15.2.1, `-O2 -std=c++17`, single-threaded
- **Block:** 16 tokens x 8 layers x 8 kv heads x 128 dim, fp16 -> 512 KiB/block
- **Pool:** 128 blocks (64 MiB)
- **Figures:** median of 3 runs

| Operation                           |   ns/op | ops/sec | Notes                                         |
| ----------------------------------- | ------: | ------: | --------------------------------------------- |
| `allocate` + `release`              |     6.4 |   157 M | Free-list pop/push, no memory touched          |
| `fork_sequence` (8 shared blocks)   |    57.5 |    17 M | Table copy plus 8 `retain` calls, ~7 ns/block  |
| First write to a shared block (CoW) |   7,780 |   128 K | Dominated by the 512 KiB `memcpy`              |
| Scheduler admit + retire a request  |   161.8 |   6.2 M | Amortized enqueue, schedule, step, and cancel  |

Reading these:

- **Allocation is free relative to everything else.** At 6.4 ns it is a
  free-list pop; block memory is allocated once up front by the pool.
- **Fork is cheap enough that parallel sampling is not allocator-bound.**
  ~7 ns per shared block is a refcount increment plus a vector copy.
- **Copy-on-write is pure memory bandwidth**, roughly 68 GB/s for a 512 KiB
  block. It scales linearly with block size, so it is the one number that moves
  if the block shape changes. Reducing *how often* CoW fires matters more than
  making the copy faster.
- **Scheduler overhead is ~162 ns per request**, which is noise next to any real
  model forward pass. Cache pressure, not scheduling cost, is what limits batch
  size.

These cover the allocator core only. They do not include attention compute: the
CPU `paged_attention_decode` path is a correctness reference with no blocking or
vectorization, so its runtime is not a meaningful optimization baseline.

## Workload Replay

`tools/` drives the scheduler and cache manager with real Qwen3-VL request
geometry and compares the result against a contiguous worst-case-reservation
allocator over an identically sized pool.

Image token counts are authoritative. The generator loads no model weights and
does not need torch: it calls `transformers`' own `smart_resize` with the patch
and merge sizes from the model's published `preprocessor_config.json`, because an
image's token count depends only on its source dimensions. Those dimensions come
from COCO `test2017` metadata or a documented mix of real device capture
resolutions. Text token counts and decode budgets are synthetic and labelled as
such in every trace header.

```bash
# Replay the committed traces. Pool defaults to a size derived per trace.
./build/qwenvl_trace_replay traces/bimodal.jsonl

# Shrink the pool and give preemption somewhere to evict to.
./build/qwenvl_trace_replay --pool-blocks 256 --swap-slots 256 traces/image-heavy.jsonl

# Reproduce every number in docs/performance.md.
./tools/trace_replay/run_experiments.sh

# Regenerate the traces themselves (needs network on first run).
./tools/trace_gen/generate_all.sh
```

Headline results, with full context and caveats in
[`docs/performance.md`](docs/performance.md):

- Cache utilization of **0.98-0.99** against **0.62-0.88** for the contiguous
  baseline, with internal waste bounded at 15 of 16 token slots per sequence.
- **Zero** admissions lost to external fragmentation, against up to 5,559 for the
  baseline at the same pool size.
- Effective utilization of **3.39** under 4-way parallel sampling, where branches
  share prompt blocks until they diverge.
- The shipped `preprocessor_config.json` permits **16,384 visual tokens for a
  single image**, which is 1.75 GiB of KV cache on Qwen3-VL-2B; the documented
  256-1280 token clamp brings the same image down to 140 MiB.
- Prompt-only admission over-commits the pool, so under pressure the paged engine
  needs preemption to finish a trace that the conservative baseline always
  finishes. Worst-case reservation is inefficient, but it is also a safety
  property.

## GPU Kernel

`tools/triton_kernel/` holds a Triton PagedAttention decode kernel. It consumes
exactly what [`docs/architecture.md`](docs/architecture.md) promises an execution
backend — `KVBlockLayout`'s four strides and the flattened
`[num_seqs, max_blocks_per_seq]` block table — and nothing else, so that claim is
tested rather than asserted.

```bash
# Build a pinned Python 3.12 venv, install torch/Triton, run the fixture gate.
bash tools/triton_kernel/setup_gpu_box.sh

# Validate against the same golden vectors the CPU reference produces.
.venv/bin/python tools/triton_kernel/run_fixture.py fixtures

# Block addressing past the int32 boundary, which the fixtures are too small to reach.
.venv/bin/python tools/triton_kernel/check_large_pool.py

# Benchmark at the real block shape; writes results/week15-gpu-kernel.csv.
.venv/bin/python tools/triton_kernel/benchmark.py

# Split-context kernel against that baseline; writes results/week16-context-partition.csv.
cd tools/triton_kernel && ../../.venv/bin/python benchmark_partitioned.py \
    ../../results/week16-context-partition.csv
```

The fixtures themselves come from `./build/qwenvl_golden_export fixtures`, packed
by `tools/golden_export/pack_npz.py` and checked by
`tools/golden_export/verify_fixtures.py`, which rebuilds attention from the
strides and the block table alone. Four of them fill a clean pool once;
`cow_and_recycled_frames` instead drives the allocator through a fork, a
copy-on-write of every one of the child's blocks, and a sequence built on frames
recycled from a released one, so the kernel is validated against a pool that
paging has actually churned rather than a static snapshot.

Measured on an NVIDIA L4, with full context and caveats in
[`docs/performance.md`](docs/performance.md):

- **The paging tax is below the measurement noise floor.** The same kernel over
  consecutive versus randomly permuted frames differs by 0.01% at batch 32, while
  a single layout varies 2.3% between its own fastest and slowest run.
- **95% of peak DRAM bandwidth** at batch 32 with a 1,280-token image context,
  measured with Nsight Compute. There is no bandwidth headroom left there.
- **8-9% of peak at batch 1**, where 16 programs cannot fill 58 SMs. Profiling
  shows DRAM utilization tracks achieved occupancy at a fixed ratio, so the lever
  is partitioning the context across programs, not pipelining the inner loop.
- **4.57x at batch 1** from doing exactly that: splitting each sequence's context
  across programs and merging the partial softmaxes lifts occupancy from 8% to
  58% and DRAM from 11% to 67%. The win decays with batch size and is gone by 16,
  where the baseline already saturates memory, so the wrapper falls back to it.
- **1.80x at batch 512** from renumbering the launch grid. The two query heads
  sharing a KV head were `num_seqs` apart in launch order; past the point where
  that exceeds what stays resident, the duplicated load stops hitting L2 and DRAM
  traffic doubles. Making the query head the fast axis is a three-line fix, and it
  buys what fusing the grouped-query heads would have, without halving the grid.

The block pool is a single contiguous slab with frame `i` at
`i * block_stride_bytes()`, so a `PhysicalBlockId` is an offset rather than a
handle to an independent allocation. That is what the kernel's
`physical_id * elements_per_block` addressing needs, and it lets the whole pool
mirror to the device in one transfer instead of a per-frame gather.

`python/live_pool_check.py` runs the kernel against a pool the C++ allocator is
managing, with no fixture in the loop: Python owns the slab, `MemoryAllocator`
adopts it, and the cache manager forks a sequence, materializes the child by
copy-on-write, and hands a released sequence's frames to another before the
kernel reads the result. It agrees with the CPU reference to 5.4e-07.

```bash
cmake -S . -B build -DQWENVL_BUILD_PYTHON=ON \
    -Dpybind11_DIR=$(.venv/bin/python -c 'import pybind11;print(pybind11.get_cmake_dir())') \
    -DPython_EXECUTABLE=$PWD/.venv/bin/python
cmake --build build -j
PYTHONPATH=build .venv/bin/python python/live_pool_check.py
```

The pool can now be device-resident. `set_copy_hook` replaces the host memcpy
behind copy-on-write, and the allocator adopts a CUDA tensor by address. Swap
and the CPU reference kernel still dereference host pointers and are unused on
that path. The Triton kernel is still not in the model forward pass: torch
attention gathers the context out of the blocks, now on-device rather than
across PCIe. See `docs/performance.md` week 19.

## Paged Cache For Transformers

`python/paged_cache.py` backs a Hugging Face model's KV cache with the allocator.
In transformers 5.x the extension point is a `CacheLayerMixin` per decoder layer
rather than a `Cache` subclass, which suits the block layout: a physical block
already spans every layer, so one block table backs all of them and each
`PagedLayer` is a window onto its own layer's stride.

Two gates decode the same prompt twice, once through `DynamicCache` and once
through paged blocks, and require the same tokens. Frames are deliberately
scattered by a spacer sequence, so a gather that ignored the block table fails
both. The first runs a small random-weight config in about four seconds and also
requires zero logit drift; the second runs the real Qwen3-VL-2B-Instruct in
bfloat16 with a 1,242-token prompt of which 1,225 are image tokens.

```bash
PYTHONPATH=build .venv/bin/python python/token_identical_check.py
PYTHONPATH=build .venv/bin/python python/qwen3vl_2b_check.py
```

The 2B gate runs a host pool, a device gather, and a device kernel. All three
are token-identical to `DynamicCache` over 20 greedy steps. Prefill still
gathers; decode writes the new token into its page and launches Triton. At this
prompt the host path costs 51 ms/step over dense and the two device paths both
cost ~11 ms/step — putting the kernel in the path did not move the number, so
that 11 ms is the Python write and dispatch, not the gather. See
`docs/performance.md` week 20.

`python/fork_sharing_check.py` covers the property paging exists for, now on the
device pool so copy-on-write goes through the hook rather than a host memcpy.
Four continuations share all 78 prompt blocks and privately own two each, so
they hold 150 MiB where four dense caches hold 552 MiB.

```bash
PYTHONPATH=build .venv/bin/python python/fork_sharing_check.py
```

## Usage

The library exposes three cooperating objects. A minimal end-to-end setup looks
like this:

```cpp
#include "qwenvl_paged/MemoryAllocator.h"
#include "qwenvl_paged/KVCacheManager.h"
#include "qwenvl_paged/Scheduler.h"
#include "qwenvl_paged/SwapBackend.h"

using namespace qwenvl_paged;

int main() {
    // 1. Describe a physical KV cache block and size the pool.
    AllocatorConfig alloc_cfg;
    alloc_cfg.block_shape = BlockShape{
        /*tokens_per_block*/ 16,
        /*num_layers*/       32,
        /*num_kv_heads*/     8,
        /*head_dim*/         128,
        /*bytes_per_element*/ 2};   // e.g. fp16/bf16
    alloc_cfg.max_blocks = 1024;

    // 2. Optional swap space, declared before the allocator that points at it.
    //    Without one, preemption parks a request but reclaims no blocks.
    HostSwapBackend swap(/*max_slots*/ 1024);

    MemoryAllocator allocator(alloc_cfg);
    allocator.set_swap_backend(&swap);

    // 3. The cache manager owns per-sequence block tables over the allocator.
    KVCacheManager cache(allocator);

    // 4. The scheduler drives continuous batching over the cache manager.
    SchedulerConfig sched_cfg;
    sched_cfg.max_active_requests = 8;
    sched_cfg.max_batch_tokens    = 4096;
    // Keep 16 blocks free after each admission, preempting to get there.
    // Zero (the default) disables automatic preemption.
    sched_cfg.preemption_watermark_blocks = 16;
    Scheduler scheduler(sched_cfg, cache, allocator);

    // 5. Enqueue a request and step the batch loop.
    Request req;
    req.request_id                 = 1;
    req.root_sequence_id           = 1;
    req.prompt_tokens              = 32;
    req.sampling.max_decode_tokens = 64;
    scheduler.enqueue(req);

    BatchPlan plan = scheduler.schedule_next();  // admits req as prefill
    scheduler.complete_step(1, /*produced_tokens*/ 1);  // prefill -> decode
    // Under cache pressure, scheduler.preempt(1, "cache pressure") swaps this
    // request's cache out and scheduler.resume(1) brings it back.
    // ... continue stepping, then scheduler.cancel(1) when finished.
}
```

Lower-level building blocks can also be used on their own, for example forking a
sequence for parallel sampling and materializing a private block on write:

```cpp
cache.create_sequence(SequenceMetadata{/*sequence_id*/ 1, /*request_id*/ 1});
cache.reserve_tokens(1, 16);              // reserve one logical block
cache.fork_sequence(1, SequenceMetadata{/*sequence_id*/ 2, /*request_id*/ 1});
// Sequences 1 and 2 now share the prompt block. Writing on branch 2 copies it:
auto physical = cache.ensure_token_writable(2, /*token_position*/ 0);
```

An execution backend reads the cache through a `CacheView`, which carries the
page table, the storage, and the element layout. `context_len` comes from the
caller because the cache manager owns reserved capacity, not committed length:

```cpp
CacheView view = *cache.cache_view(/*sequence_id*/ 1);

PagedAttentionParams params;
params.layer            = 0;
params.num_query_heads  = 8;    // grouped-query: 8 query heads over 2 kv heads
params.context_len      = 32;
params.scale            = 1.0F / std::sqrt(static_cast<float>(head_dim));

// Causal prefill is this same call per prompt position with context_len = p + 1.
bool ok = paged_attention_decode<float>(view, query.data(), params, out.data());
```

## Project Status

This is an early, actively developed prototype. The CPU allocator core,
block-table virtual memory, copy-on-write, cache lifecycle, scheduler,
swap/eviction interfaces, and a correctness-only reference PagedAttention path
are implemented and covered by the module tests under `tests/`. The memory
subsystem has been measured against real Qwen3-VL workload geometry, and a Triton
decode kernel runs on GPU, validated against the CPU golden vectors and
benchmarked at the real block shape
([`docs/performance.md`](docs/performance.md)).

Qwen3-VL-2B-Instruct now runs on the paged cache and is token-identical to
transformers' `DynamicCache`, image prompt included, on a host pool, a device
gather, and a Triton decode in the forward pass. Four sampling branches of one
prompt hold 3.67x less KV than four dense caches. Decode through the kernel is
still ~11 ms/step over dense — the same as the device gather — because the
remaining cost is Python dispatch, not attention. Four of the six
synchronization rules in
[`docs/architecture.md`](docs/architecture.md) are exercised by
`tests/SynchronizationRules.test.cpp`; the remaining two govern an asynchronous
backend that does not exist yet and are named as untested rather than faked. The
path from here is in [`docs/roadmap-phase2.md`](docs/roadmap-phase2.md).

## License

Released under the [MIT License](LICENSE). Copyright (c) 2026 The QwenVL-Paged
Authors.
