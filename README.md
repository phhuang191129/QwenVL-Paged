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
docs/architecture.md    Design document and phased roadmap
CMakeLists.txt          Build and test configuration
```

## Requirements

- A C++17 compiler (GCC, Clang, or MSVC).
- [CMake](https://cmake.org/) >= 3.16.
- Network access on the **first** configure: GoogleTest (v1.15.2) is fetched
  automatically via CMake `FetchContent`.

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
are implemented and covered by the module tests under `tests/`. GPU/Triton
backends are still planned; the integration points and the synchronization
rules they must honor are documented in
[`docs/architecture.md`](docs/architecture.md), along with the phased roadmap.

## License

Released under the [MIT License](LICENSE). Copyright (c) 2026 The QwenVL-Paged
Authors.
