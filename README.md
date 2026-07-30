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
  admission control, and preempt/resume (`Scheduler`).
- **Multimodal-aware metadata** (M-RoPE 2D/3D positional spans, image/video
  feature spans) carried through the cache manager without changing allocator
  ownership.

## Architecture

The system separates request scheduling, virtual block mapping, physical memory
ownership, and KV-cache policy so each layer can evolve independently:

| Component          | Responsibility                                                        |
| ------------------ | --------------------------------------------------------------------- |
| `Scheduler`        | Admission control and continuous-batching decisions.                  |
| `KVCacheManager`   | Sequence cache lifecycle: create, reserve, fork, CoW, release.        |
| `BlockTable`       | OS-like page table: logical block -> physical block mapping.          |
| `MemoryAllocator`  | Owns physical blocks, free lists, ref counts, and copy-on-write.      |
| `PhysicalBlock`    | Backend-neutral RAII-owned aligned cache block.                       |

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
  MemoryAllocator.h     Physical block pool + ref counting + CoW primitives
  BlockTable.h          Logical-to-physical page table
  KVCacheManager.h      Sequence cache lifecycle + multimodal metadata
  Scheduler.h           Continuous-batching scheduler
src/                    Implementations of the headers above
tests/                  GoogleTest specification tests (one per module)
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
./build/Block_test
```

## Usage

The library exposes three cooperating objects. A minimal end-to-end setup looks
like this:

```cpp
#include "qwenvl_paged/MemoryAllocator.h"
#include "qwenvl_paged/KVCacheManager.h"
#include "qwenvl_paged/Scheduler.h"

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
    MemoryAllocator allocator(alloc_cfg);

    // 2. The cache manager owns per-sequence block tables over the allocator.
    KVCacheManager cache(allocator);

    // 3. The scheduler drives continuous batching over the cache manager.
    SchedulerConfig sched_cfg;
    sched_cfg.max_active_requests = 8;
    sched_cfg.max_batch_tokens    = 4096;
    Scheduler scheduler(sched_cfg, cache, allocator);

    // 4. Enqueue a request and step the batch loop.
    Request req;
    req.request_id                 = 1;
    req.root_sequence_id           = 1;
    req.prompt_tokens              = 32;
    req.sampling.max_decode_tokens = 64;
    scheduler.enqueue(req);

    BatchPlan plan = scheduler.schedule_next();  // admits req as prefill
    scheduler.complete_step(1, /*produced_tokens*/ 1);  // prefill -> decode
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

## Project Status

This is an early, actively developed prototype. The CPU allocator core,
block-table virtual memory, copy-on-write, cache lifecycle, and scheduler are
implemented and covered by the module tests under `tests/`. GPU/Triton backends,
swap/eviction, and a reference PagedAttention kernel are planned; see the phased
roadmap in [`docs/architecture.md`](docs/architecture.md).

## License

Released under the [MIT License](LICENSE). Copyright (c) 2026 The QwenVL-Paged
Authors.
