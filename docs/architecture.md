# QwenVL-Paged Memory Management Architecture

This document defines the first CPU-only architecture target for a production-grade
PagedAttention memory subsystem. The design intentionally separates request
scheduling, virtual block mapping, physical memory ownership, and KV cache policy
so the backend can later attach CUDA kernels, Triton CPU kernels, or custom
Qwen3-VL multimodal cache layouts without rewriting the allocator core.

## System Architecture Diagram

```mermaid
flowchart TD
    R[Inference Request<br/>prompt tokens, decode budget,<br/>sampling params, multimodal metadata]
    NET[Network / API Threads<br/>async request ingress]
    SQ[Thread-safe Ingress Queue<br/>single ownership handoff]
    S[Engine Event Loop<br/>Scheduler, cache mutation,<br/>admit, preempt, resume]
    KVM[KVCacheManager<br/>sequence lifecycle, append tokens,<br/>fork for parallel sampling]
    BT[BlockTable<br/>logical block -> physical block mapping]
    MA[MemoryAllocator<br/>free list, ref counts, CoW,<br/>eviction hooks]
    PB[(Aligned Physical KV Blocks<br/>CPU memory today<br/>pinned/CUDA/Triton buffers later)]
    EX[Execution Backend<br/>CPU reference kernels now<br/>CUDA/Triton later]

    R --> NET
    NET --> SQ
    SQ --> S
    S -- scheduling decision --> KVM
    S -- batch plan --> EX
    KVM -- allocate/append/fork/free --> BT
    BT -- resolve logical slots --> MA
    MA -- owns/recycles --> PB
    PB -- block pointers/views --> EX
    EX -- token progress/cache writes --> KVM
    KVM -- cache pressure/status --> S
    MA -- capacity/refcount stats --> S
```

## Key Design Boundaries

- `Request` is an inference-level unit. It owns user-visible metadata, sampling
  parameters, multimodal feature descriptors, and decode limits, but not raw KV
  memory.
- `Scheduler` owns admission control and continuous batching decisions. It asks
  `KVCacheManager` whether a request can reserve or extend cache before placing
  it in the active batch.
- `KVCacheManager` owns sequence cache lifecycle. It creates block tables,
  appends logical tokens, forks tables for parallel sampling, and releases cache
  when requests finish or are evicted.
- `BlockTable` is the OS-like page table. It maps logical token blocks to
  physical blocks and supports shared mappings after prompt reuse or branch
  creation.
- `MemoryAllocator` owns physical blocks, reference counts, free lists, and
  copy-on-write materialization. It is the only component that mutates physical
  ownership.
- `PhysicalBlock` is backend-neutral. In the CPU prototype it references host
  memory allocated through a small aligned allocation abstraction. Later it can
  hold pinned host memory, CUDA allocation handles, stream ownership metadata,
  or Triton buffer descriptors behind the same logical interface.

## Concurrency Contract

The phase-1 allocator core is intentionally single-threaded. Network/API threads
may receive requests asynchronously, but they must transfer ownership into the
engine through an ingress queue. After admission, `Scheduler`, `KVCacheManager`,
`BlockTable`, and `MemoryAllocator` are mutated only by the engine event loop.

This keeps copy-on-write and reference counting deterministic while the virtual
memory invariants are still being developed. If the allocator is later shared
across threads, `PhysicalBlockInfo::ref_count`, free-list mutation, block-table
remapping, and `ensure_writable` must be protected by a single allocator mutex or
converted to a carefully audited atomic protocol. Until then, these classes
should be documented and tested as non-thread-safe engine-thread components.

## Phased Implementation Plan

### Week 1: Repository Foundation And Core Types

Milestones:
- Add a C++17 build skeleton, namespace policy, compiler warnings, and unit test
  framework.
- Define strongly typed identifiers for requests, logical blocks, physical
  blocks, token positions, and cache kinds.
- Implement immutable configuration structs for block size, layer count, head
  shape, dtype size, and capacity.
- Define the engine-thread ownership model and non-thread-safe core contracts.
- Add positional encoding metadata that can represent text 1D positions and
  Qwen-VL visual 2D/3D M-RoPE spans instead of assuming a flat token array.

Verification:
- Unit tests for identifier equality, config validation, and byte-size
  calculations.
- Unit tests for positional span validation across text, image, and video token
  ranges.
- CI target that builds tests with warnings enabled.

### Week 2: Physical Block Pool

Milestones:
- Implement CPU physical block allocation using RAII-owned aligned host memory.
- Centralize host allocation behind a tiny policy/deleter so replacing it with
  `cudaMallocHost`, `cudaFreeHost`, or a Triton buffer provider later is local.
- Enforce a default 256-byte alignment and round allocation sizes to alignment
  boundaries.
- Add free-list allocation and deterministic block recycling.
- Track block state, generation, reference count, and high-water memory stats.

Verification:
- Tests for allocate/free reuse, exhaustion, double-free rejection, and stable
  capacity accounting.
- Tests that every physical block pointer satisfies the configured alignment.
- Stress test that allocates and releases the full pool repeatedly.

### Week 3: BlockTable Virtual Memory Layer

Milestones:
- Implement logical-to-physical mapping, logical block growth, and lookup APIs.
- Add sequence-local block tables with sparse logical block support.
- Add invariant checks for missing mappings, invalid indices, and ownership.

Verification:
- Tests for append mapping, remapping, lookup misses, and release traversal.
- Property-style test that random logical operations preserve table invariants.

### Week 4: Copy-On-Write And Parallel Sampling

Milestones:
- Implement block-table fork for parallel sampling branches.
- Increase physical block ref counts on shared mappings.
- Add `ensure_writable` to materialize a private copy before mutation.
- Keep `retain`, `release`, and `ensure_writable` engine-thread-only in this
  phase; document that refcounts are deliberately non-atomic under the event-loop
  contract.

Verification:
- Tests for shared prompt blocks across branches.
- Tests proving writes after fork do not mutate sibling branches.
- Refcount tests for fork, partial release, and branch completion.

### Week 5: KVCacheManager Sequence Lifecycle

Milestones:
- Implement request/sequence creation, prompt prefill block reservation, decode
  append, and final release.
- Add cache views suitable for a CPU reference attention kernel.
- Preserve extension fields for Qwen3-VL multimodal feature spans and M-RoPE
  position metadata, including visual token spans that may consume hundreds of
  logical positions from one image.

Verification:
- Tests for prompt ingestion, decode token append, branch creation, and cleanup.
- Golden test with small synthetic KV tensors and logical block boundaries.

### Week 6: Continuous Batching Scheduler

Milestones:
- Implement pending, active, preempted, and finished queues.
- Add admission control based on available cache blocks and token budget.
- Add simple preemption policy for long-running requests under cache pressure.

Verification:
- Tests for FIFO admission, decode-step batching, request completion, and
  preemption/resume behavior.
- Simulation test with mixed prompt lengths and decode budgets.

### Week 7: Swap/Eviction Interfaces

Milestones:
- Add swap-state metadata without requiring GPU support yet.
- Define interfaces for future host-device migration and compressed/offloaded
  block storage.
- Add allocator callbacks for eviction candidate selection.

Verification:
- Tests for preemption metadata correctness and blocked admission recovery.
- Scheduler simulation under artificial memory pressure.

### Week 8: Backend Integration Readiness

Milestones:
- Define backend-neutral cache view structs for kernels.
- Add a correctness-only CPU reference PagedAttention path that manually walks
  `BlockTable` mappings with naive loops over scattered physical blocks.
- Document CUDA/Triton integration points and required synchronization rules.

Verification:
- End-to-end CPU test for prefill plus decode over multiple active requests.
- Golden test proving the naive CPU kernel reads logical tokens in the same order
  across block boundaries and physically scattered cache blocks.
- Benchmark harness for allocator latency, fork cost, CoW cost, and scheduler
  throughput.

## Immediate Success Criteria

- The allocator can model PagedAttention as virtual block tables over physical
  KV cache blocks.
- Parallel sampling branches share prompt cache until a branch writes.
- Continuous batching can admit, step, finish, and preempt requests based on
  allocator pressure.
- Qwen3-VL-specific metadata can be carried through the scheduler and cache
  manager without changing the allocator ownership model.
