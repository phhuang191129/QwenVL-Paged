# Phase 2 Roadmap: Qwen3-VL Integration, Profiling, And Optimization

Weeks 1–8 built a memory subsystem. Phase 2 attaches it to a real model, gives it
a real workload, and then optimizes it against measured hardware limits rather
than intuition.

[`architecture.md`](architecture.md) is the specification and carries the weeks
1–8 plan. [`system-design-walkthrough.md`](system-design-walkthrough.md) explains
the reasoning behind what exists. This document is the plan for what comes next:
weeks 9–16, with the early phases specified in detail and the later ones sketched
to the level where their first task is unambiguous.

**Status after the L4 session.** Weeks 9, 13, and 15–16 are done, plus the
device-pool / kernel / memory work recorded as weeks 17–23 in
[`performance.md`](performance.md). The GPU box is closed. **Next is week 14**
(serving policy, CPU, week-9 traces). Weeks 11–12 (fast CPU kernel) are still
open and do not need a GPU. Do not treat weeks 15–16 as upcoming.

**Contents**

1. [Where the project actually stands](#1-where-the-project-actually-stands)
2. [The numbers that drive every decision](#2-the-numbers-that-drive-every-decision)
3. [Target hardware and its ceilings](#3-target-hardware-and-its-ceilings)
4. [Guiding principles](#4-guiding-principles)
5. [Week 9: Real workload, no model](#week-9-real-workload-no-model)
6. [Week 10: Measurement rigor and the roofline](#week-10-measurement-rigor-and-the-roofline)
7. [Weeks 11–12: Optimize the CPU kernel](#weeks-1112-optimize-the-cpu-kernel)
8. [Week 13: Python bridge and real Qwen3-VL](#week-13-python-bridge-and-real-qwen3-vl)
9. [Week 14: VLM-specific serving policy](#week-14-vlm-specific-serving-policy)
10. [Weeks 15–16: Triton kernel on GPU](#weeks-1516-triton-kernel-on-gpu)
11. [Ongoing: publish and upstream](#ongoing-publish-and-upstream)
12. [Risk register](#12-risk-register)
13. [Cut order](#13-cut-order)
14. [Phase 2 success criteria](#14-phase-2-success-criteria)

---

## 1. Where The Project Actually Stands

The honest assessment, because it determines the ordering of everything below.

**What exists and is solid.** A paged KV-cache memory subsystem with block-table
translation, reference counting, copy-on-write, swap, a continuous-batching
scheduler, and a correctness-only reference attention kernel — all covered by
module tests, all single-threaded by a written-down contract.

**What is missing, and why it blocks the interesting work.**

| Gap | Consequence |
| --- | --- |
| No model | Nothing calls the subsystem, so there is no end-to-end metric to improve. |
| No workload | The benchmark exercises synthetic uniform requests, not the bimodal image/text prompt distribution a VLM actually sees. |
| No compute baseline | The README states outright that the reference kernel's runtime is not a meaningful baseline. That is honest, and it also means there is no attention number to optimize. |
| No measured hardware ceilings | Without a roofline there is no way to decide which optimizations are worth writing. |

The temptation is to start optimizing, because that is the visible part. The
ordering in this document is the opposite: build a workload, measure the machine,
derive what is worth doing, then do it. That ordering is itself the deliverable —
being able to show *why* an optimization was chosen is worth more than the
optimization.

---

## 2. The Numbers That Drive Every Decision

From the real `Qwen/Qwen3-VL-2B-Instruct` config: 28 text layers, hidden size
2048, 16 attention heads over 8 KV heads, `head_dim` 128, bf16. The vision tower
is 24 layers at hidden 1024, patch size 16, spatial merge 2, with DeepStack
fusing ViT layers 5, 11, and 17 into the language model.

Because Qwen3-VL compresses images 32× spatially, one image lands anywhere in a
**256 to 1,280 visual token** budget depending on resolution. Video spans 256 to
16,384 tokens.

Working the KV cache math through:

```
per token, per layer   = 8 kv heads x 128 head_dim x 2 bytes x 2 (K and V)
                       =   4,096 bytes
per token, all layers  = 4,096 x 28
                       = 114,688 bytes = 112 KiB
per 16-token block     = 112 KiB x 16 = 1.75 MiB
```

| Workload item | Visual tokens | Blocks | KV cache |
| --- | --: | --: | --: |
| Min-resolution image | 256 | 16 | 28 MiB |
| Max-resolution image | 1,280 | 80 | **140 MiB** |
| Short video (2,048 tokens) | 2,048 | 128 | 224 MiB |
| A 100-token text prompt | — | 7 | 11 MiB |

Three consequences that shape the whole phase:

1. **One image costs ~13× a typical text prompt.** A 4 GiB pool holds 2,340
   blocks — about 29 max-resolution images, or thousands of text requests. Cache
   pressure is not an occasional condition in VLM serving; it is the steady
   state.
2. **Prompt sizes are bimodal and span 5× within the image mode alone.** This is
   precisely the workload that breaks the strict-FIFO admission policy documented
   in walkthrough §6.10. That documented simplification becomes a measurable
   defect, and fixing it becomes a measurable contribution.
3. **Copy-on-write is 3.5× more expensive than the README's figure.** The
   baseline used an 8-layer block at 512 KiB and measured 7.8 µs. A real 28-layer
   block is 1.75 MiB, so CoW costs roughly **27 µs** at the same ~68 GB/s. The
   walkthrough's conclusion — that the lever is how *often* CoW fires, not how
   fast the copy is — gets stronger, not weaker.

Every experiment in this document uses the real 2B block shape. The 8-layer
baseline in the README stays as-is for historical comparison, but new numbers are
reported against `BlockShape{16, 28, 8, 128, 2}`.

---

## 3. Target Hardware And Its Ceilings

Development and all CPU measurements happen on one machine:

- **CPU:** AMD Ryzen 7 8845H, 8 cores / 16 threads, Zen 4. 256 KiB L1d, 8 MiB L2
  (1 MiB per core), 16 MiB shared L3.
- **ISA:** AVX-512 including `avx512_bf16` (`VDPBF16PS`) and `avx512_vnni`. Zen 4
  implements AVX-512 over two 256-bit FMA units, so 512-bit FMAs issue at
  256-bit width.
- **Memory:** 14 GiB total, ~8 GiB free, **no swap**. DDR5; exact speed to be
  confirmed in week 10.
- **iGPU:** Radeon 780M (gfx1103), no ROCm installed. **Explicitly out of scope**
  — ROCm support for gfx1103 is unofficial and the debugging cost is not
  justified when rented NVIDIA hours cost dollars.
- **Toolchain:** GCC 15.2.1, CMake, `uv` for Python environments. System Python
  is 3.14, which has no torch wheels, so a pinned 3.12 venv is required.

Estimated ceilings, all of which week 10 exists to replace with measurements:

| Ceiling | Estimate | Basis |
| --- | --- | --- |
| DRAM bandwidth | ~90 GB/s | DDR5-5600 dual channel, 2 × 5600 MT/s × 8 B |
| Observed `memcpy` bandwidth | ~68 GB/s | Week 8 CoW measurement, ~76% of peak |
| Peak fp32 FMA | ~1.0 TFLOP/s | 8 cores × 2 × 256-bit FMA × 16 FLOP × ~4.0 GHz |
| Peak bf16 (`VDPBF16PS`) | ~2.0 TFLOP/s | 2× the fp32 rate |
| Machine balance | ~11 FLOP/byte fp32, ~23 bf16 | peak FLOP/s ÷ DRAM GB/s |

The estimated balance point is the number that makes week 11 tractable, because
decode attention's arithmetic intensity is nowhere near it (§Week 10).

**GPU work** happens on rented hardware — an RTX 4090 or L4 on RunPod or Vast,
in sessions of a few hours, with a total budget of roughly $10–20. Nothing in the
CPU phases may depend on GPU availability.

---

## 4. Guiding Principles

Four rules that decide arguments before they happen.

1. **Measure before optimizing, and re-measure after.** No optimization is
   written before a measurement identifies it, and no optimization is kept
   before a re-measurement confirms the bottleneck moved. When a roofline says a
   rewrite cannot pay off, the decision not to write it is recorded as a result.
2. **One golden oracle, many backends.** The existing scalar
   `paged_attention_decode` is the reference every future backend must reproduce:
   AVX-512, Triton, quantized. Its tests in `tests/PagedAttention.test.cpp` are
   the acceptance gate, not a formality. If a backend cannot be validated against
   it, the backend is not done.
3. **Every number has a baseline and a machine spec.** A latency with no
   comparison point is marketing. Paged results are reported against a contiguous
   allocator; optimized kernels against the reference; policy changes against the
   policy they replace. Every measurement records CPU frequency, thermal state,
   and a reproduction command.
4. **Do not change the layers below what the change requires.** Weeks 1–8 earned
   a clean boundary where `MemoryAllocator` is the only component that mutates
   physical ownership. Phase 2 additions state which layer they touch and stay
   there.

---

## Week 9: Real Workload, No Model — **done**

Results in [`performance.md`](performance.md). Departures from the plan below,
and what week 9 changed about weeks 10-16, are recorded in
[Week 9 outcome](#week-9-outcome).

The processor for Qwen3-VL is a few megabytes and needs no model weights, so
exact token geometry is available immediately. This phase extracts it, replays it
through the existing scheduler, and produces the project's first honest charts —
without torch, without weights, and without touching the 8 GiB memory ceiling.

### Milestones

- **Trace generator** (`tools/trace_gen/qwen3vl_trace.py`). Load only
  `AutoProcessor` for `Qwen3-VL-2B-Instruct`. For each image in a real corpus
  (COCO val2017 or any few-hundred-image set), record `image_grid_thw`, the
  resulting visual token count, the interleaved-MRoPE position ids, and the text
  token count. Emit JSONL, one request per line:

  ```
  {"request_id": 1, "arrival_us": 0, "text_tokens": 42,
   "images": [{"grid_t": 1, "grid_h": 46, "grid_w": 62, "visual_tokens": 713}],
   "prompt_tokens": 755, "decode_budget": 128,
   "mrope_spans": [{"kind": "image", "start": 43, "length": 713,
                    "origin": [0, 0, 0], "extent": [1, 23, 31]}]}
  ```

- **Workload mixes.** At minimum: `text-only`, `image-heavy`, `bimodal` (mixed
  text and image requests, which is the interesting one), and
  `repeated-prefix` (the same system prompt and a repeating image set, which
  week 14 needs).
- **Replay driver** (`tools/trace_replay/`). Read a trace, drive
  `Scheduler` + `KVCacheManager` at a configured pool size, populate
  `SequenceMetadata` from the trace's M-RoPE spans, and emit per-step and summary
  metrics as CSV.
- **Contiguous baseline in the same driver.** Reserve worst-case context per
  request with no paging and no sharing. This is what the paged numbers are
  measured against, and it must be a real allocator that can genuinely fail an
  admission, not an analytic formula.
- **Metrics.** Peak cache bytes, mean utilization, internal waste %, requests
  admitted at a fixed pool size, preemption count, swap bytes moved, CoW count,
  and cumulative head-of-line stall time.

### Verification

- Replaying one trace twice produces byte-identical CSV. Without determinism no
  later before/after comparison means anything.
- At end of trace, invariant 15 holds: `free_blocks == total_blocks`,
  `active_blocks == 0`, `shared_blocks == 0`, `swapped_blocks == 0`.
- Paged utilization exceeds the contiguous baseline by roughly the margin
  walkthrough §1.3 predicts. **If it does not, stop and find out why before
  proceeding** — that would mean the model of the problem is wrong.
- Internal waste is bounded at one partial block per sequence, confirmed against
  the trace's exact token counts rather than assumed.

### Week 9 outcome

All four verification checks pass; the numbers are in
[`performance.md`](performance.md). Four things went differently than planned.

**The processor is not needed, only its geometry.** `AutoProcessor` for Qwen3-VL
resolves to `Qwen2VLImageProcessorFast`, which pulls in torch. It is also
unnecessary: an image's token count depends only on its source height and width,
so the generator calls `smart_resize` directly with the patch and merge sizes from
the published `preprocessor_config.json`. That keeps the authoritative code path
while removing torch and all pixel work from trace generation. For the same
reason the corpus is a set of real image *dimensions* (COCO `test2017` metadata,
1 MB) rather than real image files (COCO val2017, 778 MB).

**Exact M-RoPE position ids are deferred.** Computing them needs the model's
`get_rope_index`, which needs torch. Traces instead record the grid that
*determines* the ids, plus spans laid out by the Qwen2-VL rule Qwen3-VL inherits,
labelled `qwen2vl_rule` in the header. Validating this against the model is now a
week 13 task. Related: `PositionalEncodingSpan::stride` carries the merged grid
extent for image spans, because a raster-ordered 2D span cannot be described by a
per-token step. Nothing consumes it yet; revisit the field name in week 13.

**Arrival times are recorded but not replayed.** Week 9 has no cost model for a
step, so a virtual clock would have needed an invented step duration. The replay
is closed-loop instead: every request is enqueued up front and the pool is the
only constraint. Head-of-line stalls are therefore counted in steps, not
microseconds. Wall-clock TTFT arrives with the model in week 13.

**The contiguous baseline reserves per request, not a flat model maximum.** A flat
262,144-token reservation would be a strawman. Reserving `prompt +
max_new_tokens` is both realistic and a *stronger* opponent, and because run
lengths then vary, external fragmentation is measured rather than assumed. This
is why the measured utilization gap is 12-36 percentage points instead of the ~90
a flat reservation would produce. Trace requests carry `decode_actual` alongside
`decode_budget` so early stopping — the actual source of reservation waste — is
modelled rather than wished away.

### What week 9 changed downstream

- **Week 14 has a measured mandate, not a stylistic one.** `schedule_next` admits
  on whether a prompt fits, never on whether the prompt plus its decode budget
  fits, so it over-commits the pool and then stalls. The replay driver works
  around it by preempting on decode-growth failure, which costs 72 preemptions
  and 3,441 failed resume attempts to finish 400 requests at a 256-block pool.
  Decode-aware admission moves from "nice to have" to a numbered requirement.
- **Preemption must follow sampling branches.** `Scheduler::preempt` swaps out the
  root sequence only, so a preempted request with live forks reclaims almost
  nothing. This blocks combining parallel sampling with memory pressure.
- **Week 10 has a second block shape to report.** The real 2B block is 1.75 MiB,
  3.5x the 512 KiB shape the week 8 table used, and copy-on-write cost is pure
  memory bandwidth, so every CoW figure scales with it.
- **The pixel budget is a first-class experimental variable.** One line of
  `preprocessor_config.json` moves a single image between 140 MiB and 1.75 GiB of
  KV cache. Any later throughput claim has to state which budget it assumed.

---

## Week 10: Measurement Rigor And The Roofline

A roofline is worthless without measured roofs. This phase measures the machine,
hardens the benchmark harness enough that its numbers are defensible, and then
uses the result to decide what week 11 is allowed to do.

### Milestones

- **Measure the roofs.** A STREAM-triad DRAM bandwidth benchmark; a
  cache-resident working-set sweep to find the L1/L2/L3 bandwidth plateaus; and
  an AVX-512 FMA microbenchmark for peak fp32 and bf16 FLOP/s. Confirm actual
  DDR5 speed and channel count.
- **Control the measurement environment.** This is a laptop: pin the CPU
  governor, record frequency during each run, record thermal state, and reject
  any run whose clock drifted beyond a threshold. Silent thermal throttling
  invalidates every number otherwise.
- **Harden `bench/`.** Replace "median of 3" with warmup iterations, N
  repetitions, and median / p95 / MAD reporting. Re-run the week 8 baseline table
  with the real 2B block shape (1.75 MiB) alongside the original 512 KiB figures.
- **Derive the analytic arithmetic intensity of paged attention.** For decode,
  per (query head, context token), the kernel moves one K and one V vector and
  performs a dot product plus a weighted accumulate:

  ```
  bytes = 2 x head_dim x bytes_per_element
  flops = 4 x head_dim
  AI    = 2 / bytes_per_element        (one query head per K/V load)
        = 2G / bytes_per_element       (G query heads sharing one K/V load)
  ```

  For this model `G = 16 / 8 = 2`. So AI is **0.5 FLOP/byte in fp32 and 1.0 in
  bf16** unfused, doubling to 1.0 and 2.0 if the query-head group is fused over a
  single K/V load. Against an estimated machine balance of ~11 (fp32) to ~23
  (bf16) FLOP/byte, decode attention sits an order of magnitude into the
  memory-bound regime.
- **Do the same for prefill, separately.** Prefill attention is a GEMM over the
  whole prompt and its AI rises with sequence length, so it is compute-bound in a
  way decode is not. For a text LLM this barely matters because prefill is short.
  For a VLM a single image is a 1,280-token prefill, so **both regimes matter and
  must be rooflined separately.** This distinction is the technical core of the
  "VLM serving is a different problem" claim.
- **Plot it.** A roofline for this machine with the reference kernel's measured
  position on it, checked in as a reproducible script plus the generated figure.

### Verification

- Measured STREAM bandwidth lands within ~15% of theoretical peak; a larger gap
  means the benchmark, not the machine, is wrong.
- The reference kernel's measured GFLOP/s lands on the memory-bound roof, not
  under it by an unexplained factor. **The roofline must predict the measurement
  before it is used to predict anything else.**
- The re-run week 8 baselines reproduce the original 512 KiB figures within noise
  and show CoW scaling linearly to ~27 µs at 1.75 MiB.

### The decision this phase exists to produce

The AI of decode attention is fixed by the algorithm. Only two levers change it
rather than merely approaching the roof:

1. **Fuse the GQA query-head group** so one K/V load serves both query heads —
   a 2× AI improvement, available for free in the loop structure.
2. **Move fewer bytes** — quantized KV cache, which is week 16.

Everything else (vectorization width, softmax passes, prefetching) chases the
memory roof rather than raising it. Week 11 is scoped accordingly, and the
optimizations *not* attempted are recorded with the roofline reasoning that ruled
them out.

---

## Weeks 11–12: Optimize The CPU Kernel

Now, and only now, write a fast kernel — with a numeric target set by week 10
(a stated fraction of measured DRAM and L2 bandwidth).

### Milestones

- **Block-at-a-time iteration.** Restructure the loop to process one physical
  block per iteration, hoisting the block-table lookup out of the per-token inner
  loop. This is the direct fix for translation overhead and it also removes the
  two heap-allocated pointer vectors the reference kernel builds per call.
- **Online softmax.** Single-pass streaming max and sum, so the context is read
  once instead of twice. On a bandwidth-bound kernel, halving traffic is the
  whole game.
- **GQA group fusion**, per the week 10 decision: load each K/V vector once and
  consume it with both query heads in the group.
- **AVX-512 inner loop.** `head_dim = 128` is exactly 8 zmm registers of fp32 or
  4 of bf16 — a clean fit with no tail handling. Use `VDPBF16PS` for the bf16
  path.
- **Optional multithreading** across query heads, behind a flag, only if the
  single-threaded kernel is already close to its roof. The engine's
  single-threaded contract covers cache *mutation*; a read-only kernel
  parallelized internally does not violate it, but that argument must be written
  into `architecture.md` before the code lands.
- **Preserve the reference.** The scalar kernel stays as the oracle. Every
  variant is selected at runtime or compile time and validated against it.

### Verification

- Every existing test in `tests/PagedAttention.test.cpp` and
  `tests/EndToEnd.test.cpp` passes for each new backend, including the
  scattered-block, grouped-query, copy-on-write, and swapped-out-fault cases.
- Numeric agreement with the scalar reference within a stated fp tolerance,
  including a case with a deliberately adversarial block scattering.
- Measured position on the roofline recorded before and after each change, with
  a note on which change moved it.

### The headline result to chase

**Quantify the paging tax.** Run the same optimized kernel twice: once over
contiguous KV, once over paged KV with identical contents. The FLOPs are
identical; only the access pattern differs. Report the gap in ns/token and as a
fraction of achieved bandwidth, before and after hoisting translation out of the
inner loop.

This is the number that answers "what does PagedAttention actually cost?" — a
question the vLLM paper asserts is small and almost nobody measures in isolation.
It is the most defensible original micro-result available to this project.

---

## Week 13: Python Bridge And Real Qwen3-VL

### Milestones

- **pybind11 module** exposing `MemoryAllocator`, `KVCacheManager`, `Scheduler`,
  `BlockShape`, and stats, plus a zero-copy numpy view over a block's bytes so
  torch can write KV directly into a pool frame.
- **`PagedCache`**, subclassing the `Cache` interface in
  `transformers/cache_utils.py`. Pin an exact transformers version — that API
  changes between minor releases, and the `update()` signature must be read from
  the installed source rather than assumed.
- **Populate the multimodal metadata for real.** Build `MultimodalSpan` and
  `PositionalEncodingSpan` from actual `image_grid_thw` values, exercising the
  "carried, not interpreted" design of walkthrough §6.14 with real data for the
  first time.
- **Environment.** A `uv`-managed Python 3.12 venv, torch CPU bf16,
  `Qwen3-VL-2B-Instruct`. Plus a **tiny random-weight Qwen3-VL config** (2 layers,
  small hidden size) for the fast test loop, so correctness tests do not need
  4.4 GiB of weights or 30-second prefills.

### Verification

- **The critical gate: token-identical output.** Greedy decode through
  `PagedCache` must produce exactly the same token sequence as `DynamicCache` for
  the same prompt and image, on both the tiny model and the real 2B. Anything
  less means the paged path is silently wrong, and every performance number
  after it is meaningless.
- `n = 4` parallel sampling over one image prompt: refcount evidence that the
  prompt's ~80 blocks are stored once, with the measured process RSS delta
  against `DynamicCache` doing the same thing.
- Zero leaks after the run: invariant 15 again, now driven by a real model.

### Expected findings to write up, not hide

- **torch attention wants contiguous K/V**, so `PagedCache.update()` must gather
  blocks into a contiguous tensor per step. Measure that gather cost. It is the
  concrete, quantified argument for a native paged kernel, and it is a more
  honest framing than pretending the integration is free.
- **CPU inference is slow and that is fine.** Expect roughly 20–30 s to prefill
  one max-resolution image and a few tokens/second of decode on the 2B model. The
  bridge exists to prove correctness and measure *memory*; throughput claims wait
  for week 15's GPU.

---

## Week 14: VLM-Specific Serving Policy

The differentiating phase. Three experiments, each on a week 9 trace, each with
an honest baseline and a number on both sides.

### Milestones

- **Chunked prefill.** A 1,280-token image prompt is a 140 MiB, multi-second
  prefill that stalls the entire batch. Split it across steps in the scheduler.
  Measure TTFT for a small text request queued behind an image request, before
  and after. Also fix the documented defect that `max_batch_tokens` is only
  enforced during admission while decode tokens are added afterwards.
- **Decode-aware admission.** Measured in week 9: `schedule_next` admits on
  whether a prompt fits, so it over-commits the pool and then permanently stalls
  once decode cannot grow a sequence. The replay driver currently papers over this
  by preempting on growth failure, at a cost of 69 preemptions and 3,439 failed
  resume attempts per 400 requests at a 256-block pool. Admit against prompt plus
  a decode reserve instead, and move the recovery out of the driver. Also make
  `preempt` swap out a request's sampling branches, not just its root sequence, so
  preemption reclaims a useful number of frames when forks exist. Report the same
  pool sweep as week 9 and show the preemption count collapsing.
- **Size-aware admission.** Replace the strict-FIFO head-of-line blocking of
  §6.10 with a size-aware policy that still provably avoids starvation. Report
  throughput and p99 TTFT on the bimodal trace, against FIFO. §6.10 already
  frames FIFO as "the honest, simple baseline that the more complex policy has to
  beat" — this is that comparison.
- **Cross-request prefix caching.** A content-hash index over blocks so two
  *different* requests sharing a system prompt and an image share physical
  blocks. The sharing machinery already exists; walkthrough §8 states this needs
  no changes below `KVCacheManager`, and this phase tests that claim. The VLM win
  is much larger than the text win because the shared prefix is 80 blocks rather
  than 5. Report hit rate and blocks saved on the `repeated-prefix` trace.

### Verification

- Each policy change keeps every existing scheduler test passing, or the test is
  updated with a written justification for why the old behavior was wrong.
- Starvation freedom for the size-aware policy demonstrated by a test, not
  asserted in prose.
- Prefix-cache correctness: two requests sharing a hashed prefix produce
  identical logits to two independent requests. A hash collision or a
  stale-content bug here is silent and catastrophic, so this needs an explicit
  adversarial test.
- Zero leaks with prefix sharing active, including the case where one sharer is
  cancelled mid-decode and the other continues.

---

## Weeks 15–16: Triton Kernel On GPU

Rented NVIDIA hardware, a few hours at a time. Weeks 1–8 claimed the design was
GPU-ready in four specific places; this phase is where that claim is tested.

### Milestones

- **Export golden vectors.** Dump the existing C++ test fixtures to `.npz` so one
  set of fixtures validates the CPU and GPU kernels. This must be done *before*
  renting anything, so paid hours are spent on the kernel rather than on
  plumbing.
- **Triton paged-attention kernel** consuming exactly `KVBlockLayout`'s four
  strides and the flattened `[num_seqs, max_blocks_per_seq]` table from
  `BlockTable::entries()`. Architecture doc §"Backend Integration Points" claims
  these are the right primitives; if they are not, the finding goes back into
  that doc.
- **Profile the whole run**, then each kernel. Nsight Systems for the timeline,
  kernel launch overhead, and gaps. Nsight Compute for the memory chart, achieved
  occupancy, DRAM throughput, and a GPU roofline. **Re-profile after every change
  to confirm the bottleneck moved**, and record the cases where it did not.
- **Quantized KV cache.** `BlockShape::bytes_per_element` is already a parameter,
  so FP8 or INT8 blocks need no new plumbing in the allocator. Measure the
  bandwidth win *and* the accuracy delta on a small eval, since the whole point
  of quantization work is the tradeoff, not the speedup.

### Verification

- The Triton kernel reproduces the CPU golden vectors within a stated tolerance,
  on the same fixtures, including scattered blocks and grouped-query heads.
- Every synchronization rule in architecture doc §"Synchronization Rules" is
  exercised: notably that a frame is not recycled while a kernel still reads it,
  which means `release`, `swap_out`, and `preempt` must wait on the step's
  completion event. This is the one genuinely new constraint an async backend
  adds, and it needs a test that fails without the wait.
- Every profiling claim ships with the trace file and the command that produced
  it.

---

## Ongoing: Publish And Upstream

- **`docs/performance.md`**: every measurement, with machine spec, thermal and
  frequency conditions, baseline, and a copy-pasteable reproduction command.
  Before-and-after figures for each optimization, including the ones that did not
  work.
- **A comparative study against vLLM's block manager.** This project
  re-implements from scratch what vLLM does in production. Documenting where the
  designs agree, where they differ, and why is a piece of analysis that requires
  having built both mental models — and it is the strongest interview asset here.
- **Upstream contribution.** Two candidate targets, in order of likely success:
  a reproducible Qwen3-VL serving recipe in the style of `vllm-project/recipes`,
  or a small well-scoped fix found while reading vLLM's or SGLang's block manager
  for the comparative study. The review conversation matters more than the merge.

---

## 12. Risk Register

| Risk | Severity | Mitigation |
| --- | --- | --- |
| 8 GiB free RAM, no swap. 2B bf16 is ~4.4 GiB; one stray fp32 cast doubles it and OOMs. | High | Tiny random-weight model for the test loop, explicit dtype assertions, hard cap on experiment pool size. |
| Scope. Weeks 13–16 are each realistically two weeks for one person. | High | The cut order in §13. Ship a smaller phase completely rather than four phases partially. |
| transformers `Cache` API churn breaks `PagedCache`. | Medium | Pin an exact version; read `cache_utils.py` from the installed package, never from memory or docs. |
| Laptop thermal throttling silently invalidating measurements. | Medium | Pin governor, record clocks per run, discard runs with clock drift, treat any unreproducible number as noise. |
| Rented GPU session ends mid-experiment. | Low | Golden vectors and all scripts committed before renting; sessions scoped to one experiment each. |
| Python 3.14 has no torch wheels. | Low | `uv`-pinned 3.12 venv; already identified. |

---

## 13. Cut Order

If time runs short, drop in this order — first to go listed first:

1. Week 14's prefix caching (the largest engineering effort of the three policy
   experiments).
2. Week 16's quantized KV cache.
3. Week 11's multithreaded kernel path.
4. The vLLM comparative study, reduced from a document to a section.

Do **not** cut, in any circumstance:

- Week 9's contiguous baseline. Without it every paged number is unfalsifiable.
- Week 10's measured roofs. Without them the optimization work has no
  justification, which is exactly the weakness this phase exists to remove.
- Week 13's token-identical correctness gate. An integration that is fast and
  wrong is worse than no integration.
- Weeks 15–16's GPU port. It is the highest-value item; it goes last in the
  schedule but first in priority.

---

## 14. Phase 2 Success Criteria

- Qwen3-VL-2B generates token-identical output through the paged cache and
  through `DynamicCache`, on real images, with the KV bytes genuinely living in
  this project's block pool.
- A measured roofline for this machine, with the attention kernel's position on
  it before and after optimization, and a written record of the optimizations the
  roofline ruled out.
- The paging tax quantified: the cost of block-table translation isolated from
  the cost of attention itself.
- A serving-policy result on a real bimodal VLM workload, showing throughput or
  tail-latency improvement over the FIFO baseline the project shipped in week 6.
- A Triton kernel validated against the CPU golden vectors, profiled with
  Nsight, with its bottleneck identified and confirmed to have moved.
- Every number reproducible from a committed script.
