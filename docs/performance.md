# Measured Results

Every number here is reproducible from a committed script. Each section states
what was measured, against what baseline, and what assumption the figure depends
on. Results that contradict the project's own expectations are recorded rather
than dropped.

Roadmap context: [`roadmap-phase2.md`](roadmap-phase2.md).

## Provenance

| | |
| --- | --- |
| **CPU** | AMD Ryzen 7 8845H, 8 cores / 16 threads (Zen 4) |
| **Memory** | 14 GiB DDR5, no swap |
| **Compiler** | GCC 15.2.1, `-O2 -std=c++17`, `CMAKE_BUILD_TYPE=Release` |
| **Model geometry** | `Qwen/Qwen3-VL-2B-Instruct`: 28 layers, 8 KV heads, `head_dim` 128, bf16 |
| **Block shape** | 16 tokens x 28 layers x 8 KV heads x 128 dim x 2 B = **1.75 MiB/block** |
| **Trace generator** | `transformers==4.57.1`, Python 3.12.13 |

The week 9 results are workload and memory-accounting results. They involve no
model forward pass, so no latency figure appears in that section; "steps" is a
count of continuous-batching iterations, not time. Week 15 adds wall-clock
timings for the attention kernel itself, on its own GPU provenance. Neither is an
end-to-end throughput number: that needs the model integration in week 13.

Reproduce with:

```bash
./tools/trace_gen/generate_all.sh          # needs network on first run
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build
./tools/trace_replay/run_experiments.sh    # writes results/week9-*.csv
```

---

## Week 9: Qwen3-VL Workload Geometry And Paged vs Contiguous Allocation

### What was measured

Real Qwen3-VL request traces replayed through `Scheduler` and `KVCacheManager`,
against a contiguous worst-case-reservation allocator over an identically sized
pool of identically sized frames.

Image token counts are authoritative: they come from `transformers`' own
`smart_resize` and the patch/merge sizes in the model's published
`preprocessor_config.json`. Image *dimensions* come either from COCO
`test2017` metadata (40,670 real images) or from a documented mix of real device
capture resolutions. Text token counts, decode budgets, and mix proportions are
synthetic and labelled as such in every trace header.

The replay is closed-loop: all requests are enqueued up front and the block pool
is the only thing throttling admission. Week 9 has no cost model for a step, so
modelling arrival times would have added a parameter without adding information.

### Finding 1: the shipped pixel budget costs 1.75 GiB of KV cache per image

`preprocessor_config.json` for Qwen3-VL-2B ships `size.longest_edge = 16777216`
pixels. At 32x spatial compression that is a budget of **16,384 visual tokens for
a single image**. The Qwen3-VL documentation separately advises clamping to
256-1280 tokens. Both are one line of configuration, and on this model they
differ by 13x in cache cost:

| Pixel budget | Max visual tokens / image | Blocks | KV cache | Largest prompt seen in trace |
| --- | --: | --: | --: | --: |
| Shipped default | 16,384 | 1,024 | **1.75 GiB** | 16,288 tokens (1,774 MiB) |
| Documented clamp | 1,280 | 80 | 140 MiB | 1,247 tokens (136 MiB) |

For scale, a 100-token text prompt on the same model costs 11 MiB. A single
default-budget phone photo (4032x3024) resolves to 11,844 tokens; the same photo
under the clamp resolves to 1,230. Two default-budget images exhaust a 4 GiB KV
cache.

This is the concrete form of "VLM serving is a different problem": prompt sizes
are bimodal, span an order of magnitude within the image mode, and are set by a
preprocessing default rather than by the request.

### Finding 2: paging holds the same batch in 7-27% less cache

One pool per trace, sized so the paged batch never exhausts it mid-decode, which
leaves the baseline room as well. This isolates memory efficiency from scheduling.

| Trace mix | Image dims | Pool | Mode | Peak blocks | Peak bytes | Utilization | Tail slack |
| --- | --- | --: | --- | --: | --: | --: | --: |
| text-only | devices | 960 | paged | 332 | 581 MiB | **0.978** | 15 |
| | | | contiguous | 436 | 763 MiB | 0.617 | 516 |
| image-heavy | coco | 616 | paged | 292 | 511 MiB | **0.983** | 15 |
| | | | contiguous | 399 | 698 MiB | 0.678 | 496 |
| bimodal | devices | 912 | paged | 679 | 1,188 MiB | **0.991** | 15 |
| | | | contiguous | 770 | 1,348 MiB | 0.801 | 502 |
| image-heavy | devices | 1504 | paged | 1,010 | 1,767 MiB | **0.995** | 15 |
| | | | contiguous | 1,086 | 1,900 MiB | 0.876 | 485 |
| repeated-prefix | devices | 912 | paged | 702 | 1,228 MiB | **0.994** | 15 |
| | | | contiguous | 792 | 1,386 MiB | 0.859 | 506 |

*Utilization* is committed token positions per token slot the pool handed out,
step-weighted. *Tail slack* is the largest gap between a sequence's reserved
capacity and its committed length, in token slots.

Two things to read here:

- **The internal-waste bound holds exactly.** Paged tail slack peaks at 15 token
  slots against a 16-token block, confirming the "at most one partial block per
  sequence" claim on real workloads rather than by construction. The baseline's
  slack peaks at 485-516 slots, which is its `max_new_tokens` reservation minus
  the tokens the request actually produced.
- **The baseline here is deliberately strong.** It reserves `prompt +
  max_new_tokens` per request, not a flat model maximum, so it already benefits
  from knowing each request's declared budget. That is why its utilization is
  0.62-0.88 rather than the ~6% that reserving the full 262,144-token context
  would give. Paging still wins, but against an honest opponent the margin is
  12-36 percentage points of utilization and 7-27% of peak bytes, not 90.

### Finding 3: external fragmentation is real, and paging removes it entirely

Refused admissions where the pool had enough total free frames but not as one
contiguous run:

| Trace mix | Paged | Contiguous |
| --- | --: | --: |
| text-only | 0 | 0 |
| image-heavy (coco) | 0 | 0 |
| bimodal | 0 | 6 |
| image-heavy (devices) | 0 | 0 |
| repeated-prefix | 0 | **1,350** |

Fragmentation requires variable-length reservations, which is why the baseline
reserves per request rather than a fixed size. Under pool pressure it gets far
worse: see the sweep below, where the baseline refuses up to 5,559 admissions
this way. Paged is structurally zero, because every free block is interchangeable.

### Finding 4: sharing pushes effective utilization above 1.0

The `image-heavy-parallel4` trace requests 4 sampling branches per image request.
Branches share the prompt until one writes:

| Metric | Value |
| --- | --: |
| Forks | 1,200 |
| Copy-on-write materializations | 1,137 |
| Utilization | **3.391** |
| Peak blocks | 1,133 |
| Leak-free at end of trace | yes |

Utilization above 1.0 means each allocated token slot is backing more than one
sequence. Storing four branches of an 80-block image prompt independently would
cost 420 MiB of duplicate cache per request; here it costs one copy plus the
blocks a branch actually diverges on. Copy-on-write fires roughly once per fork,
which is the shared tail block of the prompt being written on the first decode
step, exactly as predicted.

### Finding 5: under pressure, prompt-only admission is worse than worst-case reservation

This one contradicted the expectation, and it is the most useful result of the
phase.

Shrinking the pool on a fixed workload, the paged engine fails to complete the
trace at every pool below 1,024 blocks, while the contiguous baseline completes at
every size down to 128. Reproduce with `--swap-slots 0`, which leaves preemption
unable to reclaim anything:

| Pool blocks | Paged requests completed | Contiguous requests completed |
| --: | --: | --: |
| 128 | 1 / 400 | 400 / 400 |
| 256 | 11 / 400 | 400 / 400 |
| 512 | 57 / 400 | 400 / 400 |
| 1024 | 400 / 400 | 400 / 400 |

The cause is not paging. `Scheduler::schedule_next` admits on whether a *prompt*
fits, never on whether the prompt plus its decode budget fits, so it over-admits
a batch it cannot carry to completion. With automatic preemption disabled that is
a permanent stall. The contiguous baseline cannot suffer this: reserving the
worst case up front is also a guarantee that an admitted request can finish. Its
inefficiency is its safety property.

The fix is the standard engine recovery, added to the replay driver rather than to
the scheduler: when decode cannot grow a sequence, park the newest active request
and retry on a later step. With that and a swap backend, the paged engine
completes at every pool size and beats the baseline on throughput:

| Pool blocks | Paged steps | Contiguous steps | Step delta | Paged batch | Contiguous batch | Preemptions | Baseline fragmentation failures |
| --: | --: | --: | --: | --: | --: | --: | --: |
| 192 | 22,045 | 27,963 | **-21.2%** | 3 | 2 | 36 | 1,244 |
| 256 | 14,948 | 22,271 | **-32.9%** | 4 | 3 | 72 | 3,178 |
| 384 | 10,210 | 12,562 | **-18.7%** | 5 | 5 | 15 | 4,042 |
| 512 | 7,127 | 9,903 | **-28.0%** | 7 | 6 | 56 | 5,559 |
| 768 | 4,842 | 6,179 | **-21.6%** | 8 | 8 | 13 | 3,936 |
| 1024 | 4,573 | 4,806 | -4.8% | 8 | 8 | 0 | 1,519 |

Paged utilization is 0.991-0.995 across the whole sweep; the baseline is flat at
0.876. The step advantage shrinks to noise at 1,024 blocks because at that size
neither allocator is the binding constraint and both run the full batch of 8.

**What this means for the roadmap.** Making admission itself decode-aware, rather
than relying on driver-side recovery, is now a measured requirement for week 14
rather than a stylistic preference. It also puts a number on the preemption
thrash the recovery causes: at pool 256 the driver issues 72 preemptions and
3,441 failed resume attempts to finish 400 requests, which a decode-aware
admission policy should mostly eliminate.

### Verification status

The week 9 checks in the roadmap, as asserted in `tests/TraceReplay.test.cpp`:

| Check | Status |
| --- | --- |
| Replaying a trace twice is byte-identical | pass |
| Invariant 15 (no blocks, shares, or swap slots leaked) at end of trace | pass |
| Paged utilization exceeds the contiguous baseline | pass, on all mixes |
| Internal waste bounded by one partial block per sequence | pass, 15 of 16 slots |
| Real M-RoPE and multimodal spans carried through the cache manager | pass |
| A prompt too large for the pool reports a stall instead of hanging | pass |

Full suite: 141 tests pass at the time of writing; 150 as of week 17.

### Known limitations of these numbers

- **No forward pass.** "Steps" counts batching iterations. Nothing here says how
  long a step takes, so none of it is a throughput claim.
- **Text token counts are synthetic.** Only the visual geometry is authoritative.
- **M-RoPE position layout is the Qwen2-VL rule.** Qwen3-VL's Interleaved-MRoPE
  changes frequency-band allocation rather than position ids, but this has not
  been validated against the model's own `get_rope_index`, which needs torch.
  That validation is a week 13 task.
- **Preemption does not follow sampling branches.** `Scheduler::preempt` swaps out
  the root sequence only, and forked children keep their frames, so a preempted
  request with live branches reclaims little. The parallel-sampling trace is run
  at a pool sized to avoid pressure for this reason.
- **The COCO corpus is narrow.** Its images are capped at 640 px, so they resolve
  to roughly 300 visual tokens and never exercise the high-resolution regime.
  That is why the device resolution mix exists, and why its weights are labelled
  an assumption.

---

## Week 15: Triton Decode Kernel On GPU

### Provenance

| | |
| --- | --- |
| **GPU** | NVIDIA L4 (AD104, sm_89), 58 SMs, 22.0 GiB visible, 48 MiB L2 |
| **Driver** | 595.91.07, CUDA 13.2 |
| **Stack** | torch 2.13.0+cu130, triton 3.7.1, Python 3.12.14 |
| **Block shape** | 16 tokens x 28 layers x 8 KV heads x 128 dim x 2 B = **1.75 MiB/block** |
| **Attention** | 16 query heads over 8 KV heads, `head_dim` 128, KV in bf16, query and output in fp32 |
| **Pool** | 5,120 blocks (8.75 GiB) |
| **Peak DRAM** | **300 GB/s**, from 6.25 GHz x 192-bit, agreeing with Nsight's peak-sustained figure to 0.1% |
| **Profiler** | Nsight Compute 2026.2.1 |

Reproduce with:

```bash
bash tools/triton_kernel/setup_gpu_box.sh              # venv, pinned stack, fixture gate
.venv/bin/python tools/triton_kernel/run_fixture.py fixtures
.venv/bin/python tools/triton_kernel/check_large_pool.py
.venv/bin/python tools/triton_kernel/benchmark.py      # writes results/week15-gpu-kernel.csv

# Hardware counters for one configuration (batch 32, 1,280 tokens):
cd tools/triton_kernel && sudo /opt/nvidia/nsight-compute/2026.2.1/ncu \
    --kernel-name regex:paged_attention_decode --launch-skip 8 --launch-count 1 \
    --metrics gpu__time_duration.sum,dram__bytes.sum,dram__bytes.sum.per_second,\
dram__throughput.avg.pct_of_peak_sustained_elapsed,lts__t_sector_hit_rate.pct,\
sm__warps_active.avg.pct_of_peak_sustained_active \
    ../../.venv/bin/python profile_kernel.py 32 1280
```

Every table in this section was produced at commit `854f469`, before the week 16
grid change. Re-running `benchmark.py` at HEAD moves individual rows by up to 5%
in both directions and changes no conclusion, which is the expected result: week
16 finding 1 shows that change does nothing below batch 512, and these rows stop
at 64.

### What was measured

A Triton PagedAttention decode kernel consuming exactly what
[`architecture.md`](architecture.md) promises a backend: `KVBlockLayout`'s four
strides and the flattened `[num_seqs, max_blocks_per_seq]` block table. One
program per (sequence, query head), streaming that sequence's blocks with an
online softmax.

Decode attention moves one K and one V vector per (query head, context token) and
does `4 * head_dim` FLOPs on them, so its arithmetic intensity is about
1 FLOP/byte at bf16. This L4's balance point is roughly 100 FLOP/byte
(30.3 TFLOP/s fp32 over 300 GB/s), which puts the kernel two orders of magnitude
into the memory-bound regime. Every result below is therefore reported as
achieved bandwidth against peak DRAM. No GFLOP/s figure appears, because it would
describe a limit that is nowhere near binding.

Two measurement regimes appear, and they are not interchangeable:

- **The benchmark** (`benchmark.py`) reports *unique* bytes over time — one K and
  one V vector per (KV head, context token), the traffic any correct kernel must
  move. `triton.testing.do_bench` zeroes a flush buffer before each timed
  iteration, so every row starts from a cold L2. This is the right basis for
  relative comparisons: paging tax, batch scaling.
- **Nsight Compute** reports actual DRAM bytes from hardware counters, on one
  launch after eight warmups, so its L2 is warm. This is the right basis for
  "how close to the memory roof is this kernel."

The kernel issues two loads of every K/V vector, because the two query heads
sharing a KV head load it separately. Findings 4 and 5 turn on that.

### Finding 1: the paging tax is below the measurement noise floor

The headline result, and the one the roadmap wanted: what does block-table
indirection actually cost? The identical kernel runs over the same pool with two
block tables — one giving each sequence a consecutive run of frames, one giving
it a random permutation of the same frames. Same arithmetic, same byte count,
same frames; locality is the only variable.

| Batch | Context | Consecutive | Scattered | Delta |
| --: | --: | --: | --: | --: |
| 1 | 256 | 45.4 us | 45.5 us | +0.2% |
| 8 | 256 | 89.4 us | 89.8 us | +0.4% |
| 32 | 256 | 191.4 us | 193.0 us | +0.8% |
| 64 | 256 | 400.2 us | 400.3 us | +0.0% |
| 1 | 1,280 | 194.5 us | 195.2 us | +0.4% |
| 8 | 1,280 | 318.0 us | 317.0 us | **-0.3%** |
| 32 | 1,280 | 815.4 us | 817.5 us | +0.3% |
| 64 | 1,280 | 1,670.0 us | 1,682.6 us | +0.8% |

Scattering is faster in one row, which is the tell. Repeating the two ends of the
sweep seven times each puts the effect inside run-to-run variance:

| Config | Layout | min | median | max |
| --- | --- | --: | --: | --: |
| batch 8, ctx 256 | consecutive | 89.6 | 89.7 | 89.8 us |
| | scattered | 89.8 | 89.9 | 90.2 us |
| batch 32, ctx 1,280 | consecutive | 816.0 | 819.0 | 834.8 us |
| | scattered | 817.0 | 818.9 | 840.3 us |

At batch 32 the two medians differ by 0.01% while a single layout varies by 2.3%
between its own fastest and slowest run. **The paging tax is not measurable here.**

This is not a surprise once the access pattern is written out, and the mechanism
matters more than the number. A block is 1.75 MiB, but a kernel reading one layer
and one KV head touches only 16 vectors of 256 bytes inside it, strided 2 KiB
apart. Two logically adjacent blocks put their relevant regions 1.75 MiB apart
even when the frames are consecutive. Adjacency was never buying locality at any
granularity the hardware cares about, so destroying it costs nothing.

The honest scope: this measures the cost of *scattering*, with the block-table
lookup present in both arms. It does not isolate the cost of the indirection
itself, which would need a second kernel with no block table.

### Finding 2: small batches are latency-bound, and it is severe

| Batch | Programs | ctx 256 | ctx 1,280 |
| --: | --: | --: | --: |
| 1 | 16 | 23.1 GB/s (8%) | 27.0 GB/s (9%) |
| 8 | 128 | 93.8 GB/s (31%) | 131.6 GB/s (44%) |
| 32 | 512 | 175.1 GB/s (58%) | 204.7 GB/s (**68%**) |
| 64 | 1,024 | 167.6 GB/s (56%) | 198.9 GB/s (66%) |

The grid is `(num_seqs, num_query_heads)`, so a single request launches 16
programs onto 58 SMs. Nearly three quarters of the SMs have no work at all —
hence 8-9% of peak. Finding 5 confirms with hardware counters that this is a
residency problem and not a per-program efficiency problem.
Every program also walks its whole context serially, so at 1,280 tokens one
program is stepping through 80 blocks alone.

This is the same problem vLLM solves with a second paged-attention kernel that
partitions the context across programs and reduces across the partitions. It is
the largest single optimization available here, and unlike bandwidth work it
targets the regime that actually matters for interactive single-stream serving.

### Finding 3: at serving batch sizes the kernel saturates memory

Hardware counters at batch 32 with a 1,280-token context:

| Metric | Value |
| --- | --: |
| DRAM bytes moved | 194.97 MB |
| DRAM throughput | 284.67 GB/s |
| **Percent of peak sustained** | **94.98%** |
| Kernel duration (warm L2) | 684.90 us |

**The kernel is at 95% of peak DRAM bandwidth.** For a hand-written kernel with no
tiling, no fusion, and no context partitioning, there is essentially no bandwidth
headroom left at this operating point.

The benchmark's 68% and this 95% are consistent, not contradictory. DRAM moves
194.97 MB against the algorithm's 167.8 MB of unique bytes, a 1.16x overhead from
queries, outputs, the block table, and imperfect L2 capture; and the warm-L2
duration under the profiler is 685 us against 820 us cold. Multiply those two
gaps and the figures reconcile.

**A correction this produced.** The first version of this section reported "90% of
a measured 230 GB/s copy roof." That roof was wrong. A `torch` device-to-device
copy reaches 230 GB/s and a read-only reduction 252 GB/s, but this kernel sustains
285 — so the copy was a worse bandwidth benchmark than the thing being measured,
and using it as the denominator inflated every percentage. Roadmap week 10 already
states the criterion that catches this: a measured roof more than ~15% below
theoretical means the benchmark is wrong, not the machine. The copy was 23% below.
Peak is now computed from the memory clock and bus width, which agrees with
Nsight's peak-sustained figure to 0.1%.

Batch 64 is slightly *worse* than batch 32 at both context lengths (66% vs 68%,
56% vs 58%). That inversion lines up with the working set crossing the 48 MiB L2:
32 MiB and 160 MiB at batch 32, against 64 MiB and 320 MiB at batch 64.

Converted to something an engine would feel — one decode step runs this kernel
once per layer, so 28 times:

| Batch | Context | Attention per decode step |
| --: | --: | --: |
| 1 | 1,280 | 5.44 ms |
| 32 | 1,280 | 22.9 ms |
| 64 | 1,280 | 47.2 ms |

A single stream decoding against one max-resolution image pays 5.44 ms per step
in attention alone, a ceiling of roughly 184 tokens/s before any other layer is
counted. At batch 32 the same 22.9 ms is amortized across 32 sequences.

### Finding 4: GQA fusion has no headroom at serving batch sizes, because L2 already took it

Roadmap week 10 identifies fusing the query-head group so one K/V load serves
both query heads as a 2x arithmetic-intensity win, "available for free in the
loop structure." At the batch sizes measured here that win is already taken.

Every K/V vector is loaded twice, once per query head in the group. If the second
load hits L2, the measured L2 hit rate should be almost exactly one half:

| Config | L2 hit rate |
| --- | --: |
| batch 32, ctx 1,280 (160 MiB working set) | **50.04%** |
| batch 1, ctx 1,280 (5 MiB working set) | **50.05%** |

Combined with finding 3, that settles the question at this operating point. DRAM
is at 95% of peak while carrying only the unique traffic, so fusing the group
would remove L2 traffic and load instructions but not a single DRAM byte.

The general lesson is worth keeping: the arithmetic-intensity argument that
predicted 2x models DRAM and registers, and has no term for a 48 MiB L2 sitting
between them.

**A correction, and the limit of this finding.** The first version of this section
claimed the 50% held "regardless of whether the working set fits in L2," and
recorded fusion as ruled out. Both overreached. Two data points at the same batch
size cannot support a claim about all batch sizes, and sweeping the batch shows
the hit rate is conditional — see week 16, finding 1, where it collapses to 11%
at batch 512 and costs 1.8x. What holds is narrower: the duplicate load is free
whenever the two programs sharing a KV head are resident at the same time, which
is a property of the launch geometry rather than of the working set.

### Finding 5: the small-batch ceiling is occupancy, not per-program efficiency

Two candidate explanations for batch 1 reaching only 9% of peak. Either there are
too few programs to fill the machine, or each program has too little memory
parallelism — the block loop is load, reduce, load, reduce, and the reduction is a
barrier before the next load can issue. The first is fixed by partitioning the
context across programs; the second by software-pipelining the loop. They are very
different amounts of work, so the profile was run to choose between them.

| Config | Achieved occupancy | DRAM % of peak | Ratio |
| --- | --: | --: | --: |
| batch 1, ctx 1,280 | 8.33% | 11.38% | 1.37 |
| batch 32, ctx 1,280 | 71.88% | 94.98% | 1.32 |

The ratio is essentially constant across an 8.6x change in occupancy. **The kernel
converts residency into bandwidth at a fixed rate.** If resident warps were
stalling on a dependency chain, DRAM utilization would fall away from occupancy as
the batch shrank; instead it tracks it. Achieved occupancy at batch 1 is 8.33%
against a theoretical 83.33% — the gap is that 16 programs cannot fill 58 SMs.

So the fix is context partitioning, and loop pipelining is not the lever. The
pipelining hypothesis is recorded here as tested and rejected, because the
reasoning behind it was sound and it would otherwise be proposed again.

Extrapolating the same ratio, roughly 74% occupancy saturates DRAM, which batch 32
already nearly reaches. Partitioning therefore has headroom precisely where the
kernel is weakest — single-stream and small-batch decode — and close to none where
it is already strong.

### Verification status

| Check | Status |
| --- | --- |
| Reproduces the CPU golden vectors on all four fixtures (fp32) | pass, max abs diff 2.4e-07 |
| bf16 path against a float64 reference, scattered blocks | pass, max abs diff 9.2e-08 |
| Block addressing past the int32 boundary (frame 2,341 at the real block shape) | pass |
| Nsight profile, with the command that produced it | done, findings 3-5 |
| Bottleneck identified and confirmed to have moved | identified here, moved in week 16 |
| Every synchronization rule in `architecture.md` exercised | done in week 17, with two exceptions named there |

The int32 case is worth naming: `physical_id * elements_per_block` overflowed at
frame 2,341, and roadmap section 2 sizes a 4 GiB pool at exactly 2,340 blocks. A
pool filling this L4 would need over 12,000. The `.npz` fixtures cannot reach it — their
pools are 2 to 18 blocks — so `check_large_pool.py` builds the case on device.
Before the fix the kernel raised an illegal memory access; widening only that
product to int64 costs nothing measurable (19.00 us against 19.08 us median over
five interleaved repeats).

### Known limitations of these numbers

- **Decode only.** Prefill is a different regime with rising arithmetic intensity,
  and for a VLM a single image is a 1,280-token prefill. It is not measured here.
- **Uniform context length within a batch.** Real batches are ragged, and the
  grid is gated by its slowest program, so ragged batches will be worse than
  these figures. The `batched_head_dim_128` fixture covers ragged *correctness*;
  nothing here covers ragged performance.
- **KV is bf16 but query and output are fp32.** This keeps the score reduction in
  fp32 without changing the kernel. A bf16 query changes the numerics and has not
  been validated.
- **One layer measured.** Layer 13 of 28. All layers have identical geometry, so
  this is assumed not to matter rather than shown not to.
- **Launch overhead excluded.** `do_bench` times device-side spans. Python-side
  launch cost is ~21 us per call, which is 46% of the batch-1 kernel time, so an
  engine issuing 28 of these per step would need them pipelined to hide it.
- **The benchmark and the profiler measure different things.** Benchmark rows are
  algorithmic bytes over a cold L2; Nsight rows are counted DRAM bytes over a warm
  one. Percentages from the two are not directly comparable, which is why finding
  3 reconciles them explicitly rather than quoting whichever is higher.
- **The profile is one launch of one configuration.** Findings 3-5 rest on batch 1
  and batch 32 at 1,280 tokens, profiled after eight warmup launches. Nothing here
  profiles the ragged or small-context cases.
- **This is the unoptimized kernel.** No GQA fusion, no context partitioning, no
  tiling over query heads. Findings 4 and 5 are what decide which of those is
  worth writing; week 16 measures the result.
- **Batch is swept to 64 and no further.** Findings 3 and 4 read the top of that
  range as saturated and generalize from it. Week 16 finding 1 is what happens
  when the sweep is extended.

---

## Week 16: Two Optimizations Against The Week 15 Baseline

Same GPU, same pool, same block shape, same provenance table as week 15. Both
changes are in `tools/triton_kernel/paged_attention_decode.py`; the week 15
kernel is still callable as `paged_attention_decode_triton` and every speedup
below is measured against it in the same process.

Reproduce with:

```bash
.venv/bin/python tools/triton_kernel/run_fixture.py fixtures   # golden, both paths
cd tools/triton_kernel && ../../.venv/bin/python benchmark_partitioned.py \
    ../../results/week16-context-partition.csv
```

### Finding 1: the GQA duplicate load is free only while its two programs are co-resident

Week 15 finding 4 measured a 50% L2 hit rate at batch 1 and batch 32 and
concluded the duplicate K/V load never reaches DRAM. Extending the batch sweep
shows that conclusion has a boundary. Contexts are 256 tokens throughout:

| Batch | Occupancy | DRAM % of peak | L2 hit rate | Duration |
| --: | --: | --: | --: | --: |
| 8 | 18.31% | 59.36% | 50.36% | 54.66 us |
| 32 | 71.18% | 92.09% | 50.11% | 140.58 us |
| 64 | 73.51% | 91.77% | 49.53% | 291.26 us |
| 128 | 78.36% | 88.27% | 49.53% | 607.71 us |
| 256 | 81.62% | 93.59% | 48.37% | 1.17 ms |
| 512 | 82.32% | 96.30% | **10.86%** | 3.90 ms |

Doubling the work from batch 256 to batch 512 should double the time. It goes
1.17 to 3.90 ms, **1.67x worse than linear**, and DRAM is pinned at 96% while
delivering it.

The cause is launch order, not capacity. The grid was `(num_seqs,
num_query_heads)` with the sequence on the fast axis, so the `group_size` query
heads that share a KV head sat `num_seqs` apart in the linear block order. This
L4 keeps roughly 580 blocks resident. Below batch 512 the two partners overlap
in time and the second load hits L2; at 512 they land in different waves, the
first partner's lines are evicted before the second asks for them, and both go
to DRAM.

**The fix is to renumber the grid**, making the query head the fast axis so
partners are adjacent and always co-resident. It changes three lines, adds no
arithmetic, and does not reduce the number of programs:

| batch 512, ctx 256 | Before | After |
| --- | --: | --: |
| L2 hit rate | 11.16% | 47.91% |
| DRAM bytes moved | 1.12 GB (2.09x unique) | 622 MB (1.16x unique) |
| Duration | 3.88 ms | **2.16 ms** |

The 536.9 MB of unique traffic is the denominator; before the fix DRAM carried
almost exactly twice it, after the fix it carries the same 1.16x overhead that
week 15 finding 3 measured at batch 32. Batch 128 also improves, 594.75 to
560.45 us, and batch 32 is unchanged at 140.6 us. One oddity worth naming: the
hit *rate* at batch 128 reads slightly lower afterward (49.43% to 44.98%) while
the kernel gets faster, because the rate counts every L2 access including
queries and outputs. DRAM bytes are the metric that tracks the time.

**This is what GQA fusion would have bought, for less work.** Fusion also has a
cost the arithmetic-intensity argument does not model: one program serving the
whole group means half as many programs, which is actively harmful in the
small-batch regime finding 2 addresses. Renumbering recovers the same DRAM
traffic and leaves the grid alone.

Whether the collapse is reachable depends on the workload, because KV capacity
caps the batch. After ~4.4 GiB of bf16 weights, 17.6 GiB of this L4 is left:

| Context | Per sequence | Max batch | Reaches the collapse |
| --- | --: | --: | --- |
| 1,280 tokens (max-resolution image) | 140 MiB | ~128 | no |
| 256 tokens (min-resolution image or text) | 28 MiB | ~640 | yes |

So the max-resolution image workload never sees it, and week 15's conclusion
holds there. The short-context, high-concurrency workload does see it, and
there the fix is worth 1.8x.

### Finding 2: context partitioning is worth 4.57x at batch 1 and nothing at batch 32

Week 15 finding 5 established that the small-batch ceiling is residency: 16
programs cannot fill 58 SMs. The fix is to split each sequence's context across
programs and merge the partial softmaxes in a second pass, multiplying the grid
by the partition count.

Timings are `do_bench` with a cold L2, scattered block tables, best partition
count per row. The `split=1` column is the week 15 kernel measured in the same
process, and it reproduces the committed week 15 figures to within 0.5%:

| Batch | Context | Baseline | Partitioned | Split | Speedup | % of roof |
| --: | --: | --: | --: | --: | --: | --: |
| 1 | 1,280 | 196.0 us | **42.9 us** | 27 | **4.57x** | 9% -> 41% |
| 2 | 1,280 | 202.9 us | 69.7 us | 16 | 2.91x | 17% -> 50% |
| 4 | 1,280 | 249.7 us | 127.3 us | 8 | 1.96x | 28% -> 55% |
| 8 | 1,280 | 330.9 us | 241.5 us | 4 | 1.37x | 42% -> 58% |
| 16 | 1,280 | 486.5 us | 470.0 us | 2 | 1.04x | 57% -> 59% |
| 32 | 1,280 | 820.1 us | 820.1 us | 1 | 1.00x | 68% |
| 1 | 256 | 45.2 us | **17.5 us** | 16 | **2.59x** | 8% -> 20% |
| 2 | 256 | 50.3 us | 23.8 us | 16 | 2.12x | 14% -> 29% |
| 4 | 256 | 64.4 us | 36.1 us | 8 | 1.78x | 22% -> 39% |
| 8 | 256 | 91.5 us | 59.4 us | 4 | 1.54x | 31% -> 47% |
| 16 | 256 | 111.3 us | 104.8 us | 2 | 1.06x | 50% -> 53% |
| 32 | 256 | 194.0 us | 194.0 us | 1 | 1.00x | 58% |

The shape is what week 15 predicted: everything at small batch, nothing at large.
Past batch 16 the extra partitions cost more than they return, and the wrapper
falls back to the single-partition path — which is the week 15 kernel called
directly, so the large-batch regime pays nothing at all for this change, not even
a second launch.

In engine terms, one stream decoding against one max-resolution image goes from
5.49 ms of attention per step to 1.20 ms, lifting the attention-only ceiling from
roughly 182 to 833 tokens/s.

**Choosing the split.** `choose_num_partitions` aims for 512 programs, the point
where week 15's sweep saturated DRAM, and clamps to 32. Benchmarked against the
full ladder it picks the fastest rung in 12 of 14 configurations. It misses at
batch 4 / 1,280 tokens by 0.5%, and at batch 64 / 1,280 tokens by 4.4% — there it
returns 1 while split 8 is slightly faster, because at that size partitioning has
stopped being about occupancy and is buying a little extra L2 locality instead.
The heuristic is recorded as validated within those bounds rather than assumed.

**A host sync found by the measurement.** The first version derived the split
from `int(context_lens.max())`, which copies a device tensor to the host and
synchronizes on every call. It cost ~50 us per call — more than the entire
kernel at small batch — and it inflated *both* arms, so the ratios looked
plausible while every absolute number was wrong. The split now comes from
`block_table.shape[1]`, which is already sized to the longest sequence in the
batch and is a host-side value. The tell was that the `split=1` row disagreed
with the committed week 15 baseline by a constant offset; keeping the old kernel
callable is what made that visible.

### Finding 3: the bottleneck moved, and where it moved to

Week 15 argued the kernel converts residency into bandwidth at a fixed rate, so
raising occupancy should raise DRAM utilization proportionally. Hardware counters
after the change, one launch each:

| Config | Kernel | Occupancy | DRAM % of peak | Duration |
| --- | --- | --: | --: | --: |
| batch 1, ctx 1,280 | baseline | 8.33% | 11.27% | 178.69 us |
| | partitioned, split 27 | **57.69%** | **67.41%** | 29.41 us |
| | + reduce pass | 8.32% | 24.25% | 3.58 us |
| batch 2, ctx 1,280 | baseline | 8.33% | 22.19% | 180.29 us |
| | partitioned, split 16 | **67.98%** | **78.93%** | 50.18 us |
| | + reduce pass | 8.32% | 29.12% | 3.42 us |
| batch 1, ctx 256 | baseline | 8.33% | 10.64% | 37.95 us |
| | partitioned, split 16 | 34.83% | 38.66% | 10.27 us |
| | + reduce pass | 8.32% | 17.61% | 2.91 us |

Occupancy at batch 1 goes 8.33% to 57.69% and DRAM follows it to 67.41%. That is
the week 15 model used as a prediction and then confirmed by intervention, which
is stronger evidence than the correlation it was built on.

The constant did shift, though. Week 15 measured DRAM at 1.32-1.37x occupancy and
extrapolated that ~74% occupancy would saturate memory. The partitioned kernel
runs at 1.17x, so on its own curve saturation needs closer to 85% occupancy. The
direction was right and the arithmetic was optimistic, which is the expected cost
of extrapolating a two-point fit.

**The new bottleneck is the reduce pass.** It is a fixed 2.9-3.6 us running at
8.32% occupancy, because it launches only `num_seqs * num_query_heads` programs —
the exact residency problem partitioning just solved, one level up. At batch 1
with a 1,280-token context that is 11% of the total; at 256 tokens it is 22%, and
it is why the short-context speedup is 2.59x rather than 4.57x.

It is not, however, a bandwidth problem. The merge reads 225 KB at batch 1 with a
1,280-token context, which at peak DRAM would take 0.75 us against the 3.58 us it
costs, and the counters agree: 24% of peak. Most of that time is launch and ramp,
not work. Removing it entirely would be worth 1.12x at 1,280 tokens and 1.28x at
256 — real, but an order of magnitude smaller than the partitioning itself, and
CUDA graphs reach most of the same cost for less risk. An engine issuing 28 of
these per decode step needs them regardless.

**A correction.** An earlier version of this paragraph credited a fused,
semaphore-coordinated merge to vLLM. That is wrong in both directions: vLLM's
`paged_attention_v2` launches a separate `paged_attention_v2_reduce_kernel`, and
its Triton split-KV path added in PR #19152 launches a separate `reduce_segments`
kernel. The two-pass structure here matches the reference implementation rather
than lagging it, and fusing the merge would be going past it, not catching up.

### Verification status

| Check | Status |
| --- | --- |
| Partitioned path reproduces the CPU golden vectors, splits 2/3/8 | pass, max abs diff 2.4e-07 |
| Partitions holding no tokens are weighted to zero, not nan | pass, covered by split 8 on 1-3 block fixtures |
| bf16 partitioned path vs float64, 6 ragged sequences, splits 1-32 | pass, worst 2.7e-07 |
| Block addressing past the int32 boundary, after the grid change | pass |
| `split=1` reproduces the week 15 baseline | pass, within 0.5% |
| Bottleneck identified and confirmed to have moved | pass, finding 3 |
| Every synchronization rule in `architecture.md` exercised | done in week 17, with two exceptions named there |

### Known limitations of these numbers

- **Ragged batches are validated but not benchmarked.** Correctness covers six
  sequences from 1 to 300 tokens; every timing row is uniform. Partitioning
  should help ragged batches more than uniform ones, since the grid is gated by
  its slowest program, but that is an expectation and not a measurement.
- **The partition count is chosen from the longest sequence in the batch.** One
  long sequence therefore sets the split for every short one, which is the
  conservative direction but wastes programs on the short ones.
- **`MAX_PARTITIONS` is 32 for register-pressure reasons, not measured ones.**
  The reduce kernel holds a `[MAX_PARTITIONS, head_dim]` fp32 tile. Nothing here
  establishes 32 as the right cap; it is simply past the point where the sweep
  stops improving.
- **Finding 1's sweep is at 256 tokens only.** The residency boundary depends on
  blocks per SM, which context length does not change, so the collapse should
  appear at the same batch at any context. Not verified.
- **Still decode only, still one layer, still no end-to-end number.** All week 15
  limitations that are not addressed above continue to apply.

## Week 17: The Seam Between The Allocator And The Kernel

Weeks 15 and 16 measured a kernel; weeks 1-9 measured an allocator. Nothing had
ever run the two together. This closes that, and the interesting part is that
the two halves could not have been run together at all.

### Finding 1: the allocator's memory model could not back the kernel

`PhysicalBlock` allocated its own aligned host bytes, one `std::aligned_alloc`
per frame. So a `PhysicalBlockId` was a handle to an independent allocation and
nothing constrained where frames landed relative to each other.

The decode kernel resolves a block as `physical_id * elements_per_block` with no
pointer table. That is not an implementation detail to be worked around: it is
what makes the addressing free, and `check_large_pool.py` exists specifically to
test it at the int32 boundary. Against a per-frame allocator it is meaningless.
A device backend has the same problem one level up — it wants to mirror the pool
with a transfer, and `cudaMalloc` per frame would be neither contiguous nor
cheap.

The pool is now a single aligned slab with frame `i` bound to
`i * block_stride_bytes()`, and `PhysicalBlock` is a view over that range.
Switching to `cudaMalloc` is now one allocation site. `PhysicalBlock::swap` was
removed: it exchanged storage pointers between frames while ids stayed put,
which cannot hold once the id fixes the offset. Nothing outside its own test
called it.

Nothing observable changed: 143 tests passed before and after the refactor,
`golden_export` re-exported byte-identical fixtures, and the Triton kernel
reproduced all four of them unchanged.

**One mismatch this surfaced.** The frame stride is the block byte size rounded
up to the alignment, and for the small fixtures those differ — a 128-byte block
sits in a 256-byte stride. The `.npz` fixtures pack the pool *densely* at
`elements_per_block`, so the fixture layout and the live pool layout are not the
same thing. A device mirror must use the padded stride as the kernel's
`elements_per_block` and leave the padding unread. Harmless once known, and
silently wrong otherwise.

### Finding 2: every fixture was built by a clean fill, so paging was untested

All four existing fixtures populate a freshly zeroed pool exactly once. They
therefore validate the kernel against cache contents that no fork, no
copy-on-write, and no frame recycle ever touched — which is the half of a paged
allocator worth having. A fixture built that way still passes if a fork corrupts
its parent, if copy-on-write leaves a block table pointing at the pre-copy
frame, or if a recycled frame's stale bytes leak past the context length of a
partial block.

`cow_and_recycled_frames` is generated by driving the allocator instead: a
parent, a fork whose every block is materialized by copy-on-write, and a third
sequence built on frames recycled from a released one, ending mid-block so the
tail of its last frame still holds the dead sequence's KV. The pool is prefilled
with a deterministic non-zero pattern rather than zeroed, because a zeroed tail
perturbs only the softmax denominator while a garbage tail moves the output
somewhere unmistakable.

The exporter asserts its own premises — that the child shares no frame with the
parent, that the recycled sequence really did draw a released frame, that its
last block really is partial — because without them the scenario can decay into
three ordinary sequences and still pass under a weaker property than its name
claims. That check earned its place immediately: the free list is LIFO, so on
the first attempt the *parent* took the released frames and the recycled
sequence got fresh ones. The victim's lifetime now ends immediately before the
recycled sequence allocates.

The kernel passes it, at 8.9e-08 against the CPU oracle, on the whole-context
path and at splits 2, 3, and 8.

### Finding 3: four of the six synchronization rules were testable, and two are not

`tests/SynchronizationRules.test.cpp` gives each rule in `architecture.md` a test
named for it.

| Rule | Status |
| --- | --- |
| 1. Views are borrowed, not owned | pass — a resolved block pointer changes under a copy-on-write |
| 2. Copy-on-write precedes the enqueued write | pass — forked branches share one frame at refcount 2 until `ensure_token_writable` |
| 3. No frame recycled while a kernel reads it | **cannot be tested as written** |
| 4. Reference counts stay on the engine thread | **cannot be tested as written** |
| 5. Identity is stable, placement is not | pass — a recycled id keeps its bytes and advances its generation |
| 6. Swapped-out blocks have no frame | pass — `block_bytes` and `slot` return null and the kernel refuses the whole call |

Rule 3 obliges an engine to wait on a step's completion event before releasing a
frame, and nothing in the tree produces completion events. Rule 4 forbids
touching refcounts from a stream callback, which a single-threaded process
cannot observe. Both wait on there being an asynchronous backend; neither is
faked.

Rule 3 does get a test of the hazard it exists to prevent, because that hazard is
present today: `release` publishes a frame to the free list with no grace period,
and the very next `allocate` returns it. An async backend that released on the
host thread while a kernel was still reading would hand the same bytes to two
sequences.

Rule 5 is worth a note after finding 1. Placement is now fixed by construction —
an id always addresses the same slab offset — so identity no longer distinguishes
a live observation from a stale one. The id matches and the address matches, and
only `generation` reveals that the frame has been through the free list. The
counter went from redundant to load-bearing.

### Finding 4: the kernel now reads a pool the allocator is managing

Everything above still validated the kernel against a file. `cow_and_recycled_frames`
is a better file than the other four, but the C++ side froze a pool to disk and
Python read it back, so nothing established that the kernel could read a pool the
allocator was *actively* managing.

`python/bindings.cpp` closes that. Ownership runs one way: Python allocates the
slab, `MemoryAllocator` adopts the pointer, and a `keep_alive` makes the lifetime
contract something Python cannot violate. That keeps CUDA out of a core that is
still pure C++17 and lets the pool be a tensor the kernel takes directly, which
is the same split vLLM makes between bookkeeping and storage. The C++ build does
not learn about Python: the module is behind `QWENVL_BUILD_PYTHON`, off by
default.

`python/live_pool_check.py` then runs the seam with no fixture anywhere in it. It
forks a sequence, materializes the child by copy-on-write, releases a sequence
and hands its frames to another, calls the CPU reference kernel through the
bindings, mirrors the slab to the device, and runs Triton against the block
tables C++ produced.

```
  parent   frames [11, 10, 9]  context 40
  child    frames [8, 7, 6]  (all materialized by copy-on-write)
  victim   frames [5, 4, 3]  released
  recycled frames [3, 4]  context 17, 15 stale slots past its end

mirrored the slab to device in 1 transfer of 0.75 MiB

  pass  parent                   max abs diff 5.364e-07
  pass  child (copy-on-write)    max abs diff 3.576e-07
  pass  recycled (stale tail)    max abs diff 3.576e-07
```

The single transfer is the part finding 1 paid for. It is correct only because a
physical block id is an offset into one slab, so host frame `i` and device frame
`i` land in the same place. Against the per-frame allocations this allocator used
to make it would have had to be a gather of 12 separate copies, and the kernel's
`physical_id * elements_per_block` addressing would have had nothing to compute
against.

### Finding 5: a device-resident pool is blocked on three host dereferences

The obvious next move is to adopt a device tensor instead of a host one and drop
the mirror. That does not work yet, and the reason is worth recording because it
is not visible from the allocator's interface.

Three sites in the core dereference frame bytes on the host:

| Site | What breaks |
| --- | --- |
| `MemoryAllocator::copy_block` | `std::memcpy` — copy-on-write |
| `HostSwapBackend::store` / `load` | `std::memcpy` — swap out and back in |
| `CacheView::block_bytes`, read through by `paged_attention_decode` and `slot<T>` | the CPU reference kernel, which is the oracle everything is checked against |

Adopting device memory today would compile, link, and segfault on the first
fork. The allocator's *bookkeeping* is already device-ready — ids, refcounts,
free list, generations touch no bytes — so this is three call sites needing a
copy hook, not a redesign. It is also why the pool here is host memory mirrored
per run rather than device-resident.

### Verification status

| Check | Status |
| --- | --- |
| Every frame sits at `pool_base() + id * block_stride_bytes()` | pass |
| Frames do not overlap, across the whole pool | pass |
| Slab refactor changed nothing observable | pass, byte-identical fixtures and 143 tests |
| Kernel reads a fixture pool churned by fork, copy-on-write, and recycle | pass, 8.9e-08 |
| Exporter's churn premises hold rather than being assumed | pass, asserted in the exporter |
| Allocator adopts caller-owned storage at the stride it computes | pass |
| Kernel matches the CPU oracle on a live allocator-managed pool, no fixture | pass, 5.4e-07 |
| Whole pool mirrors to device in one transfer | pass, 0.75 MiB in 1 copy |
| Synchronization rules exercised | 4 of 6, with 3 and 4 named above |

Full suite: 153 tests pass. GPU fixture checks: 20 pass.

### Known limitations of these numbers

- **The pool is host memory mirrored to the device, not device-resident.** The
  seam is connected and correct, but a serving engine cannot copy the pool per
  step. Finding 5 names the three sites that stand between here and a device
  pool.
- **No timing claim is made about the bridge.** `live_pool_check.py` is a
  correctness harness. It writes KV element by element from Python, which is far
  slower than any real path, and nothing here measures call overhead across the
  binding.
- **The churn fixture is small.** Ten frames, 4-token blocks, head_dim 8. It is
  built to make paging bugs visible, not to be representative; the live check
  runs a larger shape and the real one is exercised by the benchmarks.
- **Copy-on-write is exercised, swap is only half-exercised.** Both the churn
  fixture and the live check cover fork, copy-on-write, release, and recycle.
  Swap-out and swap-in appear only in the rule 6 tests, never in anything the GPU
  kernel reads.
- **Rules 3 and 4 remain contractual.** They are the two that matter most once a
  backend is asynchronous, and they are the two with no test.
- **Still no model.** This connects the allocator to the kernel, not either to
  Qwen3-VL. There is still no end-to-end number and no token-identical gate.

## Week 18: A Real Model On The Paged Cache

Week 17 ended with the allocator and the kernel agreeing with each other and
neither of them attached to a model. This closes half of that: Qwen3-VL
greedy-decodes with its KV in allocator-managed blocks and produces the same
tokens it produces with transformers' own cache, on a two-layer random-weight
config and on the real 2B checkpoint with an image in the prompt.

The half it does not close is the kernel, which finding 4 covers: this path uses
torch attention over gathered blocks, so nothing here is an end-to-end
performance result.

Three gates: the first for every change, the second before believing it, the
third for the property paging exists for.

```bash
PYTHONPATH=build .venv/bin/python python/token_identical_check.py   # ~4 s
PYTHONPATH=build .venv/bin/python python/qwen3vl_2b_check.py        # ~10 s
PYTHONPATH=build .venv/bin/python python/fork_sharing_check.py      # ~20 s
```

### Finding 1: the integration point moved, and the roadmap's design is stale

`docs/roadmap-phase2.md` week 13 says to subclass the `Cache` interface in
`cache_utils.py` and override `update()`. That was right for transformers 4.x.
Version 5 restructured `Cache` into a container of per-layer objects, so `Cache`
itself is now plumbing and the thing to implement is a `CacheLayerMixin` — one
object per decoder layer, dispatched to by `layers[layer_idx].update(...)`.

The replacement fits the block layout better than the thing it replaced. A
physical block already spans every layer, so one sequence's block table backs all
layers at once and a `PagedLayer` is just a window onto its own layer's stride
within the same frames. The required surface is six methods, and the reference
implementation to work against, `DynamicLayer.update`, is two `torch.cat` calls.

This is the API the roadmap warned would move, and it moved. The pin in
`setup_gpu_box.sh` is now `transformers==5.16.1`, tightened deliberately: a
silent upgrade would not fail to import, it would fail the token-identical gate.

### Finding 2: `reserve_tokens` cannot fill a partial block, and the natural way to call it is wrong

The first working version allocated 25 blocks for 30 tokens. `reserve_tokens`
rounds up and always appends, so asking it for the single new token of a decode
step appends a whole new block and strands the fifteen slots already reserved in
the tail. Counting tokens is the obvious way to call it and it is wrong; the
capacity check has to count *blocks* and ask only for the ones the table is
missing.

This is documented, but only in a comment inside `tools/trace_replay`, which is
where the same trap was hit before. It is a sharp edge in the C++ API rather
than a bug in it: no test at the C++ level fails, because at that level nobody
grows a sequence one token at a time. Every incremental-decode caller will hit
it.

### Finding 3: the tokens are identical and the logits do not drift at all

| Check | Result |
| --- | --- |
| Tokens over 24 greedy steps | identical |
| Final-logit max abs difference | 0.0 |
| Blocks allocated for 61 tokens at 16/block | 4, as predicted |

Exact zero is the right answer here rather than a suspicious one. The paged path
stores the same fp32 bytes and gathers them back in the same order, so attention
receives a bit-identical tensor and every downstream float op is unchanged. Any
nonzero drift would mean bytes were being reordered or rounded somewhere, so the
tolerance is 0 and not an epsilon.

A gate that passes is only worth what it would catch, so it was checked against
a deliberate break. Frames are forced to be non-contiguous by a spacer sequence
that takes a frame between each of ours — the real table is `[63, 61, 59, 57]` —
and with the gather patched to walk frames linearly instead of following the
block table, the run diverges on the second token and fails. Without the spacer
the frames come out adjacent and that mutation would have passed.

### Finding 4: this path does not use the Triton kernel, and the gather says why it should

`update()` has to return contiguous K/V for torch's attention to consume, so
every step gathers the whole sequence out of its blocks into a dense tensor. The
Triton kernel is not in this path at all. What this validates is the memory
subsystem under a real model, which is worth having on its own, but it is not an
end-to-end performance result and no timing here should be read as one.

The gather's shape is the argument for finishing the job. It re-copies the
entire context, per layer, per step, so a generation costs O(context × layers)
per token and is quadratic in tokens overall. On the real 2B at a 1,242-token
prompt this is measured rather than projected:

| Quantity | Measured |
| --- | --- |
| Gathered per decode step | 137 MiB |
| Gathered over 20 steps | 2.67 GiB |
| Wall clock, 20 steps | 2.4 s paged against 1.7 s with `DynamicCache` |
| Implied cost per step | ~35 ms |

Roughly 35 ms per token of pure data movement, against a kernel that decodes
these shapes in tens of microseconds. The reference run went first and absorbed
CUDA warm-up, so the true gap is if anything a little wider than 0.7 s. Note
also that this is one sequence: the gather is per sequence, so batching does not
amortize it.

### Finding 5: the real 2B is token-identical too, image prompt included

The tiny model cannot speak to the three things that only exist in a real
checkpoint: 28 layers of trained weights, where one wrong KV byte compounds
instead of washing out into noise; bfloat16 instead of float32; and a prompt
whose bulk is image tokens produced by the vision tower and positioned by
M-RoPE. `python/qwen3vl_2b_check.py` runs Qwen3-VL-2B-Instruct on all three.

```bash
PYTHONPATH=build .venv/bin/python python/qwen3vl_2b_check.py
```

| Property | Value |
| --- | --- |
| Prompt | 1,242 tokens, 1,225 of them image |
| Geometry | 28 layers, 8 KV heads, head_dim 128, bf16 |
| Blocks | 79 frames of 1.75 MiB, spanning ids 5-161 |
| Result | identical text over 20 greedy steps |

The 1,242-token prompt is deliberate: it is the ~1,280-token image prefill the
block shape in week 15 was chosen against, so this exercises the geometry the
benchmarks assume rather than a convenient small one. Both runs go through the
same `model.generate` with the same arguments and differ only in the cache
object, because a hand-rolled decode loop that got M-RoPE subtly wrong would
break the comparison in a way that looks like a cache bug.

The same block-table mutation was applied here and the model emits newlines
instead of a description, diverging at the very first token.

Two implementation notes fell out of running a real checkpoint. The slab is now
`uint8` viewed at the model's dtype through torch rather than a typed numpy
array, because numpy has no bfloat16 and every real checkpoint here is stored in
it — the pool being bytes is also what C++ already believes. And the gather
concatenates on the host before crossing to the device, so the context makes one
transfer per stream per layer rather than one per block.

### Finding 6: four branches of a 1,242-token prompt cost 1.09 prompts, not 4

This is the case paging exists for, and the one the roadmap asks to have
measured. Sampling four continuations from one image prompt means four copies of
its KV in a dense cache. With block tables and copy-on-write the branches share
every prompt block and privately own only what they write.

```bash
PYTHONPATH=build .venv/bin/python python/fork_sharing_check.py
```

| Quantity | Paged | Four `DynamicCache`s |
| --- | --- | --- |
| KV held after 4 x 20 tokens | 150 MiB | 552 MiB |
| Frames in use | 86 | equivalent of 316 |
| | 77 shared, 9 owned outright | none shared |

3.67x less memory for the same four continuations. The frame accounting is
exactly what the design predicts and is asserted rather than eyeballed: 78 prompt
blocks shared by everyone, then two private blocks per branch — the partially
filled tail block, which copy-on-write duplicates the moment a branch writes its
first token into it, plus the one block each branch grows into. Nothing else is
copied, and the parent's block table is unchanged at the end.

The branches have to actually diverge or the test proves nothing, so each is
seeded with a different one of the top-4 next tokens:

```
branch 0: 'A vibrant, pixelated gradient that smoothly transitions through the full'
branch 1: 'This is a digital gradient image composed of a smooth transition of'
branch 2: 'The image displays a smooth, continuous gradient of colors that transitions'
branch 3: 'An abstract, pixelated gradient that transitions smoothly from deep blue'
```

Each is separately required to match what the same seed produces against a
private `DynamicCache`, so the sharing is not just cheap but correct.

The saving grows with prompt length and branch count, because the shared part is
the prompt and the private part is a constant two blocks per branch. It is also
a lower bound on what matters at serving scale, where the shared prefix is
commonly a system prompt across unrelated requests rather than one prompt across
its own samples.

One counter is easy to misread and worth naming: `AllocatorStats::active_blocks`
does not mean "frames in use". Active is a frame exactly one sequence holds and
Shared is one that several do, so a frame every branch reads is in the second
bucket. Frames in use is the sum, and reading the first alone reports 9 here
instead of 86.

### Verification status

| Check | Status |
| --- | --- |
| Tiny model: tokens identical to `DynamicCache` over 24 greedy steps | pass |
| Tiny model: final-logit drift | pass, exactly 0.0 |
| Tiny model: block count matches the context length | pass, 4 blocks for 61 tokens |
| Prefill spans multiple blocks, last block partially filled | pass, 37-token prefill over 3 blocks |
| Frames physically scattered, so the gather must follow the block table | pass, `[63, 61, 59, 57]` |
| Gate fails when the gather ignores the block table | pass, diverges at token 2 |
| Real Qwen3-VL-2B, bf16, 1,225 image tokens: token-identical | pass, 20 steps |
| Real model: gate fails when the gather ignores the block table | pass, diverges at token 0 |
| Four branches share the whole prompt at fork time | pass |
| Four branches diverge, and each matches its own `DynamicCache` run | pass |
| Parent's blocks survive the branches' writes | pass |
| Frame accounting after forking | pass, 86 as predicted |
| Week 17 checks still pass | pass, 153 tests, live pool check, 20 fixture checks |

### Known limitations of these numbers

- **Batch size 1 and greedy, per sequence.** Branches share a pool but are
  stepped one at a time rather than batched, and beam reordering is refused
  outright. Nothing here measures what batching the branches would cost or save.
- **Memory is compared as KV bytes held, not RSS.** The roadmap asks for an RSS
  delta, but the dense caches live in device memory while the paged pool is host
  memory, so process RSS would not compare them. Bytes of KV resident is the
  like-for-like counter; it is also the one that stays meaningful once the pool
  moves to the device.
- **The gather is the path, not the kernel.** No number in this section reflects
  the Triton kernel, because the Triton kernel does not run in it. The host-pool
  53 ms/step is the cost of *not* having it in the path; week 19 moves the pool
  to the device and measures what remains.
- **Twenty tokens, one prompt, one image.** Enough to catch a systematic paging
  error, which is what it is for. It is not a sweep, and nothing here varies
  context length, image size, or generation length.
- **No throughput or latency claim.** The wall-clock figures compare two
  correctness harnesses on one sequence. They bound the gather cost; they are not
  a serving measurement.

## Week 19: A Device-Resident Pool

Week 17 said a device pool was blocked on three host dereferences. Only one of
those three is on the serving path: `copy_block`, which copy-on-write uses. The
swap backend and the CPU reference kernel stay host-only; this path does not
call them. `set_copy_hook` replaces the memcpy with a caller-supplied
device-to-device copy, and the allocator adopts a CUDA tensor by address because
a device tensor has no buffer protocol.

```bash
PYTHONPATH=build .venv/bin/python python/qwen3vl_2b_check.py
PYTHONPATH=build .venv/bin/python python/fork_sharing_check.py
```

### Finding 1: the device pool is token-identical, and copy-on-write still works

Same 1,242-token image prompt, same 20 greedy steps, same four-way fork. Both
the host pool and the device pool match `DynamicCache` token for token. The
fork still shares 78 prompt blocks and privately owns two per branch; the
parent's table is unchanged. The only difference is that each first write now
goes through `frames[dst].copy_(frames[src])` instead of `std::memcpy`.

### Finding 2: moving the pool off PCIe recovered most of the 53 ms, and left 12 ms that is not bandwidth

| Path | 20-step wall clock | Overhead vs dense |
| --- | ---: | ---: |
| `DynamicCache` | 0.82 s | — |
| Host pool | 1.88 s | +53.3 ms/step |
| Device pool | 1.05 s | +11.6 ms/step |

The host number is the PCIe gather from week 18, now timed against a warmed
dense run rather than absorbing CUDA setup. The device number is the same
gather on the same side of the bus. 137 MiB at the L4's ~300 GB/s peak is
0.5 ms, so about 11 ms of the remaining 12 is not the copy. It is the Python
write loop, the per-step block-table rebuild, and torch attention over a freshly
assembled tensor — the work a native paged kernel skips by reading the frames
in place.

That is the argument for the next step, and it is now a measurement rather than
a projection: the device pool made the kernel reachable, and it did not make
the gather cheap enough to leave in the path.

### Verification status

| Check | Status |
| --- | --- |
| `copy_block` hook replaces the default memcpy | pass |
| Copy-on-write routes through the hook | pass |
| Host pool still token-identical on the 2B | pass |
| Device pool token-identical on the 2B | pass |
| Device pool faster than host pool | pass, 1.05 s vs 1.88 s |
| Four-way fork on the device pool, 86 frames as predicted | pass |
| Week 18 tiny-model gate and week 17 live-pool check | pass, 155 tests |

### Known limitations of these numbers

- **Swap and the CPU oracle still dereference host pointers.** Adopting a device
  slab and then calling `swap_out` or `KVCacheManager.decode` will fault. Those
  paths are unused here and are not claimed to work.
- **The by-address constructor cannot keep the tensor alive.** A CUDA tensor
  exposes no buffer protocol, so Python has to hold `PagedPool.storage`. That is
  documented on the binding and is the one place the header's lifetime contract
  is not mechanically enforced.
- **The 12 ms is not attributed to a single site.** The table above isolates
  PCIe from everything else; it does not split the remainder into write-loop vs
  gather vs torch attention. The kernel removes all three at once, which is why
  that split was not measured before writing it.
- **Still no kernel in the forward pass.** Torch attention still consumes a
  gathered tensor. Week 19 made that gather local; it did not remove it.

## Week 20: The Triton Kernel In The Forward Pass

The kernel is decode-only, so prefill still gathers and uses torch SDPA. Decode
writes the new token into its page, skips the gather, and a registered
`qwenvl_paged` attention implementation launches the partitioned kernel against
the slab and the block table. Vision layers never get a `_paged_layer` and fall
through, so the ViT is untouched.

```bash
PYTHONPATH=build .venv/bin/python python/token_identical_check.py
PYTHONPATH=build .venv/bin/python python/qwen3vl_2b_check.py
```

### Finding 1: decode through Triton is token-identical on the tiny model and on the 2B

The tiny random-weight config matches `DynamicCache` for 24 greedy steps. The
real 2B matches for 20 steps on a 1,242-token image prompt, including a
physically scattered block table. Prefill is still torch attention over a
gather; only the 20 decode steps go through Triton.

### Finding 2: putting the kernel in the path did not beat the gather, so the 11 ms is not the gather

| Path | 20-step wall clock | Overhead vs dense |
| --- | ---: | ---: |
| `DynamicCache` | 0.82 s | — |
| Host pool | 1.85 s | +51.3 ms/step |
| Device gather | 1.04 s | +11.1 ms/step |
| Device kernel | 1.05 s | +11.4 ms/step |

Week 19 left ~11 ms after PCIe was removed and guessed it was "write loop,
block-table rebuild, and torch attention over a gathered tensor," expecting the
kernel to remove all three. It removed the third. The number did not move. The
remaining 11 ms is therefore the shared Python path — `update()` writing the
new token, pybind into the allocator, building a block-table tensor, and
dispatching attention — not the bytes Triton vs torch attention move.

The kernel's own decode at this shape is tens of microseconds. 11 ms/step
across 28 layers is ~400 µs of host work per layer, which is also why CUDA
graphs and a tighter write path are the next levers, not another kernel tweak.

### Finding 3: four-way fork through the kernel flipped one token at step 18

The same four-branch prompt that is token-identical through a device gather
diverged on one continuation at decode step 18 when that continuation ran
through Triton. The flip is stable across reruns. The single-sequence 2B
generate does not show it. Fork sharing therefore stays on the gather path,
where it already gates copy-on-write against `DynamicCache`. Week 23
resolves this: the same seed flips with no fork, and gather/SDPA reports
tokens 504 and 5916 tied at 26.625.

### Verification status

| Check | Status |
| --- | --- |
| Tiny model, Triton decode token-identical | pass, 24 steps |
| 2B generate, host / device gather / device kernel token-identical | pass, 20 steps |
| Device pool still faster than host pool | pass |
| Four-way fork on the device gather path | pass, 86 frames |
| Four-way fork through the kernel vs `DynamicCache` | not a gate; one token flip |

### Known limitations of these numbers

- **Prefill is still a gather.** There is no Triton prefill kernel. The 1,242-token
  image prompt is attended with SDPA over assembled K/V, once.
- **The 11 ms was not the gather.** Stated above. A serving loop that still
  calls `update()` from Python per layer per token will not see the kernel's
  tens of microseconds.
- **Fork through the kernel is not gated.** Finding 3.
- **No CUDA graph, no batched decode.** One sequence, one token, 28 separate
  launches per step.

## Week 21: Where The 11 ms Actually Goes

The kernel and the gather cost the same 11 ms/step. This splits that number
so the next change is chosen from a table.

```bash
PYTHONPATH=build .venv/bin/python python/decode_breakdown.py
```

Prefill is excluded. Each decode step is timed with a CUDA sync around writing
the new token, assembling the block table, launching Triton, and gathering, on
the same 1,242-token image prompt as week 20.

### Finding 1: the 11 ms is three buckets of similar size, and the kernel replaced the wrong one

| Bucket | Device gather | Device kernel |
| --- | ---: | ---: |
| Write (page the new token) | 4.16 ms/step | 3.88 ms/step |
| Gather | 4.79 ms/step | — |
| Block table | — | 3.42 ms/step |
| Triton launch | — | 3.38 ms/step |
| Rest of the model | 30.44 ms/step | 29.33 ms/step |
| Wall | 39.39 ms/step | 40.02 ms/step |
| Versus dense (28.62 ms/step) | +10.77 | +11.40 |

The model body (`other`) matches the dense step, which is the sanity check:
the 11 ms is paged-only work. Write is ~4 ms on both paths and is the cost of
`update()` — pybind `ensure_token_writable` and a slice write, 28 times. Gather
was 4.8 ms. The kernel removed it and spent 3.4 ms building a block-table
tensor plus 3.4 ms launching, which is why the wall did not move.

Launch is 120 µs per layer, not the tens of microseconds the isolated kernel
benchmark reported. That gap is the per-call scratch allocations in the
partitioned wrapper plus two launches, 28 times.

### Finding 2: the cheapest next cut is the block table, not CUDA graphs

The block table is the same for every layer of a step. Today `attend_decode`
builds it from a Python list on every layer, so 28 identical host-to-device
copies account for 3.4 ms. Building it once per token deletes that bucket
without a graph and without touching the write path.

CUDA graphs would help the 3.4 ms of launches, and they would not help the
7.3 ms of write-plus-table that sit outside any captured region. Do the table
first; the graph question is clearer once that 3.4 ms is gone.

### Finding 3: caching the table dropped it from 3.4 ms to 0.3 ms

`decode_inputs` keeps the int32 table on the pool. A remapping write compares
against a host-side copy of the frame ids, so invalidation does not sync the
device. The context-length tensor is replaced when the length changes, not
filled in place, because a previous launch may still be reading it.

| Bucket | Kernel before | Kernel after |
| --- | ---: | ---: |
| Write | 3.88 ms/step | 3.98 ms/step |
| Table | 3.42 | 0.26 |
| Launch | 3.38 | 3.41 |
| Versus dense | +11.40 | +7.57 |

The wall moved by the table bucket, not by hope. Write and launch did not.
What remains is ~4 ms of `update()` and ~3.4 ms of per-layer launch; those
are the two questions left, and they are now the same size.

### Finding 4: reuse across layers cut write and launch, and the kernel finally beat the gather

The same rule as the table: do not redo per layer what is identical for the
step. `writable_frame` is cached for the current token, so 27 of 28 layers
skip the pybind into `ensure_token_writable`. The partitioned wrapper keeps
its scratch tensors on the pool, so 27 of 28 layers skip three allocations.

| | Kernel after table cache | After reuse |
| --- | ---: | ---: |
| Write | 3.98 ms/step | 3.12 |
| Table | 0.26 | 0.25 |
| Launch | 3.41 | 2.56 |
| Versus dense | +7.57 | **+3.92** |
| Versus gather | gather was cheaper | kernel **+3.92** vs gather **+5.71** |

Write is still 3.1 ms of real stores — permute and two slice writes, 28 times.
Launch is still 90 µs/layer of two kernel launches. CUDA graphs would capture
the second; they would not capture the first. The remaining 4 ms versus dense
is those two, in that order.

### Finding 5: a decode-only write cut 1.2 ms, and launch is now the larger leftover

`update()` no longer permutes a span and walks block boundaries for a single
token. It stores `[kv_heads, dim]` at `token % tokens_per_block` through two
flat `copy_`s, using bases cached on the layer.

| | After reuse | After decode write |
| --- | ---: | ---: |
| Write | 3.12 ms/step | 1.95 |
| Table | 0.25 | 0.26 |
| Launch | 2.56 | 2.64 |
| Versus dense | +3.92 | +3.92 |

The write bucket moved; the wall-versus-dense number did not, because dense
moved with it on this run (30.98 → 30.38). Absolute kernel wall was 34.90 →
34.30 ms/step. Launch is now the larger paged bucket, and it is the one a
CUDA graph can capture. What is left of write is 70 µs/layer of two small
device copies — a store kernel is the next lever there, not more Python.

### Finding 6: one launch is slower, so the 2.6 ms is occupancy, not a second launch

`num_partitions=1` in the forward path was the cheap test of "the reduce is
paying for a problem we no longer have." Launch went 2.64 → 6.97 ms/step, and
the kernel wall versus dense went +3.92 → +8.13. The isolated week 16 result
still holds inside the model: one program per head leaves the device idle, and
that idle time is larger than a second launch plus a reduce.

The change was reverted. Partitioning stays the default. A CUDA graph would
now have to capture the partitioned pair, not a single kernel, and it would
be capturing useful device work rather than empty launch overhead.

### Finding 7: of the launch bucket, 2.7 ms is GPU time and 0.5 ms is host dispatch

CUDA events around the partitioned pair, same 20-step decode:

| Split of launch | ms/step | ms/layer |
| --- | ---: | ---: |
| Device (both kernels) | 2.69 | 96 µs |
| Host dispatch | 0.54 | 19 µs |
| Launch wall | 3.23 | 115 µs |

A CUDA graph can take the 0.54 ms and nothing of the 2.69 ms. That is the
ceiling, and it is why graphs are not the next build: they are a large
project (fixed Q and `context_len` buffers, recapture when the table moves)
for a fifth of a millisecond per layer. The remaining paged cost that is
still worth a cheap cut is the 2.0 ms write, or accepting +4–6 ms versus
dense as the price of paging on this path.

### Verification status

| Check | Status |
| --- | --- |
| Dense / gather / kernel decode-only walls reproduce the week 20 gap | pass, +11 ms |
| `other` matches the dense step | pass |
| 560 decode updates on each paged path | pass |

### Known limitations of these numbers

- **Sync around every bucket.** That is what makes the split honest, and it
  also makes each bucket a lower bound on overlapped execution. A serving loop
  that did not sync per layer could hide some of the 120 µs launches behind
  the next layer's `update()`.
- **One prompt, 20 tokens, batch 1.** Same shape as week 20 on purpose.
- **No change to the kernel or the write path.** This file is a measurement.

---

## Week 22: Device Memory Of N Shared Prefixes

Batch-1 decode cannot show the throughput paging is supposed to raise. What
this GPU can still show is the number paging exists to change: how much device
memory N live prefixes occupy when they share a prompt versus when each
carries its own copy. That is the last measurement that needed this box.
CUDA graphs were not started; week 21 already put their ceiling at 0.54 ms
of host dispatch.

```bash
PYTHONPATH=build .venv/bin/python python/concurrency_mem.py
```

Same 1,242-token image prompt, same 20 greedy tokens, eight branches seeded
from the top-8 next tokens. Eight independent `DynamicCache`s are filled and
decoded first, then dropped; then one device pool is forked eight ways and
decoded the same way. `torch.cuda.memory_allocated()` above the loaded model
is the GPU number. KV bytes held is reported next to it so a driver that
caches activations differently still has a like-for-like counter.

### Finding 1: eight shared prefixes hold 6.7x less KV, and 3.3x less GPU memory

| | Dense (8 caches) | Paged (1 pool, 8 forks) |
| --- | ---: | ---: |
| Above the 3.99 GiB model | 1113 MiB | 339 MiB |
| KV bytes held | 1103 MiB | 164 MiB |
| Frames | equivalent of 632 | 94 (77 shared, 17 owned) |

KV 6.71x, GPU allocator 3.29x. The four-way gate in week 18 was 3.67x on KV
bytes; doubling the branches nearly doubled the ratio, which is what the
design predicts. The shared part is the prompt (77 blocks, 135 MiB) and the
private part is two blocks per branch plus the leftover parent tail.

### Finding 2: the GPU allocator ratio is smaller because the pool is reserved up front

The KV counter counts frames in use. `memory_allocated` counts the whole
slab, including unused frames reserved for growth, plus whatever CUDA kept
for the forwards. That is why 164 MiB of live KV sits inside 339 MiB above
the model, and why 3.29x is the honest serving number: a serving loop pays
for the pool it reserved, not for the frames it has touched so far.

The dense side is almost all KV (1113 vs 1103). There is no reserved slack
on that path; each cache is exactly as large as the sequence it holds.

### Finding 3: this is a memory result, not a throughput result

The eight branches still step one at a time. Nothing here batches them, so
nothing here can claim tokens per second. The claim is only that eight live
continuations of one image prompt fit in 339 MiB above the weights where
eight dense caches take 1113 MiB, and that the gap is the shared prompt.

### Verification status

Re-ran every GPU gate on this box after the measurement, including the
decode-only write's missing fields on a forked `PagedLayer` (the child
copied `is_initialized` but not the cached stream bases, and the first
decode store faulted).

| Check | Status |
| --- | --- |
| 8 paged prefixes use less device memory than 8 dense caches | pass, 339 vs 1113 MiB |
| KV bytes 6.71x, frames 94 as accounted | pass |
| Four-way fork still token-identical, 86 frames | pass |
| 2B host / gather / kernel token-identical over 20 steps | pass |
| Tiny-model gather and kernel gates | pass |
| Live allocator-managed pool vs CPU reference | pass, 5.4e-07 |
| Golden fixtures, including partitioned splits | pass |
| Decode-only breakdown still +5 ms vs dense, dispatch 0.52 ms | pass |

### Known limitations of these numbers

- **Batch size 1.** Eight prefixes share a pool and are decoded sequentially.
  There is still no fused-batch throughput number.
- **The pool is over-reserved.** Slack frames are in the GPU number on
  purpose. A tighter `max_blocks` would move 3.29x toward 6.71x and would
  also be closer to an OOM.
- **`memory_allocated` at rest, not peak during a forward.** The comparison
  is resident caches after the last decode step.
- **Twenty tokens, one prompt, one image.** Same shape as week 18 on
  purpose, so the n=4 and n=8 ratios sit on the same sequence.

---

## Week 23: The Fork+Kernel Flip Was A Logit Tie

Week 20 finding 3 left four-way fork through Triton ungated after one
continuation flipped at decode step 18. That was the last GPU correctness
question. It is not a fork bug.

```bash
PYTHONPATH=build .venv/bin/python python/fork_kernel_diag.py
```

### Finding 1: the same flip happens with no fork

Seed 785 (`The`), decoded on a single sequence with no `fork_paged_cache`,
disagrees with `DynamicCache` at step 18. Isolating that branch, and
running it after the other three have already copy-on-written, produce
the same tokens. Copy-on-write and table remapping are not in the causal
path.

Seeds 32, 1986, and 2082 (`A` / `This` / `An`) match through Triton on
the same hand-rolled decode loop. The greedy 2B generate gate never
leaves the `A...` continuation, which is why it stayed token-identical.

### Finding 2: gather/SDPA ties, the kernel breaks the tie by 0.125

At the flip, both paths rank the same two tokens first:

| Path | Token 5916 | Token 504 |
| --- | ---: | ---: |
| Gather / SDPA | 26.625 | 26.625 |
| Triton | 26.625 | 26.5 |

SDPA reports an exact tie; `argmax` takes the lower id (504). The kernel
is 0.125 lower on 504 — one bf16 step at this magnitude — and so emits
5916. That is online softmax versus SDPA, not a read of the wrong frame.
The CPU reference already agrees with Triton to ~1e-7 on fixtures; the
token-identity bar against torch SDPA is stricter than that arithmetic.

`bind_paged_layers` is still required after a fork (decode writes the
child, Triton reads whichever layer object is bound). That is an API
fact, not this flip.

### Verification status

| Check | Status |
| --- | --- |
| Seeds 0, 1, 3 token-identical through Triton | pass |
| Seed 2 flips only at step 18 on `{504, 5916}` | pass, gap 0.125 |
| Same flip with no fork | pass |
| Four-way fork on gather still token-identical | pass, `fork_sharing_check.py` |

### Known limitations of these numbers

- **One prompt, four seeds, 20 tokens.** A different near-tie would look
  the same and is accepted; a flip with a logit gap above 0.25 is a
  regression and fails the script.
- **No change to the kernel.** The gather path remains the fork
  token-identity gate.

---

## GPU Session Closed

This rented L4 session is done. Everything below can be done without a
GPU. Anything that would need another box is a new project, not a
leftover step of this one.

**Closed on this box**

- Triton decode kernel vs CPU fixtures, live allocator pool, and the 2B
- Device-resident pool and copy-on-write through `set_copy_hook`
- Kernel in the Hugging Face forward pass (decode only)
- Decode breakdown: write 2.0 ms, launch 3.2 ms of which 0.5 ms is host
  dispatch
- CUDA graphs: not started; ceiling is the 0.5 ms
- `num_partitions=1`: measured slower, reverted
- N-way prefix memory: 8 forks are 6.71× KV / 3.29× GPU vs 8 dense caches
- Fork+kernel flip: logit tie, recorded rather than "fixed"

**Does not need a GPU**

- Week 14 serving policy (trace replay, scheduler)
- CPU kernel work (weeks 11–12)
- Docs, the vLLM comparison, upstream
- Host swap / CPU reference kernel (already host-only by contract)

**Would be a new GPU session, not a leftover**

- A Triton prefill kernel (prefill is a one-shot gather and is
  token-identical)
- Fused batch in the Hugging Face generate path (the kernel already
  measures batch 32; the model path is batch-1 by contract)
- Quantized KV (roadmap cut-order item)
- Device-aware swap
- A store kernel for the remaining 2 ms write
- CUDA graphs for 0.5 ms of dispatch

