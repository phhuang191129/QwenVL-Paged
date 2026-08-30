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

### Verification status

| Check | Status |
| --- | --- |
| Every frame sits at `pool_base() + id * block_stride_bytes()` | pass |
| Frames do not overlap, across the whole pool | pass |
| Refactor changed nothing observable | pass, byte-identical fixtures and 143 tests |
| Kernel reads a pool churned by fork, copy-on-write, and recycle | pass, 8.9e-08 |
| Exporter's churn premises hold rather than being assumed | pass, asserted in the exporter |
| Synchronization rules exercised | 4 of 6, with 3 and 4 named above |

Full suite: 150 tests pass. GPU fixture checks: 20 pass.

### Known limitations of these numbers

- **The pool is still host memory.** The slab makes a device-backed allocator a
  one-site change, but that site has not been changed, and no kernel has yet run
  against memory the C++ allocator owns. The seam is proven layout-compatible,
  not yet connected.
- **The churn fixture is small.** Ten frames, 4-token blocks, head_dim 8. It is
  built to make paging bugs visible, not to be representative; the real block
  shape is exercised by `batched_head_dim_128` and the benchmarks.
- **Copy-on-write is exercised, swap is only half-exercised.** The churn fixture
  covers fork, copy-on-write, release, and recycle. Swap-out and swap-in appear
  only in the rule 6 tests, never in a fixture the GPU kernel reads.
- **Rules 3 and 4 remain contractual.** They are the two that matter most once a
  backend is asynchronous, and they are the two with no test.
