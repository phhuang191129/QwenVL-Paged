#!/usr/bin/env python3
"""Compares the split-context decode kernel against the whole-context baseline.

benchmark.py measures the kernel that gives one program an entire sequence, and
found it reaches 95% of peak DRAM at batch 32 but only 9% at batch 1: sixteen
programs cannot fill 58 SMs. This measures the fix -- splitting each sequence's
context across programs and merging the partial softmaxes in a second pass -- at
the same block shape, against the same baseline, on the same pool.

The sweep runs partition counts as a fixed ladder rather than trusting
choose_num_partitions, because the point is to find out whether its heuristic
picks the right rung. Partitioning is not free: every extra partition adds a row
to the merge and another round trip through the scratch buffers, so there is a
count past which it stops paying. "auto" is reported alongside so the heuristic
can be read against the ladder it is guessing at.

Geometry, block tables, and the float64 oracle come from benchmark.py, so both
scripts describe the same kernel under the same conditions.

Contexts here are deliberately ragged. A uniform batch never produces a
partition that lies entirely past the end of a sequence, which is exactly the
case the merge has to weight to zero, so a uniform-only check would leave the
riskiest path in the new code unmeasured and unvalidated.

Usage: benchmark_partitioned.py [csv_path]
"""

import pathlib
import sys

import torch
import triton

import benchmark
from benchmark import (
    DTYPE,
    ELEMENTS_PER_BLOCK,
    HEAD_DIM,
    HEAD_STRIDE,
    KEY,
    LAYER,
    LAYER_STRIDE,
    NUM_KV_HEADS,
    NUM_LAYERS,
    NUM_QUERY_HEADS,
    SCALE,
    STREAM_STRIDE,
    TOKEN_STRIDE,
    TOKENS_PER_BLOCK,
    VALUE,
    blocks_for,
    peak_bandwidth,
    stream_view,
)
from paged_attention_decode import (
    MAX_PARTITIONS,
    choose_num_partitions,
    paged_attention_decode_partitioned,
    paged_attention_decode_triton,
)

# Batch is dense at the low end because that is the regime under test; 32 and 64
# are carried to confirm the change does not cost anything where the baseline is
# already saturating memory.
SWEEP = [(batch, context) for context in (256, 1280) for batch in (1, 2, 4, 8, 16, 32, 64)]
PARTITION_LADDER = (1, 2, 4, 8, 16, 32)

VALIDATE_SEQS, VALIDATE_CONTEXT = 6, 300
VALIDATE_TOLERANCE = 1e-3


def run_partitioned(pool, block_table, context_lens, query, num_partitions):
    return paged_attention_decode_partitioned(
        pool,
        block_table,
        context_lens,
        query,
        layer=LAYER,
        layer_stride=LAYER_STRIDE,
        stream_stride=STREAM_STRIDE,
        token_stride=TOKEN_STRIDE,
        head_stride=HEAD_STRIDE,
        tokens_per_block=TOKENS_PER_BLOCK,
        num_kv_heads=NUM_KV_HEADS,
        scale=SCALE,
        num_partitions=num_partitions,
    )


def effective_partitions(num_partitions: int, max_logical_blocks: int) -> int:
    """What the wrapper will actually launch, after it drops empty partitions."""
    num_partitions = max(1, min(num_partitions, MAX_PARTITIONS, max_logical_blocks))
    partition_blocks = -(-max_logical_blocks // num_partitions)
    return -(-max_logical_blocks // partition_blocks)


def ragged_reference(pool, block_table, context_lens, query) -> torch.Tensor:
    """Attention in float64 with a per-sequence context length."""
    group_size = NUM_QUERY_HEADS // NUM_KV_HEADS
    out = torch.empty(block_table.shape[0], NUM_QUERY_HEADS, HEAD_DIM, device="cuda", dtype=torch.float64)

    for seq, context_len in enumerate(context_lens.tolist()):
        frames = block_table[seq].tolist()
        keys = torch.cat([stream_view(pool, frame, KEY) for frame in frames])[:context_len].double()
        values = torch.cat([stream_view(pool, frame, VALUE) for frame in frames])[:context_len].double()
        for query_head in range(NUM_QUERY_HEADS):
            kv_head = query_head // group_size
            scores = (keys[:, kv_head] @ query[seq, query_head].double()) * SCALE
            out[seq, query_head] = torch.softmax(scores, dim=0) @ values[:, kv_head]

    return out


def validate(pool, generator) -> bool:
    """Checks every partition count on ragged contexts before anything is timed.

    The shortest sequence spans one logical block while the longest spans all of
    them, so at the top of the ladder most of that sequence's partitions hold no
    tokens at all.
    """
    table = benchmark.block_tables(VALIDATE_SEQS, VALIDATE_CONTEXT, generator)["scattered"]
    lengths = torch.tensor([1, 16, 17, 100, 299, 300], device="cuda", dtype=torch.int32)
    query = torch.randn(
        VALIDATE_SEQS, NUM_QUERY_HEADS, HEAD_DIM, device="cuda", dtype=torch.float32, generator=generator
    )
    expected = ragged_reference(pool, table, lengths, query)

    baseline = paged_attention_decode_triton(
        pool,
        table,
        lengths,
        query,
        layer=LAYER,
        layer_stride=LAYER_STRIDE,
        stream_stride=STREAM_STRIDE,
        token_stride=TOKEN_STRIDE,
        head_stride=HEAD_STRIDE,
        tokens_per_block=TOKENS_PER_BLOCK,
        num_kv_heads=NUM_KV_HEADS,
        scale=SCALE,
    )
    worst = float((baseline.double() - expected).abs().max())
    print(f"  baseline           max abs diff {worst:.3e}")

    for num_partitions in PARTITION_LADDER:
        actual = run_partitioned(pool, table, lengths, query, num_partitions)
        diff = float((actual.double() - expected).abs().max())
        worst = max(worst, diff)
        print(f"  split x{num_partitions:<2d}           max abs diff {diff:.3e}")

    ok = worst <= VALIDATE_TOLERANCE
    print(f"{'pass' if ok else 'FAIL'}  bf16 vs float64, {VALIDATE_SEQS} ragged seqs, scattered  "
          f"worst {worst:.3e}\n")
    return ok


def main() -> int:
    if not torch.cuda.is_available():
        print("no CUDA device visible", file=sys.stderr)
        return 1

    properties = torch.cuda.get_device_properties(0)
    roof = peak_bandwidth(properties)
    max_blocks = max(batch * blocks_for(context) for batch, context in SWEEP)

    print(f"{properties.name}, {properties.multi_processor_count} SMs, peak DRAM {roof:.0f} GB/s")
    print(f"pool {max_blocks:,} blocks ({max_blocks * ELEMENTS_PER_BLOCK * DTYPE.itemsize / 2**30:.2f} GiB)\n")

    generator = torch.Generator(device="cuda").manual_seed(0)
    pool = torch.randn(max_blocks, ELEMENTS_PER_BLOCK, device="cuda", dtype=DTYPE, generator=generator)

    if not validate(pool, generator):
        return 1

    header = (f"{'batch':>5s} {'ctx':>5s} {'split':>5s} {'programs':>9s} {'us':>8s} "
              f"{'GB/s':>7s} {'%roof':>6s} {'vs base':>8s} {'step ms':>8s} {'auto':>5s}")
    print(header)
    print("-" * len(header))

    rows = []
    for batch, context in SWEEP:
        per_seq_blocks = blocks_for(context)
        table = benchmark.block_tables(batch, context, generator)["scattered"]
        context_lens = torch.full((batch,), context, device="cuda", dtype=torch.int32)
        query = torch.randn(
            batch, NUM_QUERY_HEADS, HEAD_DIM, device="cuda", dtype=torch.float32, generator=generator
        )
        unique_bytes = batch * NUM_KV_HEADS * context * 2 * HEAD_DIM * DTYPE.itemsize
        auto = effective_partitions(
            choose_num_partitions(batch, NUM_QUERY_HEADS, per_seq_blocks), per_seq_blocks
        )

        baseline_us = None
        seen = set()
        for requested in PARTITION_LADDER:
            split = effective_partitions(requested, per_seq_blocks)
            if split in seen:
                continue
            seen.add(split)

            us = triton.testing.do_bench(
                lambda: run_partitioned(pool, table, context_lens, query, split), warmup=25, rep=100
            ) * 1000
            if split == 1:
                baseline_us = us
            gbs = unique_bytes / (us * 1e-6) / 1e9
            speedup = baseline_us / us if baseline_us else float("nan")
            print(f"{batch:5d} {context:5d} {split:5d} {batch * NUM_QUERY_HEADS * split:9d} {us:8.1f} "
                  f"{gbs:7.1f} {gbs / roof * 100:5.0f}% {speedup:7.2f}x {us * NUM_LAYERS / 1000:8.2f} "
                  f"{'<--' if split == auto else '':>5s}")
            rows.append((batch, context, split, batch * NUM_QUERY_HEADS * split, us, gbs,
                         gbs / roof * 100, speedup, us * NUM_LAYERS / 1000, int(split == auto)))
        print()

    destination = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "results/week16-context-partition.csv")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w") as csv:
        csv.write(f"# {properties.name}, {properties.multi_processor_count} SMs, "
                  f"L2 {properties.L2_cache_size / 2**20:.0f} MiB, peak DRAM {roof:.0f} GB/s\n")
        csv.write("# block 16 tokens x 28 layers x 8 kv heads x 128 dim x 2 B, 16 query heads, bf16, "
                  "scattered block table\n")
        csv.write("# speedup is against split=1, the whole-context baseline, in the same row group\n")
        csv.write("batch,context_len,partitions,programs,us_per_call,unique_gb_per_s,pct_of_roof,"
                  "speedup_vs_baseline,step_ms,chosen_by_heuristic\n")
        for row in rows:
            csv.write("{},{},{},{},{:.2f},{:.2f},{:.1f},{:.3f},{:.3f},{}\n".format(*row))
    print(f"wrote {destination}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
