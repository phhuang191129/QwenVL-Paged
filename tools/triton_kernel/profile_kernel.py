#!/usr/bin/env python3
"""Runs one benchmark configuration a handful of times, for a profiler to attach to.

benchmark.py issues thousands of launches through do_bench, and Nsight Compute
serially replays every launch it is asked to collect, so pointing it at that
script would take hours. This runs a single configuration a dozen times so
`ncu --launch-skip` can land on a steady-state launch, after Triton's JIT and the
first cold-cache pass are out of the way.

The geometry and the kernel call come from benchmark.py, so a profile always
describes the configuration the benchmark table reports.

Usage: profile_kernel.py <batch> <context_len> [consecutive|scattered] [num_partitions]

Omitting num_partitions, or passing 1, profiles the whole-context baseline.
Anything higher profiles the split-context path, which launches two kernels per
call, so --kernel-name has to match both and --launch-count doubles.

Collect with, for example:

    sudo /opt/nvidia/nsight-compute/2026.2.1/ncu \\
        --kernel-name regex:paged_attention_decode --launch-skip 8 --launch-count 1 \\
        --section SpeedOfLight --section MemoryWorkloadAnalysis \\
        --section Occupancy --section WarpStateStats \\
        .venv/bin/python tools/triton_kernel/profile_kernel.py 32 1280
"""

import sys

import torch

import benchmark
import benchmark_partitioned

LAUNCHES = 12


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__.strip().splitlines()[0], file=sys.stderr)
        return 1

    batch, context = int(sys.argv[1]), int(sys.argv[2])
    layout = sys.argv[3] if len(sys.argv) > 3 else "scattered"
    num_partitions = int(sys.argv[4]) if len(sys.argv) > 4 else 1

    generator = torch.Generator(device="cuda").manual_seed(0)
    pool = torch.randn(
        batch * benchmark.blocks_for(context),
        benchmark.ELEMENTS_PER_BLOCK,
        device="cuda",
        dtype=benchmark.DTYPE,
        generator=generator,
    )
    table = benchmark.block_tables(batch, context, generator)[layout]
    context_lens = torch.full((batch,), context, device="cuda", dtype=torch.int32)
    query = torch.randn(
        batch, benchmark.NUM_QUERY_HEADS, benchmark.HEAD_DIM, device="cuda", dtype=torch.float32, generator=generator
    )

    for _ in range(LAUNCHES):
        if num_partitions == 1:
            benchmark.run(pool, table, context_lens, query)
        else:
            benchmark_partitioned.run_partitioned(pool, table, context_lens, query, num_partitions)
    torch.cuda.synchronize()

    print(f"{LAUNCHES} launches: batch {batch}, context {context}, {layout}, split x{num_partitions}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
