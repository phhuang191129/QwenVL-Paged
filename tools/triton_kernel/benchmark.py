#!/usr/bin/env python3
"""Benchmarks the Triton decode kernel at the real Qwen3-VL-2B block shape.

run_fixture.py proves the kernel correct on toy fixtures, which say nothing about
performance: those pools are 2 to 18 blocks and their grids are a dozen programs
on 58 SMs. This runs the geometry the model actually has -- 28 layers, 8 KV heads,
head_dim 128, 16 tokens per block, bf16, so 1.75 MiB per block, per
roadmap-phase2.md section 2.

Decode attention moves one K and one V vector per (query head, context token) and
does 4 x head_dim FLOPs on them, so its arithmetic intensity is around 1 FLOP/byte
at bf16. This L4's balance point is roughly 100, which puts the kernel two orders
of magnitude into the memory-bound regime. Achieved bandwidth against peak DRAM
is therefore the only metric reported; a GFLOP/s figure would describe a limit
that is nowhere near binding.

The bandwidth here is over the bytes the *algorithm* must move, timed with a cold
L2. Actual DRAM traffic is higher and a warm L2 makes the kernel faster, so these
figures read lower than the hardware-counter view in docs/performance.md. They
answer different questions and neither substitutes for the other.

The paging tax is measured by running the identical kernel over two block tables
into the same pool: one giving each sequence a consecutive run of frames, one
giving it a random permutation. Same frames, same arithmetic, same byte count --
locality is the only variable.

triton.testing.do_bench zeroes a flush buffer before every timed iteration, so
each measurement starts from a cold L2 and every row genuinely pulls its working
set from DRAM. The working set is still reported because it decides something
else: with 16 query heads over 8 kv heads, each K/V vector is loaded twice, and
whether the second load hits L2 within the same launch depends on whether the
working set fits this L4's 48 MiB.

Nothing is measured before it is validated. The fixtures are fp32 and this runs
bf16, so the bf16 path is checked against a float64 reference first.

Usage: benchmark.py [csv_path]
"""

import pathlib
import sys

import torch
import triton

from paged_attention_decode import paged_attention_decode_triton

# Real 2B geometry, roadmap-phase2.md section 2.
TOKENS_PER_BLOCK = 16
NUM_LAYERS = 28
NUM_KV_HEADS = 8
NUM_QUERY_HEADS = 16
HEAD_DIM = 128
LAYER = 13
DTYPE = torch.bfloat16

# KVBlockLayout's [layer][K|V][token_in_block][kv_head][head_dim], in elements.
HEAD_STRIDE = HEAD_DIM
TOKEN_STRIDE = NUM_KV_HEADS * HEAD_STRIDE
STREAM_STRIDE = TOKENS_PER_BLOCK * TOKEN_STRIDE
LAYER_STRIDE = 2 * STREAM_STRIDE
ELEMENTS_PER_BLOCK = NUM_LAYERS * LAYER_STRIDE

KEY, VALUE = 0, 1
SCALE = HEAD_DIM**-0.5

# 256 is a minimum-resolution image, 1,280 the documented clamp for a maximum-
# resolution one. Batch spans one request to a batch that fills the GPU.
SWEEP = [(batch, context) for context in (256, 1280) for batch in (1, 8, 32, 64)]

VALIDATE_SEQS, VALIDATE_CONTEXT = 2, 300
VALIDATE_TOLERANCE = 1e-3


def blocks_for(context_len: int) -> int:
    return (context_len + TOKENS_PER_BLOCK - 1) // TOKENS_PER_BLOCK


def stream_view(pool: torch.Tensor, physical: int, stream: int) -> torch.Tensor:
    """One block's [token, kv_head, head_dim] slice for one stream at LAYER."""
    start = LAYER * LAYER_STRIDE + stream * STREAM_STRIDE
    return pool[physical, start : start + STREAM_STRIDE].view(TOKENS_PER_BLOCK, NUM_KV_HEADS, HEAD_DIM)


def block_tables(num_seqs: int, context_len: int, generator: torch.Generator):
    """Consecutive and permuted mappings over the same set of frames.

    Both hand every sequence the same number of frames out of the same pool, so
    the kernel does identical work either way and only locality changes.
    """
    per_seq = blocks_for(context_len)
    consecutive = torch.arange(num_seqs * per_seq, device="cuda", dtype=torch.int32)
    scattered = torch.randperm(num_seqs * per_seq, device="cuda", generator=generator).to(torch.int32)
    return {
        "consecutive": consecutive.view(num_seqs, per_seq),
        "scattered": scattered.view(num_seqs, per_seq),
    }


def run(pool, block_table, context_lens, query):
    return paged_attention_decode_triton(
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
    )


def reference(pool, block_table, context_len, query) -> torch.Tensor:
    """Attention in float64 over the same bf16 values the kernel reads.

    Upcasting the stored values rather than regenerating them means bf16
    quantization cancels out, so any disagreement is the kernel's own arithmetic.
    """
    num_seqs = block_table.shape[0]
    group_size = NUM_QUERY_HEADS // NUM_KV_HEADS
    out = torch.empty(num_seqs, NUM_QUERY_HEADS, HEAD_DIM, device="cuda", dtype=torch.float64)

    for seq in range(num_seqs):
        frames = block_table[seq].tolist()
        keys = torch.cat([stream_view(pool, frame, KEY) for frame in frames])[:context_len].double()
        values = torch.cat([stream_view(pool, frame, VALUE) for frame in frames])[:context_len].double()
        for query_head in range(NUM_QUERY_HEADS):
            kv_head = query_head // group_size
            scores = (keys[:, kv_head] @ query[seq, query_head].double()) * SCALE
            out[seq, query_head] = torch.softmax(scores, dim=0) @ values[:, kv_head]

    return out


def validate(pool, generator) -> bool:
    """Checks the bf16 path, which the fp32 .npz fixtures never exercise."""
    table = block_tables(VALIDATE_SEQS, VALIDATE_CONTEXT, generator)["scattered"]
    context_lens = torch.full((VALIDATE_SEQS,), VALIDATE_CONTEXT, device="cuda", dtype=torch.int32)
    query = torch.randn(
        VALIDATE_SEQS, NUM_QUERY_HEADS, HEAD_DIM, device="cuda", dtype=torch.float32, generator=generator
    )

    actual = run(pool, table, context_lens, query).double()
    max_diff = float((actual - reference(pool, table, VALIDATE_CONTEXT, query)).abs().max())
    ok = max_diff <= VALIDATE_TOLERANCE
    print(
        f"{'pass' if ok else 'FAIL'}  bf16 vs float64 reference  "
        f"{VALIDATE_SEQS} seq x {VALIDATE_CONTEXT} tokens, scattered  max abs diff {max_diff:.3e}"
    )
    return ok


def peak_bandwidth(properties) -> float:
    """Peak DRAM bandwidth in GB/s from the memory clock and bus width.

    Measuring the roof with a torch-level copy understates it badly -- on an L4
    that returns 230 GB/s, and this kernel itself sustains 285 -- so a copy makes
    a poor denominator. This arithmetic agrees with Nsight Compute's
    peak-sustained figure to 0.1%.
    """
    return 2 * properties.memory_clock_rate * 1e3 * properties.memory_bus_width / 8 / 1e9


def copy_bandwidth() -> float:
    """What a plain device-to-device copy reaches, reported for reference only."""
    src = torch.randn(1 << 28, device="cuda", dtype=torch.float32)
    dst = torch.empty_like(src)
    ms = triton.testing.do_bench(lambda: dst.copy_(src), warmup=10, rep=100)
    bandwidth = 2 * src.numel() * 4 / (ms * 1e-3) / 1e9
    del src, dst
    torch.cuda.empty_cache()
    return bandwidth


def main() -> int:
    if not torch.cuda.is_available():
        print("no CUDA device visible", file=sys.stderr)
        return 1

    properties = torch.cuda.get_device_properties(0)
    max_blocks = max(batch * blocks_for(context) for batch, context in SWEEP)
    pool_bytes = max_blocks * ELEMENTS_PER_BLOCK * DTYPE.itemsize

    print(f"{properties.name}, {properties.multi_processor_count} SMs, {properties.total_memory / 2**30:.1f} GiB")
    print(f"block {ELEMENTS_PER_BLOCK * DTYPE.itemsize / 2**20:.2f} MiB, pool {max_blocks:,} blocks "
          f"({pool_bytes / 2**30:.2f} GiB)")

    roof = peak_bandwidth(properties)
    print(f"peak {roof:.0f} GB/s from {properties.memory_clock_rate / 1e6:.2f} GHz x "
          f"{properties.memory_bus_width}-bit; a plain copy reaches {copy_bandwidth():.0f} GB/s\n")

    generator = torch.Generator(device="cuda").manual_seed(0)
    pool = torch.randn(max_blocks, ELEMENTS_PER_BLOCK, device="cuda", dtype=DTYPE, generator=generator)

    if not validate(pool, generator):
        return 1
    print()

    header = (f"{'layout':12s} {'batch':>5s} {'ctx':>5s} {'programs':>9s} {'KV MiB':>7s} {'L2 fit':>7s} "
              f"{'us':>8s} {'GB/s':>8s} {'%roof':>6s} {'step ms':>8s}")
    print(header)
    print("-" * len(header))

    rows = []
    for batch, context in SWEEP:
        context_lens = torch.full((batch,), context, device="cuda", dtype=torch.int32)
        query = torch.randn(
            batch, NUM_QUERY_HEADS, HEAD_DIM, device="cuda", dtype=torch.float32, generator=generator
        )
        # Bytes any correct kernel must pull from DRAM: one K and one V vector
        # per (kv head, context token). The kernel currently issues twice this,
        # because the two query heads sharing a kv head load it separately.
        unique_bytes = batch * NUM_KV_HEADS * context * 2 * HEAD_DIM * DTYPE.itemsize
        l2_fit = "no" if unique_bytes > properties.L2_cache_size else "yes"

        for layout, table in block_tables(batch, context, generator).items():
            us = triton.testing.do_bench(lambda: run(pool, table, context_lens, query), warmup=25, rep=100) * 1000
            gbs = unique_bytes / (us * 1e-6) / 1e9
            # One decode step runs this kernel once per layer.
            step_ms = us * NUM_LAYERS / 1000
            print(f"{layout:12s} {batch:5d} {context:5d} {batch * NUM_QUERY_HEADS:9d} "
                  f"{unique_bytes / 2**20:7.0f} {l2_fit:>7s} "
                  f"{us:8.1f} {gbs:8.1f} {gbs / roof * 100:5.0f}% {step_ms:8.2f}")
            rows.append((layout, batch, context, batch * NUM_QUERY_HEADS, unique_bytes / 2**20,
                         l2_fit, us, gbs, gbs / roof * 100, step_ms))

    destination = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "results/week15-gpu-kernel.csv")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w") as csv:
        csv.write(f"# {properties.name}, {properties.multi_processor_count} SMs, "
                  f"L2 {properties.L2_cache_size / 2**20:.0f} MiB, peak DRAM {roof:.0f} GB/s\n")
        csv.write("# block 16 tokens x 28 layers x 8 kv heads x 128 dim x 2 B, 16 query heads, bf16\n")
        csv.write("layout,batch,context_len,programs,kv_mib,fits_l2,us_per_call,unique_gb_per_s,pct_of_roof,step_ms\n")
        for row in rows:
            csv.write("{},{},{},{},{:.0f},{},{:.2f},{:.2f},{:.1f},{:.3f}\n".format(*row))
    print(f"\nwrote {destination}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
