#!/usr/bin/env python3
"""Checks block addressing at a pool size the .npz fixtures cannot reach.

The golden fixtures run pools of 2 to 18 blocks, so they only ever exercise the
first few thousand elements of the address arithmetic. At the real Qwen3-VL-2B
block shape one block is 917,504 elements, so physical block 2,341 is the first
whose base offset does not fit in int32 -- and roadmap-phase2.md section 2 sizes
a 4 GiB pool at exactly 2,340 blocks. Any pool sized for a 23 GiB L4 is far past
that, so this is the first thing a realistic configuration hits.

Shipping this as an .npz would mean committing 8 GiB, so the case is built on the
device and checked against torch instead of against the CPU kernel. The sequence
deliberately straddles the boundary: its first logical block is addressable in
int32 and its second is not, so a kernel that wraps returns an answer that is
right for part of the context and wrong for the rest.

Usage: check_large_pool.py
"""

import sys

import torch

from paged_attention_decode import paged_attention_decode_triton

# The real 2B geometry from roadmap-phase2.md section 2.
TOKENS_PER_BLOCK = 16
NUM_LAYERS = 28
NUM_KV_HEADS = 8
NUM_QUERY_HEADS = 16
HEAD_DIM = 128
LAYER = 13  # Non-zero, so a dropped layer_stride term also shows up here.
CONTEXT_LEN = 24  # Two logical blocks, the second partially filled.

# KVBlockLayout's [layer][K|V][token_in_block][kv_head][head_dim], in elements.
HEAD_STRIDE = HEAD_DIM
TOKEN_STRIDE = NUM_KV_HEADS * HEAD_STRIDE
STREAM_STRIDE = TOKENS_PER_BLOCK * TOKEN_STRIDE
LAYER_STRIDE = 2 * STREAM_STRIDE
ELEMENTS_PER_BLOCK = NUM_LAYERS * LAYER_STRIDE

KEY, VALUE = 0, 1
TOLERANCE = 1e-4


def stream_view(pool: torch.Tensor, physical: int, stream: int) -> torch.Tensor:
    """Returns one block's [token, kv_head, head_dim] slice for one stream."""
    start = LAYER * LAYER_STRIDE + stream * STREAM_STRIDE
    region = pool[physical, start : start + STREAM_STRIDE]
    return region.view(TOKENS_PER_BLOCK, NUM_KV_HEADS, HEAD_DIM)


def main() -> int:
    if not torch.cuda.is_available():
        print("no CUDA device visible", file=sys.stderr)
        return 1

    last_safe_block = (2**31 - 1) // ELEMENTS_PER_BLOCK
    physical_ids = [last_safe_block, last_safe_block + 1]
    num_blocks = last_safe_block + 2
    pool_bytes = num_blocks * ELEMENTS_PER_BLOCK * 4

    print(f"elements per block   {ELEMENTS_PER_BLOCK:,}")
    print(f"last int32-safe block {last_safe_block:,}")
    print(f"pool                 {num_blocks:,} blocks, {pool_bytes / 2**30:.2f} GiB fp32")
    print(f"sequence maps to     {physical_ids}")

    free_bytes = torch.cuda.mem_get_info()[0]
    if free_bytes < pool_bytes * 1.05:
        print(f"needs {pool_bytes / 2**30:.1f} GiB, only {free_bytes / 2**30:.1f} GiB free", file=sys.stderr)
        return 1

    generator = torch.Generator(device="cuda").manual_seed(0)
    randn = lambda *shape: torch.randn(*shape, device="cuda", dtype=torch.float32, generator=generator)

    pool = torch.zeros(num_blocks, ELEMENTS_PER_BLOCK, device="cuda", dtype=torch.float32)
    for physical in physical_ids:
        stream_view(pool, physical, KEY).copy_(randn(TOKENS_PER_BLOCK, NUM_KV_HEADS, HEAD_DIM))
        stream_view(pool, physical, VALUE).copy_(randn(TOKENS_PER_BLOCK, NUM_KV_HEADS, HEAD_DIM))

    block_table = torch.tensor([physical_ids], device="cuda", dtype=torch.int32)
    context_lens = torch.tensor([CONTEXT_LEN], device="cuda", dtype=torch.int32)
    query = randn(1, NUM_QUERY_HEADS, HEAD_DIM)
    scale = HEAD_DIM**-0.5

    # Reference straight from the values written above, so it depends on the
    # block table only through which frames were filled.
    keys = torch.cat([stream_view(pool, physical, KEY) for physical in physical_ids])[:CONTEXT_LEN]
    values = torch.cat([stream_view(pool, physical, VALUE) for physical in physical_ids])[:CONTEXT_LEN]
    group_size = NUM_QUERY_HEADS // NUM_KV_HEADS
    expected = torch.empty_like(query[0], dtype=torch.float64)
    for query_head in range(NUM_QUERY_HEADS):
        kv_head = query_head // group_size
        scores = (keys[:, kv_head].double() @ query[0, query_head].double()) * scale
        weights = torch.softmax(scores, dim=0)
        expected[query_head] = weights @ values[:, kv_head].double()

    try:
        out = paged_attention_decode_triton(
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
            scale=scale,
        )
        torch.cuda.synchronize()
    except RuntimeError as error:
        # A wrapped offset can land outside the allocation entirely, which the
        # driver reports rather than quietly returning wrong numbers.
        print(f"FAIL  kernel raised {error}")
        return 1

    max_diff = float((out[0].double() - expected).abs().max())
    ok = max_diff <= TOLERANCE
    print(f"{'pass' if ok else 'FAIL'}  large pool  max abs diff {max_diff:.3e}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
