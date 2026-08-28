#!/usr/bin/env python3
"""Runs the Triton decode kernel against the golden fixtures and reports timing.

This is the first thing to run on the rented GPU box, before anything else:
correctness against fixtures/*.npz gates every later profiling claim, per
docs/roadmap-phase2.md's "the Triton kernel reproduces the CPU golden vectors"
verification requirement for weeks 15-16.

Usage: run_fixture.py [fixture_root]   (default: fixtures)
"""

import pathlib
import sys
import time

import numpy as np
import torch

from paged_attention_decode import paged_attention_decode_triton

TOLERANCE = 1e-3  # fp32 kernel vs. fp32 CPU reference; generous for softmax reordering.


def run_one(path: pathlib.Path) -> bool:
    with np.load(path) as fixture:
        kv_pool = torch.from_numpy(fixture["kv_pool"]).cuda()
        block_table = torch.from_numpy(fixture["block_table"]).cuda()
        context_lens = torch.from_numpy(fixture["context_lens"]).cuda()
        query = torch.from_numpy(fixture["query"]).cuda()
        expected = fixture["output"]

        kwargs = dict(
            layer=int(fixture["layer"]),
            layer_stride=int(fixture["layer_stride"]),
            stream_stride=int(fixture["stream_stride"]),
            token_stride=int(fixture["token_stride"]),
            head_stride=int(fixture["head_stride"]),
            tokens_per_block=int(fixture["tokens_per_block"]),
            num_kv_heads=int(fixture["num_kv_heads"]),
            scale=float(fixture["scale"]),
        )

    # Warmup: first call pays Triton's JIT compilation cost, which would
    # otherwise dominate the timing of a single-shot kernel this small.
    out = paged_attention_decode_triton(kv_pool, block_table, context_lens, query, **kwargs)
    torch.cuda.synchronize()

    iters = 200
    start = time.perf_counter()
    for _ in range(iters):
        out = paged_attention_decode_triton(kv_pool, block_table, context_lens, query, **kwargs)
    torch.cuda.synchronize()
    elapsed_us = (time.perf_counter() - start) / iters * 1e6

    actual = out.cpu().numpy()
    max_diff = float(np.abs(actual - expected).max())
    ok = max_diff <= TOLERANCE
    print(f"{'pass' if ok else 'FAIL'}  {path.name}  max abs diff {max_diff:.3e}  {elapsed_us:.1f} us/call")
    return ok


def main() -> int:
    if not torch.cuda.is_available():
        print("no CUDA device visible", file=sys.stderr)
        return 1

    root = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "fixtures")
    paths = sorted(root.glob("*.npz"))
    if not paths:
        print(f"no fixtures under {root}", file=sys.stderr)
        return 1

    results = [run_one(path) for path in paths]
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
