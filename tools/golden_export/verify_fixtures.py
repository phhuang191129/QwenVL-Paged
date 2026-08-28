#!/usr/bin/env python3
"""Reproduces the golden output from a fixture using only the fixture's arrays.

This is the acceptance gate for the export format, not a test of the CPU
kernel. Roadmap weeks 15-16 claim that KVBlockLayout's four strides plus the
flattened block table are everything an execution backend needs to read a paged
KV cache. This script holds itself to exactly that: it addresses the pool by
stride arithmetic and resolves logical tokens through the block table, with no
knowledge of how the C++ side laid the bytes out. If it can rebuild attention,
a Triton kernel has enough to work with, and the same arrays can validate it.

Run before renting a GPU, so paid hours go to the kernel rather than to finding
out the fixtures were unusable.

Usage: verify_fixtures.py [fixture_root]   (default: fixtures)
"""

import pathlib
import sys

import numpy as np

KEY, VALUE = 0, 1


def read_kv(fixture, physical_block: int, stream: int, token_in_block: int, kv_head: int):
    """Addresses one head vector by stride arithmetic alone."""
    offset = (
        int(fixture["layer"]) * int(fixture["layer_stride"])
        + stream * int(fixture["stream_stride"])
        + token_in_block * int(fixture["token_stride"])
        + kv_head * int(fixture["head_stride"])
    )
    head_dim = int(fixture["head_dim"])
    return fixture["kv_pool"][physical_block, offset : offset + head_dim]


def attend(fixture, sequence: int) -> np.ndarray:
    tokens_per_block = int(fixture["tokens_per_block"])
    num_kv_heads = int(fixture["num_kv_heads"])
    num_query_heads = int(fixture["num_query_heads"])
    head_dim = int(fixture["head_dim"])
    context_len = int(fixture["context_lens"][sequence])
    scale = float(fixture["scale"])
    group_size = num_query_heads // num_kv_heads

    out = np.zeros((num_query_heads, head_dim), dtype=np.float32)
    for query_head in range(num_query_heads):
        kv_head = query_head // group_size
        query = fixture["query"][sequence, query_head].astype(np.float64)

        keys = np.empty((context_len, head_dim), dtype=np.float64)
        values = np.empty((context_len, head_dim), dtype=np.float64)
        for token in range(context_len):
            physical = int(fixture["block_table"][sequence, token // tokens_per_block])
            if physical < 0:
                raise ValueError(f"sequence {sequence} token {token} maps to padding")
            token_in_block = token % tokens_per_block
            keys[token] = read_kv(fixture, physical, KEY, token_in_block, kv_head)
            values[token] = read_kv(fixture, physical, VALUE, token_in_block, kv_head)

        scores = (keys @ query) * scale
        weights = np.exp(scores - scores.max())
        out[query_head] = (weights @ values) / weights.sum()

    return out


def main() -> int:
    root = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "fixtures")
    paths = sorted(root.glob("*.npz"))
    if not paths:
        print(f"no fixtures under {root}", file=sys.stderr)
        return 1

    failures = 0
    for path in paths:
        with np.load(path) as fixture:
            worst = 0.0
            for sequence in range(len(fixture["context_lens"])):
                expected = fixture["output"][sequence]
                worst = max(worst, float(np.abs(attend(fixture, sequence) - expected).max()))

            # The golden output is float32 written by a kernel that accumulates
            # in float32, so agreement is bounded by float32 rounding rather
            # than by anything algorithmic.
            ok = worst <= 1e-5
            failures += 0 if ok else 1
            print(f"{'pass' if ok else 'FAIL'}  {path.name}  max abs diff {worst:.3e}")

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
