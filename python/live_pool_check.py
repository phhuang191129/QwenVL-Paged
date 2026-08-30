#!/usr/bin/env python3
"""Runs the Triton kernel against a pool the C++ allocator is managing.

Everything before this validated the kernel against `.npz` snapshots: the C++
exporter built a pool, froze it to disk, and Python read the file. That proves
the kernel agrees with the reference on a static array, which leaves the
question this script exists to answer -- whether the kernel can read a pool the
allocator is actively managing, with copy-on-write and frame recycling having
happened in the real C++ paths rather than being baked into a file.

So there is no fixture here. Python allocates the slab, `MemoryAllocator` adopts
the pointer, and the cache manager forks a sequence, materializes the child by
copy-on-write, releases a sequence and hands its frames to another. The CPU
reference kernel is then called through the same bindings, on the same live
pool, and the whole slab is mirrored to the device in a single transfer for the
Triton kernel to read with the block tables C++ produced.

The single transfer is the part worth watching. It is only correct because a
physical block id is an offset into one slab, so host frame `i` and device frame
`i` land at the same place; with the per-frame allocations this allocator used to
make it would have to be a gather, and the kernel's `physical_id *
elements_per_block` addressing would have no meaning at all.

Contexts are chosen so the last sequence ends mid-block, leaving the tail of its
final frame holding the released sequence's KV. Nothing zeroes the pool, so that
tail is real data a kernel ignoring `context_len` would read and be wrong about.

Usage:
    cmake -S . -B build -DQWENVL_BUILD_PYTHON=ON \
        -Dpybind11_DIR=$(.venv/bin/python -c 'import pybind11;print(pybind11.get_cmake_dir())') \
        -DPython_EXECUTABLE=$PWD/.venv/bin/python
    cmake --build build -j
    PYTHONPATH=build .venv/bin/python python/live_pool_check.py
"""

import pathlib
import sys

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "tools" / "triton_kernel"))

import qwenvl_paged as qp
from paged_attention_decode import paged_attention_decode_triton

TOKENS_PER_BLOCK = 16
NUM_LAYERS = 2
NUM_KV_HEADS = 2
HEAD_DIM = 128
NUM_QUERY_HEADS = 4
LAYER = 1
SCALE = 1.0 / np.sqrt(HEAD_DIM)

# Peak demand is parent 3 + child 3 + victim 3; the victim's frames are back on
# the free list before the recycled sequence claims two of them. The spares are
# never allocated and keep their prefill garbage.
POOL_BLOCKS = 12

# 40 tokens spans three blocks with the last one part-full; 17 spans two and
# leaves fifteen slots of the dead sequence's KV exposed past its context.
PARENT_CONTEXT = 40
RECYCLED_CONTEXT = 17

TOLERANCE = 1e-4


def make_config() -> qp.AllocatorConfig:
    config = qp.AllocatorConfig()
    config.block_shape.tokens_per_block = TOKENS_PER_BLOCK
    config.block_shape.num_layers = NUM_LAYERS
    config.block_shape.num_kv_heads = NUM_KV_HEADS
    config.block_shape.head_dim = HEAD_DIM
    config.block_shape.bytes_per_element = 4
    config.max_blocks = POOL_BLOCKS
    return config


def fill(cache, layout, pool, sequence, context_len, rng):
    """Writes a sequence's whole context, taking copy-on-write into account.

    The frame has to be re-resolved per token rather than cached per block:
    ensure_token_writable is the call that materializes a shared block, so the
    frame a token belongs to is not known until it is asked for.
    """
    for layer in range(NUM_LAYERS):
        for stream in (qp.KVStream.Key, qp.KVStream.Value):
            for token in range(context_len):
                frame = cache.ensure_token_writable(sequence, token)
                assert frame is not None, f"ensure_token_writable failed at token {token}"
                for head in range(NUM_KV_HEADS):
                    offset = layout.element_offset(layer, stream, token % TOKENS_PER_BLOCK, head)
                    assert offset is not None, "element_offset out of range"
                    pool[frame, offset:offset + HEAD_DIM] = rng.standard_normal(HEAD_DIM, dtype=np.float32)


def churn(cache, layout, pool, rng):
    """Drives the allocator through fork, copy-on-write, release, and reuse."""
    parent, child, victim, recycled = 1, 2, 100, 3

    assert cache.create_sequence(parent, parent)
    assert cache.reserve_tokens(parent, PARENT_CONTEXT)
    fill(cache, layout, pool, parent, PARENT_CONTEXT, rng)
    parent_frames = cache.block_table(parent)

    assert cache.fork_sequence(parent, child, child)
    shared = cache.block_table(child)
    assert shared == parent_frames, "the fork did not share the parent's frames"
    fill(cache, layout, pool, child, PARENT_CONTEXT, rng)

    # The free list is LIFO, so the victim has to die immediately before the
    # recycled sequence allocates or an earlier sequence takes its frames.
    assert cache.create_sequence(victim, victim)
    assert cache.reserve_tokens(victim, PARENT_CONTEXT)
    fill(cache, layout, pool, victim, PARENT_CONTEXT, rng)
    victim_frames = cache.block_table(victim)
    cache.release_sequence(victim)

    assert cache.create_sequence(recycled, recycled)
    assert cache.reserve_tokens(recycled, RECYCLED_CONTEXT)
    fill(cache, layout, pool, recycled, RECYCLED_CONTEXT, rng)

    child_frames = cache.block_table(child)
    recycled_frames = cache.block_table(recycled)

    assert not set(child_frames) & set(parent_frames), "copy-on-write did not materialize"
    assert set(recycled_frames) & set(victim_frames), "no frame was actually recycled"
    assert RECYCLED_CONTEXT % TOKENS_PER_BLOCK != 0, "the stale tail is not exposed"

    print(f"  parent   frames {parent_frames}  context {PARENT_CONTEXT}")
    print(f"  child    frames {child_frames}  (all materialized by copy-on-write)")
    print(f"  victim   frames {victim_frames}  released")
    print(f"  recycled frames {recycled_frames}  context {RECYCLED_CONTEXT}, "
          f"{TOKENS_PER_BLOCK - RECYCLED_CONTEXT % TOKENS_PER_BLOCK} stale slots past its end")

    return [(parent, PARENT_CONTEXT), (child, PARENT_CONTEXT), (recycled, RECYCLED_CONTEXT)]


def main() -> int:
    if not torch.cuda.is_available():
        print("no CUDA device visible", file=sys.stderr)
        return 1

    config = make_config()
    layout = qp.KVBlockLayout()
    layout.shape = config.block_shape

    stride_bytes = qp.block_stride_for(config)
    elements_per_block = stride_bytes // 4

    # Python owns the storage; the allocator only adopts the pointer. Prefilled
    # with noise rather than zeros so anything the kernel should not read is
    # data that would visibly move the answer.
    rng = np.random.default_rng(0)
    flat = rng.standard_normal(qp.pool_bytes_for(config) // 4, dtype=np.float32)
    pool = flat.reshape(POOL_BLOCKS, elements_per_block)

    allocator = qp.MemoryAllocator(config, flat)
    cache = qp.KVCacheManager(allocator)
    assert not allocator.owns_pool()

    print(f"pool {POOL_BLOCKS} frames x {stride_bytes // 1024} KiB, adopted from numpy "
          f"({allocator.pool_bytes() / 2**20:.2f} MiB)")
    sequences = churn(cache, layout, pool, rng)

    queries = rng.standard_normal((len(sequences), NUM_QUERY_HEADS, HEAD_DIM), dtype=np.float32)
    oracle = np.stack([
        cache.decode(sequence, queries[i], LAYER, NUM_QUERY_HEADS, context_len, SCALE)
        for i, (sequence, context_len) in enumerate(sequences)
    ])

    # One transfer of the whole slab. Correct only because frame i sits at
    # i * stride on both sides.
    device_pool = torch.from_numpy(pool).cuda()
    print(f"\nmirrored the slab to device in 1 transfer of {pool.nbytes / 2**20:.2f} MiB")

    max_blocks = max(len(cache.block_table(sequence)) for sequence, _ in sequences)
    table = np.full((len(sequences), max_blocks), -1, dtype=np.int32)
    for i, (sequence, _) in enumerate(sequences):
        frames = cache.block_table(sequence)
        table[i, :len(frames)] = frames

    actual = paged_attention_decode_triton(
        device_pool,
        torch.from_numpy(table).cuda(),
        torch.tensor([length for _, length in sequences], dtype=torch.int32, device="cuda"),
        torch.from_numpy(queries).cuda(),
        layer=LAYER,
        layer_stride=layout.layer_stride(),
        stream_stride=layout.stream_stride(),
        token_stride=layout.token_stride(),
        head_stride=layout.head_stride(),
        tokens_per_block=TOKENS_PER_BLOCK,
        num_kv_heads=NUM_KV_HEADS,
        scale=SCALE,
    ).cpu().numpy()

    print()
    worst = 0.0
    names = ("parent", "child (copy-on-write)", "recycled (stale tail)")
    for i, name in enumerate(names):
        diff = float(np.abs(actual[i] - oracle[i]).max())
        worst = max(worst, diff)
        print(f"  {'pass' if diff <= TOLERANCE else 'FAIL'}  {name:<24s} max abs diff {diff:.3e}")

    ok = worst <= TOLERANCE
    print(f"\n{'pass' if ok else 'FAIL'}  Triton kernel vs CPU reference on a live allocator-managed "
          f"pool, worst {worst:.3e}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
