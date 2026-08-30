#!/usr/bin/env python3
"""GPU memory of N continuations: one paged pool versus N dense caches.

This is the number paging exists to change, and it is the last measurement
that needs this GPU. Batch-1 decode cannot show throughput; what it can show
is how much device memory N live prefixes occupy when they share a prompt
versus when each carries its own copy.

N independent DynamicCaches are filled and decoded first, then dropped, then
one pool is forked N ways and decoded the same way. Peak allocator bytes
above the loaded model are the comparison. KV-bytes held is reported
alongside so a driver that caches activations differently still has a
like-for-like counter.

Usage:
    PYTHONPATH=build .venv/bin/python python/concurrency_mem.py
"""

import pathlib
import sys

import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from transformers.cache_utils import Cache, DynamicLayer

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from paged_cache import DEFAULT_TOKENS_PER_BLOCK, build_paged_cache, fork_paged_cache
from fork_sharing_check import decode_from, dense_bytes, prefill
from qwen3vl_2b_check import DTYPE, MODEL_ID, NEW_TOKENS, build_inputs

BRANCHES = 8


def gpu_bytes():
    torch.cuda.synchronize()
    return torch.cuda.memory_allocated()


def main() -> int:
    if not torch.cuda.is_available():
        print("no CUDA device visible", file=sys.stderr)
        return 1

    torch.cuda.reset_peak_memory_stats()
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = Qwen3VLForConditionalGeneration.from_pretrained(MODEL_ID, dtype=DTYPE).cuda().eval()
    text = model.config.text_config
    inputs = build_inputs(processor).to("cuda")
    prompt = inputs["input_ids"].shape[1]
    after_model = gpu_bytes()

    print(f"{MODEL_ID}: prompt {prompt}, {BRANCHES} branches x {NEW_TOKENS} tokens")
    print(f"model   {after_model / 2**30:.2f} GiB allocated")

    scratch = Cache(layers=[DynamicLayer() for _ in range(text.num_hidden_layers)])
    seeds = prefill(model, inputs, scratch).topk(BRANCHES).indices.tolist()
    del scratch
    torch.cuda.empty_cache()

    dense_caches = []
    for seed in seeds:
        cache = Cache(layers=[DynamicLayer() for _ in range(text.num_hidden_layers)])
        prefill(model, inputs, cache)
        decode_from(model, cache, seed)
        dense_caches.append(cache)
    dense_gpu = gpu_bytes() - after_model
    dense_kv = sum(dense_bytes(cache) for cache in dense_caches)
    print(f"dense   {dense_gpu / 2**20:.0f} MiB above the model, "
          f"{dense_kv / 2**20:.0f} MiB of KV in {BRANCHES} caches")

    del dense_caches
    torch.cuda.empty_cache()

    prompt_blocks = -(-prompt // DEFAULT_TOKENS_PER_BLOCK)
    pool_blocks = prompt_blocks + BRANCHES * (2 + NEW_TOKENS // DEFAULT_TOKENS_PER_BLOCK) + 8
    paged, pool = build_paged_cache(text, max_blocks=pool_blocks, dtype=DTYPE, device="cuda")
    prefill(model, inputs, paged)
    for index, seed in enumerate(seeds):
        branch = fork_paged_cache(paged, pool, child_id=100 + index)
        decode_from(model, branch, seed)
    paged_gpu = gpu_bytes() - after_model
    paged_kv = pool.frames_in_use() * pool.allocator.block_stride_bytes()
    print(f"paged   {paged_gpu / 2**20:.0f} MiB above the model, "
          f"{paged_kv / 2**20:.0f} MiB of KV in {pool.frames_in_use()} frames "
          f"({pool.allocator.stats().shared_blocks} shared)")

    kv_ratio = dense_kv / paged_kv if paged_kv else 0
    gpu_ratio = dense_gpu / paged_gpu if paged_gpu else 0
    print(f"saving  KV {kv_ratio:.2f}x, GPU allocator {gpu_ratio:.2f}x")

    ok = paged_kv < dense_kv and paged_gpu < dense_gpu
    print(f"\n{'pass' if ok else 'FAIL'}  {BRANCHES} shared prefixes use less device memory "
          f"than {BRANCHES} dense caches")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
