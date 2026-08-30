#!/usr/bin/env python3
"""Four continuations of one image prompt, sharing the prompt's blocks.

This is the case paging exists for. Sampling n continuations from one prompt
means n copies of the prompt's KV in a dense cache, and the prompt here is 1,242
tokens of mostly image. With block tables and copy-on-write the branches share
every prompt block and privately own only what they write, which for a decode is
the partially filled tail block and whatever they grow into.

The branches have to actually diverge or the test proves nothing, so each is
seeded with a different one of the top-4 next tokens and then decoded greedily.
Correctness is per branch: the same seed run against a private `DynamicCache`
must produce the same continuation.

Both sides run the same hand-written decode loop, which matters. The loop leans
on the model to carry M-RoPE state across steps, and if it got that wrong the
run would still be a fair comparison -- the error would land identically on both
sides -- but the tokens would not match `generate`. `qwen3vl_2b_check.py` is the
gate that pins the cache against `generate`; this one pins forking against a
dense cache.

Frames are not scattered here. The other two gates cover that, and a spacer
sequence would sit in the middle of the frame accounting this script exists to
report. The pool is device-resident, so each first write goes through the copy
hook rather than a host memcpy -- the path a serving loop would actually take.

Usage:
    PYTHONPATH=build .venv/bin/python python/fork_sharing_check.py
"""

import pathlib
import sys

import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from transformers.cache_utils import Cache, DynamicLayer

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from paged_cache import DEFAULT_TOKENS_PER_BLOCK, build_paged_cache, fork_paged_cache
from qwen3vl_2b_check import DTYPE, MODEL_ID, build_inputs

BRANCHES = 4
NEW_TOKENS = 20


def prefill(model, inputs, cache):
    with torch.no_grad():
        return model(**inputs, past_key_values=cache, use_cache=True).logits[0, -1]


def decode_from(model, cache, seed):
    """Feeds `seed` into a filled cache and greedily continues from there."""
    tokens = [seed]
    token = torch.tensor([[seed]], device="cuda")
    with torch.no_grad():
        for _ in range(NEW_TOKENS - 1):
            logits = model(input_ids=token, past_key_values=cache, use_cache=True).logits[:, -1]
            token = logits.argmax(-1, keepdim=True)
            tokens.append(int(token))
    return tokens


def dense_bytes(cache):
    return sum(layer.keys.numel() + layer.values.numel() for layer in cache.layers) * DTYPE.itemsize


def main() -> int:
    if not torch.cuda.is_available():
        print("no CUDA device visible", file=sys.stderr)
        return 1

    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = Qwen3VLForConditionalGeneration.from_pretrained(MODEL_ID, dtype=DTYPE).cuda().eval()
    text = model.config.text_config

    inputs = build_inputs(processor).to("cuda")
    prompt_tokens = inputs["input_ids"].shape[1]
    prompt_blocks = -(-prompt_tokens // DEFAULT_TOKENS_PER_BLOCK)

    print(f"{MODEL_ID}: prompt {prompt_tokens} tokens, {BRANCHES} branches "
          f"x {NEW_TOKENS} tokens")

    # One prefill decides the branch seeds, so every branch starts from the same
    # distribution and differs only in which of the top-4 tokens it took.
    scratch = Cache(layers=[DynamicLayer() for _ in range(text.num_hidden_layers)])
    seeds = prefill(model, inputs, scratch).topk(BRANCHES).indices.tolist()
    print(f"seeds   {seeds} -> {[processor.decode([s]) for s in seeds]}")

    reference, reference_caches = [], []
    for seed in seeds:
        cache = Cache(layers=[DynamicLayer() for _ in range(text.num_hidden_layers)])
        prefill(model, inputs, cache)
        reference.append(decode_from(model, cache, seed))
        reference_caches.append(cache)

    pool_blocks = prompt_blocks + BRANCHES * (2 + NEW_TOKENS // DEFAULT_TOKENS_PER_BLOCK) + 8
    # Device-resident: this is the path that has to go through the copy hook
    # rather than a host memcpy, and it is the one a serving loop would use.
    paged, pool = build_paged_cache(text, max_blocks=pool_blocks, dtype=DTYPE, device="cuda")
    prefill(model, inputs, paged)
    prompt_frames = pool.block_table(pool.root_id)

    actual, shared_at_fork = [], []
    for index, seed in enumerate(seeds):
        branch = fork_paged_cache(paged, pool, child_id=100 + index)
        shared_at_fork.append(pool.block_table(100 + index) == prompt_frames)
        actual.append(decode_from(model, branch, seed))

    frames = pool.frames_in_use()
    stats = pool.allocator.stats()
    paged_bytes = frames * pool.allocator.block_stride_bytes()
    dense = sum(dense_bytes(cache) for cache in reference_caches)

    # Every branch shares all but the tail block, then privately owns that tail
    # plus whatever it grows into.
    private = [f for index in range(BRANCHES)
               for f in pool.block_table(100 + index) if f not in prompt_frames]
    expected_frames = prompt_blocks + len(private)

    print(f"prompt  {prompt_blocks} blocks, shared by all {BRANCHES} branches")
    print(f"private {len(private)} blocks total, {len(private) // BRANCHES} per branch "
          f"(the tail block copied on write, plus growth)")
    print(f"frames  {frames} in use: {stats.shared_blocks} shared by more than one "
          f"branch, {stats.active_blocks} owned outright")
    print(f"held    paged {paged_bytes / 2**20:.0f} MiB, "
          f"dense {dense / 2**20:.0f} MiB across {BRANCHES} caches")
    print(f"saving  {dense / paged_bytes:.2f}x")

    for index, got in enumerate(actual):
        print(f"  branch {index}: {processor.decode(got[:12], skip_special_tokens=True)!r}")

    branches_match = actual == reference
    all_distinct = len({tuple(t) for t in actual}) == BRANCHES
    parent_intact = pool.block_table(pool.root_id) == prompt_frames
    accounting = frames == expected_frames

    print()
    print(f"  {'pass' if all(shared_at_fork) else 'FAIL'}  every branch shared the whole "
          f"prompt at fork time")
    print(f"  {'pass' if all_distinct else 'FAIL'}  the {BRANCHES} branches actually diverged")
    print(f"  {'pass' if branches_match else 'FAIL'}  every branch matches its DynamicCache run")
    print(f"  {'pass' if parent_intact else 'FAIL'}  the parent's blocks survived the branches' writes")
    print(f"  {'pass' if accounting else 'FAIL'}  {frames} frames in use, expected "
          f"{expected_frames}")
    if not branches_match:
        for index, (want, got) in enumerate(zip(reference, actual)):
            if want != got:
                print(f"    branch {index} diverged at step "
                      f"{next(i for i, (a, b) in enumerate(zip(got, want)) if a != b)}")

    ok = all(shared_at_fork) and all_distinct and branches_match and parent_intact and accounting
    print(f"\n{'pass' if ok else 'FAIL'}  {BRANCHES}-way fork sharing on {MODEL_ID}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
