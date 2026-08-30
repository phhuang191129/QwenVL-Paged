#!/usr/bin/env python3
"""Greedy-decodes a Qwen3-VL twice and demands the same tokens both times.

The two runs differ only in where the KV lives: once in transformers' own
`DynamicCache`, once in blocks the C++ allocator hands out. Everything the
memory subsystem does -- block allocation, the partial tail of the last block,
copy-on-write, frame recycling -- has to be invisible from the model's side, and
the only honest way to state that is that the sampled tokens are identical.

Weights are random and tiny, which is the point: a two-layer model runs in
seconds, so this can gate every change, and a wrong answer here is a bug in the
paging rather than a property of any particular checkpoint. Logit drift is
reported alongside the tokens because identical tokens with drifting logits
would mean the gate is passing by luck of the argmax.

Usage:
    PYTHONPATH=build .venv/bin/python python/token_identical_check.py
"""

import pathlib
import sys
import time

import torch
from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration
from transformers.cache_utils import Cache, DynamicLayer

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from paged_cache import build_paged_cache

# 37 makes prefill itself straddle three blocks, so the write loop has to split
# one call across block boundaries rather than always landing inside one frame.
# 37 + 24 = 61 leaves the last block three-quarters full, which is the case a
# kernel or gather that trusted block count over context length would get wrong.
PROMPT_TOKENS = 37
NEW_TOKENS = 24
MAX_BLOCKS = 64
TOLERANCE = 0.0


def tiny_model():
    config = Qwen3VLConfig(
        text_config=dict(
            vocab_size=256, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
            num_attention_heads=4, num_key_value_heads=2, head_dim=16,
            max_position_embeddings=512,
        ),
        vision_config=dict(
            hidden_size=32, intermediate_size=64, depth=2, num_heads=2, patch_size=16,
            temporal_patch_size=2, out_hidden_size=64,
        ),
    )
    torch.manual_seed(0)
    return Qwen3VLForConditionalGeneration(config).eval(), config


def greedy(model, prompt, cache):
    """Runs prefill then one token at a time, returning tokens and final logits."""
    tokens, logits = [], None
    with torch.no_grad():
        output = model(input_ids=prompt, past_key_values=cache, use_cache=True)
        for _ in range(NEW_TOKENS):
            logits = output.logits[:, -1, :]
            nxt = logits.argmax(-1, keepdim=True)
            tokens.append(int(nxt))
            output = model(input_ids=nxt, past_key_values=cache, use_cache=True)
    return tokens, logits


def main() -> int:
    model, config = tiny_model()
    text = config.text_config
    print(f"tiny Qwen3-VL: {text.num_hidden_layers} layers, "
          f"{text.num_key_value_heads} kv heads, head_dim {text.head_dim}, "
          f"{sum(p.numel() for p in model.parameters())} params")

    torch.manual_seed(1)
    prompt = torch.randint(0, text.vocab_size, (1, PROMPT_TOKENS))

    reference = Cache(layers=[DynamicLayer() for _ in range(text.num_hidden_layers)])
    started = time.perf_counter()
    expected, expected_logits = greedy(model, prompt, reference)
    reference_seconds = time.perf_counter() - started

    paged, pool = build_paged_cache(text, max_blocks=MAX_BLOCKS, scatter=True)
    started = time.perf_counter()
    actual, actual_logits = greedy(model, prompt, paged)
    paged_seconds = time.perf_counter() - started

    frames = pool.block_table(pool.root_id)
    total_tokens = PROMPT_TOKENS + NEW_TOKENS
    expected_frames = -(-total_tokens // pool.tokens_per_block)
    gathered = sum(layer.gathered_elements for layer in paged.layers)

    print(f"prompt {PROMPT_TOKENS} tokens, decoded {NEW_TOKENS}")
    print(f"blocks  {len(frames)} frames {frames} for {total_tokens} tokens "
          f"at {pool.tokens_per_block}/block")
    print(f"gather  {gathered / 2**20 * 4:.2f} MiB copied across "
          f"{(NEW_TOKENS + 1) * text.num_hidden_layers} update calls")
    print(f"time    reference {reference_seconds:.2f}s, paged {paged_seconds:.2f}s")

    drift = float((actual_logits - expected_logits).abs().max())
    tokens_match = actual == expected
    blocks_match = len(frames) == expected_frames
    # Frames spanning more ids than they occupy is what "the spacer punched
    # holes" looks like, and it is the premise the gather is being tested under.
    scattered = max(frames) - min(frames) + 1 > len(frames)

    print()
    print(f"  {'pass' if blocks_match else 'FAIL'}  allocated {len(frames)} blocks, "
          f"expected {expected_frames}")
    print(f"  {'pass' if scattered else 'FAIL'}  frames are physically scattered, "
          f"so the gather must follow the block table")
    print(f"  {'pass' if drift <= TOLERANCE else 'FAIL'}  final-logit drift {drift:.3e}")
    print(f"  {'pass' if tokens_match else 'FAIL'}  token-identical over {NEW_TOKENS} steps")
    if not tokens_match:
        print(f"    reference {expected}")
        print(f"    paged     {actual}")

    ok = tokens_match and blocks_match and scattered and drift <= TOLERANCE
    print(f"\n{'pass' if ok else 'FAIL'}  paged KV cache vs DynamicCache on Qwen3-VL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
