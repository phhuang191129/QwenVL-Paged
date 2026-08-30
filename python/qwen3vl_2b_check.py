#!/usr/bin/env python3
"""The token-identical gate on the real Qwen3-VL-2B, with an image in the prompt.

The tiny-model gate in `token_identical_check.py` proves the paging is right on
a two-layer random-weight config. It cannot say anything about the three things
that only exist in a real checkpoint: 28 layers of real weights where a single
wrong KV byte compounds instead of washing out, bfloat16 instead of float32, and
a prompt whose bulk is image tokens produced by the vision tower and addressed
by M-RoPE rather than ordinary positions.

Generation goes through `model.generate` rather than a hand-rolled loop, on
purpose. A VL model's position handling at each step is not something to
reimplement in a test whose job is to isolate the cache: if this loop and the
reference loop differ anywhere but the cache, the comparison means nothing. Both
runs call the same generate with the same arguments and differ only in which
cache object they are handed.

The pool is host memory, so every step gathers the context across PCIe. That is
slow and expected; see docs/performance.md week 18 for why it is the design and
not a bug.

Usage:
    PYTHONPATH=build .venv/bin/python python/qwen3vl_2b_check.py
"""

import pathlib
import sys
import time

import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from transformers.cache_utils import Cache, DynamicLayer

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from paged_cache import DEFAULT_TOKENS_PER_BLOCK, build_paged_cache

MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"
NEW_TOKENS = 20

# The processor turns a 32x32 pixel patch into one token, so this lands around
# 1,225 image tokens -- the ~1,280-token image prefill the block shape in
# docs/performance.md was chosen against. A smaller image passes just as well
# and exercises far fewer blocks.
IMAGE_SIZE = 1120
DTYPE = torch.bfloat16


def make_image():
    """A deterministic image, so the prompt is fixed without a network fetch."""
    y, x = np.mgrid[0:IMAGE_SIZE, 0:IMAGE_SIZE]
    channels = [
        (x * 255 // IMAGE_SIZE),
        (y * 255 // IMAGE_SIZE),
        ((x ^ y) & 0xFF),
    ]
    return Image.fromarray(np.stack(channels, axis=-1).astype(np.uint8))


def build_inputs(processor):
    messages = [{
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": "Describe this image in one sentence."},
        ],
    }]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return processor(text=[text], images=[make_image()], return_tensors="pt")


def generate(model, inputs, cache):
    with torch.no_grad():
        return model.generate(
            **inputs,
            past_key_values=cache,
            use_cache=True,
            do_sample=False,
            num_beams=1,
            max_new_tokens=NEW_TOKENS,
            min_new_tokens=NEW_TOKENS,
        )


def main() -> int:
    if not torch.cuda.is_available():
        print("no CUDA device visible", file=sys.stderr)
        return 1

    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = Qwen3VLForConditionalGeneration.from_pretrained(MODEL_ID, dtype=DTYPE).cuda().eval()
    text = model.config.text_config

    inputs = build_inputs(processor).to("cuda")
    prompt_tokens = inputs["input_ids"].shape[1]
    image_tokens = int((inputs["input_ids"] == model.config.image_token_id).sum())

    print(f"{MODEL_ID}: {text.num_hidden_layers} layers, "
          f"{text.num_key_value_heads} kv heads, head_dim {text.head_dim}, {DTYPE}")
    print(f"prompt {prompt_tokens} tokens, {image_tokens} of them image, "
          f"decoding {NEW_TOKENS}")

    reference_cache = Cache(layers=[DynamicLayer() for _ in range(text.num_hidden_layers)])
    started = time.perf_counter()
    expected = generate(model, inputs, reference_cache)
    reference_seconds = time.perf_counter() - started

    # Room for the prompt, the new tokens, and the spacer frame taken between
    # each of ours to keep the block table non-contiguous.
    blocks = -(-(prompt_tokens + NEW_TOKENS) // DEFAULT_TOKENS_PER_BLOCK)
    paged, pool = build_paged_cache(text, max_blocks=2 * blocks + 4, scatter=True, dtype=DTYPE)
    print(f"pool    {2 * blocks + 4} frames x "
          f"{pool.allocator.block_stride_bytes() / 2**20:.2f} MiB "
          f"= {pool.allocator.pool_bytes() / 2**20:.0f} MiB")

    started = time.perf_counter()
    actual = generate(model, inputs, paged)
    paged_seconds = time.perf_counter() - started

    frames = pool.block_table(pool.root_id)
    gathered = sum(layer.gathered_elements for layer in paged.layers) * DTYPE.itemsize

    expected_new = expected[0, prompt_tokens:].tolist()
    actual_new = actual[0, prompt_tokens:].tolist()

    print(f"blocks  {len(frames)} frames for {prompt_tokens + NEW_TOKENS} tokens, "
          f"span {min(frames)}-{max(frames)}")
    print(f"gather  {gathered / 2**30:.2f} GiB copied to the device across "
          f"{NEW_TOKENS * text.num_hidden_layers} update calls")
    print(f"time    reference {reference_seconds:.1f}s, paged {paged_seconds:.1f}s")
    print(f"\nreference: {processor.decode(expected_new, skip_special_tokens=True)!r}")
    print(f"paged:     {processor.decode(actual_new, skip_special_tokens=True)!r}")

    tokens_match = actual_new == expected_new
    blocks_match = len(frames) == blocks
    scattered = max(frames) - min(frames) + 1 > len(frames)

    print()
    print(f"  {'pass' if blocks_match else 'FAIL'}  allocated {len(frames)} blocks, "
          f"expected {blocks}")
    print(f"  {'pass' if scattered else 'FAIL'}  frames are physically scattered")
    print(f"  {'pass' if tokens_match else 'FAIL'}  token-identical over "
          f"{NEW_TOKENS} steps at {text.num_hidden_layers} layers in {DTYPE}")
    if not tokens_match:
        first = next(i for i, (a, b) in enumerate(zip(actual_new, expected_new)) if a != b)
        print(f"    diverged at step {first}")

    ok = tokens_match and blocks_match and scattered
    print(f"\n{'pass' if ok else 'FAIL'}  paged KV cache vs DynamicCache on {MODEL_ID}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
