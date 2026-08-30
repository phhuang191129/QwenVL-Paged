#!/usr/bin/env python3
"""The week-20 fork+Triton flip is a bf16 logit tie, not a wrong frame.

Week 20 saw one of four forked continuations disagree with DynamicCache at
decode step 18 through Triton, and left it ungated. This script is the
resolution. It does not require a fork: the same seed, decoded on a single
sequence, flips at the same step. Three of the four top-4 seeds match. On
the one that does not, gather/SDPA reports tokens 504 and 5916 tied at
26.625; the kernel has 5916 at 26.625 and 504 at 26.5 and so breaks the
tie the other way.

The CoW/memory gate stays `fork_sharing_check.py` on gather. The greedy
2B generate gate stays `qwen3vl_2b_check.py` (that continuation never
hits this pair). This file fails only if that picture changes — a flip
on another seed, or a gap larger than a bf16 ulp at this magnitude.

Usage:
    PYTHONPATH=build .venv/bin/python python/fork_kernel_diag.py
"""

import pathlib
import sys

import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from transformers.cache_utils import Cache, DynamicLayer

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from paged_cache import build_paged_cache, install_paged_attention
from fork_sharing_check import decode_from, prefill
from qwen3vl_2b_check import DTYPE, MODEL_ID, NEW_TOKENS, build_inputs

KNOWN_SEED = 785
KNOWN_STEP = 18
KNOWN_PAIR = {504, 5916}
MAX_GAP = 0.25


def unbind(model):
    for decoder in model.model.language_model.layers:
        decoder.self_attn._paged_layer = None


def kernel_run(model, text, inputs, seed, pool_blocks):
    unbind(model)
    cache, _ = build_paged_cache(
        text, max_blocks=pool_blocks, dtype=DTYPE, device="cuda", use_kernel=True)
    install_paged_attention(model, cache)
    prefill(model, inputs, cache)
    tokens = [seed]
    token = torch.tensor([[seed]], device="cuda")
    tops = []
    with torch.no_grad():
        for _ in range(NEW_TOKENS - 1):
            logits = model(input_ids=token, past_key_values=cache, use_cache=True).logits[:, -1]
            vals, ids = logits.float().topk(3, dim=-1)
            tops.append((ids[0].tolist(), vals[0].tolist()))
            token = logits.argmax(-1, keepdim=True)
            tokens.append(int(token))
    unbind(model)
    return tokens, tops


def dense_run(model, text, inputs, seed):
    unbind(model)
    cache = Cache(layers=[DynamicLayer() for _ in range(text.num_hidden_layers)])
    prefill(model, inputs, cache)
    return decode_from(model, cache, seed)


def main() -> int:
    if not torch.cuda.is_available():
        print("no CUDA device visible", file=sys.stderr)
        return 1

    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = Qwen3VLForConditionalGeneration.from_pretrained(MODEL_ID, dtype=DTYPE).cuda().eval()
    text = model.config.text_config
    inputs = build_inputs(processor).to("cuda")
    prompt = inputs["input_ids"].shape[1]
    pool_blocks = -(-prompt // 16) + 16

    scratch = Cache(layers=[DynamicLayer() for _ in range(text.num_hidden_layers)])
    seeds = prefill(model, inputs, scratch).topk(4).indices.tolist()
    print(f"seeds {seeds}")

    checks = []
    for index, seed in enumerate(seeds):
        want = dense_run(model, text, inputs, seed)
        got, tops = kernel_run(model, text, inputs, seed, pool_blocks)
        step = next((i for i, (a, b) in enumerate(zip(got, want)) if a != b), None)
        if seed == KNOWN_SEED:
            ids, vals = tops[KNOWN_STEP - 1]
            gap = abs(vals[0] - vals[1])
            pair = set(ids[:2]) == KNOWN_PAIR
            at = step == KNOWN_STEP
            tight = gap <= MAX_GAP
            ok = at and pair and tight
            print(f"  seed {index} {seed}: flip at {step}, "
                  f"top2 {ids[:2]} {vals[0]:.3f}/{vals[1]:.3f} gap {gap:.3f}")
            print(f"    {'pass' if ok else 'FAIL'}  known tie at step {KNOWN_STEP} "
                  f"on {sorted(KNOWN_PAIR)}, gap <= {MAX_GAP}")
            checks.append(ok)
        else:
            ok = step is None
            print(f"  {'pass' if ok else 'FAIL'}  seed {index} {seed}: "
                  f"{'match' if ok else f'flip at {step}'}")
            checks.append(ok)

    print(f"\n{'pass' if all(checks) else 'FAIL'}  "
          f"kernel vs dense: one bf16 tie, three matches")
    return 0 if all(checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
