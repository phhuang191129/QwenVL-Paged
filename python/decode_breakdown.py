#!/usr/bin/env python3
"""Splits the 11 ms/step the kernel did not remove.

Week 20 put Triton in the decode path and the overhead versus DynamicCache
stayed at 11 ms/step. This times the three paged-only pieces of one decode
step — writing the new token into a page, building the block table, launching
Triton — against the same step through a dense cache, so the next optimization
is chosen from a table rather than a guess.

Prefill is excluded. The loop is one token at a time after a filled cache, with
a CUDA sync around each bucket so host timers see device work.

Usage:
    PYTHONPATH=build .venv/bin/python python/decode_breakdown.py
"""

import pathlib
import sys
import time

import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from transformers.cache_utils import Cache, DynamicLayer

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from paged_cache import (
    DEFAULT_TOKENS_PER_BLOCK,
    PagedLayer,
    build_paged_cache,
    install_paged_attention,
    paged_attention_decode_partitioned,
)
from qwen3vl_2b_check import DTYPE, MODEL_ID, NEW_TOKENS, build_inputs


class Buckets:
    def __init__(self):
        self.ns = {name: 0.0 for name in ("update", "gather", "table", "launch")}
        self.calls = 0

    def add(self, name, seconds):
        self.ns[name] += seconds


def instrument(buckets: Buckets):
    orig_update = PagedLayer.update
    orig_gather = PagedLayer._gather
    orig_attend = PagedLayer.attend_decode

    def update(self, key_states, value_states, *args, **kwargs):
        decode = key_states.shape[-2] == 1
        if not decode:
            return orig_update(self, key_states, value_states, *args, **kwargs)
        torch.cuda.synchronize()
        started = time.perf_counter()
        out = orig_update(self, key_states, value_states, *args, **kwargs)
        torch.cuda.synchronize()
        buckets.add("update", time.perf_counter() - started)
        buckets.calls += 1
        return out

    def gather(self):
        torch.cuda.synchronize()
        started = time.perf_counter()
        out = orig_gather(self)
        torch.cuda.synchronize()
        buckets.add("gather", time.perf_counter() - started)
        return out

    def attend(self, query, scale):
        q = query[:, :, 0, :].contiguous()
        torch.cuda.synchronize()
        started = time.perf_counter()
        table, context = self.pool.decode_inputs(self.sequence_id, self.length)
        torch.cuda.synchronize()
        buckets.add("table", time.perf_counter() - started)

        layout = self.pool.layout
        torch.cuda.synchronize()
        started = time.perf_counter()
        out = paged_attention_decode_partitioned(
            self.pool.frames,
            table,
            context,
            q,
            layer=self.layer_idx,
            layer_stride=layout.layer_stride(),
            stream_stride=layout.stream_stride(),
            token_stride=layout.token_stride(),
            head_stride=layout.head_stride(),
            tokens_per_block=self.pool.tokens_per_block,
            num_kv_heads=self.pool.num_kv_heads,
            scale=scale,
            scratch=self.pool._scratch,
        )
        torch.cuda.synchronize()
        buckets.add("launch", time.perf_counter() - started)
        return out.unsqueeze(1).contiguous()

    PagedLayer.update = update
    PagedLayer._gather = gather
    PagedLayer.attend_decode = attend
    return orig_update, orig_gather, orig_attend


def restore(origs):
    PagedLayer.update, PagedLayer._gather, PagedLayer.attend_decode = origs


def decode_loop(model, cache, first_logits, steps):
    """One token at a time from a filled cache. Returns wall seconds."""
    token = first_logits.argmax(-1, keepdim=True)
    torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.no_grad():
        for _ in range(steps):
            logits = model(input_ids=token, past_key_values=cache, use_cache=True).logits[:, -1]
            token = logits.argmax(-1, keepdim=True)
    torch.cuda.synchronize()
    return time.perf_counter() - started


def prefill(model, inputs, cache):
    with torch.no_grad():
        return model(**inputs, past_key_values=cache, use_cache=True).logits[:, -1]


def report(title, wall, buckets: Buckets | None, steps, layers, dense_wall=None):
    per_step = wall / steps * 1e3
    print(f"\n{title}")
    print(f"  wall    {per_step:6.2f} ms/step  ({wall:.3f}s / {steps} tokens)")
    if dense_wall is not None:
        print(f"  vs dense{ (wall - dense_wall) / steps * 1e3:+6.2f} ms/step")
    if buckets is None:
        return
    write = buckets.ns["update"] - buckets.ns["gather"]
    rows = (
        ("write", write),
        ("gather", buckets.ns["gather"]),
        ("table", buckets.ns["table"]),
        ("launch", buckets.ns["launch"]),
    )
    accounted = sum(seconds for _, seconds in rows)
    other = wall - accounted
    print(f"  {'bucket':<8s}  ms/step   ms/layer    share of wall")
    for name, seconds in rows + (("other", other),):
        print(f"  {name:<8s} {seconds / steps * 1e3:7.2f}  "
              f"{seconds / steps / layers * 1e3:8.2f}  "
              f"{seconds / wall * 100:5.1f}%")
    print(f"  calls   {buckets.calls} decode updates "
          f"(expected {steps * layers})")


def main() -> int:
    if not torch.cuda.is_available():
        print("no CUDA device visible", file=sys.stderr)
        return 1

    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = Qwen3VLForConditionalGeneration.from_pretrained(MODEL_ID, dtype=DTYPE).cuda().eval()
    text = model.config.text_config
    layers = text.num_hidden_layers
    inputs = build_inputs(processor).to("cuda")
    prompt = inputs["input_ids"].shape[1]
    blocks = -(-(prompt + NEW_TOKENS) // DEFAULT_TOKENS_PER_BLOCK) + 4

    print(f"{MODEL_ID}: {layers} layers, prompt {prompt}, {NEW_TOKENS} decode steps")

    # Dense, then kernel. One discarded pass each so compile and allocator
    # setup are not in the table.
    dense = Cache(layers=[DynamicLayer() for _ in range(layers)])
    logits = prefill(model, inputs, dense)
    decode_loop(model, dense, logits, 1)

    dense = Cache(layers=[DynamicLayer() for _ in range(layers)])
    logits = prefill(model, inputs, dense)
    dense_wall = decode_loop(model, dense, logits, NEW_TOKENS)
    report("dense (DynamicCache)", dense_wall, None, NEW_TOKENS, layers)

    def make_paged(use_kernel):
        cache, _ = build_paged_cache(
            text, max_blocks=blocks, dtype=DTYPE, device="cuda", use_kernel=use_kernel)
        if use_kernel:
            install_paged_attention(model, cache)
        return cache

    def run_paged(use_kernel, title):
        buckets = Buckets()
        origs = instrument(buckets)
        cache = make_paged(use_kernel)
        logits = prefill(model, inputs, cache)
        decode_loop(model, cache, logits, 1)
        # Fresh cache so the measured steps are the first 20 decodes, not a
        # continuation of the warmup token. Prefill of that cache still hits
        # the instrumented _gather, so the counters start after it returns.
        cache = make_paged(use_kernel)
        logits = prefill(model, inputs, cache)
        buckets.ns = {name: 0.0 for name in buckets.ns}
        buckets.calls = 0
        wall = decode_loop(model, cache, logits, NEW_TOKENS)
        restore(origs)
        report(title, wall, buckets, NEW_TOKENS, layers, dense_wall)
        return wall

    run_paged(False, "device gather")
    run_paged(True, "device kernel")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
