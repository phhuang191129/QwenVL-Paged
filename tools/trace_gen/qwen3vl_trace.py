#!/usr/bin/env python3
"""Generate Qwen3-VL request traces for the paged-cache replay driver.

The visual geometry in these traces is authoritative: image token counts come
from `transformers`' own `smart_resize` plus the patch and merge sizes in the
model's published `preprocessor_config.json`, so a trace states exactly how many
KV cache token slots a given image would consume at inference time.

No model weights and no torch are required. Image token count depends only on
source (height, width), never on pixel content, so the generator works from a
corpus of real image *dimensions* rather than real image files.

What is real and what is synthetic is recorded in the trace header:

  * real       - image dimensions (COCO) or documented capture resolutions,
                 `smart_resize` geometry, patch/merge sizes, KV block math.
  * synthetic  - text token counts, decode budgets, request mix proportions,
                 and the weights of the `devices` resolution mix.

M-RoPE position layout follows the Qwen2-VL rule that Qwen3-VL inherits for
position *ids* (text advances all three axes together; an image occupies a
grid_h x grid_w patch of the h/w axes at a fixed t, and the next text token
resumes at origin + max(grid_h, grid_w)). Qwen3-VL's Interleaved-MRoPE changes
how frequency bands are allocated across those axes, not the ids themselves.
Validating this against the model's own `get_rope_index` needs torch and is a
week 13 task; until then the header labels the layout as `qwen2vl_rule`.

Usage:
    uv run --no-project python tools/trace_gen/qwen3vl_trace.py \
        --mix bimodal --pixel-budget recommended --num-requests 400 \
        --out traces/bimodal-recommended.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize

DEFAULT_MODEL = "Qwen/Qwen3-VL-2B-Instruct"

COCO_DIMENSIONS_URL = "http://images.cocodataset.org/annotations/image_info_test2017.zip"
COCO_DIMENSIONS_MEMBER = "annotations/image_info_test2017.json"

# Real-world capture resolutions as (height, width, weight). The resolutions are
# real device output; the weights are an assumption, which is why the header
# labels this source as partly synthetic.
DEVICE_RESOLUTIONS = [
    (3024, 4032, 22),  # phone photo, landscape
    (4032, 3024, 18),  # phone photo, portrait
    (1080, 1920, 14),  # 1080p screenshot / video frame
    (1440, 2560, 10),  # 1440p screenshot
    (2160, 3840, 6),   # 4K screenshot / video frame
    (2200, 1700, 9),   # scanned letter page at 200 dpi
    (3300, 2550, 5),   # scanned letter page at 300 dpi
    (720, 1280, 8),    # webcam / low-res capture
    (4000, 6000, 3),   # DSLR
    (480, 640, 5),     # thumbnail / legacy
]

# Token budgets. `default` is what the published preprocessor_config.json ships;
# `recommended` is the 256-1280 visual token clamp the Qwen3-VL docs advise. The
# gap between them is the point of the experiment.
RECOMMENDED_MIN_TOKENS = 256
RECOMMENDED_MAX_TOKENS = 1280

MIXES = ("text-only", "image-heavy", "bimodal", "repeated-prefix")


@dataclass(frozen=True)
class Geometry:
    """Model geometry that determines cache cost per token."""

    model: str
    patch_size: int
    merge_size: int
    temporal_patch_size: int
    min_pixels: int
    max_pixels: int
    num_layers: int
    num_kv_heads: int
    head_dim: int
    bytes_per_element: int

    @property
    def pixels_per_token(self) -> int:
        return (self.patch_size * self.merge_size) ** 2

    @property
    def kv_bytes_per_token(self) -> int:
        return (
            self.num_kv_heads * self.head_dim * self.bytes_per_element * 2 * self.num_layers
        )


def fetch_json(model: str, filename: str) -> dict:
    from huggingface_hub import hf_hub_download

    with open(hf_hub_download(repo_id=model, filename=filename), encoding="utf-8") as handle:
        return json.load(handle)


def load_geometry(model: str, pixel_budget: str) -> Geometry:
    preprocessor = fetch_json(model, "preprocessor_config.json")
    config = fetch_json(model, "config.json")
    text = config["text_config"]

    patch_size = int(preprocessor["patch_size"])
    merge_size = int(preprocessor["merge_size"])
    pixels_per_token = (patch_size * merge_size) ** 2

    if pixel_budget == "default":
        min_pixels = int(preprocessor["size"]["shortest_edge"])
        max_pixels = int(preprocessor["size"]["longest_edge"])
    else:
        min_pixels = RECOMMENDED_MIN_TOKENS * pixels_per_token
        max_pixels = RECOMMENDED_MAX_TOKENS * pixels_per_token

    return Geometry(
        model=model,
        patch_size=patch_size,
        merge_size=merge_size,
        temporal_patch_size=int(preprocessor["temporal_patch_size"]),
        min_pixels=min_pixels,
        max_pixels=max_pixels,
        num_layers=int(text["num_hidden_layers"]),
        num_kv_heads=int(text["num_key_value_heads"]),
        head_dim=int(text["head_dim"]),
        bytes_per_element=2 if text["dtype"] in ("bfloat16", "float16") else 4,
    )


def load_coco_dimensions(cache_dir: Path) -> list[tuple[int, int]]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    archive = cache_dir / "image_info_test2017.zip"
    if not archive.exists():
        print(f"downloading {COCO_DIMENSIONS_URL}", file=sys.stderr)
        urllib.request.urlretrieve(COCO_DIMENSIONS_URL, archive)

    with zipfile.ZipFile(archive) as zf:
        payload = json.loads(zf.read(COCO_DIMENSIONS_MEMBER))
    # Sorted by id so the corpus order does not depend on JSON iteration order.
    images = sorted(payload["images"], key=lambda entry: entry["id"])
    return [(int(entry["height"]), int(entry["width"])) for entry in images]


def sample_dimensions(rng: random.Random, source: str, corpus: list[tuple[int, int]]) -> tuple[int, int]:
    if source == "coco":
        return rng.choice(corpus)
    weights = [weight for _, _, weight in DEVICE_RESOLUTIONS]
    height, width, _ = rng.choices(DEVICE_RESOLUTIONS, weights=weights, k=1)[0]
    return height, width


def describe_image(geometry: Geometry, height: int, width: int) -> dict:
    """Return the exact grid and LLM token count Qwen3-VL would produce."""
    resized_h, resized_w = smart_resize(
        height,
        width,
        factor=geometry.patch_size * geometry.merge_size,
        min_pixels=geometry.min_pixels,
        max_pixels=geometry.max_pixels,
    )
    grid_h = resized_h // geometry.patch_size
    grid_w = resized_w // geometry.patch_size
    merged_h = grid_h // geometry.merge_size
    merged_w = grid_w // geometry.merge_size
    return {
        "source_h": height,
        "source_w": width,
        "resized_h": resized_h,
        "resized_w": resized_w,
        "grid_t": 1,
        "grid_h": grid_h,
        "grid_w": grid_w,
        "merged_h": merged_h,
        "merged_w": merged_w,
        "visual_tokens": merged_h * merged_w,
    }


def sample_text_tokens(rng: random.Random, has_image: bool) -> int:
    """Synthetic: a question alongside an image is short, a text prompt is not."""
    if has_image:
        return int(rng.triangular(8, 96, 24))
    return max(16, int(rng.lognormvariate(5.2, 0.9)))


def build_spans(text_tokens: int, images: list[dict]) -> tuple[list[dict], int]:
    """Lay out M-RoPE spans and return them with the total prompt length.

    `stride` carries the merged grid extent for image spans and (1, 1, 1) for
    text spans, since a raster-ordered 2D span cannot be described by a single
    per-token step. The C++ side only carries this metadata; nothing interprets
    it yet.
    """
    spans: list[dict] = []
    token = 0
    position = 0
    # A chat template wraps the request; charge its tokens to a leading text span.
    lead_text = max(1, text_tokens)

    spans.append(
        {
            "kind": "text",
            "start_token": token,
            "token_count": lead_text,
            "origin": [position, position, position],
            "stride": [1, 1, 1],
        }
    )
    token += lead_text
    position += lead_text

    for image in images:
        spans.append(
            {
                "kind": "image",
                "start_token": token,
                "token_count": image["visual_tokens"],
                "origin": [position, position, position],
                "stride": [image["grid_t"], image["merged_h"], image["merged_w"]],
            }
        )
        token += image["visual_tokens"]
        position += max(image["merged_h"], image["merged_w"])

    return spans, token


def prefix_key(system_prompt_id: int, images: list[dict]) -> str:
    """Identity of the shareable prompt prefix, for week 14 prefix caching."""
    digest = hashlib.sha256()
    digest.update(f"system:{system_prompt_id}".encode())
    for image in images:
        digest.update(f"|img:{image['source_h']}x{image['source_w']}".encode())
    return digest.hexdigest()[:16]


def generate(args: argparse.Namespace, geometry: Geometry) -> list[dict]:
    rng = random.Random(args.seed)
    corpus = load_coco_dimensions(Path(args.cache_dir)) if args.dims == "coco" else []

    # `repeated-prefix` draws from a small pool so the same prompt prefix recurs.
    pool = [sample_dimensions(rng, args.dims, corpus) for _ in range(args.prefix_pool)]

    requests = []
    arrival_us = 0
    for index in range(args.num_requests):
        if args.mix == "text-only":
            image_count = 0
        elif args.mix == "image-heavy":
            image_count = 1 if rng.random() < 0.85 else 2
        elif args.mix == "bimodal":
            image_count = 1 if rng.random() < 0.5 else 0
        else:  # repeated-prefix
            image_count = 1

        if args.mix == "repeated-prefix":
            system_prompt_id = 0
            dims = [pool[rng.randrange(len(pool))]]
        else:
            system_prompt_id = index
            dims = [sample_dimensions(rng, args.dims, corpus) for _ in range(image_count)]

        images = [describe_image(geometry, height, width) for height, width in dims]
        text_tokens = sample_text_tokens(rng, bool(images))
        spans, prompt_tokens = build_spans(text_tokens, images)

        # A request reserves for `decode_budget` (its max_new_tokens) but usually
        # stops early on an end-of-sequence token. The gap between the two is the
        # internal reservation waste a contiguous allocator cannot reclaim.
        decode_budget = int(rng.triangular(16, 512, 96))
        decode_actual = max(1, min(decode_budget, int(decode_budget * rng.betavariate(2.0, 3.0))))

        requests.append(
            {
                "record": "request",
                "request_id": index + 1,
                "arrival_us": arrival_us,
                "text_tokens": text_tokens,
                "prompt_tokens": prompt_tokens,
                "decode_budget": decode_budget,
                "decode_actual": decode_actual,
                "num_parallel_samples": args.parallel_samples if images else 1,
                "prefix_key": prefix_key(system_prompt_id, images),
                "images": images,
                "mrope_spans": spans,
            }
        )
        arrival_us += int(rng.expovariate(1.0 / max(1, args.mean_interarrival_us)))

    return requests


def build_header(args: argparse.Namespace, geometry: Geometry, requests: list[dict]) -> dict:
    visual = [image["visual_tokens"] for request in requests for image in request["images"]]
    prompts = [request["prompt_tokens"] for request in requests]
    budgets = [request["decode_budget"] for request in requests]
    return {
        "max_decode_budget": max(budgets) if budgets else 0,
        "record": "header",
        "schema_version": 1,
        "model": geometry.model,
        "mix": args.mix,
        "seed": args.seed,
        "num_requests": len(requests),
        "dimension_source": args.dims,
        "pixel_budget": args.pixel_budget,
        "mrope_position_layout": "qwen2vl_rule",
        "text_tokens_source": "synthetic",
        "patch_size": geometry.patch_size,
        "merge_size": geometry.merge_size,
        "temporal_patch_size": geometry.temporal_patch_size,
        "min_pixels": geometry.min_pixels,
        "max_pixels": geometry.max_pixels,
        "min_visual_tokens": geometry.min_pixels // geometry.pixels_per_token,
        "max_visual_tokens": geometry.max_pixels // geometry.pixels_per_token,
        "num_layers": geometry.num_layers,
        "num_kv_heads": geometry.num_kv_heads,
        "head_dim": geometry.head_dim,
        "bytes_per_element": geometry.bytes_per_element,
        "kv_bytes_per_token": geometry.kv_bytes_per_token,
        "total_prompt_tokens": sum(prompts),
        "max_prompt_tokens": max(prompts) if prompts else 0,
        "images_total": len(visual),
        "visual_tokens_total": sum(visual),
        "visual_tokens_max": max(visual) if visual else 0,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--mix", choices=MIXES, default="bimodal")
    parser.add_argument(
        "--pixel-budget",
        choices=("default", "recommended"),
        default="recommended",
        help="'default' uses the shipped preprocessor_config.json; "
        "'recommended' clamps to the 256-1280 visual token range the docs advise.",
    )
    parser.add_argument("--dims", choices=("coco", "devices"), default="devices")
    parser.add_argument("--num-requests", type=int, default=400)
    parser.add_argument("--seed", type=int, default=20260808)
    parser.add_argument(
        "--parallel-samples",
        type=int,
        default=1,
        help="Sampling branches per image request, for copy-on-write experiments.",
    )
    parser.add_argument("--prefix-pool", type=int, default=8, help="Distinct images in the repeated-prefix mix.")
    parser.add_argument("--mean-interarrival-us", type=int, default=20000)
    parser.add_argument("--cache-dir", default="traces/.corpus")
    parser.add_argument("--out", required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    geometry = load_geometry(args.model, args.pixel_budget)
    requests = generate(args, geometry)
    header = build_header(args, geometry, requests)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for record in [header, *requests]:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    kv_per_token = geometry.kv_bytes_per_token
    print(
        f"{out}: {header['num_requests']} requests, "
        f"{header['images_total']} images, "
        f"prompt tokens max {header['max_prompt_tokens']}, "
        f"visual tokens max {header['visual_tokens_max']} "
        f"({header['visual_tokens_max'] * kv_per_token / (1 << 20):.1f} MiB KV)",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
