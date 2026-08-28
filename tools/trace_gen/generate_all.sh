#!/usr/bin/env bash
# Regenerates the standard Week 9 trace set into traces/.
#
# Requires network access on first run (model config from the Hugging Face hub,
# COCO image dimensions) and the pinned Python environment:
#
#   uv venv --python 3.12 .venv
#   VIRTUAL_ENV=$PWD/.venv uv pip install "transformers==4.57.1" pillow numpy
#
# After the first run everything needed is in the local caches, so set
# HF_HUB_OFFLINE=1 to skip hub revalidation, which is slow and can stall.
#
# Every trace is deterministic in its seed, so re-running overwrites byte-for-byte.
set -euo pipefail

cd "$(dirname "$0")/../.."

GEN=(env "VIRTUAL_ENV=$PWD/.venv" uv run --no-project python tools/trace_gen/qwen3vl_trace.py)
N="${N:-400}"

# The four mixes the replay driver reports on, at the documented 256-1280 visual
# token clamp.
"${GEN[@]}" --mix text-only      --num-requests "$N" --out traces/text-only.jsonl
"${GEN[@]}" --mix image-heavy    --num-requests "$N" --out traces/image-heavy.jsonl
"${GEN[@]}" --mix bimodal        --num-requests "$N" --out traces/bimodal.jsonl
"${GEN[@]}" --mix repeated-prefix --num-requests "$N" --out traces/repeated-prefix.jsonl

# Same workload under the pixel budget the published preprocessor_config.json
# actually ships, which is ~13x larger per image.
"${GEN[@]}" --mix bimodal --pixel-budget default --num-requests "$N" \
    --out traces/bimodal-default-budget.jsonl

# Natural-photo dimensions from COCO rather than the device resolution mix.
"${GEN[@]}" --mix image-heavy --dims coco --num-requests "$N" \
    --out traces/image-heavy-coco.jsonl

# Parallel sampling, to exercise fork and copy-on-write on real image prompts.
"${GEN[@]}" --mix image-heavy --parallel-samples 4 --num-requests "$N" \
    --out traces/image-heavy-parallel4.jsonl
