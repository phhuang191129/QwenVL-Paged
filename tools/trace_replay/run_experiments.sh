#!/usr/bin/env bash
# Reproduces every Week 9 number in docs/performance.md.
#
# Prerequisites:
#   ./tools/trace_gen/generate_all.sh          # writes traces/
#   cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build
#
# Results are rewritten from scratch each run so the CSVs always match the code
# that produced them.
set -euo pipefail

cd "$(dirname "$0")/../.."

REPLAY=./build/qwenvl_trace_replay
MIXES=results/week9-mixes.csv
SWEEP=results/week9-pool-sweep.csv

mkdir -p results
rm -f "$MIXES" "$SWEEP"

# Experiment 1: every workload mix at a pool sized so the paged batch never
# exhausts mid-decode, which is also enough room for the baseline. Isolates
# memory efficiency from scheduling effects.
"$REPLAY" --max-active 8 --csv "$MIXES" \
    traces/text-only.jsonl \
    traces/image-heavy-coco.jsonl \
    traces/bimodal.jsonl \
    traces/image-heavy.jsonl \
    traces/repeated-prefix.jsonl \
    traces/image-heavy-parallel4.jsonl

# Experiment 2: fixed workload, shrinking pool. Swap capacity matches the pool
# so preemption can actually reclaim frames. This is where memory efficiency
# turns into throughput.
for pool in 192 256 384 512 768 1024; do
    "$REPLAY" --max-active 8 --pool-blocks "$pool" --swap-slots "$pool" \
        --csv "$SWEEP" traces/image-heavy.jsonl
done

echo
echo "wrote $MIXES and $SWEEP"
