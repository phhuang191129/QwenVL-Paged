#!/usr/bin/env bash
# Reproduces every Week 14 number in docs/performance.md.
#
# Prerequisites:
#   cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build
#
# Results are rewritten from scratch each run so the CSVs always match the code
# that produced them.
set -euo pipefail

cd "$(dirname "$0")/../.."

REPLAY=./build/qwenvl_trace_replay
SWEEP=results/week14-pool-sweep.csv
CHUNK=results/week14-chunked-prefill.csv
SIZE=results/week14-size-aware.csv
PREFIX=results/week14-prefix-cache.csv
PAIR=results/week14-image-then-text.csv

mkdir -p results
rm -f "$SWEEP" "$CHUNK" "$SIZE" "$PREFIX" "$PAIR"

# Experiment 1: the week-9 pool sweep, now with decode-aware admission.
# Same pools, same swap, same image-heavy trace. Preemption thrash should
# collapse because a request is only admitted when its prompt plus decode
# budget fits.
for pool in 192 256 384 512 768 1024; do
    "$REPLAY" --max-active 8 --pool-blocks "$pool" --swap-slots "$pool" \
        --csv "$SWEEP" traces/image-heavy.jsonl
done

# Experiment 2: chunked prefill vs a single-shot image prefill. Token-cost
# TTFT, not wall clock. 256 tokens is one max-resolution image chunk;
# 65536 is the week-9 default that schedules a whole image in one step.
"$REPLAY" --max-active 8 --max-batch-tokens 65536 --paged-only \
    --csv "$CHUNK" traces/bimodal.jsonl
"$REPLAY" --max-active 8 --max-batch-tokens 256 --paged-only \
    --csv "$CHUNK" traces/bimodal.jsonl

# Experiment 3: FIFO vs size-aware on the bimodal mix. FIFO is the default;
# --size-aware skips a head that does not fit, bounded by --skip-limit.
"$REPLAY" --max-active 8 --paged-only --csv "$SIZE" traces/bimodal.jsonl
"$REPLAY" --max-active 8 --size-aware --skip-limit 8 --paged-only \
    --csv "$SIZE" traces/bimodal.jsonl
# Same policy, tight pool: FIFO head-of-line only appears under cache pressure.
"$REPLAY" --max-active 8 --pool-blocks 256 --swap-slots 256 --paged-only \
    --csv "$SIZE" traces/bimodal.jsonl
"$REPLAY" --max-active 8 --pool-blocks 256 --swap-slots 256 --size-aware \
    --skip-limit 8 --paged-only --csv "$SIZE" traces/bimodal.jsonl

# The plan's TTFT pair: one 1,292-token image, then a 326-token text request.
PAIR_TRACE=$(mktemp)
head -n 3 traces/bimodal.jsonl | sed 's/"num_requests": 400/"num_requests": 2/' > "$PAIR_TRACE"
"$REPLAY" --max-active 2 --max-batch-tokens 65536 --pool-blocks 256 --paged-only \
    --csv "$PAIR" "$PAIR_TRACE"
"$REPLAY" --max-active 2 --max-batch-tokens 256 --pool-blocks 256 --paged-only \
    --csv "$PAIR" "$PAIR_TRACE"
rm -f "$PAIR_TRACE"

# Experiment 4: cross-request prefix sharing on the mix built for it.
"$REPLAY" --max-active 8 --paged-only --csv "$PREFIX" traces/repeated-prefix.jsonl
"$REPLAY" --max-active 8 --prefix-cache --paged-only \
    --csv "$PREFIX" traces/repeated-prefix.jsonl

echo
echo "wrote $SWEEP, $CHUNK, $SIZE, $PREFIX, and $PAIR"
