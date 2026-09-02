#!/usr/bin/env bash
# Reproduces the Week 11 kernel comparison and the layer-major rows in
# docs/performance.md.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p results
./build/qwenvl_roofline --attention --csv results/week11-kernels.csv
echo "wrote results/week11-kernels.csv"
