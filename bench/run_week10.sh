#!/usr/bin/env bash
# Reproduces every Week 10 number in docs/performance.md.
#
# Prerequisites:
#   cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build
# Pinning the governor needs root and is skipped when unavailable.
set -euo pipefail

cd "$(dirname "$0")/.."

ENV_OUT=results/week10-env.txt
ALLOC=results/week10-allocator.csv
ROOFS=results/week10-roofs.csv
FIG=results/week10-roofline.svg

mkdir -p results
rm -f "$ENV_OUT" "$ALLOC" "$ROOFS" "$FIG"

{
    echo "=== date ==="
    date --iso-8601=seconds
    echo
    echo "=== lscpu ==="
    lscpu || true
    echo
    echo "=== meminfo ==="
    head -n 5 /proc/meminfo || true
    echo
    echo "=== governor / freq ==="
    for f in /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor \
             /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq \
             /sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq \
             /sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_min_freq; do
        echo -n "$f: "
        cat "$f" 2>/dev/null || echo unavailable
    done
    echo
    echo "=== thermal ==="
    for z in /sys/class/thermal/thermal_zone*; do
        echo -n "$z $(cat "$z/type" 2>/dev/null): "
        cat "$z/temp" 2>/dev/null || echo unavailable
    done
    echo
    echo "=== memory devices ==="
    if command -v dmidecode >/dev/null && dmidecode -t memory >/tmp/week10-dmi.txt 2>/dev/null; then
        grep -E 'Size:|Type:|Speed:|Configured Memory Speed:|Locator:' /tmp/week10-dmi.txt || true
    else
        echo "dmidecode unavailable or needs root"
    fi
    if command -v lshw >/dev/null; then
        lshw -class memory -short 2>/dev/null || true
    fi
} > "$ENV_OUT"

if [[ -w /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor ]]; then
    echo performance > /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor || true
fi

export OMP_NUM_THREADS=8
./build/qwenvl_paged_bench --csv "$ALLOC"
./build/qwenvl_roofline --csv "$ROOFS"
python3 bench/plot_roofline.py "$ROOFS" "$FIG"

echo
echo "wrote $ENV_OUT, $ALLOC, $ROOFS, and $FIG"
