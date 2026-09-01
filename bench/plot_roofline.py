#!/usr/bin/env python3
"""Draw the week-10 roofline from results/week10-roofs.csv. No third-party deps."""

from __future__ import annotations

import csv
import math
import sys
from pathlib import Path


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open() as fh:
        return list(csv.DictReader(fh))


def f(row: dict[str, str], key: str) -> float:
    return float(row[key] or 0.0)


def svg_path(points: list[tuple[float, float]]) -> str:
    if not points:
        return ""
    parts = [f"M {points[0][0]:.2f} {points[0][1]:.2f}"]
    for x, y in points[1:]:
        parts.append(f"L {x:.2f} {y:.2f}")
    return " ".join(parts)


def main() -> int:
    src = Path(sys.argv[1] if len(sys.argv) > 1 else "results/week10-roofs.csv")
    dst = Path(sys.argv[2] if len(sys.argv) > 2 else "results/week10-roofline.svg")
    rows = [r for r in read_rows(src) if r.get("kind") != "env"]

    stream = next((r for r in rows if r["name"] == "STREAM triad OpenMP"), None)
    if stream is None:
        stream = next((r for r in rows if r["kind"] == "stream"), None)
    fp32 = next((r for r in rows if r["name"] == "AVX-512 FMA fp32"), None)
    bf16 = next((r for r in rows if r["name"] == "AVX-512 VDPBF16PS"), None)
    decode = [r for r in rows if r["kind"] == "attention_decode"]
    prefill = [r for r in rows if r["kind"] == "attention_prefill"]

    peak_fp32 = f(fp32, "gflops") if fp32 else 0.0
    peak_bf16 = f(bf16, "gflops") if bf16 else 0.0
    stream_gbs = f(stream, "gbs") if stream else 0.0

    # Log-log plot: AI on x (0.1 .. 100), GFLOP/s on y (0.1 .. 4000)
    ai_min, ai_max = 0.1, 100.0
    y_min, y_max = 0.1, max(4000.0, peak_bf16 * 1.2, peak_fp32 * 1.2)
    left, top, width, height = 70.0, 30.0, 520.0, 360.0

    def x_of(ai: float) -> float:
        ai = min(max(ai, ai_min), ai_max)
        return left + width * (math.log10(ai) - math.log10(ai_min)) / (
            math.log10(ai_max) - math.log10(ai_min)
        )

    def y_of(gflops: float) -> float:
        gflops = min(max(gflops, y_min), y_max)
        return top + height * (1.0 - (math.log10(gflops) - math.log10(y_min)) / (
            math.log10(y_max) - math.log10(y_min)
        ))

    parts: list[str] = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="720" height="460" '
        'font-family="DejaVu Sans, sans-serif" font-size="12">',
        '<rect width="720" height="460" fill="white"/>',
        '<text x="360" y="22" text-anchor="middle" font-size="14">'
        "Week 10 roofline — Ryzen 7 8845H, reference paged attention</text>",
        f'<rect x="{left}" y="{top}" width="{width}" height="{height}" '
        'fill="none" stroke="#333"/>',
    ]

    for decade in (0.1, 1, 10, 100):
        x = x_of(decade)
        parts.append(
            f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top + height}" '
            'stroke="#ddd"/>'
        )
        parts.append(
            f'<text x="{x:.1f}" y="{top + height + 16}" text-anchor="middle">'
            f"{decade:g}</text>"
        )
    for g in (0.1, 1, 10, 100, 1000):
        y = y_of(g)
        parts.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{left + width}" y2="{y:.1f}" '
            'stroke="#ddd"/>'
        )
        parts.append(
            f'<text x="{left - 8}" y="{y + 4:.1f}" text-anchor="end">{g:g}</text>'
        )
    parts.append(
        f'<text x="{left + width / 2}" y="{top + height + 36}" '
        'text-anchor="middle">Arithmetic intensity (FLOP/byte)</text>'
    )
    parts.append(
        f'<text transform="translate(18 {top + height / 2}) rotate(-90)" '
        'text-anchor="middle">GFLOP/s</text>'
    )

    ais = [10 ** (math.log10(ai_min) + i / 40 * (math.log10(ai_max) - math.log10(ai_min)))
           for i in range(41)]
    if stream_gbs > 0:
        ridge = [(x_of(ai), y_of(stream_gbs * ai)) for ai in ais]
        parts.append(
            f'<path d="{svg_path(ridge)}" fill="none" stroke="#1f77b4" '
            'stroke-width="2"/>'
        )
        parts.append(
            f'<text x="{x_of(0.3)}" y="{y_of(stream_gbs * 0.3) - 6}" '
            f'fill="#1f77b4">DRAM {stream_gbs:.1f} GB/s</text>'
        )
    if peak_fp32 > 0:
        parts.append(
            f'<line x1="{left}" y1="{y_of(peak_fp32):.1f}" x2="{left + width}" '
            f'y2="{y_of(peak_fp32):.1f}" stroke="#d62728" stroke-dasharray="6 3"/>'
        )
        parts.append(
            f'<text x="{left + width - 4}" y="{y_of(peak_fp32) - 6}" '
            f'text-anchor="end" fill="#d62728">fp32 {peak_fp32:.0f} GFLOP/s</text>'
        )
    if peak_bf16 > 0:
        parts.append(
            f'<line x1="{left}" y1="{y_of(peak_bf16):.1f}" x2="{left + width}" '
            f'y2="{y_of(peak_bf16):.1f}" stroke="#ff7f0e" stroke-dasharray="2 3"/>'
        )
        parts.append(
            f'<text x="{left + width - 4}" y="{y_of(peak_bf16) - 6}" '
            f'text-anchor="end" fill="#ff7f0e">bf16 {peak_bf16:.0f} GFLOP/s</text>'
        )

    def mark(row: dict[str, str], color: str, label: str) -> None:
        ai = f(row, "ai")
        gflops = f(row, "gflops")
        if ai <= 0 or gflops <= 0:
            return
        parts.append(
            f'<circle cx="{x_of(ai):.1f}" cy="{y_of(gflops):.1f}" r="5" fill="{color}"/>'
        )
        parts.append(
            f'<text x="{x_of(ai) + 8:.1f}" y="{y_of(gflops) + 4:.1f}" fill="{color}">'
            f"{label}</text>"
        )

    for row in decode:
        mark(row, "#2ca02c", row["name"])
    for row in prefill:
        mark(row, "#9467bd", row["name"])

    parts.append("</svg>")
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text("\n".join(parts) + "\n")
    print(f"wrote {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
