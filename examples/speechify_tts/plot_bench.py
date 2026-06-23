#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Plot sglang-omni vs vllm-omni SpeechifyTTS benchmark comparison.

Parses the vllm-omni baseline ``Summary`` blocks from
``/tmp/bench_h100/logs_branch022/{length}_c{c}.log`` and (optionally) the
sglang-omni JSON cells written by ``bench.py`` (``{length}_c{c}.json``), then
renders a 3-row (short/medium/long) x 5-panel comparison figure.

The script runs end-to-end using ONLY the baseline logs; if ``--sglang-dir`` is
empty or missing, the sglang series is simply left out and the baseline series
is plotted alone.

Usage::

    python examples/speechify_tts/plot_bench.py \
        --sglang-dir /tmp/bench_h100/sglang \
        --out /tmp/bench_h100/compare_sglang_vs_vllm.png
"""

from __future__ import annotations

import argparse
import json
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

LENGTHS = ["short", "medium", "long"]
CONCURRENCIES = [1, 2, 4, 8]

LENGTH_LABELS = {
    "short": "SHORT  (~52 chars / ~3s audio)",
    "medium": "MEDIUM  (~211 chars / ~13s audio)",
    "long": "LONG  (~530 chars / ~30s audio)",
}

SGLANG_COLOR = "#2ca02c"          # sglang streaming  — solid green
SGLANG_NOSTREAM_COLOR = "#ff7f0e"  # sglang non-stream — dash-dot orange
VLLM_COLOR = "#1f77b4"            # vllm-omni baseline — dashed blue

# Summary block regex. Each numeric group is a float (the trailing unit char is
# matched but not captured).
_RE_SUMMARY = re.compile(
    r"Summary\s*\(\s*(?P<nreq>\d+)\s*requests,\s*concurrency=(?P<conc>\d+)\s*\):\s*"
    r"TTFA\s*avg/p50/p95/p99:\s*"
    r"(?P<ttfa_avg>[\d.]+)s\s*/\s*(?P<ttfa_p50>[\d.]+)s\s*/\s*"
    r"(?P<ttfa_p95>[\d.]+)s\s*/\s*(?P<ttfa_p99>[\d.]+)s\s*"
    r"E2E\s*avg/p50/p95/p99:\s*"
    r"(?P<e2e_avg>[\d.]+)s\s*/\s*(?P<e2e_p50>[\d.]+)s\s*/\s*"
    r"(?P<e2e_p95>[\d.]+)s\s*/\s*(?P<e2e_p99>[\d.]+)s\s*"
    r"Avg audio duration:\s*(?P<avg_duration>[\d.]+)s\s*"
    r"Avg RTF:\s*(?P<avg_rtf>[\d.]+)x\s*"
    r"Wall time:\s*(?P<wall_time>[\d.]+)s\s*"
    r"Throughput:\s*(?P<throughput>[\d.]+)x",
)


# --- parsing ---------------------------------------------------------------

def parse_baseline_log(path: str) -> dict | None:
    """Parse a single vllm-omni log file into a normalized cell dict."""
    with open(path, "r", encoding="utf-8") as fh:
        text = fh.read()
    m = _RE_SUMMARY.search(text)
    if not m:
        return None
    g = m.groupdict()
    return {
        "ttfa": {
            "avg": float(g["ttfa_avg"]),
            "p50": float(g["ttfa_p50"]),
            "p95": float(g["ttfa_p95"]),
            "p99": float(g["ttfa_p99"]),
        },
        "e2e": {
            "avg": float(g["e2e_avg"]),
            "p50": float(g["e2e_p50"]),
            "p95": float(g["e2e_p95"]),
            "p99": float(g["e2e_p99"]),
        },
        "avg_duration": float(g["avg_duration"]),
        "avg_rtf": float(g["avg_rtf"]),
        "wall_time": float(g["wall_time"]),
        "throughput": float(g["throughput"]),
    }


def load_baseline(log_dir: str) -> dict:
    """Return {(length, concurrency): cell} for all parseable baseline cells."""
    cells: dict = {}
    for length in LENGTHS:
        for c in CONCURRENCIES:
            path = os.path.join(log_dir, f"{length}_c{c}.log")
            if not os.path.isfile(path):
                continue
            cell = parse_baseline_log(path)
            if cell is not None:
                cells[(length, c)] = cell
    return cells


def load_sglang(sglang_dir: str | None) -> dict:
    """Return {(length, concurrency): cell} for all sglang JSON cells found."""
    cells: dict = {}
    if not sglang_dir or not os.path.isdir(sglang_dir):
        return cells
    for length in LENGTHS:
        for c in CONCURRENCIES:
            path = os.path.join(sglang_dir, f"{length}_c{c}.json")
            if not os.path.isfile(path):
                continue
            try:
                with open(path, "r", encoding="utf-8") as fh:
                    cells[(length, c)] = json.load(fh)
            except (json.JSONDecodeError, OSError):
                continue
    return cells


# --- metric extractors ------------------------------------------------------

# (title, y-label, extractor, value-format)
PANELS = [
    ("Time to First Audio p50", "seconds",
     lambda d: d["ttfa"]["p50"], "{:.3f}"),
    ("End-to-End Latency p50", "seconds",
     lambda d: d["e2e"]["p50"], "{:.2f}"),
    # Baseline logs only carry p95/p99 (no p90), so use p95 and label it.
    ("E2E p95", "seconds",
     lambda d: d["e2e"]["p95"], "{:.2f}"),
    ("Throughput", "x realtime",
     lambda d: d["throughput"], "{:.1f}"),
    ("Real-Time Factor (RTF)", "x (lower is better)",
     lambda d: d["avg_rtf"], "{:.3f}"),
]


def _series(cells: dict, length: str, extractor) -> tuple[list[int], list[float]]:
    xs, ys = [], []
    for c in CONCURRENCIES:
        cell = cells.get((length, c))
        if cell is None:
            continue
        try:
            ys.append(float(extractor(cell)))
            xs.append(c)
        except (KeyError, TypeError, ValueError):
            continue
    return xs, ys


# --- plotting ---------------------------------------------------------------

def make_figure(series: list[dict], out_path: str, dpi: int = 150) -> None:
    """``series`` is a list of dicts: {cells, label, color, ls, marker, dy}."""
    nrows = len(LENGTHS)
    ncols = len(PANELS)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(4.0 * ncols, 3.4 * nrows), squeeze=False
    )

    for r, length in enumerate(LENGTHS):
        for c_idx, (title, ylabel, extractor, vfmt) in enumerate(PANELS):
            ax = axes[r][c_idx]

            ys_all: list[float] = []
            for s in series:
                xs, ys = _series(s["cells"], length, extractor)
                if not xs:
                    continue
                ax.plot(xs, ys, color=s["color"], linestyle=s["ls"], marker=s["marker"],
                        markersize=5, linewidth=1.8, label=s["label"])
                for x, y in zip(xs, ys):
                    ax.annotate(vfmt.format(y), (x, y), textcoords="offset points",
                                xytext=(0, s["dy"]), ha="center", fontsize=6.5,
                                color=s["color"])
                ys_all += ys

            ax.set_xscale("log", base=2)
            ax.set_xticks(CONCURRENCIES)
            ax.set_xticklabels([str(c) for c in CONCURRENCIES])
            ax.set_xlim(CONCURRENCIES[0] * 0.85, CONCURRENCIES[-1] * 1.15)
            ax.grid(True, which="both", linestyle=":", alpha=0.4)
            ax.set_ylabel(ylabel, fontsize=8)
            if r == 0:
                ax.set_title(title, fontsize=10, fontweight="bold")
            if r == nrows - 1:
                ax.set_xlabel("concurrency", fontsize=8)
            ax.tick_params(labelsize=8)

            if ys_all:
                lo, hi = min(ys_all), max(ys_all)
                span = (hi - lo) or (abs(hi) or 1.0)
                ax.set_ylim(lo - 0.22 * span, hi + 0.22 * span)

        # Row-group label on the left.
        axes[r][0].annotate(
            LENGTH_LABELS[length], xy=(0, 0.5), xytext=(-58, 0),
            textcoords="offset points", xycoords="axes fraction",
            rotation=90, va="center", ha="center", fontsize=11, fontweight="bold",
        )

    handles = [
        plt.Line2D([0], [0], color=s["color"], linestyle=s["ls"], marker=s["marker"],
                   label=s["label"])
        for s in series
    ]
    fig.legend(handles=handles, loc="upper center", ncol=len(handles),
               bbox_to_anchor=(0.5, 0.985), fontsize=11, frameon=True)

    fig.suptitle(
        "sglang-omni vs vllm-omni (H100) — SpeechifyTTS MoE 4B",
        fontsize=15, fontweight="bold", y=1.0,
    )
    fig.tight_layout(rect=(0.02, 0.0, 1.0, 0.95))

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(
        description="Plot sglang-omni vs vllm-omni SpeechifyTTS comparison"
    )
    p.add_argument("--baseline-dir", default="/tmp/bench_h100/logs_branch022",
                   help="Directory of vllm-omni {length}_c{c}.log files.")
    p.add_argument("--sglang-dir", default=None,
                   help="Directory of sglang-omni STREAMING {length}_c{c}.json cells.")
    p.add_argument("--sglang-nostream-dir", default=None,
                   help="Directory of sglang-omni NON-STREAMING {length}_c{c}.json cells.")
    p.add_argument("--out", default="/tmp/bench_h100/compare_sglang_vs_vllm.png")
    p.add_argument("--dpi", type=int, default=150)
    args = p.parse_args()

    baseline = load_baseline(args.baseline_dir)
    sglang = load_sglang(args.sglang_dir)
    sglang_ns = load_sglang(args.sglang_nostream_dir)

    expected = len(LENGTHS) * len(CONCURRENCIES)
    print(f"Parsed {len(baseline)}/{expected} baseline cells from {args.baseline_dir}")
    print(f"Loaded {len(sglang)}/{expected} sglang streaming cells from "
          f"{args.sglang_dir or '(none)'}")
    print(f"Loaded {len(sglang_ns)}/{expected} sglang non-stream cells from "
          f"{args.sglang_nostream_dir or '(none)'}")

    if not baseline and not sglang and not sglang_ns:
        raise SystemExit("No data to plot.")

    series = []
    if sglang:
        series.append({"cells": sglang, "label": "sglang-omni (streaming)",
                       "color": SGLANG_COLOR, "ls": "-", "marker": "o", "dy": 7})
    if sglang_ns:
        series.append({"cells": sglang_ns, "label": "sglang-omni (non-streaming)",
                       "color": SGLANG_NOSTREAM_COLOR, "ls": "-.", "marker": "^", "dy": 7})
    if baseline:
        series.append({"cells": baseline, "label": "vllm-omni baseline (streaming)",
                       "color": VLLM_COLOR, "ls": "--", "marker": "s", "dy": -13})

    make_figure(series, args.out, dpi=args.dpi)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
