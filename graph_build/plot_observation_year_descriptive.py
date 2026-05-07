#!/usr/bin/env python3
"""
Descriptive plots by year for high-quality observations strictly inside the Americas flyway corridor.

Input:
  stats_dedup_gridded_by_year.csv (repository root; produced by run_312_313.py)
Output:
  graph_build/figures/observations_by_year_descriptive.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
IN_CSV = BASE / "stats_dedup_gridded_by_year.csv"
OUT_DIR = BASE / "graph_build" / "figures"
OUT_PNG = OUT_DIR / "observations_by_year_descriptive.png"


def main() -> None:
    if not IN_CSV.exists():
        raise SystemExit(f"Missing input: {IN_CSV}")

    df = pd.read_csv(IN_CSV).sort_values("year")
    years = df["year"].to_numpy(dtype=int)
    n = df["rows_after_dedup_gridded"].to_numpy(dtype=np.int64)
    total = int(n.sum())
    share = n / total * 100.0
    cum_share = np.cumsum(n) / total * 100.0

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.sans-serif": ["DejaVu Sans"],
            "axes.unicode_minus": False,
            "figure.facecolor": "#ffffff",
        }
    )

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12.5, 8.8), dpi=180, sharex=True)
    fig.patch.set_facecolor("#ffffff")

    # Panel A: yearly counts (bars) + YoY growth (line)
    ax1.bar(years, n, color="#4e79a7", alpha=0.9, width=0.82, label="Yearly observations")
    ax1.set_ylabel("Observation count")
    ax1.grid(axis="y", alpha=0.25, linestyle=":")
    ax1.set_title("A. Yearly observations and YoY change", loc="left", fontsize=12)

    yoy = np.empty_like(n, dtype=float)
    yoy[0] = np.nan
    yoy[1:] = (n[1:] - n[:-1]) / n[:-1] * 100.0
    ax1r = ax1.twinx()
    ax1r.plot(years, yoy, color="#e15759", marker="o", linewidth=1.8, markersize=3.8, label="YoY growth (%)")
    ax1r.set_ylabel("YoY growth (%)", color="#e15759")
    ax1r.tick_params(axis="y", colors="#e15759")

    # Panel B: yearly share + cumulative share
    ax2.bar(years, share, color="#59a14f", alpha=0.82, width=0.82, label="Yearly share (%)")
    ax2.plot(years, cum_share, color="#f28e2b", marker="o", linewidth=2.0, markersize=3.8, label="Cumulative share (%)")
    ax2.set_ylabel("Share (%)")
    ax2.set_xlabel("Year")
    ax2.grid(axis="y", alpha=0.25, linestyle=":")
    ax2.set_title("B. Yearly share and cumulative share", loc="left", fontsize=12)
    ax2.set_ylim(0, 102)

    p50_idx = int(np.argmax(cum_share >= 50.0))
    p80_idx = int(np.argmax(cum_share >= 80.0))
    ax2.axhline(50, color="#999999", linestyle="--", linewidth=0.8, alpha=0.7)
    ax2.axhline(80, color="#999999", linestyle="--", linewidth=0.8, alpha=0.7)
    ax2.text(years[p50_idx], 51.5, f"50% reached in {years[p50_idx]}", fontsize=9, color="#555555")
    ax2.text(years[p80_idx], 81.5, f"80% reached in {years[p80_idx]}", fontsize=9, color="#555555")

    fig.suptitle(
        f"Observations inside Americas flyway corridor by year (2000–2025, total {total:,})",
        fontsize=14,
        y=0.98,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(OUT_PNG, dpi=220)
    plt.close(fig)
    print(f"Wrote: {OUT_PNG}")


if __name__ == "__main__":
    main()
