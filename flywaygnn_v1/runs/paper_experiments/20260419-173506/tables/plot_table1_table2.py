#!/usr/bin/env python3
"""Publication-style bar charts from table1 / table2 CSV (mean±std cells)."""

from __future__ import annotations

import csv
import math
import re
from pathlib import Path


def parse_pm(cell: str) -> tuple[float | None, float | None]:
    cell = (cell or "").strip()
    if not cell or cell in ("—", "-", "NA", "n/a"):
        return None, None
    if "±" in cell:
        a, b = cell.split("±", 1)
        return float(a), float(b)
    m = re.match(r"^([+-]?\d*\.?\d+)\s*\+/-\s*([+-]?\d*\.?\d+)$", cell)
    if m:
        return float(m.group(1)), float(m.group(2))
    return float(cell), 0.0


def load_table1(path: Path) -> tuple[list[str], list[str], list[list[float]], list[list[float]]]:
    rows = list(csv.DictReader(path.read_text(encoding="utf-8").splitlines()))
    models = [r["model"].replace("_", " ") for r in rows]
    skip = {"model", "n_seeds", "eval_domain_note"}
    metrics = [k for k in rows[0].keys() if k not in skip]
    means: list[list[float]] = []
    stds: list[list[float]] = []
    for r in rows:
        mrow, srow = [], []
        for k in metrics:
            mu, sd = parse_pm(r.get(k, ""))
            if mu is None:
                mrow.append(float("nan"))
                srow.append(0.0)
            else:
                mrow.append(mu)
                srow.append(sd if sd is not None else 0.0)
        means.append(mrow)
        stds.append(srow)
    keep_idx = [j for j in range(len(metrics)) if any(math.isfinite(means[i][j]) for i in range(len(models)))]
    metrics = [metrics[j] for j in keep_idx]
    means = [[row[j] for j in keep_idx] for row in means]
    stds = [[row[j] for j in keep_idx] for row in stds]
    return models, metrics, means, stds


def load_table2(path: Path) -> tuple[list[str], list[str], list[list[float]], list[list[float]]]:
    rows = list(csv.DictReader(path.read_text(encoding="utf-8").splitlines()))
    models = [r["model"].replace("_", " ") for r in rows]
    skip = {
        "model",
        "n_seeds",
        "n_species_eval_m5",
        "eval_domain_note",
        "win_tie_loss_vs_vanilla_species_auprc_eval_m5",
        "auprc_eval_m5_one_sided_p_vs_vanilla",
    }
    metrics = [k for k in rows[0].keys() if k not in skip]
    means, stds = [], []
    for r in rows:
        mrow, srow = [], []
        for k in metrics:
            mu, sd = parse_pm(r.get(k, ""))
            mrow.append(mu if mu is not None else float("nan"))
            srow.append(sd if sd is not None else 0.0)
        means.append(mrow)
        stds.append(srow)
    return models, metrics, means, stds


def plot_grouped_bars(
    models: list[str],
    metrics: list[str],
    means: list[list[float]],
    stds: list[list[float]],
    *,
    title: str,
    ylabel: str,
    out_path: Path,
    colors: list[str],
) -> None:
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import numpy as np

    mpl.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    n_m, n_met = len(models), len(metrics)
    x = np.arange(n_met, dtype=float)
    width = min(0.22, 0.78 / max(n_m, 1))
    fig, ax = plt.subplots(figsize=(max(7.2, 1.15 * n_met + 2.2), 4.35))

    for mi, name in enumerate(models):
        offset = (mi - (n_m - 1) / 2) * width
        y = np.array([means[mi][j] for j in range(n_met)], dtype=float)
        e = np.array([stds[mi][j] for j in range(n_met)], dtype=float)
        ax.bar(
            x + offset,
            y,
            width,
            yerr=e,
            capsize=3,
            label=name,
            color=colors[mi % len(colors)],
            edgecolor="white",
            linewidth=0.65,
            zorder=2,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([m.replace("_", " ") for m in metrics], rotation=24, ha="right", fontsize=9)
    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=10)
    ymax = float(np.nanmax(np.array(means, dtype=float) + np.array(stds, dtype=float)))
    ax.set_ylim(0, min(1.08, ymax * 1.2 + 0.02))
    ax.grid(axis="y", alpha=0.3, zorder=0)
    ax.legend(loc="upper right", framealpha=0.95, ncol=1)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close(fig)


def main() -> None:
    here = Path(__file__).resolve().parent
    t1 = here / "table1_main_metrics_test_domain.csv"
    t2 = here / "table2_eval_m5.csv"
    out_dir = here / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    colors = ["#2c7fb8", "#7fcdbb", "#fdae61"]

    models, metrics, means, stds = load_table1(t1)
    plot_grouped_bars(
        models,
        metrics,
        means,
        stds,
        title="Table 1 — Extrapolation test domain (φ < τ), mean ± std over seeds",
        ylabel="Score",
        out_path=out_dir / "table1_main_metrics.png",
        colors=colors,
    )

    models2, metrics2, means2, stds2 = load_table2(t2)
    plot_grouped_bars(
        models2,
        metrics2,
        means2,
        stds2,
        title="Table 2 — Eval-m5 subset (macro-AUPRC / macro-AUC): mean ± std over seeds",
        ylabel="Score",
        out_path=out_dir / "table2_eval_m5.png",
        colors=colors,
    )

    print("Wrote:", out_dir / "table1_main_metrics.png")
    print("Wrote:", out_dir / "table2_eval_m5.png")


if __name__ == "__main__":
    main()
