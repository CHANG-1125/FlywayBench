#!/usr/bin/env python3
"""Visualize `paper_experiment_report.json`: grid search + test-domain baselines (mean±std)."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _is_nan(x: object) -> bool:
    return isinstance(x, float) and math.isnan(x)


def _load_report(path: Path) -> dict:
    # Python json allows NaN/Infinity when allow_nan=True (default).
    return json.loads(path.read_text(encoding="utf-8"))


def _mean_std(block: dict | None, key: str) -> tuple[float, float]:
    if not block or key not in block:
        return float("nan"), float("nan")
    m = block[key].get("mean")
    s = block[key].get("std")
    if m is None or _is_nan(m):
        return float("nan"), float("nan")
    sm = 0.0 if s is None or _is_nan(s) else float(s)
    return float(m), sm


def plot_test_comparison(report: dict, out_path: Path) -> None:
    models = [
        ("FlywayGNN", "flywaygnn_test_mean_std", "#2c7fb8"),
        ("Vanilla GCN ($E_s$)", "vanilla_gcn_test_mean_std", "#7fcdbb"),
        ("XGBoost", "xgboost_test_mean_std", "#fdae61"),
    ]
    metrics = [
        ("macro-AUPRC (eval-m5)", "auprc_macro_eval_m5"),
        ("macro-AUC (eval-m5)", "auroc_macro_eval_m5"),
    ]

    n_m = len(metrics)
    means = np.zeros((len(models), n_m))
    stds = np.zeros((len(models), n_m))
    for mi, (_, key, _) in enumerate(models):
        blk = report.get(key) or {}
        for j, (_, mk) in enumerate(metrics):
            m, s = _mean_std(blk, mk)
            means[mi, j] = m
            stds[mi, j] = s

    valid_j = [j for j in range(n_m) if np.any(np.isfinite(means[:, j]))]
    if not valid_j:
        return
    metrics_v = [metrics[j] for j in valid_j]
    means_v = means[:, valid_j]
    stds_v = stds[:, valid_j]
    n_mv = len(valid_j)

    x = np.arange(n_mv)
    width = 0.26
    fig, ax = plt.subplots(figsize=(10, 5))
    for mi, (name, _, color) in enumerate(models):
        offset = (mi - 1) * width
        yerr = np.nan_to_num(stds_v[mi], nan=0.0)
        y = means_v[mi].copy()
        mask = np.isfinite(y)
        ax.bar(
            x[mask] + offset,
            y[mask],
            width,
            yerr=yerr[mask],
            capsize=3,
            label=name,
            color=color,
            edgecolor="white",
            linewidth=0.5,
        )
    ax.set_xticks(x)
    ax.set_xticklabels([m[0] for m in metrics_v], rotation=15, ha="right")
    ax.set_ylabel("Score (test domain, $\\phi < \\tau$)")
    ax.set_title("Test-domain performance: mean ± std over seeds")
    ax.legend(loc="upper right", fontsize=9)
    ymax = float(np.nanmax(means_v + stds_v)) if np.any(np.isfinite(means_v + stds_v)) else 0.1
    ax.set_ylim(0, min(1.0, max(0.08, ymax * 1.15)))
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_grid_heatmaps(report: dict, out_path: Path) -> None:
    trials = report.get("grid_trials") or []
    if not trials:
        return
    ks = sorted({int(t["K"]) for t in trials})
    lrs = sorted({float(t["lr"]) for t in trials})
    hids = sorted({int(t["hidden_dim"]) for t in trials})

    n_k = len(ks)
    n_cols = min(2, n_k)
    n_rows = int(math.ceil(n_k / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), squeeze=False)
    lr_labels = [f"{lr:g}" for lr in lrs]
    hid_labels = [str(h) for h in hids]

    for idx, K in enumerate(ks):
        r, c = divmod(idx, n_cols)
        ax = axes[r][c]
        mat = np.full((len(lrs), len(hids)), np.nan)
        for t in trials:
            if int(t["K"]) != K:
                continue
            i = lrs.index(float(t["lr"]))
            j = hids.index(int(t["hidden_dim"]))
            v = (t.get("val") or {}).get("auprc_macro")
            if v is not None and not (isinstance(v, float) and math.isnan(v)):
                mat[i, j] = float(v)
        if np.all(~np.isfinite(mat)):
            ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
            continue
        vmin, vmax = float(np.nanmin(mat)), float(np.nanmax(mat))
        if not math.isfinite(vmin) or not math.isfinite(vmax) or vmin == vmax:
            vmax = vmin + 1e-6
        im = ax.imshow(mat, aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax)
        ax.set_xticks(range(len(hids)))
        ax.set_xticklabels(hid_labels)
        ax.set_yticks(range(len(lrs)))
        ax.set_yticklabels(lr_labels)
        ax.set_xlabel("hidden dim")
        ax.set_ylabel("learning rate")
        ax.set_title(f"Val macro-AUPRC (grid), K={K}")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    for idx in range(len(ks), n_rows * n_cols):
        r, c = divmod(idx, n_cols)
        axes[r][c].axis("off")
    fig.suptitle("Hyperparameter grid (model selection on validation)", y=1.02, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_per_seed_lines(report: dict, out_path: Path) -> None:
    seeds = report.get("seeds") or list(range(5))
    series = [
        ("FlywayGNN", "per_seed_test_flywaygnn", "#2c7fb8"),
        ("Vanilla GCN", "per_seed_test_vanilla_gcn", "#7fcdbb"),
        ("XGBoost", "per_seed_test_xgboost", "#fdae61"),
    ]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    for name, key, color in series:
        rows = report.get(key) or []
        if not rows:
            continue
        auprc_m5 = [
            float(r["auprc_macro_eval_m5"]) if r.get("auprc_macro_eval_m5") is not None else float("nan") for r in rows
        ]
        auc_m5 = [
            float(r["auroc_macro_eval_m5"]) if r.get("auroc_macro_eval_m5") is not None else float("nan") for r in rows
        ]
        xs = seeds[: len(auprc_m5)]
        ax1.plot(xs, auprc_m5, "o-", color=color, label=name, linewidth=2, markersize=6)
        ax2.plot(xs, auc_m5, "o-", color=color, label=name, linewidth=2, markersize=6)
    ax1.set_xlabel("random seed")
    ax1.set_ylabel("macro-AUPRC (eval-m5)")
    ax1.set_title("Eval-m5: macro-AUPRC per seed")
    ax1.grid(alpha=0.3)
    ax1.legend(fontsize=9)
    ax2.set_xlabel("random seed")
    ax2.set_ylabel("macro-AUC (eval-m5)")
    ax2.set_title("Eval-m5: macro-AUC per seed")
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_channel_alpha(report: dict, out_path: Path) -> None:
    alphas = report.get("flywaygnn_alpha_last_seeds") or []
    if not alphas:
        return
    last = alphas[-1]
    l1 = last.get("layer1_alpha_SF") or [0.5, 0.5]
    l2 = last.get("layer2_alpha_SF") or [0.5, 0.5]
    fig, ax = plt.subplots(figsize=(6, 3.5))
    x = np.arange(2)
    w = 0.35
    ax.bar(x - w / 2, [l1[0], l2[0]], w, label=r"$\alpha_S$ (spatial)", color="#3182bd")
    ax.bar(x + w / 2, [l1[1], l2[1]], w, label=r"$\alpha_F$ (flyway)", color="#e6550d")
    ax.set_xticks(x)
    ax.set_xticklabels(["Layer 1", "Layer 2"])
    ax.set_ylabel("softmax weight")
    ax.set_title("FlywayGNN: learned channel weights (last saved run)")
    ax.legend()
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_ef_vs_val(report: dict, out_path: Path) -> None:
    trials = report.get("grid_trials") or []
    if not trials:
        return
    xs = [int(t.get("E_f_size", 0)) for t in trials]
    ys = []
    for t in trials:
        v = (t.get("val") or {}).get("auprc_macro")
        ys.append(float(v) if v is not None and not (isinstance(v, float) and math.isnan(v)) else float("nan"))
    fig, ax = plt.subplots(figsize=(6.5, 4))
    sc = ax.scatter(xs, ys, c=[float(t["lr"]) for t in trials], cmap="coolwarm", s=80, alpha=0.85, edgecolors="k", linewidths=0.3)
    ax.set_xlabel(r"$|\mathcal{E}_F|$ (undirected edge pairs × 2 in builder set)")
    ax.set_ylabel("Val macro-AUPRC")
    ax.set_title("Grid trials: corridor edge count vs validation score")
    plt.colorbar(sc, ax=ax, label="learning rate")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def write_summary_md(report: dict, fig_dir: Path, out_md: Path) -> None:
    best = report.get("best_config") or {}
    lines = [
        "# Paper experiment snapshot",
        "",
        f"- **Best val macro-AUPRC (grid):** {report.get('best_val_macro_auprc_grid')}",
        f"- **Best config:** `{json.dumps(best)}`",
        "",
        "## Figures",
        "",
    ]
    for name in sorted(fig_dir.glob("*.png")):
        lines.append(f"- ![{name.stem}]({name.name})")
    out_md.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--report-json",
        type=Path,
        default=None,
        help="Path to paper_experiment_report.json (default: newest under runs/paper_experiments/)",
    )
    ap.add_argument("--out-dir", type=Path, default=None, help="Figure output directory (default: <report_dir>/figures)")
    args = ap.parse_args()

    if args.report_json is None:
        root = Path(__file__).resolve().parent / "runs" / "paper_experiments"
        candidates = sorted(root.glob("*/paper_experiment_report.json"), key=lambda p: p.stat().st_mtime)
        if not candidates:
            raise SystemExit(f"No report found under {root}")
        report_path = candidates[-1]
    else:
        report_path = args.report_json
    if not report_path.is_file():
        raise SystemExit(f"Missing report: {report_path}")

    report = _load_report(report_path)
    out_dir = args.out_dir or (report_path.parent / "figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_test_comparison(report, out_dir / "test_domain_mean_std.png")
    plot_grid_heatmaps(report, out_dir / "grid_val_macro_auprc_heatmaps.png")
    plot_per_seed_lines(report, out_dir / "per_seed_test_macro.png")
    plot_channel_alpha(report, out_dir / "flywaygnn_channel_alpha.png")
    plot_ef_vs_val(report, out_dir / "grid_ef_size_vs_val_auprc.png")
    write_summary_md(report, out_dir, out_dir / "README_figures.md")

    print("Wrote figures to:", out_dir.resolve())
    for p in sorted(out_dir.glob("*.png")):
        print(" ", p.name)


if __name__ == "__main__":
    main()
