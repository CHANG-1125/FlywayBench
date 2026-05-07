"""
Paired seed-level comparison: FlywayGNN vs Vanilla GCN on test-domain metrics.

- mean ± std per model (same as report aggregates)
- paired difference per seed
- bootstrap percentile CI for E[diff] (resample seeds with replacement)
- paired t-test (two-sided) and Wilcoxon signed-rank (two-sided + Flyway>Vanilla one-sided)
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

try:
    from scipy import stats
except ImportError:  # pragma: no cover
    stats = None  # type: ignore[assignment]


def _row_vec(rows: list[dict[str, float]], key: str) -> np.ndarray:
    out = []
    for r in rows:
        v = r.get(key)
        if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
            out.append(float("nan"))
        else:
            out.append(float(v))
    return np.asarray(out, dtype=np.float64)


def _mean_std(x: np.ndarray) -> tuple[float, float]:
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan"), float("nan")
    return float(x.mean()), float(x.std(ddof=1)) if x.size > 1 else (float(x.mean()), 0.0)


def bootstrap_mean_ci(
    d: np.ndarray,
    *,
    n_boot: int,
    alpha: float,
    rng: np.random.Generator,
) -> tuple[float, float]:
    """Percentile CI for the mean of paired differences (nonparametric bootstrap over seeds)."""
    d = d[np.isfinite(d)]
    n = int(d.size)
    if n == 0:
        return float("nan"), float("nan")
    if n == 1:
        return float(d[0]), float(d[0])
    means = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n, endpoint=False)
        means[b] = float(d[idx].mean())
    lo = float(np.percentile(means, 100 * alpha / 2))
    hi = float(np.percentile(means, 100 * (1 - alpha / 2)))
    return lo, hi


def paired_compare_flyway_vs_vanilla(
    flyway_rows: list[dict[str, float]],
    vanilla_rows: list[dict[str, float]],
    *,
    seeds: list[int],
    metrics: tuple[str, ...] = ("auprc_macro", "f1_macro"),
    bootstrap_b: int = 10_000,
    alpha: float = 0.05,
    rng_seed: int = 42,
) -> dict[str, Any]:
    if len(flyway_rows) != len(vanilla_rows):
        raise ValueError("Mismatched number of Flyway vs Vanilla rows")
    n = len(flyway_rows)
    rng = np.random.default_rng(rng_seed)

    out: dict[str, Any] = {
        "n_seeds": n,
        "seeds": list(seeds),
        "alpha": alpha,
        "bootstrap_B": bootstrap_b,
        "scipy_available": stats is not None,
        "metrics": {},
    }

    lines_en: list[str] = []
    if stats is None:
        lines_en.append(
            "(scipy not installed: skipped paired t-test and Wilcoxon; only Bootstrap CI for mean paired difference. `pip install scipy`)"
        )

    for key in metrics:
        f = _row_vec(flyway_rows, key)
        v = _row_vec(vanilla_rows, key)
        mask = np.isfinite(f) & np.isfinite(v)
        d = (f - v)[mask]
        n_eff = int(d.size)
        if n_eff < 2:
            continue
        mf, sf = _mean_std(f[mask])
        mv, sv = _mean_std(v[mask])
        md, sd = _mean_std(d)
        seeds_used = [int(seeds[i]) for i in range(min(len(seeds), len(mask))) if mask[i]]

        lo, hi = bootstrap_mean_ci(d, n_boot=bootstrap_b, alpha=alpha, rng=rng)

        t_p = wilcox_two = wilcox_greater = float("nan")
        if stats is not None and n_eff >= 2:
            try:
                t_p = float(stats.ttest_rel(f[mask], v[mask], nan_policy="omit").pvalue)
            except Exception:
                t_p = float("nan")
            try:
                if n_eff >= 2 and np.any(d != 0):
                    wilcox_two = float(stats.wilcoxon(d, alternative="two-sided").pvalue)
            except Exception:
                wilcox_two = float("nan")
            try:
                if n_eff >= 2 and np.any(d > 0):
                    wilcox_greater = float(stats.wilcoxon(d, alternative="greater").pvalue)
            except Exception:
                wilcox_greater = float("nan")

        ci_excludes_zero = np.isfinite(lo) and np.isfinite(hi) and (lo > 0 or hi < 0)
        ci_flyway_greater = np.isfinite(lo) and lo > 0
        one_sided_sig = np.isfinite(wilcox_greater) and wilcox_greater < alpha

        label_map = {
            "auprc_macro": "macro-AUPRC",
            "f1_macro": "macro-F1",
            "auprc_micro": "micro-AUPRC",
            "f1_micro": "micro-F1",
            "auroc_micro": "micro-AUROC",
            "auroc_macro": "macro-AUROC",
            "auprc_macro_eval_m5": "macro-AUPRC (eval-m5 subset)",
            "auroc_macro_eval_m5": "macro-AUC (eval-m5 subset)",
        }
        metric_name = label_map.get(key, key)
        lines_en.append(
            f"[{metric_name}] paired seeds n={n_eff}: Flyway mean {mf:.4f}±{sf:.4f}, Vanilla mean {mv:.4f}±{sv:.4f}; "
            f"mean paired diff (Flyway−Vanilla) {md:+.4f}±{sd:.4f}; Bootstrap({bootstrap_b}) "
            f"{(1-alpha)*100:.0f}% CI for mean diff [{lo:+.4f}, {hi:+.4f}]."
        )
        if ci_flyway_greater:
            lines_en.append("  CI strictly above 0 (consistent with Flyway > Vanilla).")
        elif ci_excludes_zero:
            lines_en.append("  CI excludes 0 but favors Vanilla (upper bound < 0).")
        else:
            lines_en.append("  CI includes 0: mean paired difference not significantly away from 0 (nonparametric Bootstrap).")
        if stats is not None and np.isfinite(t_p):
            lines_en.append(f"  Paired t-test (two-sided) p={t_p:.4g}.")
        if stats is not None and np.isfinite(wilcox_two):
            lines_en.append(
                f"  Wilcoxon signed-rank (two-sided) p={wilcox_two:.4g}; one-sided Flyway>Vanilla p="
                f"{wilcox_greater if np.isfinite(wilcox_greater) else float('nan'):.4g}."
            )

        out["metrics"][key] = {
            "label": metric_name,
            "flyway_mean_std": {"mean": mf, "std": sf},
            "vanilla_mean_std": {"mean": mv, "std": sv},
            "paired_diff_per_seed": [float(x) for x in d.tolist()],
            "paired_seeds_used": seeds_used,
            "diff_mean": md,
            "diff_std": sd,
            "bootstrap_ci_mean_diff": {"lower": lo, "upper": hi, "level": 1 - alpha},
            "paired_ttest_two_sided_pvalue": t_p,
            "wilcoxon_two_sided_pvalue": wilcox_two,
            "wilcoxon_greater_flyway_pvalue": wilcox_greater,
            "flags": {
                "ci_mean_diff_excludes_zero": bool(ci_excludes_zero),
                "ci_mean_diff_entirely_positive_flyway_better": bool(ci_flyway_greater),
                "wilcoxon_greater_p_lt_alpha": bool(one_sided_sig),
            },
        }

    out["narrative_en"] = "\n".join(lines_en)
    return out


def _load_report(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    ap = argparse.ArgumentParser(description="Paired FlywayGNN vs Vanilla stats from paper_experiment_report.json")
    ap.add_argument("--report-json", type=Path, required=True)
    ap.add_argument("--bootstrap-b", type=int, default=10_000)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--rng-seed", type=int, default=42)
    ap.add_argument(
        "--metrics",
        type=str,
        default="auprc_macro_eval_m5,auroc_macro_eval_m5",
        help="Comma-separated metric keys present in per-seed dicts",
    )
    args = ap.parse_args()

    rep = _load_report(args.report_json)
    fly = rep.get("per_seed_test_flywaygnn") or []
    van = rep.get("per_seed_test_vanilla_gcn") or []
    seeds = rep.get("seeds") or list(range(len(fly)))
    mkeys = tuple(c.strip() for c in args.metrics.split(",") if c.strip())
    block = paired_compare_flyway_vs_vanilla(
        fly, van, seeds=seeds, metrics=mkeys, bootstrap_b=args.bootstrap_b, alpha=args.alpha, rng_seed=args.rng_seed
    )
    out_path = args.report_json.parent / "flyway_vs_vanilla_paired_stats.json"
    out_path.write_text(json.dumps(block, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    print(block["narrative_en"])
    print("\nWrote:", out_path)


if __name__ == "__main__":
    main()
