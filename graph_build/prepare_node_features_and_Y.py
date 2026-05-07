#!/usr/bin/env python3
"""
Prepare graph inputs:
  1) Node features: read GEE-export CSV, align to grid_cells_active_baseline;
     impute NDVI/LST missings with median on a train subset; DEM missing → 0 m (sea level);
     add binary missing indicators per imputed dimension.
  2) Label matrix: expand baseline_Y_static_2000_2025.csv (long table of positives) to a 4554×155 binary wide matrix.

Writes under graph_build/: *_imputed.csv, Y_*.csv, split_meta.json, species_column_order.json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
PIPE = BASE.parent
GRID_ORDER = PIPE / "grid_cells_active_baseline.csv"
BASELINE_Y = PIPE / "baseline_Y_static_2000_2025.csv"
DEFAULT_FEATURES = PIPE.parent / "grid_node_features_2000_2025.csv"


def _median_nonnull(s: pd.Series) -> float:
    v = pd.to_numeric(s, errors="coerce")
    m = float(v.median())
    if np.isnan(m):
        m = float(v.dropna().median())
    return m


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--features",
        type=Path,
        default=DEFAULT_FEATURES,
        help="GEE-exported grid_node_features CSV",
    )
    ap.add_argument("--train-frac", type=float, default=0.7, help="Fraction of nodes used to estimate imputation stats")
    ap.add_argument("--seed", type=int, default=42, help="RNG seed for train subset split")
    args = ap.parse_args()

    out_dir = BASE
    out_dir.mkdir(parents=True, exist_ok=True)

    grid = pd.read_csv(GRID_ORDER)
    n = len(grid)
    if n == 0:
        raise SystemExit(f"Empty grid list: {GRID_ORDER}")

    feat_path = args.features
    if not feat_path.exists():
        raise SystemExit(f"Missing features CSV: {feat_path}")

    raw = pd.read_csv(feat_path)
    drop_cols = [c for c in ("system:index", ".geo", "mask_keep") if c in raw.columns]
    if drop_cols:
        raw = raw.drop(columns=drop_cols)

    if "DEM_mean" in raw.columns and "DEM" not in raw.columns:
        raw = raw.rename(columns={"DEM_mean": "DEM"})

    need = {"cell_lat", "cell_lon", "NDVI_mean", "LST_day_mean", "DEM", "water_frac"}
    miss_cols = need - set(raw.columns)
    if miss_cols:
        raise SystemExit(f"Features CSV missing columns: {sorted(miss_cols)}")

    merged = grid.merge(raw, on=["cell_lat", "cell_lon"], how="left", validate="one_to_one")
    if merged["NDVI_mean"].isna().all() and raw.shape[0] != n:
        raise SystemExit("Merge failed: check cell_lat/cell_lon alignment with grid list.")
    if len(merged) != n:
        raise SystemExit(f"Row count after merge {len(merged)} != grid {n}")

    rng = np.random.default_rng(args.seed)
    train_mask = rng.random(n) < args.train_frac
    train_idx = np.where(train_mask)[0]

    ndvi0 = pd.to_numeric(merged["NDVI_mean"], errors="coerce")
    lst0 = pd.to_numeric(merged["LST_day_mean"], errors="coerce")
    dem0 = pd.to_numeric(merged["DEM"], errors="coerce")
    wf = pd.to_numeric(merged["water_frac"], errors="coerce")

    med_ndvi = _median_nonnull(ndvi0.iloc[train_idx])
    med_lst = _median_nonnull(lst0.iloc[train_idx])
    if np.isnan(med_ndvi):
        med_ndvi = _median_nonnull(ndvi0)
    if np.isnan(med_lst):
        med_lst = _median_nonnull(lst0)

    miss_ndvi = ndvi0.isna().astype(np.int8)
    miss_lst = lst0.isna().astype(np.int8)
    miss_dem = dem0.isna().astype(np.int8)

    ndvi_f = ndvi0.fillna(med_ndvi)
    lst_f = lst0.fillna(med_lst)
    dem_f = dem0.fillna(0.0)

    out = pd.DataFrame(
        {
            "node_idx": np.arange(n, dtype=np.int32),
            "cell_lat": merged["cell_lat"].astype(np.int32),
            "cell_lon": merged["cell_lon"].astype(np.int32),
            "NDVI_mean": ndvi_f.astype(np.float64),
            "LST_day_mean": lst_f.astype(np.float64),
            "DEM": dem_f.astype(np.float64),
            "water_frac": wf.astype(np.float64),
            "miss_NDVI": miss_ndvi,
            "miss_LST": miss_lst,
            "miss_DEM": miss_dem,
            "is_train_impute": train_mask.astype(np.int8),
        }
    )

    feat_csv = out_dir / "node_features_imputed_4554.csv"
    out.to_csv(feat_csv, index=False, lineterminator="\n")

    y_long = pd.read_csv(BASELINE_Y)
    if "y" not in y_long.columns or "SCIENTIFIC NAME" not in y_long.columns:
        raise SystemExit(f"Unexpected baseline columns: {y_long.columns.tolist()}")
    species = sorted(y_long["SCIENTIFIC NAME"].astype(str).str.strip().unique())
    if len(species) != int(y_long["SCIENTIFIC NAME"].nunique()):
        raise SystemExit("Species name normalization produced duplicates.")

    y_wide = (
        y_long.pivot_table(
            index=["cell_lat", "cell_lon"],
            columns="SCIENTIFIC NAME",
            values="y",
            aggfunc="max",
            fill_value=0,
        )
        .reindex(columns=species, fill_value=0)
        .astype(np.int8)
    )
    y_wide = y_wide.reset_index()
    y_ordered = grid.merge(y_wide, on=["cell_lat", "cell_lon"], how="left", validate="one_to_one")
    sp_cols = [c for c in y_ordered.columns if c not in ("cell_lat", "cell_lon")]
    y_ordered[sp_cols] = y_ordered[sp_cols].fillna(0).astype(np.int8)
    if y_ordered[sp_cols].sum().sum() != int((y_long["y"] == 1).sum()):
        raise SystemExit("Y matrix positive count mismatch after pivot; check baseline file.")

    y_csv = out_dir / "Y_binary_4554x155.csv"
    y_ordered.to_csv(y_csv, index=False, lineterminator="\n")

    meta = {
        "n_nodes": n,
        "n_species": len(species),
        "feature_columns": [
            "NDVI_mean",
            "LST_day_mean",
            "DEM",
            "water_frac",
            "miss_NDVI",
            "miss_LST",
            "miss_DEM",
        ],
        "d_x": 7,
        "imputation": {
            "NDVI_LST": "median on train nodes (non-null); fallback global median if needed",
            "DEM_na": "fill 0.0 (sea level)",
            "indicators": "miss_NDVI, miss_LST, miss_DEM are 1 iff value was NaN before imputation",
        },
        "train_frac_for_impute_stats": args.train_frac,
        "seed": args.seed,
        "median_ndvi_train": med_ndvi,
        "median_lst_train": med_lst,
        "n_miss_ndvi": int(miss_ndvi.sum()),
        "n_miss_lst": int(miss_lst.sum()),
        "n_miss_dem": int(miss_dem.sum()),
        "paths": {
            "node_features": str(feat_csv),
            "Y_wide": str(y_csv),
            "species_order": str(out_dir / "species_column_order.json"),
        },
    }
    (out_dir / "split_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    (out_dir / "species_column_order.json").write_text(
        json.dumps(species, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("Wrote:", feat_csv)
    print("Wrote:", y_csv)
    print("Wrote:", out_dir / "split_meta.json")
    print("d_x =", meta["d_x"], "| species =", len(species))


if __name__ == "__main__":
    main()
