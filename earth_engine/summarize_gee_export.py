#!/usr/bin/env python3
"""Validate and summarize a downloaded GEE export CSV (no habitat masking)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parent
IN_CSV = BASE / "output" / "grid_node_features_2000_2025.csv"
OUT = BASE / "output" / "gee_feature_export_summary.csv"

if not IN_CSV.exists():
    raise SystemExit(f"Missing: {IN_CSV}")

df = pd.read_csv(IN_CSV)
base_cols = ["cell_lat", "cell_lon", "NDVI_mean", "LST_day_mean", "water_frac"]
missing = [c for c in base_cols if c not in df.columns]
if "DEM_mean" not in df.columns and "DEM" in df.columns:
    df = df.rename(columns={"DEM": "DEM_mean"})
elif "DEM_mean" not in df.columns:
    missing.append("DEM_mean")

summary = {
    "n_rows": len(df),
    "missing_expected_columns": ";".join(missing) if missing else "",
    "columns": ";".join(df.columns.astype(str).tolist()),
}
if "water_frac" in df.columns:
    wf = pd.to_numeric(df["water_frac"], errors="coerce")
    summary["water_frac_min"] = float(wf.min())
    summary["water_frac_max"] = float(wf.max())
    summary["water_frac_mean"] = float(wf.mean())

pd.DataFrame([summary]).to_csv(OUT, index=False, encoding="utf-8", lineterminator="\n")
print("Rows:", len(df))
if missing:
    print("NOTE: expected columns missing:", missing)
print("Wrote:", OUT)
