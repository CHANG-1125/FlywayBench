#!/usr/bin/env python3
"""
§3.1.2 Gridding: WGS84, 1°×1°, cell_lat=floor(lat), cell_lon=floor(lon).
§3.1.3 Debiasing: \\hat{c}_{s,l,t} = max count within same (species, lat, lon, calendar day); drop \\hat{c}=0 and invalid coords.
Grid–year–species intensity \\hat{c}_{i,s,t}: sum of dedup weights w in cell-year-species (numeric counts; X→1).
Static baseline y_{i,s} = I( sum_{t in T} \\hat{c}_{i,s,t} > 0 ), T=2000–2025 (26 years).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
SRC = BASE.parent / "ebird_overlap_155_scientific_americas_clipped"
OUT = BASE
DEDUP_DIR = OUT / "deduped_gridded"
CHUNK = 200_000
YEARS_T = list(range(2000, 2026))  # 2000–2025


def max_observation_count(series: pd.Series) -> object:
    n = pd.to_numeric(series, errors="coerce")
    is_x = series.astype(str).str.strip().str.upper().eq("X")
    nmax = n.max()
    if np.isfinite(nmax) and nmax > 0:
        return int(nmax) if float(nmax).is_integer() else float(nmax)
    if is_x.any():
        return "X"
    return np.nan


def chunk_dedup(chunk: pd.DataFrame) -> pd.DataFrame:
    df = chunk.copy()
    df["SCIENTIFIC NAME"] = df["SCIENTIFIC NAME"].astype(str).str.strip()
    d = pd.to_datetime(df["OBSERVATION DATE"], errors="coerce")
    df["_day"] = d.dt.normalize()
    df = df[df["_day"].notna()]
    lat = pd.to_numeric(df["LATITUDE"], errors="coerce")
    lon = pd.to_numeric(df["LONGITUDE"], errors="coerce")
    df["_lat"] = lat
    df["_lon"] = lon
    df = df[lat.notna() & lon.notna()]
    if df.empty:
        return df.iloc[0:0]

    df["cell_lat"] = np.floor(df["_lat"]).astype(int)
    df["cell_lon"] = np.floor(df["_lon"]).astype(int)

    keys = ["SCIENTIFIC NAME", "_lat", "_lon", "_day"]
    oc_max = df.groupby(keys, sort=False)["OBSERVATION COUNT"].agg(max_observation_count)
    oc_max = oc_max.rename("OBSERVATION COUNT").reset_index()
    first = df.groupby(keys, sort=False).first().reset_index()
    keep_cols = [
        "COMMON NAME",
        "COUNTRY",
        "STATE",
        "COUNTY",
        "LOCALITY",
        "OBSERVER ID",
        "SAMPLING EVENT IDENTIFIER",
        "OBSERVATION TYPE",
        "DURATION MINUTES",
        "cell_lat",
        "cell_lon",
    ]
    meta = first[keys + [c for c in keep_cols if c in first.columns]]
    out = oc_max.merge(meta, on=keys, how="left")
    out["LATITUDE"] = out["_lat"]
    out["LONGITUDE"] = out["_lon"]
    out["OBSERVATION DATE"] = out["_day"].dt.strftime("%Y-%m-%d")
    out = out.drop(columns=["_day", "_lat", "_lon"])

    out = out[out["OBSERVATION COUNT"].notna()]
    num = pd.to_numeric(out["OBSERVATION COUNT"], errors="coerce")
    is_x = out["OBSERVATION COUNT"].astype(str).str.strip().str.upper().eq("X")
    out = out[(num > 0) | is_x]
    return out


def merge_boundary_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    df["_day"] = pd.to_datetime(df["OBSERVATION DATE"], errors="coerce").dt.normalize()
    keys = ["SCIENTIFIC NAME", "LATITUDE", "LONGITUDE", "_day"]
    oc_max = df.groupby(keys, sort=False)["OBSERVATION COUNT"].agg(max_observation_count)
    oc_max = oc_max.rename("OBSERVATION COUNT").reset_index()
    first = df.groupby(keys, sort=False).first().reset_index()
    meta_cols = [
        c
        for c in [
            "COMMON NAME",
            "cell_lat",
            "cell_lon",
            "COUNTRY",
            "STATE",
            "COUNTY",
            "LOCALITY",
            "OBSERVER ID",
            "SAMPLING EVENT IDENTIFIER",
            "OBSERVATION TYPE",
            "DURATION MINUTES",
        ]
        if c in first.columns
    ]
    out = oc_max.merge(first[keys + meta_cols], on=keys, how="left")
    out["OBSERVATION DATE"] = out["_day"].dt.strftime("%Y-%m-%d")
    out = out.drop(columns=["_day"])
    num = pd.to_numeric(out["OBSERVATION COUNT"], errors="coerce")
    is_x = out["OBSERVATION COUNT"].astype(str).str.strip().str.upper().eq("X")
    out = out[out["OBSERVATION COUNT"].notna() & ((num > 0) | is_x)]
    return out


def process_year(year: int) -> tuple[int, int]:
    path = SRC / f"{year}.csv"
    if not path.exists():
        return 0, 0

    DEDUP_DIR.mkdir(parents=True, exist_ok=True)
    out_path = DEDUP_DIR / f"{year}.csv"
    rows_in = 0
    parts: list[pd.DataFrame] = []

    for chunk in pd.read_csv(path, chunksize=CHUNK, low_memory=False):
        rows_in += len(chunk)
        sub = chunk_dedup(chunk)
        if len(sub):
            parts.append(sub)

    if not parts:
        pd.DataFrame().to_csv(out_path, index=False)
        return rows_in, 0

    merged = pd.concat(parts, ignore_index=True)
    final = merge_boundary_duplicates(merged)
    final["year"] = year

    cols = [
        "cell_lat",
        "cell_lon",
        "COMMON NAME",
        "SCIENTIFIC NAME",
        "OBSERVATION DATE",
        "year",
        "LATITUDE",
        "LONGITUDE",
        "COUNTRY",
        "STATE",
        "COUNTY",
        "LOCALITY",
        "OBSERVATION COUNT",
        "OBSERVER ID",
        "SAMPLING EVENT IDENTIFIER",
        "OBSERVATION TYPE",
        "DURATION MINUTES",
    ]
    for c in cols:
        if c not in final.columns:
            final[c] = np.nan
    final = final[cols]
    final.to_csv(out_path, index=False, encoding="utf-8", lineterminator="\n")
    return rows_in, len(final)


def numeric_for_sum(val) -> float:
    if isinstance(val, str) and val.strip().upper() == "X":
        return 1.0
    x = pd.to_numeric(val, errors="coerce")
    return float(x) if np.isfinite(x) and x > 0 else 0.0


def main() -> None:
    if not SRC.is_dir():
        print("Missing source:", SRC, file=sys.stderr)
        sys.exit(1)

    stats = []
    for y in YEARS_T:
        ri, ro = process_year(y)
        stats.append({"year": y, "rows_in_clipped": ri, "rows_after_dedup_gridded": ro})
        print(f"{y}: {ri} -> {ro}")

    pd.DataFrame(stats).to_csv(
        OUT / "stats_dedup_gridded_by_year.csv", index=False, encoding="utf-8", lineterminator="\n"
    )

    c_rows = []
    for y in YEARS_T:
        p = DEDUP_DIR / f"{y}.csv"
        if not p.exists() or p.stat().st_size < 10:
            continue
        parts_y: list[pd.DataFrame] = []
        for chunk in pd.read_csv(
            p,
            chunksize=CHUNK,
            usecols=["cell_lat", "cell_lon", "SCIENTIFIC NAME", "OBSERVATION COUNT", "year"],
            low_memory=False,
        ):
            chunk["w"] = chunk["OBSERVATION COUNT"].map(numeric_for_sum)
            g = (
                chunk.groupby(["cell_lat", "cell_lon", "SCIENTIFIC NAME", "year"], as_index=False)["w"]
                .sum()
                .rename(columns={"w": "c_hat"})
            )
            parts_y.append(g)
        if not parts_y:
            continue
        merged_y = pd.concat(parts_y, ignore_index=True)
        agg = merged_y.groupby(["cell_lat", "cell_lon", "SCIENTIFIC NAME", "year"], as_index=False)["c_hat"].sum()
        c_rows.append(agg)

    c_hat = pd.concat(c_rows, ignore_index=True)
    c_hat.to_csv(OUT / "c_hat_grid_species_year.csv", index=False, encoding="utf-8", lineterminator="\n")

    y_tab = (
        c_hat.groupby(["cell_lat", "cell_lon", "SCIENTIFIC NAME"], as_index=False)["c_hat"]
        .sum()
        .rename(columns={"c_hat": "sum_c_hat_T"})
    )
    y_tab["y"] = (y_tab["sum_c_hat_T"] > 0).astype(int)
    y_tab.to_csv(OUT / "baseline_Y_static_2000_2025.csv", index=False, encoding="utf-8", lineterminator="\n")
    # Remove legacy filename if present to avoid confusion
    legacy = OUT / "baseline_Y_static_2000_2005.csv"
    if legacy.exists():
        legacy.unlink()

    cells = y_tab[["cell_lat", "cell_lon"]].drop_duplicates()
    cells.to_csv(OUT / "grid_cells_active_baseline.csv", index=False, encoding="utf-8", lineterminator="\n")

    summ = {
        "years_T": "2000-2025",
        "n_years_T": 26,
        "resolution_deg": 1,
        "n_grid_cells": int(len(cells)),
        "n_species_in_Y": int(y_tab["SCIENTIFIC NAME"].nunique()),
        "n_pairs_y_eq_1": int((y_tab["y"] == 1).sum()),
        "total_sum_c_hat_T": float(y_tab["sum_c_hat_T"].sum()),
    }
    pd.DataFrame([summ]).to_csv(OUT / "summary_baseline.csv", index=False, encoding="utf-8", lineterminator="\n")
    print("---", summ)


if __name__ == "__main__":
    main()
