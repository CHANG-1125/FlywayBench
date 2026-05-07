# Google Earth Engine — multi-source node features (no masking)

## Features and conventions

- **Time window**: 2000-01-01 — 2025-12-31 (aligned with the observation baseline).  
- **Spatial units**: \(1^\circ\) grid cells from `../grid_cells_active_baseline.csv` (`cell_lat`, `cell_lon`).  
- **Spatial reducer**: `reduceRegion` on each cell polygon; **NDVI / LST / DEM use `scale=500` m (mean)**; **`water_frac` uses `scale=30` m** (water area summed then divided by cell area).  

| Column | Meaning |
|--------|---------|
| `NDVI_mean` | MODIS MOD13A2, multi-year mean after simple QA masking |
| `LST_day_mean` | MOD11A2 daytime LST multi-year mean (°C) |
| `DEM_mean` | Copernicus GLO30 mean elevation (m) |
| `water_frac` | JRC GSW `max_extent` fractional water area in cell (**continuous**, no hard mask) |

## Prerequisites

1. `pip install earthengine-api pandas`  
2. `earthengine authenticate`  
3. Optional project: `export EARTHENGINE_PROJECT=ee-your-project-id`

## Run

```bash
cd /path/to/FlywayBench/earth_engine
python3 gee_export_node_features.py
```

Save the downloaded CSV as:

`output/grid_node_features_2000_2025.csv`

## Validation (without re-running GEE)

```bash
python3 summarize_gee_export.py
```

Writes `output/gee_feature_export_summary.csv`.

## Column naming (legacy exports)

If an older export used **`DEM`** instead of `DEM_mean`, `summarize_gee_export.py` still accepts it; rename to `DEM_mean` before modeling for consistency.
