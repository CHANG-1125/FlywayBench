# GEE multi-source node features — checklist

## Status

- [x] Baseline grid 2000–2025: `../grid_cells_active_baseline.csv`
- [x] Export script: `gee_export_node_features.py`
- [x] Local validation script: `summarize_gee_export.py`
- [x] Earth Engine Python API (`earthengine-api`)
- [ ] GEE authentication (`earthengine authenticate`)
- [ ] Export task finished and CSV downloaded to `output/grid_node_features_2000_2025.csv`
- [ ] (Optional) Local check: `python3 summarize_gee_export.py`

## Steps

```bash
# 1) Authenticate (first time only)
earthengine authenticate

# 2) Submit export tasks
cd /path/to/FlywayBench/earth_engine
python3 gee_export_node_features.py

# 3) After Tasks complete, download CSV to:
#    <FlywayBench>/earth_engine/output/grid_node_features_2000_2025.csv

# 4) Summarize columns (no masking)
python3 summarize_gee_export.py
```

## Outputs

- `output/grid_node_features_2000_2025.csv` (columns: `cell_lat`, `cell_lon`, `NDVI_mean`, `LST_day_mean`, `DEM_mean` or legacy `DEM`, `water_frac`; reducers: NDVI/LST/DEM at **500 m**, `water_frac` at **30 m**)
- `output/gee_feature_export_summary.csv` (from `summarize_gee_export.py`)

## Alignment with baseline grid

- Expected active grid cells: **4554** (see `../summary_baseline.csv` / `grid_cells_active_baseline.csv`)
- Export row count should match **4554** (same as `grid_cells_active_baseline.csv`).
