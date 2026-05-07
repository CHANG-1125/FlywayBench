# FlywayBench

Large-scale, reproducible **benchmark data** for migratory flyway modeling—linking observations, geography, and environmental context—is still scarce. **FlywayBench** is the accompanying codebase for a **gridded multi-species flyway dataset**: we ship an end-to-end **pipeline** (observation preprocessing, \(1^\circ\) gridding, Earth Engine–derived node features, graph construction, splits) so others can rebuild or extend those inputs under documented assumptions. **FlywayGNN** is a graph neural network we developed **on this dataset**, together with baselines and a fixed benchmark protocol (hyperparameter search, multi-seed runs, extrapolation-style evaluation, paired statistics), to check that the assembled benchmark supports meaningful model comparison rather than only showcasing one architecture.

This repository contains **software and documentation**. Large proprietary or bulky datasets are **not** tracked in git (see `.gitignore` and **[`DATA.md`](DATA.md)** for provenance, attribution, and how to obtain or regenerate inputs).

---

## Overview

| Component | Description |
|-----------|-------------|
| **Pipeline** | Debiased gridding, static labels, GEE covariates → graph-ready node features and multi-label \(Y\) |
| **Models** | FlywayGNN, Vanilla GCN baseline, XGBoost baseline |
| **Evaluation** | Geographic extrapolation split, eval-m5 metrics, seed-level aggregation, optional paired tests |

---

## Repository layout

| Path | Purpose |
|------|---------|
| [`flywaygnn_v1/`](flywaygnn_v1/) | Models, training, paper benchmark script, metrics, plotting |
| [`graph_build/`](graph_build/) | Merge GEE exports with baseline grid; build \(Y\); optional diagnostic figures |
| [`earth_engine/`](earth_engine/) | Google Earth Engine export script and docs |
| [`run_312_313.py`](run_312_313.py) | Gridding, debiasing, static baseline \(Y\) (2000–2025) |

Full annotated tree and pipeline order: **[`docs/REPOSITORY_LAYOUT.md`](docs/REPOSITORY_LAYOUT.md)**.

---

## Requirements

- **Python** 3.10+ recommended  
- Install Python deps from the repo root:

```bash
pip install -r requirements.txt
```

- **`requirements-lock.txt`** — full `pip freeze` from the maintainer machine (372 packages). Use only if you need a near-identical flat environment; it may pull packages unrelated to FlywayBench.
- **`environment.yml`** — optional Conda env with Python + `pip`; after `conda activate`, run `pip install -r requirements.txt`.

All scripts resolve paths from the repository root or `Path(__file__)` (no hard-coded `C:\…` or `/Users/…`). Regenerate inputs under your own layout as documented in **[`DATA.md`](DATA.md)**.

If **`torch` / `torch-geometric`** wheels fail for your OS/CUDA, install **PyTorch** first from the [official PyTorch site](https://pytorch.org/), then `pip install torch-geometric` and `pip install -r requirements.txt`.

---

## Quick start (after inputs exist)

Paths below assume you run commands from the **repository root**.

### 1. Grid, debias, static labels

```bash
python3 run_312_313.py
```

Writes CSVs next to `run_312_313.py` (see **Pipeline details**).

### 2. Remote sensing features (Google Earth Engine)

```bash
cd earth_engine
earthengine authenticate   # once
# optional: export EARTHENGINE_PROJECT=ee-your-project-id
python3 gee_export_node_features.py
```

See [`earth_engine/README_GEE.md`](earth_engine/README_GEE.md) for output naming (e.g. `earth_engine/output/grid_node_features_2000_2025.csv`).

### 3. Graph tables (features + \(Y\))

```bash
cd graph_build
python3 prepare_node_features_and_Y.py --features /path/to/grid_node_features_2000_2025.csv
```

Produces `graph_build/node_features_imputed_4554.csv`, `graph_build/Y_binary_4554x155.csv`, and metadata JSON files.

### 4. Train FlywayGNN

```bash
cd ../flywaygnn_v1
python3 train.py --epochs 80
```

Checkpoints: `flywaygnn_v1/runs/<timestamp>/`.

### 5. Full benchmark protocol

```bash
PYTHONUNBUFFERED=1 python3 run_paper_experiments.py 2>&1 | tee runs/paper_experiments/last_run.log
```

Smoke test:

```bash
python3 run_paper_experiments.py --quick
```

### 6. Figures from a saved run

```bash
python3 visualize_paper_experiment.py --report-json runs/paper_experiments/<run_id>/paper_experiment_report.json
```

---

## Pipeline details (methods §3.1, condensed)

- **§3.1.2 Gridding.** WGS84, \(1^\circ\times1^\circ\), `cell_lat=floor(lat)`, `cell_lon=floor(lon)` — [`run_312_313.py`](run_312_313.py).
- **§3.1.3 Debiasing & static baseline.** Within-day max within `(species, lat, lon, calendar day)`; grid–year–species aggregation; static \(y_{i,s}\) for **2000–2025**. Typical artifacts: `deduped_gridded/YYYY.csv`, `c_hat_grid_species_year.csv`, `baseline_Y_static_2000_2025.csv`, `grid_cells_active_baseline.csv`, `summary_baseline.csv`, `stats_dedup_gridded_by_year.csv`.
- **§3.1.4 Remote sensing.** Per-cell NDVI/LST/DEM means (~500 m) and `water_frac` from JRC GSW (~30 m) via GEE — [`earth_engine/README_GEE.md`](earth_engine/README_GEE.md).
- **§3.1.5 Graph-ready tables.** [`graph_build/prepare_node_features_and_Y.py`](graph_build/prepare_node_features_and_Y.py): align GEE CSV to `grid_cells_active_baseline.csv`, impute missing covariates, emit imputed features and binary \(Y\) wide matrix.

---

## External data (not shipped here)

Place alongside this repo or pass CLI paths:

| Input | Role |
|-------|------|
| `birdlife_americas_flyway/` | BirdLife Americas flyway GeoJSON(s) for corridors |
| `grid_node_features_2000_2025.csv` | GEE export (or under `earth_engine/output/`) |
| Preprocessed eBird extracts | Per [`DATA.md`](DATA.md) |

---

## Reference experiment outputs

A frozen benchmark snapshot (tables + JSON) ships under:

`flywaygnn_v1/runs/paper_experiments/20260419-173506/`

Re-running `run_paper_experiments.py` creates a new timestamped directory under `flywaygnn_v1/runs/paper_experiments/`.

---

## Default experimental settings

| Setting | Value |
|---------|--------|
| Grid | \(1^\circ\times1^\circ\), `cell_lat=floor(lat)`, `cell_lon=floor(lon)` |
| Time window | 2000-01-01 — 2025-12-31 |
| Extrapolation threshold | \(\tau=-12^\circ\) on \(\phi(v)=\texttt{cell_lat}+0.5\) |
| Graph | 4554 active cells × 155 species |
| Benchmark script | Validation macro-AUPRC for model selection; `--epochs-grid 45`, `--patience-grid 8`; final `--epochs-final 80`, `--patience-final 12` |

---

## Citation

If you use this software, cite the **FlywayBench / FlywayGNN** publication when available, and cite upstream data sources (**eBird**, **BirdLife**, **Wetlands International WPE** where relevant, **Google Earth Engine** catalog entries for MODIS, Copernicus DEM, JRC GSW). See **[`DATA.md`](DATA.md)** for attribution notes.

---

## License

Code is released under **[`LICENSE`](LICENSE)**. **Third-party data remain under their own licenses** — see **[`DATA.md`](DATA.md)**.

---

## Contributing & support

Bug reports and focused improvements are welcome via **GitHub Issues** and pull requests. For data-access or licensing questions about upstream providers, consult their official terms and [`DATA.md`](DATA.md).
