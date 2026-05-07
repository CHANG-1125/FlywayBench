# FlywayBench repository layout

This folder holds **navigation-only** documentation: a concise map of directories and entry points. Executable code and heavy artifacts live elsewhere per [`.gitignore`](../.gitignore).

---

## Top-level directory tree

```
FlywayBench/
├── LICENSE
├── README.md                 # Project overview and quick start
├── DATA.md                   # Data provenance, licensing notes, temporal coverage
├── requirements.txt          # Pinned Python deps for FlywayBench
├── requirements-lock.txt     # Full pip freeze (optional, large)
├── environment.yml           # Optional Conda bootstrap + pip
├── .gitignore
├── docs/
│   └── REPOSITORY_LAYOUT.md  # This file
├── run_312_313.py            # Gridding, debiasing, static baseline labels
├── earth_engine/             # Google Earth Engine export utilities
├── graph_build/              # Node features + Y matrix → graph-ready tables
└── flywaygnn_v1/             # Models, training, benchmarks, reference run outputs
```

---

## Subfolders (what belongs where)

### `earth_engine/`

| Item | Role |
|------|------|
| `gee_export_node_features.py` | Export grid-aligned remote-sensing summaries (users run with their own GEE credentials). |
| `summarize_gee_export.py` | Helper to summarize exported tables. |
| `README_GEE.md`, `STATUS_gee_progress.md` | Usage and progress notes. |
| `output/` | Default sink for GEE CSV exports (contents gitignored; add `.gitkeep` if you want an empty tracked folder). |

### `graph_build/`

| Item | Role |
|------|------|
| `prepare_node_features_and_Y.py` | Merge GEE outputs with active grid → imputed node features and binary \(Y\). |
| `split_meta.json`, `species_column_order.json` | Schema / column order metadata (small, tracked). |
| `plot_*.py` | Optional diagnostic figures (not required for the benchmark numbers). |

Generated CSVs here (`node_features_imputed_*.csv`, `Y_binary_*.csv`, etc.) are **ignored by git**; regenerate after you have inputs.

### `flywaygnn_v1/`

| Item | Role |
|------|------|
| `model.py` | FlywayGNN and Vanilla GCN definitions. |
| `graph_builder.py`, `data_utils.py` | Edge construction and tensor loading. |
| `train.py` | Single-run training entry point. |
| `run_paper_experiments.py` | Multi-seed benchmark + baselines + JSON report. |
| `metrics_multilabel.py`, `significance_analysis.py` | Metrics and paired comparisons. |
| `visualize_paper_experiment.py`, `section53_richness_maps.py` | Figures / optional analyses. |
| `README.md` | Module-level notes. |
| `runs/paper_experiments/20260419-173506/` | **Bundled reference snapshot** (tables + JSON); other run timestamps stay untracked. |

Training checkpoints (`*.pt`, `*.pth`) and non-reference `runs/` directories are **ignored**.

---

## Geometry data (not shipped in git)

Flyway polygons used by the corridor graph are expected beside the repo root or passed via CLI, e.g.:

- `birdlife_americas_flyway/americas_flyway_subregions.geojson`
- `birdlife_americas_flyway/americas_flyway_system.geojson`

See **[`README.md`](../README.md)** and **[`DATA.md`](../DATA.md)** for acquisition and attribution.

---

## End-to-end flow (pipeline order)

1. **`run_312_313.py`** → debiased grid + static presence baseline.  
2. **`earth_engine/gee_export_node_features.py`** → per-node covariates.  
3. **`graph_build/prepare_node_features_and_Y.py`** → `node_features_*`, `Y_binary_*`.  
4. **`flywaygnn_v1/train.py`** or **`run_paper_experiments.py`** → models and benchmark outputs.

---

## Duplicate `requirements.txt` under `flywaygnn_v1/`

If present, prefer the **repository root** [`requirements.txt`](../requirements.txt) for installs; the subdirectory copy may be legacy.
