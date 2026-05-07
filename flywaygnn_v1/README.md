# FlywayGNN v1 (minimal runnable)

Self-contained training and evaluation code for FlywayGNN (paths relative to **FlywayBench** root).

## What it does

- Loads node features from `graph_build/node_features_imputed_4554.csv`
- Loads multi-label targets from `graph_build/Y_binary_4554x155.csv`
- **Setting B (paper-aligned):** full-graph forward pass; supervision only on the train-supervised domain \( \mathcal{V}_{\mathrm{train}}=\{\phi(v)\ge\tau\} \); extrapolation domain \( \{\phi(v)<\tau\} \) is **eval-only** (no loss gradients).
  - Grid-center latitude \( \phi(v)=\texttt{cell\_lat}+0.5 \); default \( \tau=-12^\circ \) (`--lat-threshold-tau`).
  - Random train / validation split **inside** \( \phi\ge\tau \) (`--val-frac-within-train-domain`).
  - Loss mask \( D_{v,s}=\mathbb{1}_{v\in\text{train-fit}}\cdot M_{v,s} \) (elementwise with `y_mask`).
- **Z-score:** columns in `--zscore-cols` use mean/std estimated on \( \phi\ge\tau \) only, then applied to all nodes. For domain-only median imputation in CSVs, re-run `prepare_node_features_and_Y.py` upstream if needed.
- Hybrid edges:
  - `E_s`: 8-neighbor (Moore) grid edges
  - `E_f`: BirdLife **subregion** corridor edges (`corridor(v)` by point-in-polygon), species-aware north–south pairs with top-`K` sparsification. Rows with **\( \phi<\tau \)** use **zero labels** when building \( \mathcal{E}_F \) (no co-occurrence leakage into corridor edges).
- 2-layer dual-channel **symmetric-normalized GCN** (`GCNConv`) with **learnable channel weights** \( \alpha \) (softmax) and **MLP encoder** on \(X\) (paper §4.3-style: neighbor aggregation + self term).
- Masked binary cross-entropy (MBCE) on the supervised subset
- Saves checkpoint + `metrics.json` / `config.json`

## Inputs

- `../graph_build/node_features_imputed_4554.csv`
- `../graph_build/Y_binary_4554x155.csv`
- `../../birdlife_americas_flyway/americas_flyway_subregions.geojson` (default `--flyway`; adjust if your layout differs)

## Quick start

1) Dependencies (virtualenv recommended):

```bash
pip install torch torchvision torchaudio
pip install torch-geometric
pip install pandas numpy geopandas scikit-learn xgboost scipy matplotlib
```

2) Single training run:

```bash
cd /path/to/FlywayBench/flywaygnn_v1
python3 train.py --epochs 80
```

3) **Paper experiments (§5.1 grid + §5.2 eval + baselines)** — `run_paper_experiments.py`:

- Grid: `lr ∈ {1e-3,3e-3,5e-3}`, `hidden ∈ {64,128,256}`, corridor `K ∈ {3,5,8,10}`; select by **validation macro-AUPRC** (`--grid-split-seed`).
- **30 seeds by default** (`--n-seeds`, `--seed-start`) on test domain \( \phi<\tau \): FlywayGNN (best grid config), Vanilla GCN (\(E_s\) only), XGBoost (per-label). Use `--n-seeds 5` for a quicker run.
- Output: `runs/paper_experiments/<timestamp>/paper_experiment_report.json` (mean ± std + per-seed rows for the paper metric slice).

```bash
# Full protocol (CPU wall time depends on early stopping; often ~1–3 h class)
PYTHONUNBUFFERED=1 python3 run_paper_experiments.py 2>&1 | tee runs/paper_experiments/last_run.log

# Smoke test
python3 run_paper_experiments.py --quick
```

4) Outputs

- After `train.py`: `runs/<timestamp>/best_model.pt`, `metrics.json`, `config.json`.
- After `run_paper_experiments.py`: `runs/paper_experiments/<timestamp>/paper_experiment_report.json`.
- **Figures:** `python3 visualize_paper_experiment.py` (newest report by default) or `--report-json runs/paper_experiments/<id>/paper_experiment_report.json` → PNGs under that run’s `figures/`.
- **Paired Flyway vs Vanilla:** embedded as `flyway_vs_vanilla_paired_stats` and mirrored to `flyway_vs_vanilla_paired_stats.json`. Regenerate from a saved report:  
  `python3 significance_analysis.py --report-json runs/paper_experiments/<id>/paper_experiment_report.json`  
  Use `--n-seeds 30` for stable bootstrap / tests; very small *n* makes p-values exploratory only.

## Notes

- **Spatial edges** `E_s` may still cross \( \tau \) (full-grid adjacency). To hard-cut cross-domain spatial edges, add a switch in `build_spatial_edges` or the trainer.
- `y_mask` is supported in code; current bundled \(Y\) is fully observed binary.
