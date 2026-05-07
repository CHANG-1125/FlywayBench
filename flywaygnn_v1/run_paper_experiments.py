#!/usr/bin/env python3
"""
Paper §5.1: grid search on (lr, hidden_dim, K), model selection by val macro-AUPRC.
Paper §5.2: test-domain metrics; 5 seeds → mean ± std.

Also runs Vanilla GCN (E_s only) and XGBoost baselines on the same splits.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from data_utils import load_node_features_and_labels, make_geographic_domain_masks, zscore_features_train_domain
from graph_builder import (
    assign_corridor_id,
    build_flyway_corridor_edges,
    build_spatial_edges,
    edge_set_to_undirected_index,
    load_flyway_wgs84,
)
from metrics_multilabel import (
    aggregate_mean_std,
    evaluable_species_indices,
    multilabel_augmented_test_metrics,
    multilabel_metrics_paper,
    species_auprc_values,
)
from model import FlywayGNN, VanillaSpatialGCN
from significance_analysis import paired_compare_flyway_vs_vanilla

# Stored in JSON / paired stats (paper Table 3 eval-m5 block).
PAPER_REPORT_TEST_KEYS = ("auprc_macro_eval_m5", "auroc_macro_eval_m5", "n_labels_eval_m5")


def _paper_slice_test_metrics(m: dict[str, float]) -> dict[str, float]:
    out: dict[str, float] = {}
    for k in PAPER_REPORT_TEST_KEYS:
        if k not in m:
            continue
        v = m[k]
        if v is None:
            continue
        out[k] = float(v)
    return out

try:
    from xgboost import XGBClassifier
except ImportError as e:  # pragma: no cover
    raise SystemExit("pip install scikit-learn xgboost") from e


PIPE_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PIPE_ROOT.parent
DEFAULT_FEATURES = PIPE_ROOT / "graph_build" / "node_features_imputed_4554.csv"
DEFAULT_LABELS = PIPE_ROOT / "graph_build" / "Y_binary_4554x155.csv"
DEFAULT_FLYWAY = DATA_ROOT / "birdlife_americas_flyway" / "americas_flyway_subregions.geojson"


def _json_sanitize(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _json_sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_sanitize(v) for v in obj]
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    return obj


def masked_bce_with_logits(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    loss_raw = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    return (loss_raw * mask).sum() / mask.sum().clamp(min=1.0)


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _alphas_flyway(model: FlywayGNN) -> dict[str, Any]:
    a1 = F.softmax(model.layer1.raw_alpha, dim=0).detach().cpu().tolist()
    a2 = F.softmax(model.layer2.raw_alpha, dim=0).detach().cpu().tolist()
    return {"layer1_alpha_SF": a1, "layer2_alpha_SF": a2}


@torch.no_grad()
def _subset_metrics(logits: torch.Tensor, y: torch.Tensor, row_mask: torch.Tensor) -> dict[str, float]:
    prob = torch.sigmoid(logits[row_mask]).cpu().numpy()
    yt = y[row_mask].cpu().numpy()
    return multilabel_metrics_paper(yt, prob)


@torch.no_grad()
def _subset_metrics_test_with_augmented(
    logits: torch.Tensor,
    y: torch.Tensor,
    test_domain_mask: torch.Tensor,
    *,
    eval_macro_min_pos: int,
) -> dict[str, float]:
    """Full-label test metrics + eval-m5 subset (macro-AUPRC / macro-AUC over species with ≥ m positives)."""
    p_te = torch.sigmoid(logits[test_domain_mask]).cpu().numpy()
    y_te = y[test_domain_mask].cpu().numpy()
    base = multilabel_metrics_paper(y_te, p_te)
    if eval_macro_min_pos <= 0:
        return base
    aug = multilabel_augmented_test_metrics(y_te, p_te, evaluable_min_positives=eval_macro_min_pos)
    return {**base, **aug}


@torch.no_grad()
def _species_auprc_eval_m5_from_logits(
    logits: torch.Tensor,
    y: torch.Tensor,
    test_domain_mask: torch.Tensor,
    *,
    eval_macro_min_pos: int,
) -> tuple[np.ndarray, np.ndarray]:
    p_te = torch.sigmoid(logits[test_domain_mask]).cpu().numpy()
    y_te = y[test_domain_mask].cpu().numpy()
    idx = evaluable_species_indices(y_te, evaluable_min_positives=int(eval_macro_min_pos))
    vals = species_auprc_values(y_te, p_te, idx)
    return idx, vals


def train_gnn_until_patience(
    model: nn.Module,
    *,
    vanilla: bool,
    x: torch.Tensor,
    y: torch.Tensor,
    y_mask: torch.Tensor,
    d_train: torch.Tensor,
    train_fit_mask: torch.Tensor,
    val_mask: torch.Tensor,
    test_domain_mask: torch.Tensor,
    edge_index_s: torch.Tensor,
    edge_index_f: torch.Tensor,
    epochs_max: int,
    lr: float,
    weight_decay: float,
    patience: int,
    device: torch.device,
    torch_seed: int,
    eval_macro_min_pos: int = 5,
    attach_logits: bool = False,
) -> tuple[dict[str, float], dict[str, float], dict[str, Any] | None]:
    torch.manual_seed(torch_seed)
    model = model.to(device)
    x = x.to(device)
    y = y.to(device)
    y_mask = y_mask.to(device)
    d_train = d_train.to(device)
    train_fit_mask = train_fit_mask.to(device)
    val_mask = val_mask.to(device)
    test_domain_mask = test_domain_mask.to(device)
    edge_index_s = edge_index_s.to(device)
    edge_index_f = edge_index_f.to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_score = -1.0
    best_state: dict[str, Any] | None = None
    bad = 0

    for _ in range(1, epochs_max + 1):
        model.train()
        opt.zero_grad(set_to_none=True)
        if vanilla:
            logits = model(x, edge_index_s)
        else:
            logits = model(x, edge_index_s, edge_index_f)
        loss = masked_bce_with_logits(logits, y, d_train)
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            if vanilla:
                logits = model(x, edge_index_s)
            else:
                logits = model(x, edge_index_s, edge_index_f)
            vm = _subset_metrics(logits, y, val_mask)
        score = vm.get("auprc_macro", float("nan"))
        if not np.isfinite(score):
            score = -1.0
        if score > best_score + 1e-9:
            best_score = float(score)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    extras: dict[str, Any] | None = None
    with torch.no_grad():
        if vanilla:
            logits = model(x, edge_index_s)
        else:
            logits = model(x, edge_index_s, edge_index_f)
            extras = {"alphas": _alphas_flyway(model)}  # type: ignore[arg-type]
        val_m = _subset_metrics(logits, y, val_mask)
        if eval_macro_min_pos > 0:
            test_m = _subset_metrics_test_with_augmented(
                logits, y, test_domain_mask, eval_macro_min_pos=eval_macro_min_pos
            )
        else:
            test_m = _subset_metrics(logits, y, test_domain_mask)
        if attach_logits:
            if extras is None:
                extras = {}
            extras["logits_full"] = logits.detach().cpu()
    return val_m, test_m, extras


def run_xgboost_baseline(
    x: torch.Tensor,
    y: torch.Tensor,
    train_fit_mask: torch.Tensor,
    val_mask: torch.Tensor,
    test_domain_mask: torch.Tensor,
    seed: int,
    *,
    eval_macro_min_pos: int = 5,
) -> dict[str, float]:
    X_tr = x[train_fit_mask].cpu().numpy()
    y_tr = y[train_fit_mask].cpu().numpy().astype(np.int32)
    X_te = x[test_domain_mask].cpu().numpy()
    y_te = y[test_domain_mask].cpu().numpy()
    n_tr, n_lab = y_tr.shape
    prob_te = np.zeros((X_te.shape[0], n_lab), dtype=np.float64)

    for j in range(n_lab):
        yc = y_tr[:, j]
        pos = int(yc.sum())
        if pos == 0:
            prob_te[:, j] = 1e-6
            continue
        if pos == n_tr:
            prob_te[:, j] = 1.0 - 1e-6
            continue
        clf = XGBClassifier(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.1,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=1.0,
            random_state=seed + j,
            n_jobs=-1,
            tree_method="hist",
            verbosity=0,
        )
        clf.fit(X_tr, yc)
        prob_te[:, j] = clf.predict_proba(X_te)[:, 1]

    base = multilabel_metrics_paper(y_te, prob_te)
    if eval_macro_min_pos <= 0:
        return base
    aug = multilabel_augmented_test_metrics(y_te.astype(np.float64), prob_te, evaluable_min_positives=eval_macro_min_pos)
    return {**base, **aug}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Paper §5.1 grid + §5.2 metrics (5 seeds)")
    ap.add_argument("--features-csv", type=Path, default=DEFAULT_FEATURES)
    ap.add_argument("--labels-csv", type=Path, default=DEFAULT_LABELS)
    ap.add_argument("--flyway", type=Path, default=DEFAULT_FLYWAY)
    ap.add_argument("--runs-dir", type=Path, default=Path(__file__).resolve().parent / "runs" / "paper_experiments")
    ap.add_argument("--lat-threshold-tau", type=float, default=-12.0)
    ap.add_argument("--val-frac-within-train-domain", type=float, default=0.15)
    ap.add_argument("--breeding-lat-min", type=float, default=35.0)
    ap.add_argument("--winter-lat-max", type=float, default=10.0)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument(
        "--grid-split-seed",
        type=int,
        default=0,
        help="Fixed validation split for hyperparameter grid (matches paper: tune on validation)",
    )
    ap.add_argument("--epochs-grid", type=int, default=45, help="Max epochs per grid trial (early stopping)")
    ap.add_argument("--epochs-final", type=int, default=80, help="Max epochs per seed after selecting best hyperparameters")
    ap.add_argument("--patience-grid", type=int, default=8)
    ap.add_argument("--patience-final", type=int, default=12)
    ap.add_argument(
        "--n-seeds",
        type=int,
        default=30,
        help="Number of final-phase repeats (train/val splits + init vary by seed; independent of grid-split seed). Default 30 for bootstrap/paired tests",
    )
    ap.add_argument("--seed-start", type=int, default=42, help="Use seeds seed_start, seed_start+1, ... (count = --n-seeds)")
    ap.add_argument("--quick", action="store_true", help="Tiny grid, 2 seeds, short epochs (smoke test)")
    ap.add_argument(
        "--bootstrap-b",
        type=int,
        default=10_000,
        help="Bootstrap resamples for mean paired difference Flyway−Vanilla (0 skips paired-stats block)",
    )
    ap.add_argument("--stats-alpha", type=float, default=0.05, help="Alpha for Bootstrap CI and one-sided Wilcoxon")
    ap.add_argument("--stats-rng-seed", type=int, default=42, help="RNG seed for bootstrap")
    ap.add_argument(
        "--eval-macro-min-pos",
        type=int,
        default=5,
        help="eval-m5: only species with >= this many test positives enter macro-AUPRC_eval_m5 / macro-AUC_eval_m5; 0 disables this subset",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    device = _device()
    print("device:", device)

    feature_cols = [c.strip() for c in "NDVI_mean,LST_day_mean,DEM,water_frac,miss_NDVI,miss_LST,miss_DEM".split(",")]
    zscore_cols = [c.strip() for c in "NDVI_mean,LST_day_mean,DEM,water_frac".split(",")]
    zscore_indices = [feature_cols.index(c) for c in zscore_cols]

    loaded = load_node_features_and_labels(args.features_csv, args.labels_csv, feature_cols=feature_cols)
    n_nodes = loaded.x.shape[0]
    n_species = loaded.y.shape[1]
    fly = load_flyway_wgs84(args.flyway)
    lon_center = loaded.cell_lon.astype(np.float64) + 0.5
    lat_center = loaded.cell_lat.astype(np.float64) + 0.5
    train_domain_row = (lat_center >= args.lat_threshold_tau).astype(bool)

    x_base = zscore_features_train_domain(
        loaded.x, torch.from_numpy(np.asarray(train_domain_row, dtype=bool)), zscore_col_indices=zscore_indices
    )

    e_s = build_spatial_edges(loaded.cell_lat, loaded.cell_lon)
    edge_index_s = edge_set_to_undirected_index(e_s, n_nodes=n_nodes)

    if args.quick:
        lr_grid = [1e-3]
        hid_grid = [64]
        k_grid = [5]
        epochs_grid = 12
        epochs_final = 15
        patience_g = 3
        patience_f = 4
        n_seeds = 2
    else:
        lr_grid = [1e-3, 3e-3, 5e-3]
        hid_grid = [64, 128, 256]
        k_grid = [3, 5, 8, 10]
        epochs_grid = args.epochs_grid
        epochs_final = args.epochs_final
        patience_g = args.patience_grid
        patience_f = args.patience_final
        n_seeds = args.n_seeds

    seeds = [args.seed_start + i for i in range(n_seeds)]
    eval_macro_min_pos = int(args.eval_macro_min_pos)

    # ----- Phase 1: grid (fixed split seed) -----
    split_seed = int(args.grid_split_seed)
    train_domain_mask, train_fit_mask, val_mask, test_domain_mask = make_geographic_domain_masks(
        lat_center,
        tau=args.lat_threshold_tau,
        val_frac_within_train_domain=args.val_frac_within_train_domain,
        seed=split_seed,
    )
    d_train = train_fit_mask[:, None].float() * loaded.y_mask

    best_cfg: dict[str, Any] | None = None
    best_val_auprc_macro = -1.0
    grid_log: list[dict[str, Any]] = []
    torch_grid_init = 12345

    for lr in lr_grid:
        for hidden_dim in hid_grid:
            for K in k_grid:
                e_f = build_flyway_corridor_edges(
                    loaded.y.numpy(),
                    fly,
                    lon_center,
                    lat_center,
                    breeding_lat_min=args.breeding_lat_min,
                    winter_lat_max=args.winter_lat_max,
                    max_nodes_per_side=int(K),
                    train_row_mask=train_domain_row,
                )
                if not e_f:
                    e_f_use = e_s
                else:
                    e_f_use = e_f
                edge_index_f = edge_set_to_undirected_index(e_f_use, n_nodes=n_nodes)

                model = FlywayGNN(in_dim=x_base.shape[1], hidden_dim=hidden_dim, out_dim=n_species, dropout=args.dropout)
                val_m, test_m, _ = train_gnn_until_patience(
                    model,
                    vanilla=False,
                    x=x_base,
                    y=loaded.y,
                    y_mask=loaded.y_mask,
                    d_train=d_train,
                    train_fit_mask=train_fit_mask,
                    val_mask=val_mask,
                    test_domain_mask=test_domain_mask,
                    edge_index_s=edge_index_s,
                    edge_index_f=edge_index_f,
                    epochs_max=epochs_grid,
                    lr=float(lr),
                    weight_decay=args.weight_decay,
                    patience=patience_g,
                    device=device,
                    torch_seed=torch_grid_init,
                    eval_macro_min_pos=eval_macro_min_pos,
                )
                score = val_m.get("auprc_macro", float("nan"))
                row = {
                    "lr": lr,
                    "hidden_dim": hidden_dim,
                    "K": K,
                    "val": val_m,
                    "test_snapshot_same_split": _paper_slice_test_metrics(test_m),
                    "E_f_size": len(e_f),
                }
                grid_log.append(row)
                print(
                    f"[grid] lr={lr} hid={hidden_dim} K={K} val_macro_auprc={score:.4f} "
                    f"test_macro_auprc={test_m.get('auprc_macro', float('nan')):.4f}"
                )
                if np.isfinite(score) and float(score) > best_val_auprc_macro:
                    best_val_auprc_macro = float(score)
                    best_cfg = {"lr": lr, "hidden_dim": hidden_dim, "K": K, "edge_index_f": edge_index_f.cpu()}

    if best_cfg is None:
        raise RuntimeError("Grid search failed to produce a config.")

    print("\n=== Best grid config (by val macro-AUPRC) ===")
    print(json.dumps({k: v for k, v in best_cfg.items() if k != "edge_index_f"}, indent=2))

    edge_index_f_best = best_cfg["edge_index_f"].to(device)

    # ----- Phase 2: FlywayGNN @ best cfg, multiple seeds -----
    flyway_test_rows: list[dict[str, float]] = []
    flyway_species_auprc_eval_m5_by_seed: list[np.ndarray] = []
    flyway_alphas: list[dict[str, Any]] = []
    eval_species_idx_ref: np.ndarray | None = None
    for sd in seeds:
        train_domain_mask, train_fit_mask, val_mask, test_domain_mask = make_geographic_domain_masks(
            lat_center,
            tau=args.lat_threshold_tau,
            val_frac_within_train_domain=args.val_frac_within_train_domain,
            seed=int(sd),
        )
        d_train = train_fit_mask[:, None].float() * loaded.y_mask
        model = FlywayGNN(
            in_dim=x_base.shape[1], hidden_dim=int(best_cfg["hidden_dim"]), out_dim=n_species, dropout=args.dropout
        )
        _, test_m, ex = train_gnn_until_patience(
            model,
            vanilla=False,
            x=x_base,
            y=loaded.y,
            y_mask=loaded.y_mask,
            d_train=d_train,
            train_fit_mask=train_fit_mask,
            val_mask=val_mask,
            test_domain_mask=test_domain_mask,
            edge_index_s=edge_index_s,
            edge_index_f=edge_index_f_best,
            epochs_max=epochs_final,
            lr=float(best_cfg["lr"]),
            weight_decay=args.weight_decay,
            patience=patience_f,
            device=device,
            torch_seed=int(sd),
            eval_macro_min_pos=eval_macro_min_pos,
            attach_logits=True,
        )
        flyway_test_rows.append(_paper_slice_test_metrics(test_m))
        if ex and "logits_full" in ex and eval_macro_min_pos > 0:
            idx, vals = _species_auprc_eval_m5_from_logits(
                ex["logits_full"],
                loaded.y,
                test_domain_mask,
                eval_macro_min_pos=eval_macro_min_pos,
            )
            if eval_species_idx_ref is None:
                eval_species_idx_ref = idx
            elif idx.shape != eval_species_idx_ref.shape or np.any(idx != eval_species_idx_ref):
                raise RuntimeError("Eval-m5 species index mismatch across seeds.")
            flyway_species_auprc_eval_m5_by_seed.append(vals)
        if ex and "alphas" in ex:
            flyway_alphas.append(ex["alphas"])
        exs = f" eval_m5_AUPRC={test_m.get('auprc_macro_eval_m5', float('nan')):.4f}" if eval_macro_min_pos > 0 else ""
        print(
            f"[FlywayGNN seed={sd}] test macro-AUPRC={test_m.get('auprc_macro'):.4f} macro-F1={test_m.get('f1_macro'):.4f}"
            f"{exs}"
        )

    # ----- Vanilla GCN -----
    vanilla_test_rows: list[dict[str, float]] = []
    vanilla_species_auprc_eval_m5_by_seed: list[np.ndarray] = []
    for sd in seeds:
        train_domain_mask, train_fit_mask, val_mask, test_domain_mask = make_geographic_domain_masks(
            lat_center,
            tau=args.lat_threshold_tau,
            val_frac_within_train_domain=args.val_frac_within_train_domain,
            seed=int(sd),
        )
        d_train = train_fit_mask[:, None].float() * loaded.y_mask
        model = VanillaSpatialGCN(
            in_dim=x_base.shape[1], hidden_dim=int(best_cfg["hidden_dim"]), out_dim=n_species, dropout=args.dropout
        )
        _, test_m, ex = train_gnn_until_patience(
            model,
            vanilla=True,
            x=x_base,
            y=loaded.y,
            y_mask=loaded.y_mask,
            d_train=d_train,
            train_fit_mask=train_fit_mask,
            val_mask=val_mask,
            test_domain_mask=test_domain_mask,
            edge_index_s=edge_index_s,
            edge_index_f=edge_index_f_best,
            epochs_max=epochs_final,
            lr=float(best_cfg["lr"]),
            weight_decay=args.weight_decay,
            patience=patience_f,
            device=device,
            torch_seed=int(sd),
            eval_macro_min_pos=eval_macro_min_pos,
            attach_logits=True,
        )
        vanilla_test_rows.append(_paper_slice_test_metrics(test_m))
        if ex and "logits_full" in ex and eval_macro_min_pos > 0:
            idx, vals = _species_auprc_eval_m5_from_logits(
                ex["logits_full"],
                loaded.y,
                test_domain_mask,
                eval_macro_min_pos=eval_macro_min_pos,
            )
            if eval_species_idx_ref is None:
                eval_species_idx_ref = idx
            elif idx.shape != eval_species_idx_ref.shape or np.any(idx != eval_species_idx_ref):
                raise RuntimeError("Eval-m5 species index mismatch between models.")
            vanilla_species_auprc_eval_m5_by_seed.append(vals)
        exs = f" eval_m5_AUPRC={test_m.get('auprc_macro_eval_m5', float('nan')):.4f}" if eval_macro_min_pos > 0 else ""
        print(
            f"[VanillaGCN seed={sd}] test macro-AUPRC={test_m.get('auprc_macro'):.4f} macro-F1={test_m.get('f1_macro'):.4f}"
            f"{exs}"
        )

    # ----- XGBoost -----
    xgb_test_rows: list[dict[str, float]] = []
    for sd in seeds:
        train_domain_mask, train_fit_mask, val_mask, test_domain_mask = make_geographic_domain_masks(
            lat_center,
            tau=args.lat_threshold_tau,
            val_frac_within_train_domain=args.val_frac_within_train_domain,
            seed=int(sd),
        )
        xm = run_xgboost_baseline(
            x_base,
            loaded.y,
            train_fit_mask,
            val_mask,
            test_domain_mask,
            seed=int(sd),
            eval_macro_min_pos=eval_macro_min_pos,
        )
        xgb_test_rows.append(_paper_slice_test_metrics(xm))
        exs = f" eval_m5_AUPRC={xm.get('auprc_macro_eval_m5', float('nan')):.4f}" if eval_macro_min_pos > 0 else ""
        print(f"[XGBoost seed={sd}] test macro-AUPRC={xm.get('auprc_macro'):.4f} macro-F1={xm.get('f1_macro'):.4f}{exs}")

    out_dir = args.runs_dir / time.strftime("%Y%m%d-%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "protocol": "§5.1 grid on val macro-AUPRC; §5.2 eval-m5 test subset (phi<tau); n_seeds mean±std",
        "eval_protocol_extras": {
            "eval_macro_min_positives": eval_macro_min_pos,
            "auprc_macro_eval_m5": "macro-AUPRC averaged over species with >=eval_macro_min_positives positives in test domain",
            "auroc_macro_eval_m5": "macro-AUC averaged over the same species subset",
        },
        "grid_split_seed": split_seed,
        "best_config": {k: v for k, v in best_cfg.items() if k != "edge_index_f"},
        "best_val_macro_auprc_grid": best_val_auprc_macro,
        "grid_trials": grid_log,
        "seeds": seeds,
        "flywaygnn_test_mean_std": aggregate_mean_std(flyway_test_rows),
        "vanilla_gcn_test_mean_std": aggregate_mean_std(vanilla_test_rows),
        "xgboost_test_mean_std": aggregate_mean_std(xgb_test_rows),
        "per_seed_test_flywaygnn": flyway_test_rows,
        "per_seed_test_vanilla_gcn": vanilla_test_rows,
        "per_seed_test_xgboost": xgb_test_rows,
        "flywaygnn_alpha_last_seeds": flyway_alphas[-3:] if len(flyway_alphas) > 3 else flyway_alphas,
    }

    if (
        eval_macro_min_pos > 0
        and eval_species_idx_ref is not None
        and flyway_species_auprc_eval_m5_by_seed
        and vanilla_species_auprc_eval_m5_by_seed
    ):
        f_mat = np.vstack(flyway_species_auprc_eval_m5_by_seed)
        v_mat = np.vstack(vanilla_species_auprc_eval_m5_by_seed)
        f_mean = np.nanmean(f_mat, axis=0)
        v_mean = np.nanmean(v_mat, axis=0)
        d = f_mean - v_mean
        tol = 1e-12
        wins = int(np.sum(d > tol))
        losses = int(np.sum(d < -tol))
        ties = int(d.size - wins - losses)
        report["species_level_eval_m5"] = {
            "species_indices_eval_m5": eval_species_idx_ref.tolist(),
            "flyway_minus_vanilla_auprc_eval_m5_mean_by_species": d.tolist(),
            "win_tie_loss_vs_vanilla_auprc_eval_m5": {"win": wins, "tie": ties, "loss": losses, "n_species": int(d.size)},
            "win_tie_loss_str_vs_vanilla_auprc_eval_m5": f"{wins}/{ties}/{losses}",
        }

    if int(args.bootstrap_b) > 0:
        paired = paired_compare_flyway_vs_vanilla(
            flyway_test_rows,
            vanilla_test_rows,
            seeds=seeds,
            metrics=("auprc_macro_eval_m5", "auroc_macro_eval_m5"),
            bootstrap_b=int(args.bootstrap_b),
            alpha=float(args.stats_alpha),
            rng_seed=int(args.stats_rng_seed),
        )
        report["flyway_vs_vanilla_paired_stats"] = paired
        print("\n--- Paired FlywayGNN vs Vanilla GCN (test domain, same seeds) ---")
        print(paired["narrative_en"])

    (out_dir / "paper_experiment_report.json").write_text(
        json.dumps(_json_sanitize(report), indent=2, allow_nan=False), encoding="utf-8"
    )
    if int(args.bootstrap_b) > 0:
        (out_dir / "flyway_vs_vanilla_paired_stats.json").write_text(
            json.dumps(_json_sanitize(report["flyway_vs_vanilla_paired_stats"]), indent=2, ensure_ascii=False, allow_nan=False),
            encoding="utf-8",
        )
    print("\nSaved:", out_dir / "paper_experiment_report.json")


if __name__ == "__main__":
    main()
