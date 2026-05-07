#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from data_utils import (
    load_node_features_and_labels,
    make_geographic_domain_masks,
    zscore_features_train_domain,
)
from graph_builder import (
    assign_corridor_id,
    build_flyway_corridor_edges,
    build_spatial_edges,
    edge_set_to_undirected_index,
    load_flyway_wgs84,
)
from model import FlywayGNN


PIPE_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PIPE_ROOT.parent
DEFAULT_FEATURES = PIPE_ROOT / "graph_build" / "node_features_imputed_4554.csv"
DEFAULT_LABELS = PIPE_ROOT / "graph_build" / "Y_binary_4554x155.csv"
DEFAULT_FLYWAY = DATA_ROOT / "birdlife_americas_flyway" / "americas_flyway_subregions.geojson"
DEFAULT_RUNS = Path(__file__).resolve().parent / "runs"


def masked_bce_with_logits(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    loss_raw = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    loss = (loss_raw * mask).sum() / mask.sum().clamp(min=1.0)
    return loss


@torch.no_grad()
def multilabel_scores(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> dict[str, float]:
    prob = torch.sigmoid(logits)
    pred = (prob >= 0.5).float()
    valid = mask > 0.5

    tp = ((pred == 1) & (target == 1) & valid).sum().item()
    fp = ((pred == 1) & (target == 0) & valid).sum().item()
    fn = ((pred == 0) & (target == 1) & valid).sum().item()
    tn = ((pred == 0) & (target == 0) & valid).sum().item()

    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)
    acc = (tp + tn) / (tp + tn + fp + fn + 1e-9)
    return {"acc_micro": float(acc), "precision_micro": float(precision), "recall_micro": float(recall), "f1_micro": float(f1)}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="FlywayGNN v1 training (Setting B: geographic split, full-graph forward)")
    ap.add_argument("--features-csv", type=Path, default=DEFAULT_FEATURES)
    ap.add_argument("--labels-csv", type=Path, default=DEFAULT_LABELS)
    ap.add_argument("--flyway", type=Path, default=DEFAULT_FLYWAY, help="americas_flyway_subregions.geojson (WGS84)")
    ap.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--hidden-dim", type=int, default=128)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument(
        "--lat-threshold-tau",
        type=float,
        default=-12.0,
        help="Supervised train domain V_train = {φ(v)≥τ}, extrapolation test V_test = {φ(v)<τ}; φ is grid-center latitude cell_lat+0.5",
    )
    ap.add_argument(
        "--val-frac-within-train-domain",
        type=float,
        default=0.15,
        help="Random validation fraction inside V_train",
    )
    ap.add_argument("--breeding-lat-min", type=float, default=35.0)
    ap.add_argument("--winter-lat-max", type=float, default=10.0)
    ap.add_argument("--corridor-max-per-side", type=int, default=6)
    ap.add_argument(
        "--feature-cols",
        type=str,
        default="NDVI_mean,LST_day_mean,DEM,water_frac,miss_NDVI,miss_LST,miss_DEM",
    )
    ap.add_argument(
        "--zscore-cols",
        type=str,
        default="NDVI_mean,LST_day_mean,DEM,water_frac",
        help="Columns to z-score using mean/std from train domain only (must be subset of --feature-cols)",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    feature_cols = [c.strip() for c in args.feature_cols.split(",") if c.strip()]
    zscore_cols = [c.strip() for c in args.zscore_cols.split(",") if c.strip()]
    missing_z = [c for c in zscore_cols if c not in feature_cols]
    if missing_z:
        raise SystemExit(f"--zscore-cols must be subset of --feature-cols; unknown: {missing_z}")
    zscore_indices = [feature_cols.index(c) for c in zscore_cols]

    loaded = load_node_features_and_labels(args.features_csv, args.labels_csv, feature_cols=feature_cols)
    n_nodes = loaded.x.shape[0]
    n_species = loaded.y.shape[1]

    if not args.flyway.exists():
        raise SystemExit(f"Flyway GeoJSON not found: {args.flyway}")
    fly = load_flyway_wgs84(args.flyway)
    lon_center = loaded.cell_lon.astype(np.float64) + 0.5
    lat_center = loaded.cell_lat.astype(np.float64) + 0.5
    corridor_ids, corridor_names = assign_corridor_id(fly, lon_center, lat_center)
    n_unassigned = int((corridor_ids < 0).sum())

    train_domain_mask, train_fit_mask, val_mask, test_domain_mask = make_geographic_domain_masks(
        lat_center,
        tau=args.lat_threshold_tau,
        val_frac_within_train_domain=args.val_frac_within_train_domain,
        seed=args.seed,
    )
    train_domain_row = train_domain_mask.numpy().astype(bool)

    x = zscore_features_train_domain(loaded.x, train_domain_mask, zscore_col_indices=zscore_indices)

    e_s = build_spatial_edges(loaded.cell_lat, loaded.cell_lon)
    e_f = build_flyway_corridor_edges(
        loaded.y.numpy(),
        fly,
        lon_center,
        lat_center,
        breeding_lat_min=args.breeding_lat_min,
        winter_lat_max=args.winter_lat_max,
        max_nodes_per_side=args.corridor_max_per_side,
        train_row_mask=train_domain_row,
    )
    if not e_f:
        print("WARN: E_f is empty under current thresholds; fallback to E_s for flyway channel.")
    edge_index_s = edge_set_to_undirected_index(e_s, n_nodes=n_nodes)
    edge_index_f = edge_set_to_undirected_index(e_f if e_f else e_s, n_nodes=n_nodes)

    # Supervision mask D_{v,s} = 1_{v in train-fit subset} * M_{v,s} (extrapolation domain has no gradient)
    d_train = train_fit_mask[:, None].float() * loaded.y_mask
    d_val = val_mask[:, None].float() * loaded.y_mask

    model = FlywayGNN(in_dim=x.shape[1], hidden_dim=args.hidden_dim, out_dim=n_species, dropout=args.dropout)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    run_id = time.strftime("%Y%m%d-%H%M%S")
    out_dir = args.runs_dir / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    best_val = float("inf")
    best_payload = None
    history: list[dict[str, float]] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        opt.zero_grad(set_to_none=True)
        logits = model(x, edge_index_s, edge_index_f)
        loss_train = masked_bce_with_logits(logits, loaded.y, d_train)
        loss_train.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            logits = model(x, edge_index_s, edge_index_f)
            loss_val = masked_bce_with_logits(logits, loaded.y, d_val).item()
            tr_scores = multilabel_scores(logits[train_fit_mask], loaded.y[train_fit_mask], loaded.y_mask[train_fit_mask])
            va_scores = multilabel_scores(logits[val_mask], loaded.y[val_mask], loaded.y_mask[val_mask])
            te_scores = multilabel_scores(
                logits[test_domain_mask], loaded.y[test_domain_mask], loaded.y_mask[test_domain_mask]
            )

        row = {
            "epoch": epoch,
            "loss_train": float(loss_train.item()),
            "loss_val": float(loss_val),
            "train_f1_micro": tr_scores["f1_micro"],
            "val_f1_micro": va_scores["f1_micro"],
            "test_f1_micro": te_scores["f1_micro"],
        }
        history.append(row)
        print(
            f"[{epoch:03d}] train_loss={row['loss_train']:.4f} val_loss={row['loss_val']:.4f} "
            f"train_f1={row['train_f1_micro']:.4f} val_f1={row['val_f1_micro']:.4f} test_f1={row['test_f1_micro']:.4f}"
        )

        if loss_val < best_val:
            best_val = loss_val
            best_payload = {
                "model_state_dict": model.state_dict(),
                "feature_cols": feature_cols,
                "zscore_cols": zscore_cols,
                "species_names": loaded.species_names,
                "args": vars(args),
                "n_nodes": int(n_nodes),
                "n_species": int(n_species),
                "edge_counts": {"E_s": len(e_s), "E_f": len(e_f), "E_all": len(e_s.union(e_f))},
                "corridor": {
                    "flyway_geojson": str(args.flyway),
                    "n_unassigned_nodes": n_unassigned,
                    "corridor_names": corridor_names,
                },
                "geographic_split": {
                    "lat_threshold_tau": float(args.lat_threshold_tau),
                    "n_train_domain": int(train_domain_mask.sum().item()),
                    "n_train_fit": int(train_fit_mask.sum().item()),
                    "n_val": int(val_mask.sum().item()),
                    "n_test_domain": int(test_domain_mask.sum().item()),
                },
            }

    if best_payload is None:
        raise RuntimeError("Training failed: no best checkpoint captured.")

    ckpt_path = out_dir / "best_model.pt"
    torch.save(best_payload, ckpt_path)

    model.eval()
    with torch.no_grad():
        logits = model(x, edge_index_s, edge_index_f)
        final_train = multilabel_scores(logits[train_fit_mask], loaded.y[train_fit_mask], loaded.y_mask[train_fit_mask])
        final_val = multilabel_scores(logits[val_mask], loaded.y[val_mask], loaded.y_mask[val_mask])
        final_test = multilabel_scores(
            logits[test_domain_mask], loaded.y[test_domain_mask], loaded.y_mask[test_domain_mask]
        )

    report = {
        "run_id": run_id,
        "protocol": "Setting B: full-graph forward, supervision only on train subset within V_train; extrapolation on V_test",
        "best_val_loss": best_val,
        "n_nodes": int(n_nodes),
        "n_species": int(n_species),
        "feature_cols": feature_cols,
        "zscore_cols": zscore_cols,
        "lat_threshold_tau": float(args.lat_threshold_tau),
        "split_sizes": {
            "train_domain_phi_ge_tau": int(train_domain_mask.sum().item()),
            "train_fit": int(train_fit_mask.sum().item()),
            "val_within_train_domain": int(val_mask.sum().item()),
            "test_domain_phi_lt_tau": int(test_domain_mask.sum().item()),
        },
        "edge_counts": {"E_s": len(e_s), "E_f": len(e_f), "E_all": len(e_s.union(e_f))},
        "ef_label_scope": "Y rows masked to zero outside V_train (phi>=tau) when building E_f",
        "zscore_scope": "mean/std from all nodes with phi>=tau",
        "corridor": {
            "flyway_geojson": str(args.flyway),
            "n_unassigned_nodes": n_unassigned,
            "corridor_names": corridor_names,
            "nodes_per_corridor": {
                corridor_names[i]: int((corridor_ids == i).sum()) for i in range(len(corridor_names))
            },
        },
        "channels": {
            "spatial": "GCNConv symmetric norm on E_s (neighbor-only; no graph self-loop)",
            "flyway": "GCNConv symmetric norm on E_f (neighbor-only)",
            "fusion": "h = ReLU(α_S·GCN_S + α_F·GCN_F + W_self·x); α = softmax(learned)",
            "encoder": "2-layer MLP on raw X before GCN stacks",
        },
        "final_scores": {"train_fit": final_train, "val": final_val, "test_extrapolation_domain": final_test},
        "history": history,
    }

    (out_dir / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (out_dir / "config.json").write_text(json.dumps(vars(args), indent=2, default=str), encoding="utf-8")

    print("Saved checkpoint:", ckpt_path)
    print("Saved metrics:", out_dir / "metrics.json")


if __name__ == "__main__":
    main()
