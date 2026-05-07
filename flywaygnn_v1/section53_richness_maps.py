#!/usr/bin/env python3
"""
Paper §5.3-style figures: summed predicted probability (“richness”) on the South American extrapolation domain.

- Panel A: 1×3 — XGBoost | Vanilla GCN | FlywayGNN (shared color scale).
- Panel B (optional): FlywayGNN − Vanilla GCN (RdBu).
- Panel C (optional): zoom inset inside a bbox (cartopy if available, else matplotlib).

Requires PyTorch, torch-geometric, geopandas, numpy, pandas, matplotlib; **xgboost** + **scikit-learn** for the left column.
On macOS without libomp: install OpenMP (`brew install libomp`) or use a conda env with working XGBoost.
Without importable XGBoost the script exits with an error and install hints. Use `--fast` for shorter CPU runs.
Extra qualitative figure: `--row3-per-panel-vmax` writes `fig5_3a_richness_row3_per_panel_vmax.png` (per-column vmax).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from data_utils import load_node_features_and_labels, make_geographic_domain_masks, zscore_features_train_domain
from graph_builder import build_flyway_corridor_edges, build_spatial_edges, edge_set_to_undirected_index, load_flyway_wgs84
from model import FlywayGNN, VanillaSpatialGCN

PIPE_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PIPE_ROOT.parent
DEFAULT_FEATURES = PIPE_ROOT / "graph_build" / "node_features_imputed_4554.csv"
DEFAULT_LABELS = PIPE_ROOT / "graph_build" / "Y_binary_4554x155.csv"
DEFAULT_FLYWAY = DATA_ROOT / "birdlife_americas_flyway" / "americas_flyway_subregions.geojson"


def masked_bce_with_logits(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    loss_raw = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    return (loss_raw * mask).sum() / mask.sum().clamp(min=1.0)


def train_gnn_fixed_epochs(
    model: nn.Module,
    *,
    vanilla: bool,
    x: torch.Tensor,
    y: torch.Tensor,
    d_train: torch.Tensor,
    edge_index_s: torch.Tensor,
    edge_index_f: torch.Tensor,
    device: torch.device,
    torch_seed: int,
    epochs: int,
    lr: float,
    weight_decay: float,
) -> torch.Tensor:
    """Train for a fixed number of epochs (no sklearn validation loop); return full-graph logits on CPU."""
    torch.manual_seed(torch_seed)
    model = model.to(device)
    x = x.to(device)
    y = y.to(device)
    d_train = d_train.to(device)
    edge_index_s = edge_index_s.to(device)
    edge_index_f = edge_index_f.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    for _ in range(int(epochs)):
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
    return logits.detach().cpu()

try:
    from xgboost import XGBClassifier

    _HAS_XGB = True
except Exception:  # ImportError / XGBoostError(libomp), etc.
    XGBClassifier = None  # type: ignore[misc, assignment]
    _HAS_XGB = False


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def xgboost_prob_all_nodes(
    x: torch.Tensor,
    y: torch.Tensor,
    train_fit_mask: torch.Tensor,
    *,
    seed: int,
) -> np.ndarray:
    """Per-label XGBoost on train_fit rows; predict probabilities for all nodes."""
    if not _HAS_XGB:
        raise RuntimeError("xgboost unavailable")
    X_tr = x[train_fit_mask].cpu().numpy()
    y_tr = y[train_fit_mask].cpu().numpy().astype(np.int32)
    X_all = x.cpu().numpy()
    n_all, n_lab = X_all.shape
    prob = np.zeros((n_all, n_lab), dtype=np.float64)
    n_tr = X_tr.shape[0]
    for j in range(n_lab):
        yc = y_tr[:, j]
        pos = int(yc.sum())
        if pos == 0:
            prob[:, j] = 1e-6
            continue
        if pos == n_tr:
            prob[:, j] = 1.0 - 1e-6
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
        prob[:, j] = clf.predict_proba(X_all)[:, 1]
    return prob


def _apply_shared_geo_aspect(axes: Any, lon: np.ndarray, lat: np.ndarray, *, equal_aspect: bool = True) -> None:
    """Shared geographic limits across subplots; optional equal lat/lon aspect."""
    lon_pad = float((lon.max() - lon.min()) * 0.02 + 1e-6)
    lat_pad = float((lat.max() - lat.min()) * 0.02 + 1e-6)
    x0, x1 = float(lon.min() - lon_pad), float(lon.max() + lon_pad)
    y0, y1 = float(lat.min() - lat_pad), float(lat.max() + lat_pad)
    for ax in axes:
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
        if equal_aspect:
            # Keep subplot boxes aligned in width/height across rows.
            # With "datalim", data limits expand instead of shrinking axis boxes.
            ax.set_aspect("equal", adjustable="datalim")
        else:
            ax.set_aspect("auto")


def _plot_row3(
    lon: np.ndarray,
    lat: np.ndarray,
    r_xgb: np.ndarray,
    r_van: np.ndarray,
    r_fly: np.ndarray,
    out_path: Path,
    *,
    use_cartopy: bool,
    left_panel_title: str = "XGBoost",
    bbox: tuple[float, float, float, float] | None = None,
) -> None:
    import matplotlib.pyplot as plt

    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
    except ImportError:
        use_cartopy = False

    if use_cartopy:
        proj = ccrs.PlateCarree()
        fig, axes = plt.subplots(1, 3, figsize=(14.5, 5.2), subplot_kw={"projection": proj})
        for ax in axes:
            ax.add_feature(cfeature.COASTLINE, linewidth=0.35)
            ax.add_feature(cfeature.BORDERS, linewidth=0.2, alpha=0.5)
            ax.set_extent([lon.min() - 2, lon.max() + 2, lat.min() - 2, lat.max() + 2], crs=proj)
    else:
        fig, axes = plt.subplots(1, 3, figsize=(15.6, 5.2))
        # Reserve space for colorbar/title before scatter + shared limits; keeps aspect after subplots_adjust.
        fig.subplots_adjust(left=0.045, right=0.995, top=0.80, bottom=0.10, wspace=0.20)

    vmax = float(max(r_xgb.max(), r_van.max(), r_fly.max()) * 1.02)
    vmin = 0.0
    titles = (left_panel_title, r"Vanilla GCN ($E_s$)", r"FlywayGNN ($E_s+E_f$)")
    arrs = (r_xgb, r_van, r_fly)
    mappable = None
    for ax, arr, ti in zip(axes, arrs, titles, strict=True):
        if use_cartopy:
            import cartopy.crs as ccrs

            mappable = ax.scatter(
                lon,
                lat,
                c=arr,
                s=48,
                cmap="magma",
                vmin=vmin,
                vmax=vmax,
                alpha=0.9,
                transform=ccrs.PlateCarree(),
            )
            _ = ti
        else:
            mappable = ax.scatter(
                lon,
                lat,
                c=arr,
                s=55,
                cmap="magma",
                vmin=vmin,
                vmax=vmax,
                alpha=0.92,
                edgecolors="none",
            )
            ax.set_xlabel("Longitude (°E)")
            ax.set_ylabel("Latitude (°N)")

        if bbox is not None:
            from matplotlib.patches import Rectangle

            lon0, lon1, lat0, lat1 = bbox
            ax.add_patch(
                Rectangle(
                    (lon0, lat0),
                    lon1 - lon0,
                    lat1 - lat0,
                    fill=False,
                    edgecolor="red",
                    linewidth=2.0,
                )
            )

    if not use_cartopy:
        # Keep true geographic aspect ratio (no distortion).
        _apply_shared_geo_aspect(axes, lon, lat, equal_aspect=True)
    else:
        fig.subplots_adjust(left=0.045, right=0.995, top=0.80, bottom=0.10, wspace=0.20)

    # Bottom horizontal colorbar for cleaner two-row composition.
    cax = fig.add_axes([0.20, 0.93, 0.60, 0.026])
    fig.colorbar(mappable, cax=cax, orientation="horizontal", label=r"$\sum_s \hat{Y}_{v,s}$ (richness)")
    fig.savefig(out_path, dpi=200, facecolor="white")
    plt.close(fig)


def _plot_row3_per_panel_vmax(
    lon: np.ndarray,
    lat: np.ndarray,
    r_xgb: np.ndarray,
    r_van: np.ndarray,
    r_fly: np.ndarray,
    out_path: Path,
    *,
    left_panel_title: str = "XGBoost",
    percentile: float = 99.5,
) -> None:
    """
    Same data/layout as fig5_3a but independent color scales per column:
    vmax_i = nanpercentile(arr_i, p)*1.02 (qualitative spatial patterns).
    Matplotlib only (not implemented under cartopy); does not overwrite fig5_3a_richness_row3.png.
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.9))
    fig.subplots_adjust(left=0.06, right=0.98, top=0.80, bottom=0.18, wspace=0.36)
    titles = (left_panel_title, r"Vanilla GCN ($E_s$)", r"FlywayGNN ($E_s+E_f$)")
    arrs = (r_xgb, r_van, r_fly)
    scatters: list[Any] = []
    for ax, arr, ti in zip(axes, arrs, titles, strict=True):
        hi = float(np.nanpercentile(arr.astype(np.float64), float(percentile)))
        vmax_i = max(hi * 1.02, 1e-6)
        sc = ax.scatter(
            lon,
            lat,
            c=arr,
            s=55,
            cmap="magma",
            vmin=0.0,
            vmax=vmax_i,
            alpha=0.92,
            edgecolors="none",
        )
        scatters.append(sc)
        ax.set_title(ti, fontsize=11)
        ax.set_xlabel("Longitude (°E)")
        ax.set_ylabel("Latitude (°N)")
    _apply_shared_geo_aspect(axes, lon, lat)
    for ax, sc in zip(axes, scatters, strict=True):
        fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, label=r"$\sum_s \hat{Y}_{v,s}$")
    fig.suptitle(
        f"Same domain, per-panel vmax (p={percentile} percentile ×1.02; qualitative only)",
        fontsize=11,
        y=0.98,
    )
    fig.savefig(out_path, dpi=200, facecolor="white")
    plt.close(fig)


def _plot_diff_rdbu(lon: np.ndarray, lat: np.ndarray, diff: np.ndarray, out_path: Path, *, use_cartopy: bool) -> None:
    import matplotlib.pyplot as plt

    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
    except ImportError:
        use_cartopy = False

    vmax = float(np.nanpercentile(np.abs(diff), 98))
    vmax = max(vmax, 1e-3)

    if use_cartopy:
        proj = ccrs.PlateCarree()
        fig, ax = plt.subplots(figsize=(6.2, 5.5), subplot_kw={"projection": proj})
        fig.subplots_adjust(left=0.10, right=0.92, top=0.92, bottom=0.12)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.35)
        ax.set_extent([lon.min() - 2, lon.max() + 2, lat.min() - 2, lat.max() + 2], crs=proj)
        sc = ax.scatter(
            lon,
            lat,
            c=diff,
            s=52,
            cmap="RdBu_r",
            vmin=-vmax,
            vmax=vmax,
            transform=ccrs.PlateCarree(),
        )
    else:
        fig, ax = plt.subplots(figsize=(6.2, 5.0))
        fig.subplots_adjust(left=0.10, right=0.92, top=0.92, bottom=0.12)
        sc = ax.scatter(lon, lat, c=diff, s=52, cmap="RdBu_r", vmin=-vmax, vmax=vmax, edgecolors="none")
        ax.set_xlabel("Longitude (°E)")
        ax.set_ylabel("Latitude (°N)")
        _apply_shared_geo_aspect((ax,), lon, lat)

    ax.set_title(r"FlywayGNN $-$ Vanilla GCN (richness)")
    fig.colorbar(sc, ax=ax, shrink=0.82, label="Δ richness")
    fig.savefig(out_path, dpi=200, facecolor="white")
    plt.close(fig)


def _plot_zoom_inset(
    lon: np.ndarray,
    lat: np.ndarray,
    r_xgb: np.ndarray,
    r_van: np.ndarray,
    r_fly: np.ndarray,
    bbox: tuple[float, float, float, float],
    out_path: Path,
    *,
    use_cartopy: bool,
    xpad_frac: float = 0.0,
    ypad_frac: float = 0.0,
    match_top_panel_aspect: bool = False,
) -> None:
    """Zoomed 1×3 row: XGBoost | Vanilla GCN | FlywayGNN (shared scale, same order)."""
    import matplotlib.pyplot as plt

    _ = use_cartopy
    lon0, lon1, lat0, lat1 = bbox
    m = (lon >= lon0) & (lon <= lon1) & (lat >= lat0) & (lat <= lat1)

    # Keep identical canvas width with fig5_3a for vertical alignment in papers.
    fig, axes = plt.subplots(1, 3, figsize=(15.6, 4.6))
    # Match fig5_3a panel geometry so upper/lower rows align visually.
    fig.subplots_adjust(left=0.045, right=0.995, top=0.80, bottom=0.10, wspace=0.20)
    vmax = float(max(r_xgb.max(), r_van.max(), r_fly.max()) * 1.02)
    ax0, ax1, ax2 = axes
    xpad = (lon1 - lon0) * float(max(xpad_frac, 0.0))
    ypad = (lat1 - lat0) * float(max(ypad_frac, 0.0))
    zx0, zx1 = lon0 - xpad, lon1 + xpad
    zy0, zy1 = lat0 - ypad, lat1 + ypad
    if match_top_panel_aspect:
        # Match the top-row panel x/y ratio by expanding zoom y-range only (no distortion).
        top_x = float(np.nanmax(lon) - np.nanmin(lon))
        top_y = float(np.nanmax(lat) - np.nanmin(lat))
        if top_x > 0 and top_y > 0:
            target_x_over_y = top_x / top_y
            cur_x = float(zx1 - zx0)
            cur_y = float(zy1 - zy0)
            if cur_y > 0:
                cur_x_over_y = cur_x / cur_y
                if cur_x_over_y > target_x_over_y:
                    desired_y = cur_x / target_x_over_y
                    extra_y = max(0.0, desired_y - cur_y)
                    zy0 -= 0.5 * extra_y
                    zy1 += 0.5 * extra_y

    sc0 = ax0.scatter(lon[m], lat[m], c=r_xgb[m], s=90, cmap="magma", vmin=0, vmax=vmax, edgecolors="none")
    # Remove per-panel titles; keep only axes and shared colorbar.
    ax0.set_xlim(zx0, zx1)
    ax0.set_ylim(zy0, zy1)
    ax0.set_aspect("equal", adjustable="datalim")
    ax0.set_xlabel("Longitude (°E)")
    ax0.set_ylabel("Latitude (°N)")

    ax1.scatter(lon[m], lat[m], c=r_van[m], s=90, cmap="magma", vmin=0, vmax=vmax, edgecolors="none")
    ax1.set_xlim(zx0, zx1)
    ax1.set_ylim(zy0, zy1)
    ax1.set_aspect("equal", adjustable="datalim")
    ax1.set_xlabel("Longitude (°E)")

    ax2.scatter(lon[m], lat[m], c=r_fly[m], s=90, cmap="magma", vmin=0, vmax=vmax, edgecolors="none")
    ax2.set_xlim(zx0, zx1)
    ax2.set_ylim(zy0, zy1)
    ax2.set_aspect("equal", adjustable="datalim")
    ax2.set_xlabel("Longitude (°E)")

    cax = fig.add_axes([0.20, 0.93, 0.60, 0.026])
    fig.colorbar(sc0, cax=cax, orientation="horizontal", label="richness")
    fig.savefig(out_path, dpi=200, facecolor="white")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent / "runs" / "section53_figs")
    ap.add_argument("--lat-threshold-tau", type=float, default=-12.0)
    ap.add_argument("--seed", type=int, default=42, help="Train/val split inside train domain + model init")
    ap.add_argument("--lr", type=float, default=0.005)
    ap.add_argument("--hidden-dim", type=int, default=256)
    ap.add_argument("--K", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--fast", action="store_true", help="epochs=28, patience=5 (quick figures)")
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--no-diff", action="store_true")
    ap.add_argument("--no-zoom", action="store_true")
    ap.add_argument("--zoom-bbox", type=str, default="-62,-48,-36,-24", help="lon0,lon1,lat0,lat1")
    ap.add_argument("--cartopy", action="store_true", help="Use cartopy coastlines if installed")
    ap.add_argument("--zoom-xpad-frac", type=float, default=0.0, help="Horizontal padding fraction for zoom bbox")
    ap.add_argument("--zoom-ypad-frac", type=float, default=0.0, help="Vertical padding fraction for zoom bbox")
    ap.add_argument(
        "--zoom-match-top-panel-aspect",
        action="store_true",
        default=False,
        help="Pad zoom panel vertically to match top-row aspect ratio (no data distortion)",
    )
    ap.add_argument(
        "--row3-per-panel-vmax",
        action="store_true",
        help="Also save fig5_3a_richness_row3_per_panel_vmax.png (per-column vmax; does not overwrite main figure)",
    )
    ap.add_argument(
        "--row3-per-panel-percentile",
        type=float,
        default=99.5,
        help="Percentile for per-column vmax when using --row3-per-panel-vmax (default 99.5)",
    )
    args = ap.parse_args()

    if args.fast:
        args.epochs = 28
        args.patience = 5

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    feature_cols = [c.strip() for c in "NDVI_mean,LST_day_mean,DEM,water_frac,miss_NDVI,miss_LST,miss_DEM".split(",")]
    zscore_cols = [c.strip() for c in "NDVI_mean,LST_day_mean,DEM,water_frac".split(",")]
    zidx = [feature_cols.index(c) for c in zscore_cols]

    loaded = load_node_features_and_labels(DEFAULT_FEATURES, DEFAULT_LABELS, feature_cols=feature_cols)
    n_nodes = loaded.x.shape[0]
    n_species = loaded.y.shape[1]
    fly = load_flyway_wgs84(DEFAULT_FLYWAY)
    lon_center = loaded.cell_lon.astype(np.float64) + 0.5
    lat_center = loaded.cell_lat.astype(np.float64) + 0.5
    train_domain_row = (lat_center >= args.lat_threshold_tau).astype(bool)
    x_base = zscore_features_train_domain(
        loaded.x, torch.from_numpy(train_domain_row.astype(bool)), zscore_col_indices=zidx
    )

    e_s = build_spatial_edges(loaded.cell_lat, loaded.cell_lon)
    edge_index_s = edge_set_to_undirected_index(e_s, n_nodes=n_nodes)
    e_f = build_flyway_corridor_edges(
        loaded.y.numpy(),
        fly,
        lon_center,
        lat_center,
        breeding_lat_min=35.0,
        winter_lat_max=10.0,
        max_nodes_per_side=int(args.K),
        train_row_mask=train_domain_row,
    )
    e_f_use = e_s if not e_f else e_f
    edge_index_f = edge_set_to_undirected_index(e_f_use, n_nodes=n_nodes)

    train_domain_mask, train_fit_mask, val_mask, test_domain_mask = make_geographic_domain_masks(
        lat_center,
        tau=args.lat_threshold_tau,
        val_frac_within_train_domain=args.val_frac,
        seed=int(args.seed),
    )
    d_train = train_fit_mask[:, None].float() * loaded.y_mask
    device = _device()

    eis = edge_index_s.to(device)
    eif = edge_index_f.to(device)

    print("Training FlywayGNN …")
    model_f = FlywayGNN(in_dim=x_base.shape[1], hidden_dim=args.hidden_dim, out_dim=n_species, dropout=args.dropout)
    fly_logits = train_gnn_fixed_epochs(
        model_f,
        vanilla=False,
        x=x_base,
        y=loaded.y,
        d_train=d_train,
        edge_index_s=eis,
        edge_index_f=eif,
        device=device,
        torch_seed=int(args.seed),
        epochs=args.epochs,
        lr=float(args.lr),
        weight_decay=args.weight_decay,
    )
    fly_prob = torch.sigmoid(fly_logits).numpy()

    print("Training Vanilla GCN …")
    model_v = VanillaSpatialGCN(in_dim=x_base.shape[1], hidden_dim=args.hidden_dim, out_dim=n_species, dropout=args.dropout)
    van_logits = train_gnn_fixed_epochs(
        model_v,
        vanilla=True,
        x=x_base,
        y=loaded.y,
        d_train=d_train,
        edge_index_s=eis,
        edge_index_f=eif,
        device=device,
        torch_seed=int(args.seed) + 1000,
        epochs=args.epochs,
        lr=float(args.lr),
        weight_decay=args.weight_decay,
    )
    van_prob = torch.sigmoid(van_logits).numpy()

    left_title = "XGBoost"
    print("XGBoost (per-species, full grid) …")
    try:
        if not _HAS_XGB:
            raise RuntimeError("xgboost not importable")
        xgb_prob = xgboost_prob_all_nodes(x_base, loaded.y, train_fit_mask, seed=int(args.seed))
    except Exception as exc:  # noqa: BLE001
        print(
            "ERROR: XGBoost is required for the left column. Install with:\n"
            "  pip install -U xgboost 'scikit-learn>=1.5.2'\n"
            "On macOS you may need OpenMP: brew install libomp",
            flush=True,
        )
        raise SystemExit(1) from exc

    south = lat_center < float(args.lat_threshold_tau)
    lon_s = lon_center[south]
    lat_s = lat_center[south]
    r_xgb = xgb_prob[south].sum(axis=1)
    r_van = van_prob[south].sum(axis=1)
    r_fly = fly_prob[south].sum(axis=1)

    use_c = bool(args.cartopy)
    parts = [float(x) for x in args.zoom_bbox.split(",")]
    bbox = (parts[0], parts[1], parts[2], parts[3]) if len(parts) == 4 else None

    _plot_row3(
        lon_s,
        lat_s,
        r_xgb,
        r_van,
        r_fly,
        out_dir / "fig5_3a_richness_row3.png",
        use_cartopy=use_c,
        left_panel_title=left_title,
        bbox=bbox,
    )
    print("Saved", out_dir / "fig5_3a_richness_row3.png")

    if args.row3_per_panel_vmax:
        if use_c:
            print("Skip per-panel vmax figure: not implemented with --cartopy (rerun without --cartopy)")
        else:
            extra_path = out_dir / "fig5_3a_richness_row3_per_panel_vmax.png"
            _plot_row3_per_panel_vmax(
                lon_s,
                lat_s,
                r_xgb,
                r_van,
                r_fly,
                extra_path,
                left_panel_title=left_title,
                percentile=float(args.row3_per_panel_percentile),
            )
            print("Saved", extra_path)

    if not args.no_diff:
        _plot_diff_rdbu(lon_s, lat_s, r_fly - r_van, out_dir / "fig5_3b_diff_flyway_minus_vanilla.png", use_cartopy=use_c)
        print("Saved", out_dir / "fig5_3b_diff_flyway_minus_vanilla.png")

    if not args.no_zoom:
        if bbox is not None:
            _plot_zoom_inset(
                lon_s,
                lat_s,
                r_xgb,
                r_van,
                r_fly,
                bbox,
                out_dir / "fig5_3c_zoom_bbox.png",
                use_cartopy=use_c,
                xpad_frac=float(args.zoom_xpad_frac),
                ypad_frac=float(args.zoom_ypad_frac),
                match_top_panel_aspect=bool(args.zoom_match_top_panel_aspect),
            )
            print("Saved", out_dir / "fig5_3c_zoom_bbox.png")

    meta = {
        "tau": args.lat_threshold_tau,
        "seed": args.seed,
        "lr": args.lr,
        "hidden_dim": args.hidden_dim,
        "K": args.K,
        "epochs": args.epochs,
        "patience": args.patience,
        "n_south_cells": int(south.sum()),
        "left_panel_title": left_title,
        "left_baseline": "xgboost",
        "richness_mean": {"left_baseline": float(r_xgb.mean()), "vanilla": float(r_van.mean()), "flyway": float(r_fly.mean())},
    }
    if args.row3_per_panel_vmax and not use_c:
        hi_x = float(np.nanpercentile(r_xgb.astype(np.float64), float(args.row3_per_panel_percentile)))
        hi_v = float(np.nanpercentile(r_van.astype(np.float64), float(args.row3_per_panel_percentile)))
        hi_f = float(np.nanpercentile(r_fly.astype(np.float64), float(args.row3_per_panel_percentile)))
        meta["fig5_3a_per_panel_vmax"] = {
            "path": "fig5_3a_richness_row3_per_panel_vmax.png",
            "percentile": float(args.row3_per_panel_percentile),
            "vmax_per_panel": [
                max(hi_x * 1.02, 1e-6),
                max(hi_v * 1.02, 1e-6),
                max(hi_f * 1.02, 1e-6),
            ],
        }
    (out_dir / "section53_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("Saved", out_dir / "section53_meta.json")


if __name__ == "__main__":
    main()
