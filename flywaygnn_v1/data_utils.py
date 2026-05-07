from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch


@dataclass
class LoadedData:
    x: torch.Tensor
    y: torch.Tensor
    y_mask: torch.Tensor
    cell_lat: np.ndarray
    cell_lon: np.ndarray
    species_names: list[str]


def load_node_features_and_labels(
    features_csv: Path,
    labels_csv: Path,
    feature_cols: list[str],
) -> LoadedData:
    df_x = pd.read_csv(features_csv)
    df_y = pd.read_csv(labels_csv)

    if len(df_x) != len(df_y):
        raise ValueError(f"Feature rows ({len(df_x)}) != label rows ({len(df_y)})")

    for c in ("cell_lat", "cell_lon"):
        if c not in df_x.columns or c not in df_y.columns:
            raise ValueError(f"Missing required column: {c}")

    if not np.array_equal(df_x["cell_lat"].to_numpy(), df_y["cell_lat"].to_numpy()) or not np.array_equal(
        df_x["cell_lon"].to_numpy(), df_y["cell_lon"].to_numpy()
    ):
        raise ValueError("Feature/label grid order mismatch on (cell_lat, cell_lon)")

    missing = [c for c in feature_cols if c not in df_x.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")

    species_names = [c for c in df_y.columns if c not in ("cell_lat", "cell_lon")]
    if not species_names:
        raise ValueError("No species label columns found")

    x_np = df_x[feature_cols].to_numpy(dtype=np.float32)
    y_np = df_y[species_names].to_numpy(dtype=np.float32)
    y_mask_np = np.isfinite(y_np).astype(np.float32)
    y_np = np.nan_to_num(y_np, nan=0.0)

    return LoadedData(
        x=torch.from_numpy(x_np),
        y=torch.from_numpy(y_np),
        y_mask=torch.from_numpy(y_mask_np),
        cell_lat=df_x["cell_lat"].to_numpy(dtype=np.int32),
        cell_lon=df_x["cell_lon"].to_numpy(dtype=np.int32),
        species_names=species_names,
    )


def make_node_splits(n_nodes: int, train_ratio: float, val_ratio: float, seed: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if train_ratio <= 0 or val_ratio <= 0 or (train_ratio + val_ratio) >= 1:
        raise ValueError("Require 0 < train_ratio, val_ratio and train+val < 1")

    rng = np.random.default_rng(seed)
    idx = np.arange(n_nodes, dtype=np.int64)
    rng.shuffle(idx)

    n_train = int(round(n_nodes * train_ratio))
    n_val = int(round(n_nodes * val_ratio))
    n_train = max(1, min(n_train, n_nodes - 2))
    n_val = max(1, min(n_val, n_nodes - n_train - 1))

    train_idx = idx[:n_train]
    val_idx = idx[n_train : n_train + n_val]
    test_idx = idx[n_train + n_val :]

    train_mask = torch.zeros(n_nodes, dtype=torch.bool)
    val_mask = torch.zeros(n_nodes, dtype=torch.bool)
    test_mask = torch.zeros(n_nodes, dtype=torch.bool)
    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True
    return train_mask, val_mask, test_mask


def make_geographic_domain_masks(
    lat_center: np.ndarray,
    *,
    tau: float = -12.0,
    val_frac_within_train_domain: float = 0.15,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Geographic split matching the paper: φ(v)=cell_lat+0.5; train-supervised domain V_train = {φ≥τ},
    extrapolation domain V_test = {φ<τ}. Random validation subset inside V_train; training loss only on
    the train-fit subset (validation nodes excluded from loss).

    Returns:
        train_domain_mask: φ≥τ (for z-score stats and E_f label masking)
        train_fit_mask: non-validation nodes in V_train (optimization)
        val_mask: validation nodes in V_train
        test_domain_mask: φ<τ (evaluation only, no supervised gradient)
    """
    n = int(lat_center.shape[0])
    phi = lat_center.astype(np.float64)
    train_dom = phi >= float(tau)
    test_dom = phi < float(tau)
    idx = np.nonzero(train_dom)[0]
    if idx.size < 3:
        raise ValueError("Too few nodes in train domain for train/val split")

    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    n_val = max(1, int(round(idx.size * float(val_frac_within_train_domain))))
    n_val = min(n_val, idx.size - 1)
    val_idx = idx[:n_val]
    fit_idx = idx[n_val:]

    train_domain_mask = torch.from_numpy(train_dom)
    test_domain_mask = torch.from_numpy(test_dom)
    train_fit_mask = torch.zeros(n, dtype=torch.bool)
    val_mask = torch.zeros(n, dtype=torch.bool)
    fit_t = torch.as_tensor(fit_idx, dtype=torch.long)
    val_t = torch.as_tensor(val_idx, dtype=torch.long)
    train_fit_mask[fit_t] = True
    val_mask[val_t] = True
    return train_domain_mask, train_fit_mask, val_mask, test_domain_mask


def zscore_features_train_domain(
    x: torch.Tensor,
    train_domain_mask: torch.Tensor,
    *,
    zscore_col_indices: list[int],
) -> torch.Tensor:
    """Z-score selected columns using mean/std from train-domain nodes only; apply transform to all nodes."""
    if not zscore_col_indices:
        return x
    m = train_domain_mask.to(device=x.device)
    out = x.clone()
    for j in zscore_col_indices:
        col = x[:, j][m]
        mu = col.mean()
        sig = col.std(unbiased=False).clamp(min=1e-6)
        out[:, j] = (x[:, j] - mu) / sig
    return out
