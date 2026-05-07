"""Multilabel metrics aligned with paper §5.2 (micro/macro F1, AUPRC, AUROC)."""

from __future__ import annotations

import warnings

import numpy as np

try:
    from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
except ImportError as e:  # pragma: no cover
    raise ImportError("scikit-learn is required for paper metrics: pip install scikit-learn") from e


def multilabel_metrics_paper(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    threshold: float = 0.5,
) -> dict[str, float]:
    """
    y_true, y_prob: (n_samples, n_labels), float in {0,1} / [0,1].
    Assumes same finite mask over all entries (typical binary Y with no missing).
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    y_pred = (y_prob >= threshold).astype(np.int32)

    out: dict[str, float] = {}

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        out["f1_micro"] = float(f1_score(y_true, y_pred, average="micro", zero_division=0))
        out["f1_macro"] = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

        try:
            out["auprc_micro"] = float(average_precision_score(y_true, y_prob, average="micro"))
        except ValueError:
            out["auprc_micro"] = float("nan")
        try:
            out["auprc_macro"] = float(average_precision_score(y_true, y_prob, average="macro"))
        except ValueError:
            out["auprc_macro"] = float("nan")

        for name, avg in [("auroc_micro", "micro"), ("auroc_macro", "macro")]:
            try:
                out[name] = float(roc_auc_score(y_true, y_prob, average=avg))
            except ValueError:
                out[name] = float("nan")

    return out


def multilabel_augmented_test_metrics(
    y_te: np.ndarray,
    p_te: np.ndarray,
    *,
    evaluable_min_positives: int,
) -> dict[str, float]:
    """
    Eval-m5 subset (paper Table 3 / appendix): species with ≥ evaluable_min_positives test positives.

    - auprc_macro_eval_m5, auroc_macro_eval_m5: macro average over that species set.
    - n_labels_eval_m5: |species set|.
    """
    y_te = np.asarray(y_te, dtype=np.float64)
    p_te = np.asarray(p_te, dtype=np.float64)
    out: dict[str, float] = {}

    if evaluable_min_positives > 0:
        eval_idx = evaluable_species_indices(y_te, evaluable_min_positives=evaluable_min_positives)
        aps: list[float] = []
        for j in eval_idx:
            c = y_te[:, int(j)]
            try:
                aps.append(float(average_precision_score(c, p_te[:, int(j)])))
            except ValueError:
                pass
        out["auprc_macro_eval_m5"] = float(np.mean(aps)) if aps else float("nan")
        out["n_labels_eval_m5"] = float(len(aps))
        out["auroc_macro_eval_m5"] = macro_auroc_on_species(y_te, p_te, eval_idx)
    else:
        out["auprc_macro_eval_m5"] = float("nan")
        out["n_labels_eval_m5"] = float("nan")
        out["auroc_macro_eval_m5"] = float("nan")

    return out


def evaluable_species_indices(y_true: np.ndarray, *, evaluable_min_positives: int) -> np.ndarray:
    y_true = np.asarray(y_true, dtype=np.float64)
    if y_true.ndim != 2:
        raise ValueError("y_true must be 2D [n_samples, n_labels]")
    if evaluable_min_positives <= 0:
        return np.arange(y_true.shape[1], dtype=np.int64)
    pos = y_true.sum(axis=0)
    return np.nonzero(pos >= float(evaluable_min_positives))[0].astype(np.int64)


def species_auprc_values(y_true: np.ndarray, y_prob: np.ndarray, species_idx: np.ndarray) -> np.ndarray:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    species_idx = np.asarray(species_idx, dtype=np.int64)
    out = np.full((species_idx.shape[0],), np.nan, dtype=np.float64)
    for i, j in enumerate(species_idx):
        yt = y_true[:, int(j)]
        yp = y_prob[:, int(j)]
        try:
            out[i] = float(average_precision_score(yt, yp))
        except ValueError:
            out[i] = np.nan
    return out


def macro_auroc_on_species(y_true: np.ndarray, y_prob: np.ndarray, species_idx: np.ndarray) -> float:
    vals: list[float] = []
    for j in np.asarray(species_idx, dtype=np.int64):
        try:
            vals.append(float(roc_auc_score(y_true[:, int(j)], y_prob[:, int(j)])))
        except ValueError:
            pass
    return float(np.mean(vals)) if vals else float("nan")


def aggregate_mean_std(rows: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    if not rows:
        return {}
    keys = list(rows[0].keys())
    out: dict[str, dict[str, float]] = {}
    for k in keys:
        vals: list[float] = []
        for r in rows:
            if k not in r:
                continue
            v = r[k]
            if isinstance(v, float) and np.isnan(v):
                continue
            if isinstance(v, (int, float)):
                vals.append(float(v))
        if not vals:
            out[k] = {"mean": float("nan"), "std": float("nan")}
        else:
            a = np.array(vals, dtype=np.float64)
            out[k] = {"mean": float(a.mean()), "std": float(a.std(ddof=1)) if len(vals) > 1 else 0.0}
    return out
