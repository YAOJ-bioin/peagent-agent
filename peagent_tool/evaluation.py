"""Small evaluation helpers for PEAgent prediction tables."""

from __future__ import annotations

from typing import Mapping, Optional

import numpy as np
import pandas as pd


def _as_arrays(y_true, y_score) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y_true)
    score = np.asarray(y_score, dtype=np.float64)
    if y.shape[0] != score.shape[0]:
        raise ValueError("y_true and y_score must have the same length.")
    mask = np.isfinite(score)
    y = y[mask]
    score = score[mask]
    return y, score


def safe_pearson(x, y) -> float:
    """Return Pearson correlation, or NaN when it is undefined."""

    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    x_arr = x_arr[mask]
    y_arr = y_arr[mask]
    if x_arr.size == 0 or np.allclose(x_arr.std(), 0.0) or np.allclose(y_arr.std(), 0.0):
        return float("nan")
    return float(np.corrcoef(x_arr, y_arr)[0, 1])


def safe_spearman(x, y) -> float:
    """Return Spearman correlation, or NaN when it is undefined."""

    x_arr = pd.Series(np.asarray(x, dtype=np.float64))
    y_arr = pd.Series(np.asarray(y, dtype=np.float64))
    mask = np.isfinite(x_arr.to_numpy()) & np.isfinite(y_arr.to_numpy())
    x_rank = x_arr[mask].rank(method="average").to_numpy(dtype=np.float64)
    y_rank = y_arr[mask].rank(method="average").to_numpy(dtype=np.float64)
    return safe_pearson(x_rank, y_rank)


def safe_mae(x, y) -> float:
    """Return mean absolute error, or NaN for empty finite input."""

    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    if not np.any(mask):
        return float("nan")
    return float(np.mean(np.abs(x_arr[mask] - y_arr[mask])))


def safe_auroc(y_true, y_score) -> float:
    """Return binary AUROC, or NaN when labels contain only one class."""

    y, score = _as_arrays(y_true, y_score)
    y = y.astype(np.int8)
    if y.size == 0 or np.unique(y).size < 2:
        return float("nan")
    positives = y == 1
    n_pos = int(positives.sum())
    n_neg = int(y.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = pd.Series(score).rank(method="average").to_numpy(dtype=np.float64)
    rank_sum_pos = float(ranks[positives].sum())
    auc = (rank_sum_pos - (n_pos * (n_pos + 1) / 2.0)) / float(n_pos * n_neg)
    return float(auc)


def safe_average_precision(y_true, y_score) -> float:
    """Return binary average precision, or NaN when labels contain only one class."""

    y, score = _as_arrays(y_true, y_score)
    y = y.astype(np.int8)
    if y.size == 0 or np.unique(y).size < 2:
        return float("nan")
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(-score, kind="mergesort")
    y_sorted = y[order]
    true_positive = np.cumsum(y_sorted == 1)
    ranks = np.arange(1, y_sorted.size + 1, dtype=np.float64)
    precision = true_positive / ranks
    return float(precision[y_sorted == 1].sum() / n_pos)


def binary_classification_metrics(
    y_true,
    y_score,
    prefix: Optional[str] = None,
) -> Mapping[str, float]:
    """Return AUROC and average precision for one binary scoring task."""

    names = {
        "auroc": safe_auroc(y_true, y_score),
        "average_precision": safe_average_precision(y_true, y_score),
    }
    if not prefix:
        return names
    return {"%s_%s" % (prefix, key): value for key, value in names.items()}
