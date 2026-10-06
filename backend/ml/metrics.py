"""
Shared forecast scores and intervals for the offline evals and fits.

W/D/L forecasts are (p_home, p_draw, p_away) with actuals 0/1/2.
"""

from __future__ import annotations

import math
from typing import List, Tuple

import numpy as np

EPS = 1e-12
Z95 = 1.959964


def log_loss(preds, actuals) -> float:
    total = 0.0
    for p, y in zip(preds, actuals):
        total += -math.log(max(p[y], EPS))
    return total / len(preds)


def brier(preds, actuals) -> float:
    total = 0.0
    for p, y in zip(preds, actuals):
        t = [0.0, 0.0, 0.0]
        t[y] = 1.0
        total += sum((p[k] - t[k]) ** 2 for k in range(3))
    return total / len(preds)


def per_match_ll(preds, actuals) -> np.ndarray:
    return np.array([-math.log(max(p[y], EPS)) for p, y in zip(preds, actuals)])


def per_match_brier(preds, actuals) -> np.ndarray:
    return np.array(
        [sum((p[k] - (1.0 if k == y else 0.0)) ** 2 for k in range(3)) for p, y in zip(preds, actuals)]
    )


def per_match_rps(preds, actuals) -> np.ndarray:
    """Ranked probability score over ordered outcomes H/D/A (Constantinou & Fenton 2012):
    1/(K-1) * sum_k (cumP_k - cumO_k)^2."""
    P = np.asarray(preds, dtype=float)
    onehot = np.eye(P.shape[1])[np.asarray(actuals)]
    return ((np.cumsum(P, 1) - np.cumsum(onehot, 1))[:, :-1] ** 2).sum(1) / (P.shape[1] - 1)


def rps(preds, actuals) -> float:
    return float(per_match_rps(preds, actuals).mean())


def murphy(preds, actuals, bins: int = 10) -> dict:
    """Murphy (1973) decomposition of the multi-class Brier, summed over classes:
    Brier ~= reliability - resolution + uncertainty (exact up to within-bin spread)."""
    P = np.asarray(preds, dtype=float)
    onehot = np.eye(P.shape[1])[np.asarray(actuals)]
    rel = res = unc = 0.0
    for k in range(P.shape[1]):
        p, o = P[:, k], onehot[:, k]
        b = np.minimum((p * bins).astype(int), bins - 1)
        n = np.bincount(b, minlength=bins)
        m = n > 0
        pb = np.bincount(b, p, bins)[m] / n[m]
        ob = np.bincount(b, o, bins)[m] / n[m]
        w = n[m] / len(p)
        rel += float((w * (pb - ob) ** 2).sum())
        res += float((w * (ob - o.mean()) ** 2).sum())
        unc += float(o.mean() * (1 - o.mean()))
    return {"reliability": round(rel, 4), "resolution": round(res, 4), "uncertainty": round(unc, 4),
            "brier": round(float(((P - onehot) ** 2).sum(1).mean()), 4)}


def paired_bootstrap(diff: np.ndarray, n_boot: int = 5000, seed: int = 0) -> List[float]:
    """95% percentile CI for the mean of a per-match difference."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diff), size=(n_boot, len(diff)))
    means = diff[idx].mean(axis=1)
    return [round(float(np.percentile(means, 2.5)), 4), round(float(np.percentile(means, 97.5)), 4)]


def binary_logloss(p: np.ndarray, y: np.ndarray) -> float:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def wilson(k: int, n: int, z: float = Z95) -> Tuple[float, float]:
    """Wilson score interval for k successes in n; (0, 1) when n is 0."""
    if n <= 0:
        return 0.0, 1.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)
