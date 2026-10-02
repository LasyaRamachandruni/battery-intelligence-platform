"""Flag cells whose early discharge curves look unlike normal cells.

A healthy LFP cell's Q(V) curve has a characteristic shape. A cell with a
manufacturing defect (contamination, poor electrolyte fill, a damaged
electrode) often shows a distorted curve well before its capacity drops.

Method:
1. Take each cell's discharge curve at an early cycle, normalized by its own
   capacity so the score reflects shape, not size, and averaged into 100 voltage
   bins to suppress point-to-point sensor noise.
2. Fit PCA on reference cells; keep enough components for 95% of the variance
   (at most 8), so the model describes real shape variation rather than noise.
3. Score each cell two ways: Mahalanobis distance inside the PCA subspace
   (Hotelling T^2: unusual combinations of normal variation) and the
   reconstruction error outside it (Q residual: variation never seen before).
4. Thresholds come from out-of-fold scores: each reference cell is scored by a
   model that did not see it (5-fold). In-sample scores are optimistically small,
   which would make every new cell look anomalous. A cell beyond either
   threshold is flagged.

This is the standard multivariate SPC pairing (T^2 + Q) used in process monitoring.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold

BINS = 100
MAX_COMPONENTS = 8


@dataclass
class CurveAnomalyModel:
    pca: PCA
    t2_threshold: float
    q_threshold: float

    def scores(self, curves: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x = _normalize(curves)
        z = self.pca.transform(x)
        t2 = np.sum(z**2 / self.pca.explained_variance_, axis=1)
        recon = self.pca.inverse_transform(z)
        q = np.sum((x - recon) ** 2, axis=1)
        return t2, q

    def flag(self, curves: np.ndarray) -> np.ndarray:
        t2, q = self.scores(curves)
        return (t2 > self.t2_threshold) | (q > self.q_threshold)


def _normalize(curves: np.ndarray) -> np.ndarray:
    c = np.asarray(curves, dtype=float)
    c = c / c[:, -1:]
    n = c.shape[1] // BINS * BINS
    return c[:, :n].reshape(c.shape[0], BINS, -1).mean(axis=2)


def _fit_pca(x: np.ndarray, variance: float) -> PCA:
    full = PCA(svd_solver="full").fit(x)
    k = int(np.searchsorted(np.cumsum(full.explained_variance_ratio_), variance) + 1)
    return PCA(n_components=min(k, MAX_COMPONENTS, x.shape[0] - 1), svd_solver="full").fit(x)


def fit(reference_curves: np.ndarray, variance: float = 0.95, quantile: float = 0.99,
        folds: int = 5, seed: int = 0) -> CurveAnomalyModel:
    curves = np.asarray(reference_curves, dtype=float)
    if curves.shape[0] < 2 * folds:
        raise ValueError(f"need at least {2 * folds} reference cells")
    x = _normalize(curves)

    t2_oof, q_oof = np.empty(len(x)), np.empty(len(x))
    for train, test in KFold(folds, shuffle=True, random_state=seed).split(x):
        m = CurveAnomalyModel(_fit_pca(x[train], variance), np.inf, np.inf)
        t2_oof[test], q_oof[test] = m.scores(curves[test])

    model = CurveAnomalyModel(_fit_pca(x, variance), np.inf, np.inf)
    model.t2_threshold = float(np.quantile(t2_oof, quantile))
    model.q_threshold = float(np.quantile(q_oof, quantile))
    return model
