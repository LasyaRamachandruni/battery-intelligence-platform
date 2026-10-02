"""Data and model drift monitoring.

Cells from a new production batch, supplier or charging protocol can look
different from what the model was trained on. Predictions on them may be wrong
long before any cell reaches end of life and proves it. So we watch the inputs:

- Population Stability Index (PSI) per feature, binned on the reference
  (training) quantiles.  < 0.1 stable, 0.1-0.25 moderate shift, > 0.25 major shift.
- Two-sample Kolmogorov-Smirnov test per feature, as a significance check. With
  tens of cells, PSI alone fluctuates a lot, so a retrain is only recommended for
  a shift that is both large (PSI > 0.25) and significant (KS p < 0.01).
- Prediction drift: the same PSI on the model's predicted log cycle life.
- Performance, once labels arrive: % error and interval coverage on the newly
  completed cells, compared with the error measured at training time.

The report says whether to retrain and why.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy import stats

PSI_MODERATE, PSI_MAJOR = 0.1, 0.25
KS_ALPHA = 0.01  # a shift must also be statistically significant to trigger retraining


def psi(reference: np.ndarray, current: np.ndarray, bins: int | None = None) -> float:
    """PSI with quantile bins from the reference.

    Battery test programs are small (tens of cells per batch), and textbook PSI
    (10 bins, near-zero floor for empty bins) is unstable there: one empty bin
    adds ~0.7 on its own and two samples of the *same* population can score as a
    major shift. So bins scale with sample size (about 10 cells per bin, 3 to 10
    bins) and counts get add-0.5 smoothing.
    """
    ref = np.asarray(reference, float)
    cur = np.asarray(current, float)
    if bins is None:
        bins = int(np.clip(min(ref.size, cur.size) // 10, 3, 10))
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    r_counts = np.histogram(ref, edges)[0] + 0.5
    c_counts = np.histogram(cur, edges)[0] + 0.5
    r, c = r_counts / r_counts.sum(), c_counts / c_counts.sum()
    return float(np.sum((c - r) * np.log(c / r)))


@dataclass
class FeatureDrift:
    feature: str
    psi: float
    ks_pvalue: float
    status: str


def feature_drift(reference: pd.DataFrame, current: pd.DataFrame, features: list[str]) -> list[FeatureDrift]:
    out = []
    for f in features:
        p = psi(reference[f], current[f])
        ks = stats.ks_2samp(reference[f], current[f]).pvalue
        status = "major" if p > PSI_MAJOR else "moderate" if p > PSI_MODERATE else "stable"
        out.append(FeatureDrift(f, round(p, 4), float(ks), status))
    return out


def report(model, reference: pd.DataFrame, current: pd.DataFrame, reference_error_pct: float | None = None,
           error_tolerance: float = 1.5) -> dict:
    """Drift report for `current` cells against the model's training `reference` cells."""
    drift = feature_drift(reference, current, model.features)
    ref_pred, cur_pred = np.log10(model.predict(reference)), np.log10(model.predict(current))
    pred_psi = psi(ref_pred, cur_pred)
    pred_ks = float(stats.ks_2samp(ref_pred, cur_pred).pvalue)
    reasons = [f"{d.feature} shifted (PSI {d.psi:.2f}, KS p={d.ks_pvalue:.1e})"
               for d in drift if d.status == "major" and d.ks_pvalue < KS_ALPHA]
    if pred_psi > PSI_MAJOR and pred_ks < KS_ALPHA:
        reasons.append(f"predicted life distribution shifted (PSI {pred_psi:.2f}, KS p={pred_ks:.1e})")

    perf = None
    labelled = current.dropna(subset=["cycle_life"]) if "cycle_life" in current else current.iloc[0:0]
    if len(labelled):
        y = labelled.cycle_life.to_numpy(float)
        pred = model.predict(labelled)
        lo, hi = model.interval(labelled, 0.9)
        err = float(np.mean(np.abs(pred - y) / y) * 100)
        perf = {"cells": int(len(labelled)), "mean_abs_pct_error": err,
                "interval90_coverage": float(np.mean((y >= lo) & (y <= hi)))}
        if reference_error_pct is not None and err > error_tolerance * reference_error_pct:
            reasons.append(f"error {err:.1f}% vs {reference_error_pct:.1f}% at training")
        if perf["interval90_coverage"] < 0.75:
            reasons.append(f"90% intervals cover only {perf['interval90_coverage']:.0%}")

    return {
        "reference_cells": int(len(reference)),
        "current_cells": int(len(current)),
        "features": [asdict(d) for d in drift],
        "prediction_psi": round(pred_psi, 4),
        "prediction_ks_pvalue": pred_ks,
        "performance": perf,
        "retrain_recommended": bool(reasons),
        "reasons": reasons,
    }
