"""Early-life features, following Severson et al. (2019).

The key idea: capacity barely moves in the first 100 cycles, but the *shape* of
the discharge curve does. Q_n(V) is the capacity discharged by the time voltage
reaches V on cycle n. The difference curve

    dQ(V) = Q_late(V) - Q_10(V)          (late = 100 in the paper)

summarizes how the curve changed, and its variance alone predicts cycle life
remarkably well. Features are computed only from cycles <= `late`, so a cell
can be scored as soon as it reaches that cycle.

Feature sets (the paper's three models):
  variance:  log10 var(dQ)
  discharge: log10 |min dQ|, log10 var dQ, log10 |skew dQ|, log10 |kurtosis dQ|,
             capacity at cycle 2, max capacity - capacity at cycle 2
  full:      log10 |min dQ|, log10 var dQ, slope and intercept of a linear fit to
             capacity over cycles 2..late, capacity at cycle 2, mean charge time
             of cycles 2-6, summed average temperature over cycles 2..late,
             minimum internal resistance, and resistance change from cycle 2 to late
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from ..config import EARLY_CYCLES, Lake
from ..lake import read_curves

VARIANCE = ["dq_var"]
DISCHARGE = ["dq_min", "dq_var", "dq_skew", "dq_kurt", "q2", "qmax_minus_q2"]
FULL = ["dq_min", "dq_var", "fade_slope", "fade_intercept", "q2", "chargetime_first5",
        "temp_integral", "ir_min", "ir_diff"]
FEATURE_SETS = {"variance": VARIANCE, "discharge": DISCHARGE, "full": FULL}
ALL_FEATURES = sorted(set(VARIANCE + DISCHARGE + FULL))

_EPS = 1e-12


def curve_features(q_early: np.ndarray, q_late: np.ndarray) -> dict[str, float]:
    dq = np.asarray(q_late, float) - np.asarray(q_early, float)
    return {
        "dq_min": float(np.log10(abs(dq.min()) + _EPS)),
        "dq_var": float(np.log10(np.var(dq) + _EPS)),
        "dq_skew": float(np.log10(abs(stats.skew(dq)) + _EPS)),
        "dq_kurt": float(np.log10(abs(stats.kurtosis(dq, fisher=False)) + _EPS)),
    }


def summary_features(cycles: pd.DataFrame, late: int = EARLY_CYCLES) -> dict[str, float]:
    """`cycles`: one cell's valid cycle_summary rows."""
    w = cycles[(cycles.cycle >= 2) & (cycles.cycle <= late)].sort_values("cycle")
    if len(w) < 0.8 * (late - 1):
        raise ValueError("not enough valid cycles in the early window")
    first = w.iloc[0]
    slope, intercept = np.polyfit(w.cycle, w.qd, 1)
    ir = w.ir[w.ir > 0]
    return {
        "q2": float(first.qd),
        "qmax_minus_q2": float(w.qd.max() - first.qd),
        "fade_slope": float(slope),
        "fade_intercept": float(intercept),
        "chargetime_first5": float(w.chargetime.iloc[:5].mean()),
        "temp_integral": float(w.tavg.sum()),
        "ir_min": float(ir.min()),
        "ir_diff": float(ir.iloc[-1] - ir.iloc[0]),
    }


def build(lake: Lake, late: int = EARLY_CYCLES, early: int = 10) -> pd.DataFrame:
    """One row per cell: features from cycles <= late, plus batch and cycle life."""
    cells = pd.read_parquet(lake.cells)
    summary = pd.read_parquet(lake.cycle_summary)
    curves = read_curves(lake, [early, late])
    rows = []
    for cell in cells.itertuples():
        if (cell.cell_id, early) not in curves or (cell.cell_id, late) not in curves:
            continue  # curve missing or quarantined: cell can't be scored at this window
        try:
            s = summary_features(summary[summary.cell_id == cell.cell_id], late)
        except ValueError:
            continue
        row = {"cell_id": cell.cell_id, "batch": cell.batch, "charge_policy": cell.charge_policy,
               "cycle_life": cell.cycle_life}
        row.update(curve_features(curves[(cell.cell_id, early)], curves[(cell.cell_id, late)]))
        row.update(s)
        rows.append(row)
    return pd.DataFrame(rows)


def paper_split(cell_ids: list[str]) -> dict[str, list[str]]:
    """The paper's split: batches 1+2 alternate between train and primary test,
    batch 3 is a secondary test set collected later.

    With 41 + 43 cells in batches 1 and 2 (as in the paper): train = odd positions
    (41 cells), primary test = even positions plus the last one (43), secondary = batch 3 (40).
    For other layouts the same rule is applied to whatever cells are present.
    """
    order = {"b1": 0, "b2": 1, "b3": 2}

    def key(c: str):
        b, n = c.split("c")
        return order.get(b, 9), int(n)

    ids = sorted(cell_ids, key=key)
    first_two = [c for c in ids if c.startswith(("b1", "b2"))]
    n12 = len(first_two)
    test_idx = set(range(0, n12, 2)) | {n12 - 1}
    return {
        "train": [c for i, c in enumerate(first_two) if i not in test_idx],
        "test_primary": [c for i, c in enumerate(first_two) if i in test_idx],
        "test_secondary": [c for c in ids if c.startswith("b3")],
    }
