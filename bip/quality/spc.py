"""Statistical process control for cell-level quality metrics.

Typical setup in cell manufacturing: control limits are established on a
qualified reference batch (phase I), then new batches are monitored against
those fixed limits (phase II). Limits are not recomputed from the batch being
judged, because a drifting batch would simply widen its own limits.

Chart: individuals (I-MR). Sigma is estimated from the average moving range
(MR-bar / 1.128), which is robust to slow drift within the reference data.

Run rules (Western Electric), applied in order of measurement:
  1  one point beyond 3 sigma
  2  two of three consecutive points beyond 2 sigma, same side
  3  four of five consecutive points beyond 1 sigma, same side
  4  eight consecutive points on the same side of the center line

Capability against spec limits: Cp = (USL - LSL) / 6 sigma,
Cpk = min(USL - mean, mean - LSL) / 3 sigma. Cpk >= 1.33 is a common target.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

D2 = 1.128  # d2 constant for moving ranges of size 2

# Default spec limits for the cells in this dataset (A123 APR18650M1A, 1.1 Ah nominal).
SPECS = {
    "q_cycle2_ah": (1.04, 1.11),
    "ir_early_first_ohm": (0.012, 0.021),
}


@dataclass
class ControlLimits:
    metric: str
    center: float
    sigma: float

    def zone(self, k: float) -> tuple[float, float]:
        return self.center - k * self.sigma, self.center + k * self.sigma

    @property
    def lcl(self) -> float:
        return self.zone(3)[0]

    @property
    def ucl(self) -> float:
        return self.zone(3)[1]


@dataclass
class Violation:
    index: int
    rule: int
    value: float


@dataclass
class ChartResult:
    limits: ControlLimits
    values: np.ndarray
    violations: list[Violation] = field(default_factory=list)

    @property
    def in_control(self) -> bool:
        return not self.violations


def fit_limits(reference: np.ndarray, metric: str = "") -> ControlLimits:
    x = np.asarray(reference, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 5:
        raise ValueError("need at least 5 reference points to set control limits")
    mr_bar = np.mean(np.abs(np.diff(x)))
    return ControlLimits(metric, float(np.mean(x)), float(mr_bar / D2))


def run_rules(values: np.ndarray, limits: ControlLimits) -> list[Violation]:
    x = np.asarray(values, dtype=float)
    z = (x - limits.center) / limits.sigma
    out: list[Violation] = []
    flagged: set[tuple[int, int]] = set()

    def add(i: int, rule: int):
        if (i, rule) not in flagged:
            flagged.add((i, rule))
            out.append(Violation(i, rule, float(x[i])))

    for i in range(len(z)):
        if abs(z[i]) > 3:
            add(i, 1)
        if i >= 2:
            w = z[i - 2:i + 1]
            for side in (1, -1):
                if (side * w > 2).sum() >= 2 and side * z[i] > 2:
                    add(i, 2)
        if i >= 4:
            w = z[i - 4:i + 1]
            for side in (1, -1):
                if (side * w > 1).sum() >= 4 and side * z[i] > 1:
                    add(i, 3)
        if i >= 7:
            w = z[i - 7:i + 1]
            if (w > 0).all() or (w < 0).all():
                add(i, 4)
    return sorted(out, key=lambda v: (v.index, v.rule))


def chart(values: np.ndarray, limits: ControlLimits) -> ChartResult:
    v = np.asarray(values, dtype=float)
    return ChartResult(limits, v, run_rules(v, limits))


def capability(values: np.ndarray, lsl: float, usl: float, sigma: float | None = None) -> dict:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    s = sigma if sigma is not None else float(np.std(x, ddof=1))
    mean = float(np.mean(x))
    return {
        "mean": mean,
        "sigma": s,
        "cp": (usl - lsl) / (6 * s),
        "cpk": min(usl - mean, mean - lsl) / (3 * s),
        "out_of_spec": int(((x < lsl) | (x > usl)).sum()),
    }


def monitor_batches(
    lifecycle: pd.DataFrame,
    metric: str,
    reference_batch: str = "b1",
    specs: dict[str, tuple[float, float]] | None = None,
) -> tuple[ControlLimits, pd.DataFrame, list[dict]]:
    """Set limits on the reference batch, then chart every batch against them.

    `lifecycle` is the cell_lifecycle mart (one row per cell). Cells are taken in
    channel order within a batch, the closest available proxy for build order.
    Returns the limits, a per-batch summary and a list of alerts.
    """
    specs = specs or SPECS
    df = lifecycle.copy()
    df["channel"] = df.cell_id.str.extract(r"c(\d+)$").astype(int)
    df = df.sort_values(["batch", "channel"])

    limits = fit_limits(df.loc[df.batch == reference_batch, metric].to_numpy(), metric)
    rows, alerts = [], []
    for batch, part in df.groupby("batch", sort=True):
        res = chart(part[metric].to_numpy(), limits)
        lsl, usl = specs.get(metric, (np.nan, np.nan))
        cap = capability(part[metric].to_numpy(), lsl, usl) if metric in specs else {}
        rows.append({
            "batch": batch, "metric": metric, "cells": len(part),
            "mean": float(part[metric].mean()), "violations": len(res.violations),
            "in_control": res.in_control, "cpk": cap.get("cpk"), "out_of_spec": cap.get("out_of_spec"),
        })
        cell_ids = part.cell_id.to_numpy()
        for v in res.violations:
            alerts.append({"batch": batch, "cell_id": cell_ids[v.index], "metric": metric,
                           "rule": v.rule, "value": v.value})
    return limits, pd.DataFrame(rows), alerts
