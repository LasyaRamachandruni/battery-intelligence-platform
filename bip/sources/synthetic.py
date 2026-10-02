"""Synthetic LFP cells shaped like the Severson dataset, for development, tests and CI.

NOT REAL DATA. It exists so the whole platform runs without the 3 GB download,
and so tests have known ground truth. Real results come from sources.severson.

How a cell is generated:

- A charging policy like "5.4C(40%)-3.6C" is drawn per batch; faster charging
  raises the cell's hidden degradation rate k (plus cell-to-cell variation).
- Cycle life L falls with k (log-linear, with noise).
- The discharge curve Q(V) has the LFP shape: almost all capacity comes out on
  a flat plateau near 3.3 V. As a cell ages, the plateau shifts and broadens in
  proportion to k. So between cycle 10 and 100, a fast-degrading cell's curve
  changes more, even though its total capacity has barely moved. That is the
  effect behind the paper's main feature, var(Q100(V) - Q10(V)).
- Capacity fades slowly, then accelerates past a "knee", reaching 0.88 Ah at L.
- Charge time follows the policy's C-rates; temperature rises with them;
  internal resistance grows slowly with use.

A few measurement glitches (zero capacity readings, resistance spikes) are added
on purpose, as in real cycler data, so validation has something to catch.
"""

from __future__ import annotations

import numpy as np

from ..config import END_OF_LIFE_AH, VOLTAGE_GRID, keep_curve
from . import CellRecord

BATCH_SIZES = {"b1": 41, "b2": 43, "b3": 40}


def _policy(rng: np.random.Generator, batch: str) -> tuple[str, float]:
    """Return a policy string and its average C-rate."""
    if batch == "b3":  # the third batch used gentler policies, so cells lived longer
        c1, c2 = rng.uniform(3.0, 5.5), rng.uniform(3.0, 4.8)
    else:
        c1, c2 = rng.uniform(3.6, 8.0), rng.uniform(3.6, 6.0)
    switch = int(rng.choice([10, 20, 30, 40, 50, 60, 70, 80]))
    avg = (c1 * switch + c2 * (80 - switch)) / 80
    return f"{c1:.1f}C({switch}%)-{c2:.1f}C", avg


def _shape(v: np.ndarray, center: float, width: float) -> np.ndarray:
    """Normalized discharged-capacity curve (0 at 3.5 V, 1 at 2.0 V) for an LFP cell."""
    def s(c, w):
        return 1.0 / (1.0 + np.exp(-(c - v) / w))
    raw = 0.55 * s(center, 0.012 * width) + 0.38 * s(center - 0.08, 0.03 * width) + 0.07 * s(2.9, 0.25)
    lo = 0.55 * s(center, 0.012 * width)[0] + 0.38 * s(center - 0.08, 0.03 * width)[0] + 0.07 * s(2.9, 0.25)[0]
    hi = 0.55 * s(center, 0.012 * width)[-1] + 0.38 * s(center - 0.08, 0.03 * width)[-1] + 0.07 * s(2.9, 0.25)[-1]
    return (raw - lo) / (hi - lo)


def _make_cell(rng: np.random.Generator, cell_id: str, batch: str) -> CellRecord:
    policy, c_avg = _policy(rng, batch)
    k = np.exp(0.35 * (c_avg - 5.0) + rng.normal(0, 0.35))  # hidden degradation rate
    life = int(np.clip(np.exp(6.75 - 1.05 * np.log(k) + rng.normal(0, 0.12)), 150, 2400))

    q0 = rng.normal(1.075, 0.008)
    n_obs = life + int(rng.integers(1, 15))  # testing stops shortly after end of life
    n = np.arange(1, n_obs + 1)

    # capacity fade: gentle linear phase, then a knee, hitting END_OF_LIFE_AH at `life`
    slope = rng.uniform(0.15, 0.35) * (q0 - END_OF_LIFE_AH) / life
    knee_drop = q0 - END_OF_LIFE_AH - slope * life
    p = rng.uniform(5.0, 9.0)
    qmax = q0 - slope * n - knee_drop * (n / life) ** p
    qmax += rng.normal(0, 0.0015, n.size)

    # charge time from the policy (minutes, to 80% then CV); hotter at higher C
    chargetime = 60 * 0.8 / c_avg + 2.0 + rng.normal(0, 0.15, n.size) + 0.002 * n * k
    tmax = 30 + 2.2 * c_avg + rng.normal(0, 0.4, n.size) + 0.0015 * n
    tavg = tmax - 4.0 + rng.normal(0, 0.3, n.size)
    tmin = tavg - 3.0 + rng.normal(0, 0.3, n.size)
    ir = 0.0165 + 0.0006 * rng.standard_normal() + 2.5e-7 * n * k + rng.normal(0, 1e-4, n.size)
    qc = qmax + rng.normal(0.002, 0.001, n.size)

    summary = {
        "cycle": n.astype(float),
        "qd": qmax.copy(),  # glitches below must not leak into the curves
        "qc": qc,
        "ir": ir,
        "tavg": tavg,
        "tmin": tmin,
        "tmax": tmax,
        "chargetime": chargetime,
    }

    # measurement glitches, as real cyclers produce
    for _ in range(rng.poisson(1.5)):
        i = int(rng.integers(5, n.size))
        if rng.random() < 0.5:
            summary["qd"][i] = 0.0
        else:
            summary["ir"][i] = rng.uniform(0.05, 0.2)

    curves = {}
    base_width = rng.normal(1.0, 0.03)
    for c in n:
        if not keep_curve(int(c)):
            continue
        age = c / 100.0
        center = 3.30 - 0.004 * k * age
        width = base_width * (1 + 0.05 * k * age)
        q = qmax[c - 1] * _shape(VOLTAGE_GRID, center, width) + rng.normal(0, 3e-4, VOLTAGE_GRID.size)
        temp = tavg[c - 1] + 3.0 * _shape(VOLTAGE_GRID, center, width)
        curves[int(c)] = (q.astype(np.float32), temp.astype(np.float32))

    return CellRecord(cell_id, batch, policy, life, summary, curves, source="synthetic")


def generate(seed: int = 0, batch_sizes: dict[str, int] | None = None) -> list[CellRecord]:
    rng = np.random.default_rng(seed)
    cells = []
    for batch, size in (batch_sizes or BATCH_SIZES).items():
        for i in range(size):
            cells.append(_make_cell(rng, f"{batch}c{i}", batch))
    return cells
