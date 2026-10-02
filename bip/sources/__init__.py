"""Data sources. Every source yields the same in-memory `CellRecord`s."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class CellRecord:
    cell_id: str  # e.g. "b1c0": batch 1, channel 0
    batch: str  # "b1", "b2", "b3"
    charge_policy: str
    cycle_life: int  # cycles until capacity fell to 0.88 Ah
    # per-cycle summary, aligned arrays
    summary: dict[str, np.ndarray] = field(default_factory=dict)
    # cycle number -> (Qdlin, Tdlin) on config.VOLTAGE_GRID
    curves: dict[int, tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)
    source: str = "severson2019"


SUMMARY_FIELDS = ("cycle", "qd", "qc", "ir", "tavg", "tmin", "tmax", "chargetime")
