"""Shared constants and lakehouse paths.

The cells are A123 APR18650M1A lithium iron phosphate (LFP) cells, nominal
capacity 1.1 Ah, cycled under different fast-charging policies until they lost
20% of their capacity (Severson et al., Nature Energy 2019).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

NOMINAL_CAPACITY_AH = 1.1
END_OF_LIFE_AH = 0.88  # 80% of nominal: the paper's definition of end of life

# Discharge curves are stored as capacity Q(V) on a fixed voltage grid, which is
# what the paper's "Qdlin" field contains: 1000 points from 3.5 V down to 2.0 V.
VOLTAGE_GRID = np.linspace(3.5, 2.0, 1000)

# Early-prediction window: features may only use cycles up to this one.
EARLY_CYCLES = 100

# Curves are kept for the early window plus a sparse sample afterwards (for plots
# and drift monitoring). Keeping every cycle's curve would be ~100x larger.
CURVE_EVERY_N_AFTER_EARLY = 50


@dataclass(frozen=True)
class Lake:
    """Locations of the lakehouse tables under one data directory."""

    root: Path

    @property
    def raw(self) -> Path:
        return self.root / "raw"

    @property
    def lake(self) -> Path:
        return self.root / "lake"

    @property
    def cells(self) -> Path:
        return self.lake / "cells.parquet"

    @property
    def cycle_summary(self) -> Path:
        return self.lake / "cycle_summary"

    @property
    def curves(self) -> Path:
        return self.lake / "curves"

    @property
    def quarantine(self) -> Path:
        return self.lake / "quarantine.parquet"

    @property
    def warehouse(self) -> Path:
        return self.root / "warehouse.duckdb"

    @property
    def features(self) -> Path:
        return self.lake / "features.parquet"


def keep_curve(cycle: int) -> bool:
    return cycle <= EARLY_CYCLES or cycle % CURVE_EVERY_N_AFTER_EARLY == 0
