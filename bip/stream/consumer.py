"""Streaming consumer: early quality checks and a life prediction per cell.

Per cell, the consumer keeps only the state it needs (cycle summaries up to
cycle 100 and the curves at cycles 10 and 100), and:

- at cycle 2:   checks discharge capacity against the SPC limits and the spec,
                emitting an alert to `quality.alerts` if it is out of control;
- at cycle 10:  scores the discharge curve with the anomaly model (if given);
- at cycle 100: computes the features with the same function used in training
                and emits a cycle-life prediction with a 90% interval to
                `predictions.cycle_life`; the cell's state is then dropped.

Duplicate (cell, cycle) messages, which at-least-once delivery can produce, are
ignored. Cycles arriving out of order are fine: state is keyed by cycle number.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..config import EARLY_CYCLES
from ..ml.features import cell_features
from .broker import Broker
from .cycler import TOPIC_CYCLES

TOPIC_PREDICTIONS = "predictions.cycle_life"
TOPIC_ALERTS = "quality.alerts"


@dataclass
class CellState:
    cycles: dict[int, dict] = field(default_factory=dict)
    curves: dict[int, np.ndarray] = field(default_factory=dict)
    predicted: bool = False


class StreamProcessor:
    def __init__(self, broker: Broker, model, capacity_limits=None, capacity_spec=(1.04, 1.11),
                 anomaly_model=None, early: int = 10, late: int = EARLY_CYCLES):
        self.broker = broker
        self.model = model
        self.limits = capacity_limits  # spc.ControlLimits for cycle-2 capacity, optional
        self.spec = capacity_spec
        self.anomaly = anomaly_model
        self.early, self.late = early, late
        self.state: dict[str, CellState] = {}
        self.duplicates = 0
        self.done: set[str] = set()

    def handle(self, msg: dict) -> None:
        cell, cycle = msg["cell_id"], int(msg["cycle"])
        if cycle > self.late:
            return
        if cell in self.done:  # already predicted: anything up to the late cycle is a redelivery
            self.duplicates += 1
            return
        st = self.state.setdefault(cell, CellState())
        if cycle in st.cycles:
            self.duplicates += 1
            return
        st.cycles[cycle] = {k: msg[k] for k in ("cycle", "qd", "qc", "ir", "tavg", "tmin", "tmax", "chargetime")}
        if "qdlin" in msg and cycle in (self.early, self.late):
            st.curves[cycle] = np.asarray(msg["qdlin"], float)

        if cycle == 2:
            self._check_capacity(cell, msg["qd"])
        if cycle == self.early and self.anomaly is not None and self.early in st.curves:
            if bool(self.anomaly.flag(st.curves[self.early][None, :])[0]):
                self._alert(cell, "curve_anomaly", cycle, None)
        if self._ready(st):
            self._predict(cell, st)

    def _ready(self, st: CellState) -> bool:
        return (not st.predicted and self.early in st.curves and self.late in st.curves
                and len(st.cycles) >= 0.8 * (self.late - 1))

    def _check_capacity(self, cell: str, qd: float) -> None:
        lo, hi = self.spec
        if not lo <= qd <= hi:
            self._alert(cell, "capacity_out_of_spec", 2, qd)
        elif self.limits is not None and not self.limits.lcl <= qd <= self.limits.ucl:
            self._alert(cell, "capacity_out_of_control", 2, qd)

    def _alert(self, cell: str, kind: str, cycle: int, value) -> None:
        self.broker.send(TOPIC_ALERTS, cell, {"cell_id": cell, "alert": kind, "cycle": cycle, "value": value})

    def _predict(self, cell: str, st: CellState) -> None:
        cycles = pd.DataFrame(st.cycles.values())
        try:
            f = cell_features(st.curves[self.early], st.curves[self.late], cycles, self.late)
        except ValueError:
            return
        df = pd.DataFrame([f])
        lo, hi = self.model.interval(df, 0.9)
        self.broker.send(TOPIC_PREDICTIONS, cell, {
            "cell_id": cell, "at_cycle": self.late, "predicted_cycle_life": float(self.model.predict(df)[0]),
            "lower90": float(lo[0]), "upper90": float(hi[0]), "model": self.model.name})
        st.predicted = True
        self.done.add(cell)
        del self.state[cell]  # free memory: nothing else is needed for this cell

    def run(self, max_messages: int | None = None) -> int:
        n = 0
        for msg in self.broker.consume(TOPIC_CYCLES, max_messages):
            self.handle(msg)
            n += 1
        self.broker.flush()
        return n
