"""Replay lakehouse data as a live cycler feed.

A battery cycler emits one record per cell per cycle. This producer replays
stored cells in "wall-clock" order: cycle 1 of every cell, then cycle 2 of
every cell, and so on, the way a rack of channels running in parallel would.
Records carry the cycle summary, plus the discharge curve on cycles where one
is stored (every early cycle, then every 50th).

Topic `cycler.cycles`, key = cell_id.
"""

from __future__ import annotations

import pandas as pd
import pyarrow.parquet as pq

from ..config import Lake
from .broker import Broker

TOPIC_CYCLES = "cycler.cycles"


def replay(lake: Lake, broker: Broker, cells: list[str] | None = None, max_cycle: int | None = None) -> int:
    summary = pd.read_parquet(lake.cycle_summary)
    if cells is not None:
        summary = summary[summary.cell_id.isin(cells)]
    if max_cycle is not None:
        summary = summary[summary.cycle <= max_cycle]

    curves = {}
    for path in sorted(lake.curves.glob("*.parquet")):
        t = pq.read_table(path, columns=["cell_id", "cycle", "qdlin"]).to_pandas()
        if cells is not None:
            t = t[t.cell_id.isin(cells)]
        if max_cycle is not None:
            t = t[t.cycle <= max_cycle]
        for row in t.itertuples():
            curves[(row.cell_id, int(row.cycle))] = list(map(float, row.qdlin))

    sent = 0
    for row in summary.sort_values(["cycle", "cell_id"]).itertuples():
        msg = {"cell_id": row.cell_id, "batch": row.batch, "cycle": int(row.cycle), "qd": row.qd, "qc": row.qc,
               "ir": row.ir, "tavg": row.tavg, "tmin": row.tmin, "tmax": row.tmax, "chargetime": row.chargetime}
        curve = curves.get((row.cell_id, int(row.cycle)))
        if curve is not None:
            msg["qdlin"] = curve
        broker.send(TOPIC_CYCLES, row.cell_id, msg)
        sent += 1
    broker.flush()
    return sent
