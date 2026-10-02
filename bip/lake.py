"""Write cell records into the Parquet lakehouse, validating as we go.

Tables (under data/lake/):

    cells.parquet                 one row per cell: batch, policy, cycle life, data flags
    cycle_summary/<batch>.parquet one row per cell x cycle: capacity, resistance, temperature, charge time
    curves/<batch>.parquet        discharge curves Q(V) and T(V) on the 1000-point voltage grid
    quarantine.parquet            every rejected cycle or curve, with the reason

Cyclers produce occasional glitches (a zero capacity reading, a resistance
spike). One bad cycle would wreck a fitted fade slope, so invalid cycles are
moved to quarantine, not silently dropped and not passed downstream.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .config import EARLY_CYCLES, NOMINAL_CAPACITY_AH, VOLTAGE_GRID, Lake
from .sources import SUMMARY_FIELDS, CellRecord

# per-cycle validity limits for LFP 18650 cells
LIMITS = {
    "qd": (0.0, 1.3 * NOMINAL_CAPACITY_AH),
    "qc": (0.0, 1.3 * NOMINAL_CAPACITY_AH),
    "ir": (0.0, 0.05),          # ohms; healthy cells are ~0.015-0.02
    "tavg": (10.0, 70.0),       # deg C
    "tmin": (10.0, 70.0),
    "tmax": (10.0, 70.0),
    "chargetime": (0.0, 120.0), # minutes
}


@dataclass
class WriteStats:
    cells: int = 0
    cycles: int = 0
    curves: int = 0
    quarantined_cycles: int = 0
    quarantined_curves: int = 0
    cells_without_early_window: int = 0


def cycle_issues(df: pd.DataFrame) -> pd.Series:
    """Reason string per row ('' when the row is valid)."""
    reasons = pd.Series("", index=df.index)
    for col, (lo, hi) in LIMITS.items():
        bad = ~df[col].between(lo, hi, inclusive="neither") | df[col].isna()
        reasons[bad] += f"{col}_out_of_range;"
    dup = df.duplicated(["cell_id", "cycle"], keep="first")
    reasons[dup] += "duplicate_cycle;"
    return reasons.str.rstrip(";")


def curve_issue(q: np.ndarray, t: np.ndarray) -> str:
    if q.shape != VOLTAGE_GRID.shape or t.shape != VOLTAGE_GRID.shape:
        return "wrong_length"
    if not (np.isfinite(q).all() and np.isfinite(t).all()):
        return "non_finite"
    # discharged capacity can only grow as voltage falls; allow small sensor noise,
    # but measure the drop from the running maximum so slow declines are caught too
    if np.max(np.maximum.accumulate(q) - q) > 0.01:
        return "non_monotonic"
    if q[-1] <= 0.1 or q[-1] > 1.3 * NOMINAL_CAPACITY_AH:
        return "implausible_capacity"
    return ""


def write(cells: list[CellRecord], lake: Lake) -> WriteStats:
    stats = WriteStats()
    lake.cycle_summary.mkdir(parents=True, exist_ok=True)
    lake.curves.mkdir(parents=True, exist_ok=True)
    for old in list(lake.cycle_summary.glob("*.parquet")) + list(lake.curves.glob("*.parquet")):
        old.unlink()  # the lake is rebuilt from the source each run

    cell_rows, quarantine = [], []
    by_batch_summary: dict[str, list[pd.DataFrame]] = {}
    by_batch_curves: dict[str, list[dict]] = {}

    for cell in cells:
        df = pd.DataFrame({f: np.asarray(cell.summary[f], dtype=float) for f in SUMMARY_FIELDS})
        df.insert(0, "cell_id", cell.cell_id)
        df.insert(1, "batch", cell.batch)
        df["cycle"] = df["cycle"].astype(int)
        reasons = cycle_issues(df)
        bad = reasons != ""
        for _, row in df[bad].iterrows():
            quarantine.append({"cell_id": cell.cell_id, "kind": "cycle", "cycle": int(row.cycle),
                               "reason": reasons[_]})
        good = df[~bad]
        by_batch_summary.setdefault(cell.batch, []).append(good)

        valid_early = int((good.cycle <= EARLY_CYCLES).sum())
        has_early = valid_early >= int(0.9 * EARLY_CYCLES)
        stats.cells_without_early_window += not has_early

        for cycle, (q, t) in sorted(cell.curves.items()):
            issue = curve_issue(np.asarray(q), np.asarray(t))
            if issue:
                quarantine.append({"cell_id": cell.cell_id, "kind": "curve", "cycle": cycle, "reason": issue})
                stats.quarantined_curves += 1
                continue
            by_batch_curves.setdefault(cell.batch, []).append(
                {"cell_id": cell.cell_id, "cycle": cycle,
                 "qdlin": np.asarray(q, np.float32), "tdlin": np.asarray(t, np.float32)})

        cell_rows.append({
            "cell_id": cell.cell_id,
            "batch": cell.batch,
            "charge_policy": cell.charge_policy,
            "cycle_life": int(cell.cycle_life),
            "cycles_recorded": int(len(df)),
            "valid_cycles": int(len(good)),
            "has_early_window": has_early,
            "source": cell.source,
        })
        stats.cells += 1
        stats.cycles += len(good)
        stats.quarantined_cycles += int(bad.sum())

    pd.DataFrame(cell_rows).to_parquet(lake.cells, index=False)
    for batch, frames in by_batch_summary.items():
        pd.concat(frames, ignore_index=True).to_parquet(lake.cycle_summary / f"{batch}.parquet", index=False)
    curve_type = pa.list_(pa.float32())
    for batch, rows in by_batch_curves.items():
        table = pa.table({
            "cell_id": pa.array([r["cell_id"] for r in rows], pa.string()),
            "cycle": pa.array([r["cycle"] for r in rows], pa.int32()),
            "qdlin": pa.array([r["qdlin"] for r in rows], curve_type),
            "tdlin": pa.array([r["tdlin"] for r in rows], curve_type),
        })
        pq.write_table(table, lake.curves / f"{batch}.parquet")
        stats.curves += len(rows)

    pd.DataFrame(quarantine, columns=["cell_id", "kind", "cycle", "reason"]).to_parquet(lake.quarantine, index=False)
    (lake.lake / "write_stats.json").write_text(pd.Series(asdict(stats)).to_json(indent=2))
    return stats


def read_curves(lake: Lake, cycles: list[int] | None = None) -> dict[tuple[str, int], np.ndarray]:
    """{(cell_id, cycle): Qdlin} for the requested cycles (all stored cycles if None)."""
    out = {}
    for path in sorted(lake.curves.glob("*.parquet")):
        table = pq.read_table(path, columns=["cell_id", "cycle", "qdlin"])
        cell_ids = table.column("cell_id").to_pylist()
        cyc = table.column("cycle").to_pylist()
        q = table.column("qdlin").to_pylist()
        for c, n, curve in zip(cell_ids, cyc, q):
            if cycles is None or n in cycles:
                out[(c, n)] = np.asarray(curve, dtype=np.float64)
    return out
