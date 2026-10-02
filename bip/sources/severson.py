"""Load the real Severson et al. (2019) battery cycling dataset.

Download the three batch files from https://data.matr.io/1/ into data/raw/:

    2017-05-12_batchdata_updated_struct_errorcorrect.mat   (batch 1)
    2017-06-30_batchdata_updated_struct_errorcorrect.mat   (batch 2)
    2018-04-12_batchdata_updated_struct_errorcorrect.mat   (batch 3)

They are MATLAB v7.3 files (HDF5), read here with h5py. The cleaning below
reproduces the authors' own loading code
(github.com/rdbraatz/data-driven-prediction-of-battery-cycle-life-before-capacity-degradation):

- batch 1: drop five cells that never reached 80% capacity (b1c8, b1c10, b1c12, b1c13, b1c22);
- five batch-1 cells continued testing in batch 2; their batch-2 records
  (b2c7, b2c8, b2c9, b2c15, b2c16) are appended to b1c0-b1c4 and their cycle lives extended;
- batch 3: drop six noisy channels (b3c2, b3c23, b3c32, b3c37, b3c42, b3c43).

That leaves the paper's 124 cells (41 + 43 + 40).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..config import keep_curve
from . import CellRecord

BATCH_FILES = {
    "b1": "2017-05-12_batchdata_updated_struct_errorcorrect.mat",
    "b2": "2017-06-30_batchdata_updated_struct_errorcorrect.mat",
    "b3": "2018-04-12_batchdata_updated_struct_errorcorrect.mat",
}
DROP = {
    "b1": {"b1c8", "b1c10", "b1c12", "b1c13", "b1c22"},
    "b3": {"b3c2", "b3c23", "b3c32", "b3c37", "b3c42", "b3c43"},
}
CARRY_OVER = [  # (batch-2 cell, batch-1 cell, extra cycles)
    ("b2c7", "b1c0", 662),
    ("b2c8", "b1c1", 981),
    ("b2c9", "b1c2", 1060),
    ("b2c15", "b1c3", 208),
    ("b2c16", "b1c4", 482),
]
_SUMMARY_KEYS = {"IR": "ir", "QCharge": "qc", "QDischarge": "qd", "Tavg": "tavg",
                 "Tmin": "tmin", "Tmax": "tmax", "chargetime": "chargetime", "cycle": "cycle"}


def _read_batch(path: Path, batch: str) -> dict[str, CellRecord]:
    import h5py  # optional dependency: pip install ".[real-data]"

    cells: dict[str, CellRecord] = {}
    with h5py.File(path, "r") as f:
        b = f["batch"]
        for i in range(b["summary"].shape[0]):
            cell_id = f"{batch}c{i}"
            if cell_id in DROP.get(batch, set()):
                continue
            cycle_life = int(np.asarray(f[b["cycle_life"][i, 0]][()]).ravel()[0])
            policy = np.asarray(f[b["policy_readable"][i, 0]][()]).tobytes()[::2].decode()
            s = f[b["summary"][i, 0]]
            summary = {ours: np.hstack(s[theirs][0, :].tolist()).astype(float)
                       for theirs, ours in _SUMMARY_KEYS.items()}
            cyc = f[b["cycles"][i, 0]]
            curves = {}
            for j in range(cyc["Qdlin"].shape[0]):
                cycle = j + 1  # the authors' cycles['9'] is cycle 10
                if keep_curve(cycle):
                    qd = np.hstack(f[cyc["Qdlin"][j, 0]][()]).astype(np.float32)
                    td = np.hstack(f[cyc["Tdlin"][j, 0]][()]).astype(np.float32)
                    curves[cycle] = (qd, td)
            cells[cell_id] = CellRecord(cell_id, batch, policy, cycle_life, summary, curves)
    return cells


def _carry_over(b1: dict[str, CellRecord], b2: dict[str, CellRecord]) -> None:
    for b2_id, b1_id, extra in CARRY_OVER:
        if b2_id not in b2 or b1_id not in b1:
            continue
        first, cont = b1[b1_id], b2.pop(b2_id)
        n_cycles = len(first.summary["cycle"])  # the authors offset both summary and curves by this
        for k, values in cont.summary.items():
            add = values + n_cycles if k == "cycle" else values
            first.summary[k] = np.hstack([first.summary[k], add])
        for cycle, curve in cont.curves.items():
            if keep_curve(cycle + n_cycles):
                first.curves[cycle + n_cycles] = curve
        first.cycle_life += extra


def load(raw_dir: str | Path) -> list[CellRecord]:
    raw_dir = Path(raw_dir)
    missing = [name for name in BATCH_FILES.values() if not (raw_dir / name).exists()]
    if missing:
        raise FileNotFoundError(
            "missing batch files in " + str(raw_dir) + ": " + ", ".join(missing)
            + "\nDownload them from https://data.matr.io/1/"
        )
    batches = {batch: _read_batch(raw_dir / name, batch) for batch, name in BATCH_FILES.items()}
    _carry_over(batches["b1"], batches["b2"])
    return [cell for batch in ("b1", "b2", "b3") for cell in batches[batch].values()]
