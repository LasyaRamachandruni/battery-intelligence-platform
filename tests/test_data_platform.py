"""Phase 1: sources, lakehouse validation, dbt models."""

import numpy as np
import pandas as pd
import pytest

from bip.config import END_OF_LIFE_AH, VOLTAGE_GRID, Lake
from bip.lake import curve_issue, cycle_issues, read_curves, write
from bip.sources import CellRecord
from bip.sources.synthetic import generate


# --- synthetic generator --------------------------------------------------------

def test_synthetic_matches_paper_layout(synthetic_cells):
    by_batch = pd.Series([c.batch for c in synthetic_cells]).value_counts().to_dict()
    assert by_batch == {"b1": 41, "b2": 43, "b3": 40}  # the paper's 124 cells


def test_synthetic_capacity_crosses_end_of_life_at_cycle_life(synthetic_cells):
    for cell in synthetic_cells[:20]:
        qd = cell.summary["qd"]
        clean = qd[qd > 0.5]  # skip injected glitches
        assert clean[0] > END_OF_LIFE_AH
        crossing = int(np.argmax(qd[(qd > 0.5)] < END_OF_LIFE_AH)) + 1
        assert abs(crossing - cell.cycle_life) <= max(50, 0.1 * cell.cycle_life)


def test_synthetic_early_curve_change_predicts_life(synthetic_cells):
    # the mechanism behind the paper's headline feature should be present
    x = [np.log10(np.var(c.curves[100][0] - c.curves[10][0])) for c in synthetic_cells]
    y = [np.log10(c.cycle_life) for c in synthetic_cells]
    assert np.corrcoef(x, y)[0, 1] < -0.8


def test_synthetic_is_deterministic():
    a, b = generate(seed=5, batch_sizes={"b1": 3}), generate(seed=5, batch_sizes={"b1": 3})
    assert [c.cycle_life for c in a] == [c.cycle_life for c in b]
    assert np.array_equal(a[0].curves[50][0], b[0].curves[50][0])


# --- validation -----------------------------------------------------------------

def test_cycle_issues_flags_glitches_and_duplicates():
    df = pd.DataFrame({
        "cell_id": ["a"] * 4, "cycle": [1, 2, 2, 3],
        "qd": [1.07, 0.0, 1.07, 1.07], "qc": [1.07] * 4, "ir": [0.016, 0.016, 0.016, 0.3],
        "tavg": [30.0] * 4, "tmin": [28.0] * 4, "tmax": [33.0] * 4, "chargetime": [10.0] * 4,
    })
    reasons = cycle_issues(df).tolist()
    assert reasons[0] == ""
    assert "qd_out_of_range" in reasons[1]
    assert "duplicate_cycle" in reasons[2]
    assert "ir_out_of_range" in reasons[3]


def test_curve_issue_rules():
    good = np.linspace(0, 1.07, VOLTAGE_GRID.size)
    assert curve_issue(good, good) == ""
    assert curve_issue(good[:10], good[:10]) == "wrong_length"
    assert curve_issue(good[::-1].copy(), good) == "non_monotonic"
    nan = good.copy()
    nan[3] = np.nan
    assert curve_issue(nan, good) == "non_finite"


def test_lake_write_quarantines_and_keeps_early_curves(lake, synthetic_cells):
    q = pd.read_parquet(lake.quarantine)
    assert (q.kind == "cycle").sum() > 0  # the generator injects glitches
    summary = pd.read_parquet(lake.cycle_summary)
    assert summary.qd.between(0, 1.43, inclusive="neither").all()
    assert not summary.duplicated(["cell_id", "cycle"]).any()
    curves = read_curves(lake, [10, 100])
    assert len(curves) == 2 * len(synthetic_cells)
    assert next(iter(curves.values())).shape == VOLTAGE_GRID.shape


def test_lake_write_is_idempotent(tmp_path):
    cells = generate(seed=1, batch_sizes={"b1": 4})
    lake = Lake(tmp_path)
    first, second = write(cells, lake), write(cells, lake)
    assert first == second
    assert len(pd.read_parquet(lake.cells)) == 4


def test_cells_without_early_window_are_flagged(tmp_path):
    short = CellRecord("b9c0", "b9", "4C-4C", 60,
                       {f: np.full(60, v) for f, v in [("cycle", 0), ("qd", 1.0), ("qc", 1.0), ("ir", 0.016),
                                                        ("tavg", 30), ("tmin", 28), ("tmax", 33), ("chargetime", 10)]})
    short.summary["cycle"] = np.arange(1, 61, dtype=float)
    stats = write([short], Lake(tmp_path))
    assert stats.cells_without_early_window == 1
    assert not pd.read_parquet(Lake(tmp_path).cells).has_early_window.iloc[0]


# --- real-data loader on files with the real .mat structure ----------------------------

def _fake_mat(path, batch_cells: int, cycles: int = 3, life: int = 500):
    """Write a small HDF5 file laid out like the paper's MATLAB v7.3 batch files."""
    h5py = pytest.importorskip("h5py")
    rng = np.random.default_rng(0)
    with h5py.File(path, "w") as f:
        refs = f.create_group("#refs#")
        counter = iter(range(10**6))

        def ds(data):
            d = refs.create_dataset(str(next(counter)), data=np.asarray(data))
            return d.ref

        n = batch_cells
        b = f.create_group("batch")
        ref_t = h5py.ref_dtype
        cl, pol, summ, cyc = (b.create_dataset(k, (n, 1), dtype=ref_t) for k in ("cycle_life", "policy_readable", "summary", "cycles"))
        for i in range(n):
            cl[i, 0] = ds([[life + i]])
            pol[i, 0] = ds(np.frombuffer("5.4C(40%)-3.6C".encode("utf-16-le"), dtype=np.uint16).reshape(-1, 1))
            g = refs.create_group(f"s{i}")
            for key, val in [("IR", 0.016), ("QCharge", 1.07), ("QDischarge", 1.07), ("Tavg", 31.0),
                             ("Tmin", 29.0), ("Tmax", 34.0), ("chargetime", 10.0)]:
                g.create_dataset(key, data=np.full((1, cycles), val))
            g.create_dataset("cycle", data=np.arange(1, cycles + 1, dtype=float).reshape(1, -1))
            summ[i, 0] = g.ref
            cg = refs.create_group(f"c{i}")
            for key in ("I", "Qc", "Qd", "T", "V", "discharge_dQdV", "t"):
                r = cg.create_dataset(key, (cycles, 1), dtype=ref_t)
                for j in range(cycles):
                    r[j, 0] = ds(rng.random((5, 1)))
            for key in ("Qdlin", "Tdlin"):
                r = cg.create_dataset(key, (cycles, 1), dtype=ref_t)
                for j in range(cycles):
                    r[j, 0] = ds(np.linspace(0, 1.07, 1000).reshape(-1, 1))
            cyc[i, 0] = cg.ref


def test_severson_loader_applies_authors_cleaning(tmp_path):
    from bip.sources.severson import BATCH_FILES, load

    _fake_mat(tmp_path / BATCH_FILES["b1"], 23)  # needs b1c22
    _fake_mat(tmp_path / BATCH_FILES["b2"], 17)  # needs b2c16
    _fake_mat(tmp_path / BATCH_FILES["b3"], 44)  # needs b3c43
    cells = {c.cell_id: c for c in load(tmp_path)}

    assert len(cells) == (23 - 5) + (17 - 5) + (44 - 6)
    assert {"b1c8", "b1c22", "b2c7", "b2c16", "b3c2", "b3c43"}.isdisjoint(cells)
    # b1c0 continues with b2c7's cycles: 3 + 3 cycles, life extended by 662
    b1c0 = cells["b1c0"]
    assert list(b1c0.summary["cycle"]) == [1, 2, 3, 4, 5, 6]
    assert b1c0.cycle_life == 500 + 662
    assert set(b1c0.curves) == {1, 2, 3, 4, 5, 6}
    assert b1c0.charge_policy == "5.4C(40%)-3.6C"
    assert b1c0.curves[1][0].shape == (1000,)


def test_severson_loader_reports_missing_files(tmp_path):
    from bip.sources.severson import load

    with pytest.raises(FileNotFoundError, match="data.matr.io"):
        load(tmp_path)


# --- dbt -----------------------------------------------------------------------

def test_dbt_build_passes_and_builds_marts(lake):
    import duckdb

    from bip.cli import main

    assert main(["--data-dir", str(lake.root), "transform"]) == 0
    con = duckdb.connect(str(lake.warehouse))
    lifecycle = con.sql("select * from cell_lifecycle").df()
    assert len(lifecycle) == 124 and lifecycle.cell_id.is_unique
    # faster early fade should go with shorter life
    assert lifecycle.fade_slope_ah_per_cycle.corr(np.log(lifecycle.cycle_life)) > 0.3
    batches = con.sql("select batch, cycle_life_p50 from batch_quality order by batch").df()
    assert list(batches.batch) == ["b1", "b2", "b3"]
    con.close()
