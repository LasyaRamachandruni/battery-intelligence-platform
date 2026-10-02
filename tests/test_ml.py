"""Phase 3: features, models, intervals, selection and the train command."""

import joblib
import numpy as np
import pandas as pd
import pytest

from bip.config import Lake
from bip.lake import write
from bip.ml import features as F
from bip.ml import models as M
from bip.ml.train import split_frames, train_all
from bip.sources.synthetic import generate


@pytest.fixture(scope="module")
def feats(lake):
    return F.build(lake)


def test_curve_features_on_known_curves():
    q10 = np.linspace(0, 1.0, 1000)
    f = F.curve_features(q10, q10 - 0.01)  # uniform 10 mAh loss: no variance
    assert f["dq_min"] == pytest.approx(np.log10(0.01))
    assert f["dq_var"] < -10
    g = F.curve_features(q10, q10 - np.linspace(0, 0.02, 1000))
    assert g["dq_var"] == pytest.approx(np.log10(np.var(np.linspace(0, 0.02, 1000))))


def test_summary_features_fit_the_fade_line():
    cyc = np.arange(1, 121)
    df = pd.DataFrame({"cycle": cyc, "qd": 1.08 - 1e-4 * cyc, "ir": 0.016 + 1e-6 * cyc,
                       "chargetime": 10.0, "tavg": 30.0})
    s = F.summary_features(df)
    assert s["fade_slope"] == pytest.approx(-1e-4)
    assert s["q2"] == pytest.approx(1.08 - 2e-4)
    assert s["ir_diff"] == pytest.approx(98e-6)
    assert s["temp_integral"] == pytest.approx(30.0 * 99)


def test_features_ignore_everything_after_the_early_window(tmp_path):
    cells = generate(seed=3, batch_sizes={"b1": 6})
    a = Lake(tmp_path / "a")
    write(cells, a)
    for cell in cells:  # change the future: later capacity, later curves, labels kept
        late = cell.summary["cycle"] > 100
        cell.summary["qd"][late] *= 0.97
        for c in list(cell.curves):
            if c > 100:
                cell.curves[c] = (cell.curves[c][0] * 0.9, cell.curves[c][1])
    b = Lake(tmp_path / "b")
    write(cells, b)
    fa, fb = F.build(a), F.build(b)
    pd.testing.assert_frame_equal(fa[F.ALL_FEATURES], fb[F.ALL_FEATURES])


def test_paper_split(feats):
    split = F.paper_split(feats.cell_id.tolist())
    assert {k: len(v) for k, v in split.items()} == {"train": 41, "test_primary": 43, "test_secondary": 40}
    all_ids = sum(split.values(), [])
    assert len(all_ids) == len(set(all_ids)) == 124
    assert "b2c42" in split["test_primary"]  # the paper adds the last batch-2 cell to the test set


def test_models_beat_baselines_and_intervals_are_calibrated(feats):
    fitted, results, selected = train_all(feats)
    primary = {n: r["test_primary"]["mean_abs_pct_error"] for n, r in results.items()}
    assert primary["variance"] < 0.5 * primary["mean_baseline"]
    assert primary["variance"] < primary["capacity_baseline"]
    for split in ("test_primary", "test_secondary"):
        cov = results[selected][split]["interval90_coverage"]
        assert 0.75 <= cov <= 1.0, f"90% interval covered {cov:.0%} on {split}"


def test_selection_does_not_look_at_test_labels(feats):
    _, _, selected = train_all(feats)
    scrambled = feats.copy()
    parts = split_frames(feats)
    test_ids = set(parts["test_primary"].cell_id) | set(parts["test_secondary"].cell_id)
    mask = scrambled.cell_id.isin(test_ids)
    scrambled.loc[mask, "cycle_life"] = np.random.default_rng(0).permutation(scrambled.loc[mask, "cycle_life"].to_numpy())
    _, _, selected_scrambled = train_all(scrambled)
    assert selected == selected_scrambled


def test_interval_width_grows_with_coverage(feats):
    parts = split_frames(feats)
    cols, est = M.model_specs()["variance"]
    m = M.fit("variance", cols, est, parts["train"])
    assert m.interval_half_width[0.9] >= m.interval_half_width[0.8]
    lo, hi = m.interval(parts["test_primary"], 0.9)
    pred = m.predict(parts["test_primary"])
    assert ((lo < pred) & (pred < hi)).all()


def test_train_command_writes_model_and_reports(lake, tmp_path):
    from bip.cli import main

    assert main(["--data-dir", str(lake.root), "train", "--out", str(tmp_path / "reports"),
                 "--artifacts", str(tmp_path / "artifacts")]) == 0
    model = joblib.load(tmp_path / "artifacts/life_model.joblib")
    preds = pd.read_csv(tmp_path / "reports/model/predictions.csv")
    feats = pd.read_parquet(lake.features)
    assert np.allclose(model.predict(feats), preds.predicted, rtol=1e-3)
    assert set(preds.split) == {"train", "test_primary", "test_secondary"}
    for name in ("metrics.json", "early_window.csv", "importance.csv", "predicted_vs_observed.png"):
        assert (tmp_path / "reports/model" / name).exists()
