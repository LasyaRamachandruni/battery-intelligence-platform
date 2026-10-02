"""Phase 2: SPC, curve anomaly detection, Weibull reliability."""

import numpy as np
import pandas as pd
import pytest

from bip.config import VOLTAGE_GRID
from bip.lake import read_curves
from bip.quality import anomaly, reliability, spc


# --- SPC ---------------------------------------------------------------------------

def test_limits_estimate_sigma_from_moving_range():
    rng = np.random.default_rng(0)
    lim = spc.fit_limits(rng.normal(10, 2, 5000))
    assert lim.center == pytest.approx(10, abs=0.1)
    assert lim.sigma == pytest.approx(2, rel=0.05)
    assert lim.ucl == pytest.approx(lim.center + 3 * lim.sigma)


def test_in_control_process_rarely_alarms_on_rule_1():
    rng = np.random.default_rng(1)
    lim = spc.ControlLimits("x", 0.0, 1.0)
    res = spc.chart(rng.normal(0, 1, 20000), lim)
    rule1 = sum(v.rule == 1 for v in res.violations) / 20000
    assert rule1 == pytest.approx(0.0027, abs=0.0015)  # 3-sigma false alarm rate


def test_each_run_rule_fires():
    lim = spc.ControlLimits("x", 0.0, 1.0)
    rules = lambda x: {v.rule for v in spc.run_rules(np.array(x, float), lim)}  # noqa: E731
    assert 1 in rules([0, 0, 3.5])
    assert 2 in rules([0, 2.5, 0.1, 2.5])
    assert 3 in rules([1.5, 1.5, 0.2, 1.5, 1.5])
    assert 4 in rules([0.3] * 8)
    assert rules([0.3, -0.2, 0.5, -0.4, 0.1, -0.1, 0.2, -0.3]) == set()


def test_capability_indices():
    cap = spc.capability(np.array([0.9, 1.0, 1.1] * 10), lsl=0.4, usl=1.6, sigma=0.1)
    assert cap["cp"] == pytest.approx(2.0)
    assert cap["cpk"] == pytest.approx(2.0)
    off = spc.capability(np.full(10, 1.3), lsl=0.4, usl=1.6, sigma=0.1)
    assert off["cpk"] == pytest.approx(1.0)


def test_monitor_flags_a_shifted_batch():
    rng = np.random.default_rng(2)
    rows = []
    for batch, shift in (("b1", 0.0), ("b2", 0.0), ("b3", 0.02)):  # b3 drifted by ~2.5 sigma
        for i in range(40):
            rows.append({"cell_id": f"{batch}c{i}", "batch": batch,
                         "q_cycle2_ah": 1.075 + shift + rng.normal(0, 0.008)})
    limits, summary, alerts = spc.monitor_batches(pd.DataFrame(rows), "q_cycle2_ah")
    by = summary.set_index("batch")
    assert by.loc["b3", "violations"] > 5 * max(1, by.loc["b2", "violations"])
    assert any(a["batch"] == "b3" for a in alerts)
    assert by.loc["b3", "cpk"] < by.loc["b1", "cpk"]


# --- anomaly detection ----------------------------------------------------------------

def _curves(lake, cycle=10):
    curves = read_curves(lake, [cycle])
    ids = sorted({c for c, _ in curves})
    return ids, np.stack([curves[(c, cycle)] for c in ids])


def test_anomaly_model_flags_distorted_curves(lake):
    ids, x = _curves(lake)
    ref = np.array([not c.startswith("b3") for c in ids])
    model = anomaly.fit(x[ref])

    held_out = model.flag(x[~ref])
    assert held_out.mean() < 0.15  # normal cells from another batch mostly pass

    bad = x[~ref][:5].copy()
    # a defect that lowers the voltage plateau: capacity comes out ~60 mV later
    shift = int(0.06 / abs(VOLTAGE_GRID[1] - VOLTAGE_GRID[0]))
    bad = np.concatenate([np.zeros((5, shift)), bad[:, :-shift]], axis=1)
    bad[:, -1] = x[~ref][:5, -1]
    assert model.flag(bad).all()


def test_anomaly_scores_are_size_invariant(lake):
    _, x = _curves(lake)
    model = anomaly.fit(x)
    t2a, qa = model.scores(x[:3])
    t2b, qb = model.scores(x[:3] * 0.95)  # same shape, 5% less capacity
    assert np.allclose(t2a, t2b) and np.allclose(qa, qb)


# --- Weibull reliability -----------------------------------------------------------------

def _sample(beta, eta, n, seed=0):
    return eta * np.random.default_rng(seed).weibull(beta, n)


def test_weibull_recovers_parameters():
    w = reliability.fit(_sample(3.0, 1000, 2000))
    assert w.beta == pytest.approx(3.0, rel=0.06)
    assert w.eta == pytest.approx(1000, rel=0.03)


def test_censoring_is_handled_and_naive_approach_is_biased():
    life = _sample(3.0, 1000, 2000, seed=1)
    t, failed = reliability.censor_at(life, 900)  # analysis while ~half are still on test
    assert 0.4 < failed.mean() < 0.7
    proper = reliability.fit(t, failed)
    naive = reliability.fit(t)  # pretends survivors failed at the cutoff
    assert proper.eta == pytest.approx(1000, rel=0.05)
    assert naive.eta < 0.9 * 1000


def test_b_life_and_interval():
    w = reliability.fit(_sample(3.0, 1000, 400, seed=2))
    b10 = w.b_life(0.10)
    assert w.cdf(b10) == pytest.approx(0.10)
    true_b10 = 1000 * (-np.log(0.9)) ** (1 / 3)
    lo, hi = w.b_life_interval(0.10)
    assert lo < true_b10 < hi
    assert lo < b10 < hi


def test_group_comparison():
    same = {"a": (_sample(3, 1000, 150, 3), np.ones(150, bool)), "b": (_sample(3, 1000, 150, 4), np.ones(150, bool))}
    diff = {"a": (_sample(3, 1000, 150, 5), np.ones(150, bool)), "b": (_sample(3, 1400, 150, 6), np.ones(150, bool))}
    assert reliability.compare_groups(same)["p_value"] > 0.01
    assert reliability.compare_groups(diff)["p_value"] < 1e-4


def test_kaplan_meier_matches_empirical_survival_without_censoring():
    t = np.array([100, 200, 200, 300, 400.0])
    times, surv = reliability.kaplan_meier(t)
    assert list(times) == [100, 200, 300, 400]
    assert np.allclose(surv, [0.8, 0.4, 0.2, 0.0])


def test_kaplan_meier_with_censoring():
    times, surv = reliability.kaplan_meier([100, 200, 300, 400], [True, False, True, True])
    # after the censored cell at 200, 2 remain at risk at 300
    assert np.allclose(surv, [0.75, 0.375, 0.0])


# --- end to end ------------------------------------------------------------------------

def test_quality_command_writes_reports(lake, tmp_path):
    import json

    from bip.cli import main

    assert main(["--data-dir", str(lake.root), "transform"]) == 0
    assert main(["--data-dir", str(lake.root), "quality", "--out", str(tmp_path)]) == 0
    out = tmp_path / "quality"
    for name in ("spc_summary.csv", "spc_alerts.csv", "anomalies.csv", "weibull.json",
                 "control_chart_q_cycle2_ah.png", "weibull_probability.png"):
        assert (out / name).stat().st_size > 0
    weibull = json.loads((out / "weibull.json").read_text())
    assert weibull["all_cells"]["beta"] > 1  # capacity fade is a wear-out failure mode
    assert set(weibull["per_batch"]) == {"b1", "b2", "b3"}
