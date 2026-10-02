"""Phase 4: registry, drift, retraining, the API and the streaming path."""

import numpy as np
import pandas as pd
import pytest

from bip.lake import read_curves
from bip.ml import features as F
from bip.ml.train import split_frames, train_all
from bip.mlops import drift
from bip.mlops.retrain import evaluate, split_new_batch
from bip.stream.broker import InMemoryBroker
from bip.stream.consumer import TOPIC_ALERTS, TOPIC_PREDICTIONS, StreamProcessor
from bip.stream.cycler import TOPIC_CYCLES, replay


@pytest.fixture(scope="module")
def feats(lake):
    return F.build(lake)


@pytest.fixture(scope="module")
def model(feats):
    fitted, _, selected = train_all(feats, seed=0)
    return fitted[selected]


@pytest.fixture(scope="module")
def model_file(model, tmp_path_factory):
    import joblib

    path = tmp_path_factory.mktemp("artifacts") / "life_model.joblib"
    joblib.dump(model, path)
    return path


# drift ---------------------------------------------------------------------

def test_drift_rule_false_alarm_and_detection_rates():
    """PSI alone is noisy with 40 cells, so the rule also requires KS significance."""
    from scipy import stats

    rng = np.random.default_rng(0)

    def alarm(a, b):
        return drift.psi(a, b) > drift.PSI_MAJOR and stats.ks_2samp(a, b).pvalue < drift.KS_ALPHA

    false_alarms = np.mean([alarm(rng.normal(size=40), rng.normal(size=40)) for _ in range(300)])
    detected = np.mean([alarm(rng.normal(size=40), rng.normal(1, 1, size=40)) for _ in range(300)])
    assert false_alarms < 0.03
    assert detected > 0.7


def test_drift_report_same_vs_shifted_population(model, feats):
    parts = split_frames(feats)
    same = drift.report(model, parts["train"], parts["test_primary"].drop(columns="cycle_life"))
    assert not same["retrain_recommended"]
    assert same["performance"] is None  # no labels, no performance check

    shifted = parts["test_primary"].copy()
    shifted["dq_var"] += 1.0  # ten times the capacity-curve variance
    report = drift.report(model, parts["train"], shifted, reference_error_pct=10.0)
    assert report["retrain_recommended"]
    assert any("dq_var" in r for r in report["reasons"])
    assert report["performance"]["cells"] == len(shifted)


# retraining ----------------------------------------------------------------

def test_split_new_batch_is_disjoint_and_covers_all(feats):
    new = feats[feats.batch.eq("b3")]
    adopt, holdout = split_new_batch(new)
    assert set(adopt.cell_id).isdisjoint(holdout.cell_id)
    assert len(adopt) + len(holdout) == len(new)
    assert abs(len(adopt) - len(holdout)) <= 1


def test_retrain_decision_is_consistent(model, feats):
    decision, challenger = evaluate(model, feats, "b3")
    improved = decision.challenger_error_pct < 0.95 * decision.champion_error_pct
    assert decision.promote == (improved and decision.challenger_coverage >= 0.75)
    assert decision.holdout_cells > 0 and challenger.features


# registry ------------------------------------------------------------------

def test_registry_round_trip(model, model_file, tmp_path, monkeypatch):
    pytest.importorskip("mlflow")
    import json

    from bip.mlops import registry

    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    metrics = tmp_path / "metrics.json"
    metrics.write_text(json.dumps({"selected_model": model.name, "models": {model.name: {
        "test_primary": {"mean_abs_pct_error": 10.0}}}}))
    root = tmp_path / "mlruns"
    v1 = registry.log_and_register(model_file, metrics, root=root)
    assert registry.champion_version(root) is None
    registry.promote(v1, root)
    v2 = registry.log_and_register(model_file, metrics, root=root)
    assert (v1, v2) == (1, 2) and registry.champion_version(root) == 1

    loaded = registry.load_life_model("champion", root)
    df = pd.DataFrame([{f: 0.0 for f in model.features}])
    np.testing.assert_allclose(loaded.predict(df), model.predict(df))


# API -----------------------------------------------------------------------

@pytest.fixture()
def client(model_file, monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from bip.mlops import serve

    monkeypatch.delenv("BIP_MODEL_URI", raising=False)
    monkeypatch.setenv("BIP_MODEL_PATH", str(model_file))
    serve.get_model.cache_clear()
    yield TestClient(serve.app)
    serve.get_model.cache_clear()


def _raw_request(lake, cell):
    curves = read_curves(lake, [10, 100])
    summary = pd.read_parquet(lake.cycle_summary)
    rows = summary[(summary.cell_id == cell) & summary.cycle.between(2, 100)]
    return {"q_cycle10": curves[(cell, 10)].tolist(), "q_cycle100": curves[(cell, 100)].tolist(),
            "cycles": rows[["cycle", "qd", "ir", "chargetime", "tavg"]].to_dict("records")}


def test_api_health_and_model(client, model):
    assert client.get("/health").json() == {"status": "ok", "model_loaded": True}
    info = client.get("/model").json()
    assert info["model"] == model.name and info["features"] == model.features


def test_api_raw_and_feature_paths_agree(client, lake, feats, model):
    cell = feats.cell_id.iloc[50]
    raw = client.post("/predict/early-cycles", json=_raw_request(lake, cell))
    assert raw.status_code == 200, raw.text
    row = feats[feats.cell_id == cell]
    by_features = client.post("/predict/features", json={"features": row[model.features].iloc[0].to_dict()})
    assert raw.json()["predicted_cycle_life"] == pytest.approx(by_features.json()["predicted_cycle_life"], rel=1e-4)
    assert raw.json()["lower90"] < raw.json()["predicted_cycle_life"] < raw.json()["upper90"]


def test_api_rejects_bad_input(client, lake, feats):
    req = _raw_request(lake, feats.cell_id.iloc[0])
    assert client.post("/predict/early-cycles", json={**req, "q_cycle10": [1.0] * 10}).status_code == 422
    assert client.post("/predict/early-cycles", json={**req, "cycles": req["cycles"][:20]}).status_code == 422
    bad = [dict(c) for c in req["cycles"]]
    bad[5]["ir"] = -1.0
    assert client.post("/predict/early-cycles", json={**req, "cycles": bad}).status_code == 422
    assert client.post("/predict/features", json={"features": {"nope": 1.0}}).status_code == 422


# streaming -----------------------------------------------------------------

CELLS = ["b2c1", "b2c2", "b2c3", "b2c4", "b2c5", "b2c6"]


def test_stream_predictions_match_batch(lake, feats, model):
    broker = InMemoryBroker()
    sent = replay(lake, broker, CELLS, max_cycle=100)
    assert sent > 0
    StreamProcessor(broker, model).run()
    preds = {p["cell_id"]: p for p in broker.consume(TOPIC_PREDICTIONS)}
    assert set(preds) == set(CELLS)
    expected = dict(zip(feats.cell_id, model.predict(feats)))
    for cell, p in preds.items():
        assert p["predicted_cycle_life"] == pytest.approx(expected[cell], abs=0.5)
        assert p["lower90"] < p["predicted_cycle_life"] < p["upper90"]


def test_stream_ignores_redelivered_messages(lake, model):
    broker = InMemoryBroker()
    replay(lake, broker, CELLS, max_cycle=100)
    msgs = list(broker.consume(TOPIC_CYCLES))
    # at-least-once delivery: replay every 7th record right after the original
    redelivered = []
    for i, m in enumerate(msgs):
        redelivered.append(m)
        if i % 7 == 0:
            redelivered.append(dict(m))
    for m in redelivered:
        broker.send(TOPIC_CYCLES, m["cell_id"], m)
    proc = StreamProcessor(broker, model)
    proc.run()
    assert proc.duplicates == len(redelivered) - len(msgs)
    assert len(list(broker.consume(TOPIC_PREDICTIONS))) == len(CELLS)
    assert proc.state == {}  # state freed once a cell is predicted


def test_stream_raises_capacity_alerts(lake, model):
    from bip.quality.spc import fit_limits

    broker = InMemoryBroker()
    replay(lake, broker, CELLS, max_cycle=100)
    msgs = list(broker.consume(TOPIC_CYCLES))
    for m in msgs:
        if m["cell_id"] == "b2c3" and m["cycle"] == 2:
            m["qd"] = 1.00  # below the 1.04 Ah spec limit
        broker.send(TOPIC_CYCLES, m["cell_id"], m)
    limits = fit_limits(np.array([m["qd"] for m in msgs if m["cycle"] == 2 and m["cell_id"] != "b2c3"]))
    StreamProcessor(broker, model, capacity_limits=limits).run()
    alerts = list(broker.consume(TOPIC_ALERTS))
    assert [(a["cell_id"], a["alert"]) for a in alerts] == [("b2c3", "capacity_out_of_spec")]
