"""The `bip` command.

    bip ingest --source severson     # real data: the three .mat files in data/raw/
    bip ingest --source synthetic    # synthetic cells, no download needed
    bip transform                    # dbt build: models + data tests
    bip quality                      # SPC, curve anomalies, Weibull reliability
    bip train                        # features, model comparison, selected model + analyses
    bip register --promote           # log the trained model to MLflow, make it the champion
    bip drift --batch b3             # input / prediction / performance drift for a batch
    bip retrain --batch b3           # champion vs challenger on a new batch; promote if better
    bip serve                        # prediction API (FastAPI)
    bip stream                       # replay cycler data through the streaming consumer
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

from .config import Lake

ROOT = Path(__file__).resolve().parent.parent
DBT_DIR = ROOT / "dbt"


def _lake(args) -> Lake:
    return Lake(Path(args.data_dir).resolve())


def cmd_ingest(args) -> int:
    from .lake import write

    lake = _lake(args)
    if args.source == "severson":
        from .sources.severson import load
        cells = load(lake.raw)
    else:
        from .sources.synthetic import generate
        cells = generate(seed=args.seed)
    stats = write(cells, lake)
    print(json.dumps(asdict(stats), indent=2))
    return 0


def cmd_transform(args) -> int:
    from dbt.cli.main import dbtRunner

    lake = _lake(args)
    os.environ["BIP_LAKE"] = str(lake.lake)
    os.environ["BIP_WAREHOUSE"] = str(lake.warehouse)
    result = dbtRunner().invoke(["build", "--project-dir", str(DBT_DIR), "--profiles-dir", str(DBT_DIR), "--quiet"])
    print("dbt build: " + ("passed" if result.success else "FAILED"))
    return 0 if result.success else 1


def _label(args) -> str:
    import pandas as pd

    lake = _lake(args)
    sources = set(pd.read_parquet(lake.cells, columns=["source"]).source)
    return "Synthetic data (pipeline demo)" if sources == {"synthetic"} else "Severson et al. 2019 cells"


def cmd_quality(args) -> int:
    from .quality.report import run

    result = run(_lake(args), Path(args.out) / "quality", _label(args), args.warranty_cycles)
    print(json.dumps({k: v for k, v in result.items() if k != "weibull"}, indent=2))
    print(json.dumps(result["weibull"]["all_cells"], indent=2))
    return 0


def cmd_train(args) -> int:
    from .ml.train import run

    result = run(_lake(args), Path(args.out) / "model", args.artifacts, _label(args), args.seed)
    print(json.dumps(result, indent=2))
    return 0


def _champion(args):
    """The serving model: the registry champion if there is one, else the trained joblib file."""
    import joblib

    if getattr(args, "registry", False):
        from .mlops.registry import champion_version, load_life_model

        if champion_version(args.mlruns) is not None:
            return load_life_model("champion", args.mlruns), "registry:champion"
    return joblib.load(Path(args.artifacts) / "life_model.joblib"), str(Path(args.artifacts) / "life_model.joblib")


def _features(args):
    import pandas as pd

    from .ml.features import build

    lake = _lake(args)
    return pd.read_parquet(lake.features) if lake.features.exists() else build(lake)


def cmd_register(args) -> int:
    from .mlops.registry import champion_version, log_and_register, promote

    reports = Path(args.reports) / "model"
    version = log_and_register(Path(args.artifacts) / "life_model.joblib", reports / "metrics.json", reports,
                               root=args.mlruns, tags={"data": _label(args)})
    if args.promote or champion_version(args.mlruns) is None:
        promote(version, args.mlruns)
        print(f"registered version {version} and made it the champion")
    else:
        print(f"registered version {version} (champion unchanged)")
    return 0


def cmd_drift(args) -> int:
    from .ml.train import split_frames
    from .mlops.drift import report

    model, source = _champion(args)
    feats = _features(args)
    reference = split_frames(feats)["train"]
    current = feats[feats.batch.eq(args.batch)] if args.batch else split_frames(feats)[args.split]
    if not args.with_labels:
        current = current.drop(columns="cycle_life")
    ref_err = None
    metrics_path = Path(args.reports) / "model" / "metrics.json"
    if metrics_path.exists():
        m = json.loads(metrics_path.read_text())
        ref_err = m["models"].get(model.name, {}).get("test_primary", {}).get("mean_abs_pct_error")
    result = {"model_source": source, "current": args.batch or args.split,
              **report(model, reference, current, reference_error_pct=ref_err)}
    out = Path(args.reports) / "mlops"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"drift_{args.batch or args.split}.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k: result[k] for k in ("current", "prediction_psi", "retrain_recommended", "reasons")}, indent=2))
    return 0


def cmd_retrain(args) -> int:
    import joblib

    from .mlops.retrain import decision_dict, evaluate

    champion, source = _champion(args)
    decision, challenger = evaluate(champion, _features(args), args.batch, args.min_improvement, seed=args.seed)
    result = {"champion_source": source, "new_batch": args.batch, **decision_dict(decision)}
    out = Path(args.reports) / "mlops"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"retrain_{args.batch}.json").write_text(json.dumps(result, indent=2))
    if decision.promote:
        path = Path(args.artifacts) / "challenger.joblib"
        joblib.dump(challenger, path)
        if args.registry:
            from .mlops.registry import log_and_register, promote

            metrics = out / "challenger_metrics.json"
            metrics.write_text(json.dumps({"selected_model": challenger.name, "selection": "challenger",
                                           "models": {challenger.name: {"holdout": {
                                               "mean_abs_pct_error": decision.challenger_error_pct,
                                               "interval90_coverage": decision.challenger_coverage}}}}))
            version = log_and_register(path, metrics, root=args.mlruns, run_name=f"retrain-{args.batch}",
                                       tags={"trigger": f"new batch {args.batch}"})
            promote(version, args.mlruns)
            result["promoted_version"] = version
        else:
            joblib.dump(challenger, Path(args.artifacts) / "life_model.joblib")
    print(json.dumps(result, indent=2))
    return 0


def cmd_serve(args) -> int:
    import uvicorn

    if args.registry:
        os.environ.setdefault("BIP_MODEL_URI", "models:/battery-cycle-life@champion")
        os.environ.setdefault("BIP_MLRUNS", args.mlruns)
    else:
        os.environ.setdefault("BIP_MODEL_PATH", str(Path(args.artifacts) / "life_model.joblib"))
    uvicorn.run("bip.mlops.serve:app", host=args.host, port=args.port)
    return 0


def stream_monitors(lake: Lake, reference_batch: str = "b1"):
    """Cycle-2 capacity control limits and the cycle-10 curve anomaly model, both fit on the reference batch."""
    import numpy as np
    import pandas as pd

    from .lake import read_curves
    from .quality import anomaly, spc

    summary = pd.read_parquet(lake.cycle_summary)
    ref = summary[(summary.batch == reference_batch) & (summary.cycle == 2)].sort_values("cell_id")
    limits = spc.fit_limits(ref.qd.to_numpy(), "qd_cycle2")
    curves = read_curves(lake, [10])
    ref_cells = set(ref.cell_id)
    x = np.stack([q for (cell, _), q in sorted(curves.items()) if cell in ref_cells])
    return limits, anomaly.fit(x)


def cmd_stream(args) -> int:
    from .stream.broker import InMemoryBroker, KafkaBroker
    from .stream.consumer import TOPIC_ALERTS, TOPIC_PREDICTIONS, StreamProcessor
    from .stream.cycler import replay

    lake = _lake(args)
    broker = KafkaBroker(args.bootstrap) if args.bootstrap else InMemoryBroker()
    cells = args.cells.split(",") if args.cells else None
    if args.role in ("produce", "both"):
        print(f"sent {replay(lake, broker, cells, max_cycle=args.max_cycle)} cycle records")
    if args.role in ("consume", "both"):
        model, _ = _champion(args)
        limits, anomaly_model = stream_monitors(lake)
        proc = StreamProcessor(broker, model, capacity_limits=limits, anomaly_model=anomaly_model)
        n = proc.run()
        print(f"consumed {n} records, {proc.duplicates} duplicates ignored")
        if isinstance(broker, InMemoryBroker):
            preds = list(broker.consume(TOPIC_PREDICTIONS))
            alerts = list(broker.consume(TOPIC_ALERTS))
            print(f"{len(preds)} predictions, {len(alerts)} alerts")
            for a in alerts[:20]:
                print("  ALERT", a["cell_id"], a["alert"], a["value"])
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="bip", description="Battery Intelligence Platform")
    p.add_argument("--data-dir", default="data")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("ingest", help="load cells into the Parquet lakehouse")
    s.add_argument("--source", choices=["severson", "synthetic"], default="severson")
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(fn=cmd_ingest)

    sub.add_parser("transform", help="dbt build (models + data tests)").set_defaults(fn=cmd_transform)

    q = sub.add_parser("quality", help="SPC, curve anomaly detection and Weibull reliability")
    q.add_argument("--out", default="reports")
    q.add_argument("--warranty-cycles", type=int, default=500)
    q.set_defaults(fn=cmd_quality)

    t = sub.add_parser("train", help="train and compare cycle-life models")
    t.add_argument("--out", default="reports")
    t.add_argument("--artifacts", default="artifacts")
    t.add_argument("--seed", type=int, default=0)
    t.set_defaults(fn=cmd_train)

    def mlops_args(x, registry_default=False):
        x.add_argument("--artifacts", default="artifacts")
        x.add_argument("--reports", default="reports")
        x.add_argument("--mlruns", default="mlruns")
        x.add_argument("--registry", action=argparse.BooleanOptionalAction, default=registry_default,
                       help="use the MLflow champion instead of artifacts/life_model.joblib")

    r = sub.add_parser("register", help="log the trained model to MLflow and register it")
    mlops_args(r)
    r.add_argument("--promote", action="store_true", help="make the new version the champion")
    r.set_defaults(fn=cmd_register)

    d = sub.add_parser("drift", help="drift report for a batch or split against the training cells")
    mlops_args(d)
    d.add_argument("--batch")
    d.add_argument("--split", default="test_primary", choices=["test_primary", "test_secondary"])
    d.add_argument("--with-labels", action="store_true", help="include performance drift (labels available)")
    d.set_defaults(fn=cmd_drift)

    rt = sub.add_parser("retrain", help="champion/challenger evaluation on a new batch")
    mlops_args(rt)
    rt.add_argument("--batch", required=True)
    rt.add_argument("--min-improvement", type=float, default=0.05)
    rt.add_argument("--seed", type=int, default=0)
    rt.set_defaults(fn=cmd_retrain)

    sv = sub.add_parser("serve", help="run the prediction API")
    mlops_args(sv)
    sv.add_argument("--host", default="127.0.0.1")
    sv.add_argument("--port", type=int, default=8000)
    sv.set_defaults(fn=cmd_serve)

    st = sub.add_parser("stream", help="replay cycler data and run the streaming consumer")
    mlops_args(st)
    st.add_argument("--role", choices=["both", "produce", "consume"], default="both")
    st.add_argument("--bootstrap", help="Kafka bootstrap servers; in-memory broker if omitted")
    st.add_argument("--cells", help="comma-separated cell IDs (default: all)")
    st.add_argument("--max-cycle", type=int, default=100)
    st.set_defaults(fn=cmd_stream)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
