"""MLflow tracking and model registry.

Each training run is logged with its parameters, per-split metrics and report
files. The model is logged as an MLflow pyfunc and registered as a new version
of `battery-cycle-life`. The version serving production carries the alias
`champion`; aliases (not version numbers) are what the API loads, so promotion
and rollback are a single alias move.

Tracking defaults to a local SQLite store (needed for the registry) under
`mlruns/`; point MLFLOW_TRACKING_URI at a server to share it.
"""

from __future__ import annotations

import inspect
import json
import os
from pathlib import Path

import joblib
import pandas as pd

MODEL_NAME = "battery-cycle-life"
CHAMPION = "champion"


def tracking_uri(root: str | Path = "mlruns") -> str:
    if os.environ.get("MLFLOW_TRACKING_URI"):
        return os.environ["MLFLOW_TRACKING_URI"]
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{root / 'mlflow.db'}"


def _mlflow(root):
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    import mlflow

    mlflow.set_tracking_uri(tracking_uri(root))
    return mlflow


class LifeModelPyfunc:  # wrapped by mlflow.pyfunc.PythonModel at log time
    """Pyfunc predict: features DataFrame -> predicted cycle life with 90% interval."""

    def load_context(self, context):
        self.model = joblib.load(context.artifacts["life_model"])

    def predict(self, context, model_input: pd.DataFrame, params=None) -> pd.DataFrame:
        lo, hi = self.model.interval(model_input, 0.9)
        return pd.DataFrame({"predicted": self.model.predict(model_input), "lower90": lo, "upper90": hi})


def _pyfunc_class():
    import mlflow.pyfunc

    return type("LifeModelPythonModel", (LifeModelPyfunc, mlflow.pyfunc.PythonModel), {})


def log_and_register(model_path: str | Path, metrics_json: str | Path, reports_dir: str | Path | None = None,
                     root: str | Path = "mlruns", run_name: str = "train", tags: dict | None = None) -> int:
    """Log a trained model with its metrics; register it. Returns the new model version."""
    mlflow = _mlflow(root)
    model = joblib.load(model_path)
    metrics = json.loads(Path(metrics_json).read_text())
    selected = metrics["selected_model"]
    mlflow.set_experiment("battery-cycle-life")

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tags({"selected_model": selected, **(tags or {})})
        mlflow.log_params({"model": model.name, "features": ",".join(model.features),
                           "selection": metrics.get("selection", "")})
        mlflow.log_metric("cv_rmse_log10", model.cv_rmse_log10)
        for split, values in metrics["models"][selected].items():
            if isinstance(values, dict):
                for k, v in values.items():
                    mlflow.log_metric(f"{split}.{k}", float(v))
        if reports_dir and Path(reports_dir).exists():
            mlflow.log_artifacts(str(reports_dir), artifact_path="reports")

        kw = {"name": "model"} if "name" in inspect.signature(mlflow.pyfunc.log_model).parameters else {"artifact_path": "model"}
        input_example = pd.DataFrame([{f: 0.0 for f in model.features}])
        info = mlflow.pyfunc.log_model(python_model=_pyfunc_class()(), artifacts={"life_model": str(model_path)},
                                       registered_model_name=MODEL_NAME, input_example=input_example, **kw)
    version = int(info.registered_model_version)
    from mlflow import MlflowClient

    MlflowClient().set_model_version_tag(MODEL_NAME, str(version), "run_id", run.info.run_id)
    return version


def promote(version: int, root: str | Path = "mlruns", alias: str = CHAMPION) -> None:
    _mlflow(root)
    from mlflow import MlflowClient

    MlflowClient().set_registered_model_alias(MODEL_NAME, alias, str(version))


def champion_version(root: str | Path = "mlruns") -> int | None:
    _mlflow(root)
    from mlflow import MlflowClient
    from mlflow.exceptions import MlflowException

    try:
        return int(MlflowClient().get_model_version_by_alias(MODEL_NAME, CHAMPION).version)
    except MlflowException:
        return None


def load_life_model(version_or_alias: str | int = CHAMPION, root: str | Path = "mlruns"):
    """Return the underlying LifeModel (not the pyfunc wrapper) for a registered version."""
    mlflow = _mlflow(root)
    ref = f"models:/{MODEL_NAME}@{version_or_alias}" if isinstance(version_or_alias, str) \
        else f"models:/{MODEL_NAME}/{version_or_alias}"
    local = mlflow.artifacts.download_artifacts(ref)
    candidates = list(Path(local).rglob("*.joblib"))
    if not candidates:
        raise FileNotFoundError(f"no model artifact found for {ref}")
    return joblib.load(candidates[0])
