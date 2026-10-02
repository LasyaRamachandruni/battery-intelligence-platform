"""Prediction API.

    uvicorn bip.mlops.serve:app            (or: bip serve)

Model source, in order of preference:
  BIP_MODEL_URI   an MLflow model reference, e.g. models:/battery-cycle-life@champion
  BIP_MODEL_PATH  a joblib file written by `bip train` (default artifacts/life_model.joblib)

Endpoints
  GET  /health                 liveness + whether a model is loaded
  GET  /model                  which model is serving and what it expects
  POST /predict/features       precomputed features -> cycle life + 90% interval
  POST /predict/early-cycles   raw early-cycle data -> features -> cycle life + interval

The early-cycles endpoint computes features with bip.ml.features.cell_features,
the same function used in training, so serving can't drift from training.
"""

from __future__ import annotations

import os
from functools import lru_cache

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

from ..config import EARLY_CYCLES, VOLTAGE_GRID
from ..ml.features import cell_features

app = FastAPI(title="Battery cycle-life prediction", version="1.0")


@lru_cache(maxsize=1)
def get_model():
    uri = os.environ.get("BIP_MODEL_URI")
    if uri:
        from .registry import load_life_model, tracking_uri  # noqa: F401

        alias_or_version = uri.rsplit("@", 1)[-1] if "@" in uri else int(uri.rsplit("/", 1)[-1])
        model = load_life_model(alias_or_version, os.environ.get("BIP_MLRUNS", "mlruns"))
        return model, uri
    import joblib

    path = os.environ.get("BIP_MODEL_PATH", "artifacts/life_model.joblib")
    return joblib.load(path), path


class Prediction(BaseModel):
    predicted_cycle_life: float
    lower90: float
    upper90: float
    model: str
    model_source: str


class FeaturesRequest(BaseModel):
    features: dict[str, float]


class CycleRecord(BaseModel):
    cycle: int
    qd: float = Field(gt=0, lt=1.5, description="discharge capacity, Ah")
    ir: float = Field(gt=0, lt=0.05, description="internal resistance, ohm")
    chargetime: float = Field(gt=0, lt=120, description="minutes")
    tavg: float = Field(gt=10, lt=70, description="average temperature, deg C")


class EarlyCyclesRequest(BaseModel):
    q_cycle10: list[float] = Field(description=f"Qdlin at cycle 10: {VOLTAGE_GRID.size} points from 3.5 V to 2.0 V")
    q_cycle100: list[float] = Field(description="Qdlin at cycle 100, same grid")
    cycles: list[CycleRecord] = Field(description="summary records for cycles 2-100")

    @field_validator("q_cycle10", "q_cycle100")
    @classmethod
    def _grid(cls, v):
        if len(v) != VOLTAGE_GRID.size:
            raise ValueError(f"expected {VOLTAGE_GRID.size} points on the voltage grid, got {len(v)}")
        return v


def _predict(df: pd.DataFrame) -> Prediction:
    model, source = get_model()
    missing = [f for f in model.features if f not in df]
    if missing:
        raise HTTPException(422, f"missing features: {', '.join(missing)}")
    lo, hi = model.interval(df, 0.9)
    return Prediction(predicted_cycle_life=float(model.predict(df)[0]), lower90=float(lo[0]),
                      upper90=float(hi[0]), model=model.name, model_source=str(source))


@app.get("/health")
def health():
    try:
        get_model()
        return {"status": "ok", "model_loaded": True}
    except Exception as exc:  # report, don't crash, so orchestrators see the reason
        return {"status": "degraded", "model_loaded": False, "error": str(exc)}


@app.get("/model")
def model_info():
    model, source = get_model()
    return {"model": model.name, "source": str(source), "features": model.features,
            "cv_rmse_log10": model.cv_rmse_log10, "interval_coverages": sorted(model.interval_half_width)}


@app.post("/predict/features", response_model=Prediction)
def predict_features(req: FeaturesRequest):
    return _predict(pd.DataFrame([req.features]))


@app.post("/predict/early-cycles", response_model=Prediction)
def predict_early_cycles(req: EarlyCyclesRequest):
    cycles = pd.DataFrame([c.model_dump() for c in req.cycles])
    if cycles.empty or cycles.cycle.max() < EARLY_CYCLES:
        raise HTTPException(422, f"need summary records up to cycle {EARLY_CYCLES}")
    try:
        f = cell_features(np.array(req.q_cycle10), np.array(req.q_cycle100), cycles)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    return _predict(pd.DataFrame([f]))
