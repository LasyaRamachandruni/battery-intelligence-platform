"""Cycle-life models with calibrated prediction intervals.

All models predict log10(cycle life); errors are reported back in cycles.

Models
  mean_baseline       predicts the training mean (anything useful must beat this)
  capacity_baseline   linear model on capacity at cycle 2 and early fade slope only:
                      what you could do without looking at curve shape
  variance/discharge/full
                      elastic net on the paper's three feature sets (standardized,
                      regularization chosen by cross-validation)
  gbm                 gradient-boosted trees on all features

Intervals: cross-conformal (CV+) style. Out-of-fold absolute residuals on the
training set (in log space) give the half-width that covers the requested share
of unseen cells; an interval is prediction x/÷ 10^half_width. With ~40 training
cells this is the honest way to get intervals: no distributional assumption,
and coverage can be checked on the test sets.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import ElasticNetCV, LinearRegression
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .features import ALL_FEATURES, FEATURE_SETS

CAPACITY_ONLY = ["q2", "fade_slope"]


def model_specs(seed: int = 0) -> dict[str, tuple[list[str], object]]:
    enet = lambda: make_pipeline(StandardScaler(), ElasticNetCV(  # noqa: E731
        l1_ratio=[0.1, 0.5, 0.9, 1.0], cv=4, max_iter=50000, random_state=seed))
    specs = {
        "mean_baseline": (["dq_var"], DummyRegressor(strategy="mean")),
        "capacity_baseline": (CAPACITY_ONLY, make_pipeline(StandardScaler(), LinearRegression())),
    }
    for name, cols in FEATURE_SETS.items():
        specs[name] = (cols, enet())
    specs["gbm"] = (ALL_FEATURES, GradientBoostingRegressor(
        n_estimators=300, max_depth=2, learning_rate=0.05, subsample=0.8, random_state=seed))
    return specs


@dataclass
class LifeModel:
    name: str
    features: list[str]
    estimator: object
    interval_half_width: dict[float, float] = field(default_factory=dict)  # coverage -> log10 half width
    cv_rmse_log10: float = float("nan")  # out-of-fold error on the training set: used for model selection

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return 10 ** self.estimator.predict(df[self.features].to_numpy())

    def interval(self, df: pd.DataFrame, coverage: float = 0.9) -> tuple[np.ndarray, np.ndarray]:
        h = self.interval_half_width[coverage]
        log_pred = self.estimator.predict(df[self.features].to_numpy())
        return 10 ** (log_pred - h), 10 ** (log_pred + h)


def fit(name: str, features: list[str], estimator, train: pd.DataFrame,
        coverages=(0.8, 0.9), folds: int = 5, seed: int = 0) -> LifeModel:
    x = train[features].to_numpy()
    y = np.log10(train.cycle_life.to_numpy(float))

    oof = np.empty_like(y)
    for tr, te in KFold(folds, shuffle=True, random_state=seed).split(x):
        oof[te] = clone(estimator).fit(x[tr], y[tr]).predict(x[te])
    resid = np.abs(y - oof)
    n = len(resid)
    # finite-sample conformal quantile: the ceil((n+1) * coverage)-th smallest residual
    widths = {c: float(np.sort(resid)[min(n - 1, int(np.ceil((n + 1) * c)) - 1)]) for c in coverages}

    cv_rmse = float(np.sqrt(np.mean((y - oof) ** 2)))
    return LifeModel(name, list(features), clone(estimator).fit(x, y), widths, cv_rmse)


def metrics(model: LifeModel, df: pd.DataFrame, coverage: float = 0.9) -> dict:
    y = df.cycle_life.to_numpy(float)
    pred = model.predict(df)
    lo, hi = model.interval(df, coverage)
    err = pred - y
    return {
        "cells": int(len(df)),
        "rmse_cycles": float(np.sqrt(np.mean(err**2))),
        "mean_abs_pct_error": float(np.mean(np.abs(err) / y) * 100),
        "median_abs_pct_error": float(np.median(np.abs(err) / y) * 100),
        f"interval{int(coverage * 100)}_coverage": float(np.mean((y >= lo) & (y <= hi))),
        f"interval{int(coverage * 100)}_median_width_cycles": float(np.median(hi - lo)),
    }
