"""Train, compare and select cycle-life models; write results and the selected model.

Model selection uses only the training set (cross-validated error). Test sets
are scored for reporting, never used to choose. Picking the model with the best
test score would leak the test set into the choice and overstate accuracy.

Outputs (default reports/model/):
  metrics.json           per model x split: RMSE, % error, interval coverage and width
  predictions.csv        every cell's prediction and 90% interval from the selected model
  early_window.csv/.png  error when predicting from cycles 20, 40, ... 100
  importance.csv/.png    permutation importance of the 9-feature "full" model's features
  predicted_vs_observed.png
and artifacts/life_model.joblib (the selected model, for serving).
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from .. import plots
from ..config import Lake
from . import features as F
from . import models as M

BASELINES = {"mean_baseline", "capacity_baseline"}
SPLITS = ("train", "test_primary", "test_secondary")


def split_frames(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    split = F.paper_split(df.cell_id.tolist())
    return {k: df[df.cell_id.isin(v)].reset_index(drop=True) for k, v in split.items()}


def train_all(df: pd.DataFrame, seed: int = 0) -> tuple[dict[str, M.LifeModel], dict, str]:
    parts = split_frames(df)
    fitted, results = {}, {}
    for name, (cols, est) in M.model_specs(seed).items():
        model = M.fit(name, cols, est, parts["train"], seed=seed)
        fitted[name] = model
        results[name] = {"cv_rmse_log10": model.cv_rmse_log10,
                         **{s: M.metrics(model, parts[s]) for s in SPLITS if len(parts[s])}}
    candidates = {n: m for n, m in fitted.items() if n not in BASELINES}
    selected = min(candidates, key=lambda n: candidates[n].cv_rmse_log10)
    return fitted, results, selected


def early_window_study(lake: Lake, windows=(20, 40, 60, 80, 100), seed: int = 0) -> pd.DataFrame:
    """How early can life be predicted? Refit the variance and full models using
    only cycles <= n, for each n."""
    rows = []
    for n in windows:
        df = F.build(lake, late=n)
        parts = split_frames(df)
        for name in ("variance", "full"):
            cols, est = M.model_specs(seed)[name]
            m = M.fit(name, cols, est, parts["train"], seed=seed)
            for s in ("test_primary", "test_secondary"):
                if len(parts[s]):
                    rows.append({"window_cycles": n, "model": name, "split": s,
                                 "mean_abs_pct_error": M.metrics(m, parts[s])["mean_abs_pct_error"]})
    return pd.DataFrame(rows)


def _scatter_png(model: M.LifeModel, parts: dict, path: Path, label: str):
    fig, ax = plots.figure("Predicted vs observed cycle life",
                           f"{label}; model '{model.name}' using cycles 1-100, bars: 90% intervals",
                           "Observed cycle life", "Predicted cycle life", size=(6.6, 5.4))
    for color, s in zip(plots.SERIES, SPLITS):
        d = parts[s]
        if not len(d):
            continue
        pred = model.predict(d)
        lo, hi = model.interval(d, 0.9)
        ax.errorbar(d.cycle_life, pred, yerr=[pred - lo, hi - pred], fmt="o", markersize=4, color=color,
                    ecolor=color, elinewidth=0.8, alpha=0.85, label=s.replace("_", " "))
    lim = [100, 3000]
    ax.plot(lim, lim, color=plots.INK_2, linewidth=1, linestyle="--")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    plots.save(fig, path)


def _window_png(study: pd.DataFrame, path: Path, label: str):
    fig, ax = plots.figure("How early can cycle life be predicted?",
                           f"{label}; error on the primary test set", "Last cycle used for features",
                           "Mean absolute % error")
    d = study[study.split == "test_primary"]
    last_window = d.window_cycles.max()
    ends = d[d.window_cycles == last_window].set_index("model").mean_abs_pct_error
    for color, (name, part) in zip(plots.SERIES, d.groupby("model")):
        part = part.sort_values("window_cycles")
        ax.plot(part.window_cycles, part.mean_abs_pct_error, marker="o", color=color, linewidth=2, label=f"{name} model")
        last = part.iloc[-1]
        above = last.mean_abs_pct_error >= ends.max()  # keep the two end labels apart
        ax.annotate(f"{last.mean_abs_pct_error:.1f}%", (last.window_cycles, last.mean_abs_pct_error),
                    xytext=(6, 7 if above else -7), textcoords="offset points", fontsize=9, color=plots.INK,
                    va="bottom" if above else "top")
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, fontsize=9)
    plots.save(fig, path)


def _importance_png(imp: pd.DataFrame, path: Path, label: str, model_name: str):
    imp = imp.sort_values("importance")
    fig, ax = plots.figure("Which early-life signals matter", f"{label}; permutation importance, model '{model_name}'",
                           "Increase in error when the feature is shuffled (log10 cycles)", "")
    ax.barh(imp.feature, imp.importance, xerr=imp["std"], color=plots.SERIES[0], height=0.6,
            error_kw={"ecolor": plots.INK_2, "elinewidth": 0.8})
    ax.grid(axis="x", color=plots.GRID)
    ax.grid(axis="y", visible=False)
    plots.save(fig, path)


def run(lake: Lake, out_dir: str | Path, artifact_dir: str | Path, label: str, seed: int = 0) -> dict:
    out, art = Path(out_dir), Path(artifact_dir)
    out.mkdir(parents=True, exist_ok=True)
    art.mkdir(parents=True, exist_ok=True)

    df = F.build(lake)
    df.to_parquet(lake.features, index=False)
    fitted, results, selected = train_all(df, seed)
    model = fitted[selected]
    parts = split_frames(df)

    (out / "metrics.json").write_text(json.dumps({"selected_model": selected, "selection": "lowest cross-validated "
                                                  "error on the training set", "models": results}, indent=2))
    lo, hi = model.interval(df, 0.9)
    split_of = {c: s for s, d in parts.items() for c in d.cell_id}
    pd.DataFrame({"cell_id": df.cell_id, "split": df.cell_id.map(split_of), "cycle_life": df.cycle_life,
                  "predicted": model.predict(df).round(1), "lower90": lo.round(1), "upper90": hi.round(1)}
                 ).to_csv(out / "predictions.csv", index=False)
    _scatter_png(model, parts, out / "predicted_vs_observed.png", label)

    study = early_window_study(lake, seed=seed)
    study.to_csv(out / "early_window.csv", index=False)
    _window_png(study, out / "early_window.png", label)

    # feature study on the full model (the selected model may use only one feature)
    full, test = fitted["full"], parts["test_primary"]
    pi = permutation_importance(full.estimator, test[full.features].to_numpy(),
                                np.log10(test.cycle_life.to_numpy(float)), n_repeats=30, random_state=seed,
                                scoring="neg_root_mean_squared_error")
    imp = pd.DataFrame({"feature": full.features, "importance": pi.importances_mean, "std": pi.importances_std})
    imp.sort_values("importance", ascending=False).to_csv(out / "importance.csv", index=False)
    _importance_png(imp, out / "importance.png", label, "full")

    joblib.dump(model, art / "life_model.joblib")
    return {"selected_model": selected, "test_primary": results[selected]["test_primary"],
            "test_secondary": results[selected].get("test_secondary")}
