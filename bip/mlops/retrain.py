"""Champion / challenger retraining.

When a new batch's cells start reaching end of life, their labels are new
training data and also the fairest test of the current model. The pipeline:

1. Split the new batch's labelled cells by channel into `adopt` (added to the
   training data) and `holdout` (never trained on by either model).
2. Train a challenger on the original training cells + `adopt`, using the same
   cross-validated model selection as the original training.
3. Score champion and challenger on `holdout`.
4. Promote the challenger only if it is better by at least `min_improvement`
   (relative % error) and its 90% intervals still cover at least
   `min_coverage` of the holdout cells. Otherwise keep the champion.

The decision and both scores are logged, so every promotion can be audited.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from ..ml import models as M
from ..ml.train import BASELINES, split_frames


@dataclass
class Decision:
    promote: bool
    reason: str
    champion_error_pct: float
    challenger_error_pct: float
    challenger_coverage: float
    challenger_model: str
    adopted_cells: int
    holdout_cells: int


def _channel(c: str) -> int:
    return int(c.split("c")[-1])


def split_new_batch(new: pd.DataFrame, holdout_share: float = 0.5) -> tuple[pd.DataFrame, pd.DataFrame]:
    new = new.assign(_ch=new.cell_id.map(_channel)).sort_values("_ch")
    step = max(2, int(round(1 / holdout_share)))
    holdout_mask = (np.arange(len(new)) % step) == 0
    return new[~holdout_mask].drop(columns="_ch"), new[holdout_mask].drop(columns="_ch")


def train_challenger(train: pd.DataFrame, seed: int = 0) -> M.LifeModel:
    best = None
    for name, (cols, est) in M.model_specs(seed).items():
        if name in BASELINES:
            continue
        m = M.fit(name, cols, est, train, seed=seed)
        if best is None or m.cv_rmse_log10 < best.cv_rmse_log10:
            best = m
    return best


def evaluate(champion: M.LifeModel, features: pd.DataFrame, new_batch: str, min_improvement: float = 0.05,
             min_coverage: float = 0.75, seed: int = 0) -> tuple[Decision, M.LifeModel]:
    parts = split_frames(features[~features.batch.eq(new_batch)])
    new = features[features.batch.eq(new_batch)].dropna(subset=["cycle_life"])
    adopt, holdout = split_new_batch(new)
    challenger = train_challenger(pd.concat([parts["train"], adopt], ignore_index=True), seed)

    champ = M.metrics(champion, holdout)
    chall = M.metrics(challenger, holdout)
    better = chall["mean_abs_pct_error"] < (1 - min_improvement) * champ["mean_abs_pct_error"]
    covered = chall["interval90_coverage"] >= min_coverage
    if better and covered:
        reason = "challenger is more accurate on the new batch and its intervals are calibrated"
    elif not better:
        reason = f"challenger not at least {min_improvement:.0%} more accurate"
    else:
        reason = f"challenger intervals cover only {chall['interval90_coverage']:.0%} of holdout cells"
    decision = Decision(better and covered, reason, champ["mean_abs_pct_error"], chall["mean_abs_pct_error"],
                        chall["interval90_coverage"], challenger.name, len(adopt), len(holdout))
    return decision, challenger


def decision_dict(d: Decision) -> dict:
    return asdict(d)
