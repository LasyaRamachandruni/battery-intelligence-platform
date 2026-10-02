"""Run the quality and reliability analyses and write tables + charts.

Outputs (default reports/quality/):
  spc_summary.csv, spc_alerts.csv   control-chart results per batch and metric
  anomalies.csv                     curve anomaly scores per cell
  weibull.json                      Weibull fits per batch and overall, B10, warranty risk
  control_chart_<metric>.png        individuals chart across batches
  weibull_probability.png           Weibull probability plot with fitted lines
"""

from __future__ import annotations

import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from .. import plots
from ..config import Lake
from ..lake import read_curves
from . import anomaly, reliability, spc

METRICS = ("q_cycle2_ah", "ir_early_first_ohm")
LABELS = {"q_cycle2_ah": "Discharge capacity at cycle 2 (Ah)", "ir_early_first_ohm": "Internal resistance (ohm)"}


def _lifecycle(lake: Lake) -> pd.DataFrame:
    con = duckdb.connect(str(lake.warehouse))
    try:
        return con.sql("select * from cell_lifecycle").df()
    finally:
        con.close()


def control_chart_png(df: pd.DataFrame, metric: str, limits: spc.ControlLimits, alerts: list[dict], path: Path, label: str):
    df = df.copy()
    df["channel"] = df.cell_id.str.extract(r"c(\d+)$").astype(int)
    df = df.sort_values(["batch", "channel"]).reset_index(drop=True)
    fig, ax = plots.figure(f"Control chart: {LABELS.get(metric, metric).split(' (')[0].lower()}",
                           f"{label}; limits set on batch b1, cells in channel order", "Cell", LABELS.get(metric, metric))
    for i, (batch, part) in enumerate(df.groupby("batch", sort=True)):
        ax.plot(part.index, part[metric], marker="o", markersize=3.5, linewidth=1,
                color=plots.SERIES[i % 3], label=batch)
    for y, style in ((limits.center, "-"), (limits.ucl, "--"), (limits.lcl, "--")):
        ax.axhline(y, color=plots.INK_2, linewidth=1, linestyle=style)
    for name, y in (("UCL", limits.ucl), ("LCL", limits.lcl)):
        ax.annotate(name, (0, y), xytext=(2, 3), textcoords="offset points", fontsize=8, color=plots.INK_2, va="bottom")
    flagged = df[df.cell_id.isin({a["cell_id"] for a in alerts})]
    ax.scatter(flagged.index, flagged[metric], s=60, facecolors="none", edgecolors=plots.CRITICAL,
               linewidths=1.5, label="Rule violation", zorder=5)
    ax.legend(frameon=False, fontsize=9, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.14))
    plots.save(fig, path)


def weibull_png(groups: dict[str, np.ndarray], fits: dict[str, reliability.WeibullFit], path: Path, label: str):
    fig, ax = plots.figure("Weibull probability plot of cycle life", f"{label}; points: median ranks, lines: ML fits",
                           "Cycle life (log scale)", "ln(-ln(1 - F))")
    for i, (name, life) in enumerate(sorted(groups.items())):
        t = np.sort(life)
        f = (np.arange(1, t.size + 1) - 0.3) / (t.size + 0.4)
        color = plots.SERIES[i % 3]
        ax.scatter(np.log(t), np.log(-np.log(1 - f)), s=14, color=color, label=f"{name}: beta={fits[name].beta:.1f}, eta={fits[name].eta:.0f}")
        grid = np.linspace(t.min() * 0.8, t.max() * 1.1, 50)
        ax.plot(np.log(grid), np.log(-np.log(fits[name].survival(grid))), color=color, linewidth=1.5)
    ticks = [200, 500, 1000, 2000]
    ax.set_xticks(np.log(ticks))
    ax.set_xticklabels([str(t) for t in ticks])
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    plots.save(fig, path)


def run(lake: Lake, out_dir: str | Path, label: str, warranty_cycles: int = 500) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    life = _lifecycle(lake)

    # 1. SPC
    summaries, all_alerts = [], []
    for metric in METRICS:
        limits, summary, alerts = spc.monitor_batches(life, metric)
        summaries.append(summary)
        all_alerts += alerts
        control_chart_png(life, metric, limits, alerts, out / f"control_chart_{metric}.png", label)
    pd.concat(summaries).to_csv(out / "spc_summary.csv", index=False)
    pd.DataFrame(all_alerts, columns=["batch", "cell_id", "metric", "rule", "value"]).to_csv(out / "spc_alerts.csv", index=False)

    # 2. curve anomalies at cycle 10, reference = batch b1
    curves = read_curves(lake, [10])
    ids = sorted({c for c, _ in curves})
    x = np.stack([curves[(c, 10)] for c in ids])
    ref = np.array([c.startswith("b1") for c in ids])
    model = anomaly.fit(x[ref])
    t2, q = model.scores(x)
    flags = model.flag(x)
    pd.DataFrame({"cell_id": ids, "t2": t2, "q_residual": q, "flagged": flags}).to_csv(out / "anomalies.csv", index=False)

    # 3. Weibull per batch and overall
    groups = {b: part.cycle_life.to_numpy(float) for b, part in life.groupby("batch")}
    comparison = reliability.compare_groups({b: (t, np.ones_like(t, bool)) for b, t in groups.items()})
    fits = comparison["fits"]
    pooled = comparison["pooled"]

    def describe(w: reliability.WeibullFit) -> dict:
        lo, hi = w.b_life_interval(0.10)
        return {"beta": round(w.beta, 3), "eta": round(w.eta, 1), "cells": w.n, "failures": w.failures,
                "b10": round(w.b_life(0.10), 1), "b10_95ci": [round(lo, 1), round(hi, 1)],
                f"fail_by_{warranty_cycles}_cycles": round(float(w.cdf(warranty_cycles)), 4)}

    weibull = {"per_batch": {b: describe(f) for b, f in fits.items()}, "all_cells": describe(pooled),
               "batches_differ_lr_test": {"statistic": round(comparison["statistic"], 2), "dof": comparison["dof"],
                                          "p_value": comparison["p_value"]}}
    (out / "weibull.json").write_text(json.dumps(weibull, indent=2))
    weibull_png(groups, fits, out / "weibull_probability.png", label)

    return {"spc_alerts": len(all_alerts), "anomalies_flagged": int(flags.sum()),
            "anomaly_components": int(model.pca.n_components_), "weibull": weibull}
