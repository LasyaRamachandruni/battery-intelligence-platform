"""Battery Intelligence dashboard.

    streamlit run dashboard/app.py

Reads the warehouse built by `bip transform` and the reports written by
`bip quality`, `bip train` and `bip drift`. Point it at another run with
BIP_DATA_DIR / BIP_REPORTS_DIR.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import altair as alt
import duckdb
import pandas as pd
import streamlit as st

DATA = Path(os.environ.get("BIP_DATA_DIR", "data"))
REPORTS = Path(os.environ.get("BIP_REPORTS_DIR", "reports"))
SPLIT_COLORS = alt.Scale(domain=["train", "test_primary", "test_secondary"],
                         range=["#9aa0a6", "#2a78d6", "#eb6834"])
BATCH_COLORS = alt.Scale(domain=["b1", "b2", "b3"], range=["#2a78d6", "#eb6834", "#1baf7a"])

st.set_page_config(page_title="Battery Intelligence", layout="wide")


@st.cache_data
def query(sql: str) -> pd.DataFrame:
    with duckdb.connect(str(DATA / "warehouse.duckdb"), read_only=True) as con:
        return con.sql(sql).df()


def read_csv(path: str) -> pd.DataFrame | None:
    p = REPORTS / path
    return pd.read_csv(p) if p.exists() else None


def read_json(path: str) -> dict | None:
    p = REPORTS / path
    return json.loads(p.read_text()) if p.exists() else None


if not (DATA / "warehouse.duckdb").exists():
    st.error("No warehouse found. Run `bip ingest`, `bip transform`, `bip quality` and `bip train` first.")
    st.stop()

cells = query("select * from cell_lifecycle")
synthetic = set(cells.source) == {"synthetic"}

st.title("Battery Intelligence")
st.caption(("Synthetic cells generated to exercise the pipeline, not real measurements. "
            if synthetic else "Severson et al. 2019 LFP/graphite cells (124 cells, 3 batches). ")
           + f"{len(cells)} cells · {cells.batch.nunique()} batches")

overview, quality, model, mlops = st.tabs(["Fleet", "Quality", "Cycle-life model", "Monitoring"])

with overview:
    bq = query("select * from batch_quality order by batch")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Cells", len(cells))
    c2.metric("Median cycle life", f"{cells.cycle_life.median():.0f}")
    c3.metric("Shortest / longest", f"{cells.cycle_life.min():.0f} / {cells.cycle_life.max():.0f}")
    c4.metric("Mean capacity at cycle 2", f"{cells.q_cycle2_ah.mean():.3f} Ah")

    st.subheader("Capacity fade")
    batches = st.multiselect("Batches", sorted(cells.batch.unique()), default=sorted(cells.batch.unique()))
    fade = query("select cell_id, batch, cycle, capacity_retention from capacity_fade")
    fade = fade[fade.batch.isin(batches)]
    st.altair_chart(
        alt.Chart(fade).mark_line(opacity=0.45, strokeWidth=1).encode(
            x=alt.X("cycle:Q", title="Cycle"),
            y=alt.Y("capacity_retention:Q", title="Capacity retention vs cycle 2", scale=alt.Scale(zero=False)),
            color=alt.Color("batch:N", scale=BATCH_COLORS),
            detail="cell_id:N", tooltip=["cell_id", "cycle", alt.Tooltip("capacity_retention", format=".3f")],
        ).properties(height=380),
        width="stretch",
    )
    st.subheader("Batch summary")
    st.dataframe(bq, hide_index=True, width="stretch")

with quality:
    spc = read_csv("quality/spc_summary.csv")
    if spc is None:
        st.info("Run `bip quality` to populate this tab.")
    else:
        st.subheader("Statistical process control")
        st.caption("Control limits are set on batch b1; every batch is charted against them in channel order.")
        st.dataframe(spc.round(4), hide_index=True, width="stretch")
        cols = st.columns(2)
        for col, name in zip(cols, ["q_cycle2_ah", "ir_early_first_ohm"]):
            img = REPORTS / "quality" / f"control_chart_{name}.png"
            if img.exists():
                col.image(str(img))
        alerts = read_csv("quality/spc_alerts.csv")
        if alerts is not None and len(alerts):
            st.markdown("**Rule violations** (Western Electric rules 1-4)")
            st.dataframe(alerts, hide_index=True, width="stretch")

        st.subheader("Discharge-curve anomalies")
        anomalies = read_csv("quality/anomalies.csv")
        if anomalies is not None:
            flagged = anomalies[anomalies.flagged]
            st.write(f"{len(flagged)} of {len(anomalies)} cells flagged by the PCA T² / Q-residual model "
                     "(99% out-of-fold limits).")
            if len(flagged):
                st.dataframe(flagged, hide_index=True)

        st.subheader("Reliability (Weibull)")
        wb = read_json("quality/weibull.json")
        if wb:
            rows = [{"group": g, **{k: v for k, v in d.items() if k != "b10_95ci"},
                     "b10_95ci": f"{d['b10_95ci'][0]:.0f}-{d['b10_95ci'][1]:.0f}"}
                    for g, d in {**wb["per_batch"], "all": wb["all_cells"]}.items()]
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
            lr = wb.get("batches_differ_lr_test")
            if lr:
                st.caption(f"Likelihood-ratio test that batches share one Weibull: χ²={lr['statistic']:.1f}, "
                           f"df={lr['dof']}, p={lr['p_value']:.1e}")
            img = REPORTS / "quality" / "weibull_probability.png"
            if img.exists():
                st.image(str(img), width=640)

with model:
    metrics = read_json("model/metrics.json")
    preds = read_csv("model/predictions.csv")
    if metrics is None or preds is None:
        st.info("Run `bip train` to populate this tab.")
    else:
        sel = metrics["selected_model"]
        m = metrics["models"][sel]["test_primary"]
        st.subheader(f"Selected model: {sel}")
        st.caption(metrics.get("selection", ""))
        c1, c2, c3 = st.columns(3)
        c1.metric("Test error (mean abs %)", f"{m['mean_abs_pct_error']:.1f}%")
        c2.metric("RMSE", f"{m['rmse_cycles']:.0f} cycles")
        c3.metric("90% interval coverage", f"{m['interval90_coverage']:.0%}")

        table = pd.DataFrame([{"model": k, **{f"{s} %err": v[s]["mean_abs_pct_error"]
                                              for s in ("train", "test_primary", "test_secondary") if s in v},
                               "cv_rmse_log10": v.get("cv_rmse_log10")} for k, v in metrics["models"].items()])
        st.dataframe(table.round(3), hide_index=True, width="stretch")

        st.subheader("Predicted vs observed")
        lim = [0, float(max(preds.cycle_life.max(), preds.upper90.max())) * 1.05]
        base = alt.Chart(preds).encode(x=alt.X("cycle_life:Q", title="Observed cycle life", scale=alt.Scale(domain=lim)))
        st.altair_chart(
            (base.mark_rule(opacity=0.25).encode(y=alt.Y("lower90:Q", scale=alt.Scale(domain=lim)), y2="upper90:Q",
                                                 color=alt.Color("split:N", scale=SPLIT_COLORS))
             + base.mark_circle(size=40).encode(y=alt.Y("predicted:Q", title="Predicted cycle life"), color=alt.Color("split:N", scale=SPLIT_COLORS),
                                                tooltip=["cell_id", "split", "cycle_life", "predicted", "lower90",
                                                         "upper90"])
             + alt.Chart(pd.DataFrame({"x": lim, "y": lim})).mark_line(color="#888", strokeDash=[4, 4])
             .encode(x="x:Q", y="y:Q")).properties(height=420),
            width="stretch",
        )

        st.subheader("Look up a cell")
        cell = st.selectbox("Cell", preds.cell_id)
        row = preds[preds.cell_id == cell].iloc[0]
        st.write(f"**{cell}** ({row.split}): predicted **{row.predicted:.0f}** cycles "
                 f"(90% interval {row.lower90:.0f}-{row.upper90:.0f}), observed {row.cycle_life}.")

        cols = st.columns(2)
        for col, name in zip(cols, ["early_window.png", "importance.png"]):
            img = REPORTS / "model" / name
            if img.exists():
                col.image(str(img))

with mlops:
    reports = sorted((REPORTS / "mlops").glob("drift_*.json")) if (REPORTS / "mlops").exists() else []
    if not reports:
        st.info("Run `bip drift --batch b3` (or `--split test_primary`) to populate this tab.")
    for path in reports:
        r = json.loads(path.read_text())
        status = "Retrain recommended" if r["retrain_recommended"] else "No action needed"
        st.subheader(f"Drift: {r['current']} — {status}")
        for reason in r["reasons"]:
            st.write(f"- {reason}")
        table = pd.DataFrame(r["features"])
        table["ks_pvalue"] = table.ks_pvalue.map(lambda v: f"{v:.1e}")
        st.dataframe(table.round(3), hide_index=True)
        st.caption(f"Prediction PSI {r['prediction_psi']:.2f} · {r['current_cells']} cells vs "
                   f"{r['reference_cells']} training cells")
    for path in sorted((REPORTS / "mlops").glob("retrain_*.json")) if (REPORTS / "mlops").exists() else []:
        d = json.loads(path.read_text())
        st.subheader(f"Champion vs challenger on {d['new_batch']}")
        c1, c2, c3 = st.columns(3)
        c1.metric("Champion error", f"{d['champion_error_pct']:.1f}%")
        c2.metric("Challenger error", f"{d['challenger_error_pct']:.1f}%")
        c3.metric("Decision", "Promote" if d["promote"] else "Keep champion")
        st.caption(d["reason"])
