# Architecture

```
                         ┌──────────────────────────── batch path ─────────────────────────────┐
 .mat files (Severson)   │                                                                      │
 or synthetic cells ──▶ ingest ──▶ Parquet lakehouse ──▶ dbt (DuckDB) ──▶ quality ──▶ reports    │
                         │   validate      │ cells           staging          SPC        dashboard │
                         │   quarantine    │ cycle_summary   marts            anomalies            │
                         │                 │ curves          data tests       Weibull              │
                         │                 ▼                                                       │
                         │             features ──▶ train ──▶ MLflow registry ──▶ API (FastAPI)    │
                         │                            ▲          (champion)                         │
                         │                 drift ─────┤                                            │
                         │                 retrain ───┘  champion vs challenger                    │
                         └──────────────────────────────────────────────────────────────────────┘
                         ┌──────────────────────────── stream path ────────────────────────────┐
 cycler replay ──▶ Kafka `cycler.cycles` ──▶ consumer ──▶ `quality.alerts` (cycle 2, cycle 10)   │
                                              (state per cell) ──▶ `predictions.cycle_life` (cycle 100)
                         └──────────────────────────────────────────────────────────────────────┘
```

## Data platform (`bip/sources`, `bip/lake.py`, `dbt/`)

**Sources.** `sources.severson` reads the three MATLAB v7.3 batch files with h5py
and applies the cleaning from the paper's own code: five batch-1 cells with no
usable life are dropped, five batch-2 cells that were moved from batch 1
mid-test are stitched back onto their batch-1 history, and six batch-3 cells
with bad channels are dropped, which leaves 124 cells. `sources.synthetic` produces
the same `CellRecord` structure with known ground truth, so tests and CI run
without the 3 GB download.

**Lakehouse.** Parquet, one file per batch:

| table | grain | notes |
|---|---|---|
| `cells.parquet` | cell | batch, charging policy, cycle life, source |
| `cycle_summary/<batch>.parquet` | cell × cycle | capacity, resistance, temperatures, charge time |
| `curves/<batch>.parquet` | cell × cycle | Q(V) on a fixed 1000-point grid, `list<float32>`; every cycle up to 100, then every 50th |
| `quarantine.parquet` | rejected record | reason, value |

Validation is applied on write. A cycle row with an out-of-range value (zero capacity,
resistance spikes, impossible temperatures) goes to quarantine and is not dropped silently.
Curves are rejected for wrong length, non-finite values, non-monotonic
cumulative capacity, or implausible totals. `write_stats.json` records counts.

**Warehouse.** dbt-duckdb reads the Parquet files in place as external sources.
- `stg_cells`, `stg_cycles`: typing, plus capacity retention relative to cycle 2.
- `cell_lifecycle`: one row per cell with early-life metrics (cycle-2
  capacity, early fade slope, first resistance, charge time) and milestones
  (cycles to 95% and 90% retention).
- `batch_quality`: per-batch means, spreads and life percentiles.
- `capacity_fade`: the fade curves, downsampled for plotting.

There are 21 data tests: uniqueness and not-null checks, generic `between` and
`expression_is_true` tests, and a singular test checking that each cell's
recorded cycle life is consistent with its fade curve.

## Quality and reliability (`bip/quality`)

- **SPC** (`spc.py`): individuals / moving-range charts. Limits come from a reference
  batch and use MR̄/1.128 for sigma. Western Electric rules 1–4 are applied, and Cp/Cpk
  are computed against spec limits. Cells are charted in channel order, which is the
  closest proxy for build order the data has.
- **Curve anomalies** (`anomaly.py`): curves are normalized and binned to 100 points,
  then modeled with PCA (95% variance, at most 8 components). Each cell gets
  Hotelling T² and Q-residual scores. Thresholds are 99th percentiles of
  *out-of-fold* scores. In-sample thresholds flagged every new cell, because the
  model fits its own training noise.
- **Reliability** (`reliability.py`): Weibull MLE with right censoring.
  It reports B10 life with a delta-method confidence interval, a
  likelihood-ratio test of whether batches share one distribution, Kaplan–Meier
  curves, and the fraction expected to fail within a warranty horizon.

## Lifetime model (`bip/ml`)

Features come from cycles 1–100 only. A test guarantees this: changing every value after cycle 100 leaves
the features identical.
- **Curve features**: log variance, minimum, skew and kurtosis of
  ΔQ(V) = Q₁₀₀(V) − Q₁₀(V). This is the signal from the paper.
- **Summary features**: linear fade slope and intercept over cycles 2–100, cycle-2
  capacity, max − cycle-2 capacity, minimum resistance and resistance change, charge time over the first five cycles, temperature
  integral.

`cell_features()` is the one implementation of these features. Training, the API
and the stream consumer all call it.

The models are two baselines (mean, capacity-only), three elastic nets on the paper's feature
sets (variance, discharge, full) and gradient boosting. Every model predicts log₁₀ cycle life.
The split is the paper's: 41 train / 43 primary test / 40 secondary test cells. **Selection uses
5-fold cross-validated error on the training cells only**, so test labels never
choose the model. A test checks this by shuffling test labels and asserting that the selection
doesn't change.

Intervals are cross-conformal, built from the absolute out-of-fold residuals in log space.
The training run also writes an early-window study (how accuracy changes with 20…100
cycles of data) and permutation importance.

## MLOps (`bip/mlops`)

- **Registry** (`registry.py`). Every training run is logged to MLflow with its parameters,
  per-split metrics and report files. The model is logged as a pyfunc that returns
  predictions with intervals, and registered as `battery-cycle-life`. Production is
  whichever version carries the alias `champion`, so promotion and rollback each take one alias move.
- **Drift** (`drift.py`). Each model feature gets a PSI and a two-sample KS test against the training cells,
  and the predicted-life distribution gets the same. Once labels arrive, the report also checks error and
  interval coverage. Battery programs have tens of cells per batch. At that
  size, textbook PSI calls two samples of the *same* population a major shift
  more than 15% of the time. So bins scale with sample size, counts are smoothed, and a
  retrain is recommended only when the shift is both large (PSI > 0.25) and
  significant (KS p < 0.01). In simulation at n = 40, that rule gives under 1% false alarms and
  catches a one-standard-deviation shift about 85% of the time.
- **Retraining** (`retrain.py`). A new batch's labelled cells are split by channel
  into `adopt` and `holdout`. A challenger is trained on the original training
  cells plus `adopt`, with the same cross-validated selection as the original model. Champion and challenger
  are both scored on `holdout`, which neither has seen. The challenger is promoted only if
  it is at least 5% more accurate and its 90% intervals still cover ≥ 75% of
  the holdout cells.
- **Serving** (`serve.py`). FastAPI with pydantic validation. `/predict/features`
  takes precomputed features. `/predict/early-cycles` takes raw Q(V) curves and
  cycle summaries, computes the features, and returns 422 on malformed input. The API loads
  the registry champion or a joblib file.

## Streaming (`bip/stream`)

A cycler emits one record per cell per cycle. `cycler.replay` turns stored data
into that feed in wall-clock order, with cycle 1 of every cell first, then cycle 2, and so on. The feed goes
through a broker interface with two implementations: Kafka (Redpanda in
`docker-compose.yml`) and in-memory (tests). Messages are keyed by cell ID, so
each cell's cycles stay on one partition in order.

The consumer keeps the minimum state per cell. At **cycle 2** it checks capacity
against the spec and the SPC limits. At **cycle 10** it scores the discharge curve with the
anomaly model. At **cycle 100** it computes features and publishes a prediction with its
interval. After the prediction it frees that cell's state. Delivery is at least once, so the consumer
ignores redelivered (cell, cycle) messages. Tests check three things:
- stream predictions match the batch pipeline;
- every duplicate is counted and ignored;
- an out-of-spec cell raises exactly one alert.
