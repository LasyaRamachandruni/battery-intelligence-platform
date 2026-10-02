"""The `bip` command.

    bip ingest --source severson     # real data: the three .mat files in data/raw/
    bip ingest --source synthetic    # synthetic cells, no download needed
    bip transform                    # dbt build: models + data tests
    bip quality                      # SPC, curve anomalies, Weibull reliability
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
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
