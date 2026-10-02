"""The `bip` command.

    bip ingest --source severson     # real data: the three .mat files in data/raw/
    bip ingest --source synthetic    # synthetic cells, no download needed
    bip transform                    # dbt build: models + data tests
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


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="bip", description="Battery Intelligence Platform")
    p.add_argument("--data-dir", default="data")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("ingest", help="load cells into the Parquet lakehouse")
    s.add_argument("--source", choices=["severson", "synthetic"], default="severson")
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(fn=cmd_ingest)

    sub.add_parser("transform", help="dbt build (models + data tests)").set_defaults(fn=cmd_transform)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
