"""CLI for validating raw files and creating station caches."""

from __future__ import annotations

import argparse
import json

from aqfl.config import load_config
from aqfl.data.pipeline import prepare_dataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base.yaml")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    manifest = prepare_dataset(load_config(args.config))
    print(json.dumps({"status": manifest["status"], "stations": manifest["station_count"], "rows": manifest["total_rows"]}, indent=2))


if __name__ == "__main__":
    main()
