"""Raw-file discovery, validation, hashing, and manifest generation."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from aqfl.data.schema import RAW_COLUMNS


@dataclass(frozen=True)
class RawFileRecord:
    path: str
    name: str
    station: str
    size_bytes: int
    sha256: str
    rows: int
    start: str
    end: str
    missing_total: int
    target_missing: int


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def discover_raw_files(root: Path, pattern: str = "PRSA_Data_*.csv") -> list[Path]:
    """Discover only official per-station PRSA files."""
    files = sorted(path.resolve() for path in root.rglob(pattern) if path.is_file())
    unrelated = {"data.csv", "test.csv"}
    if any(path.name.lower() in unrelated for path in files):
        raise AssertionError("Unrelated top-level CSV was selected")
    return files


def inspect_raw_file(path: Path, expected_rows: int | None = None) -> tuple[pd.DataFrame, RawFileRecord]:
    frame = pd.read_csv(path)
    if list(frame.columns) != RAW_COLUMNS:
        raise ValueError(f"Unexpected schema in {path.name}: {list(frame.columns)}")
    if expected_rows is not None and len(frame) != expected_rows:
        raise ValueError(f"Expected {expected_rows} rows in {path.name}, found {len(frame)}")
    stations = frame["station"].dropna().astype(str).unique().tolist()
    if len(stations) != 1:
        raise ValueError(f"Each raw file must contain exactly one station: {path.name}")
    timestamp = pd.to_datetime(frame[["year", "month", "day", "hour"]])
    if timestamp.duplicated().any():
        raise ValueError(f"Duplicate hourly keys in {path.name}")
    expected = pd.date_range(timestamp.min(), timestamp.max(), freq="h")
    if len(expected) != len(timestamp) or not (timestamp.to_numpy() == expected.to_numpy()).all():
        raise ValueError(f"Hourly sequence is incomplete or unordered in {path.name}")
    frame = frame.copy()
    frame["timestamp"] = timestamp
    record = RawFileRecord(
        path=str(path),
        name=path.name,
        station=stations[0],
        size_bytes=path.stat().st_size,
        sha256=sha256_file(path),
        rows=len(frame),
        start=str(timestamp.min()),
        end=str(timestamp.max()),
        missing_total=int(frame.isna().sum().sum()),
        target_missing=int(frame["PM2.5"].isna().sum()),
    )
    return frame, record


def validate_and_load_raw(config: dict[str, Any], root: Path) -> tuple[dict[str, pd.DataFrame], list[dict[str, Any]]]:
    data_cfg = config["data"]
    files = discover_raw_files(root, str(data_cfg.get("file_glob", "PRSA_Data_*.csv")))
    expected_files = int(data_cfg["expected_files"])
    if len(files) != expected_files:
        raise ValueError(f"Expected {expected_files} PRSA files, found {len(files)}")

    station_frames: dict[str, pd.DataFrame] = {}
    records: list[dict[str, Any]] = []
    for path in files:
        frame, record = inspect_raw_file(path, int(data_cfg["expected_rows_per_station"]))
        if record.start != str(pd.Timestamp(data_cfg["expected_start"])):
            raise ValueError(f"Unexpected start timestamp in {path.name}: {record.start}")
        if record.end != str(pd.Timestamp(data_cfg["expected_end"])):
            raise ValueError(f"Unexpected end timestamp in {path.name}: {record.end}")
        if record.station in station_frames:
            raise ValueError(f"Duplicate station across files: {record.station}")
        station_frames[record.station] = frame
        records.append(asdict(record))

    total_rows = sum(len(frame) for frame in station_frames.values())
    if total_rows != int(data_cfg["expected_total_rows"]):
        raise ValueError(f"Expected {data_cfg['expected_total_rows']} rows, found {total_rows}")
    return dict(sorted(station_frames.items())), records
