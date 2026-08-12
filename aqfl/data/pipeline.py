"""End-to-end preparation of validated, leakage-safe station caches."""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from aqfl.config import resolve_data_root, resolve_project_path
from aqfl.data.manifest import validate_and_load_raw
from aqfl.data.preprocessing import (
    fit_global_scaler,
    fit_station_imputer,
    scale_features,
    scale_target,
    transform_station,
)
from aqfl.data.schema import (
    MISSING_SOURCE_COLUMNS,
    NONNEGATIVE_COLUMNS,
    TARGET_COLUMN,
    WIND_DEGREES,
)
from aqfl.data.windows import WindowSplit, build_window_split


def _mask(frame: pd.DataFrame, start: str, end: str) -> np.ndarray:
    timestamps = frame["timestamp"]
    return ((timestamps >= pd.Timestamp(start)) & (timestamps <= pd.Timestamp(end))).to_numpy()


def _save_split(path: Path, split: WindowSplit) -> None:
    path.mkdir(parents=True, exist_ok=True)
    np.save(path / "x.npy", split.x, allow_pickle=False)
    np.save(path / "y.npy", split.y, allow_pickle=False)
    np.save(path / "y_raw.npy", split.y_raw, allow_pickle=False)
    np.save(path / "target_timestamp_ns.npy", split.target_timestamp_ns, allow_pickle=False)
    np.save(path / "persistence.npy", split.persistence, allow_pickle=False)
    np.save(path / "seasonal_naive.npy", split.seasonal_naive, allow_pickle=False)


def _split_boundaries(config: dict[str, Any]) -> dict[str, dict[str, str]]:
    return {name: dict(config["data"][name]) for name in ("train", "val", "test")}


def prepare_dataset(config: dict[str, Any]) -> dict[str, Any]:
    raw_root = resolve_data_root(config)
    manifest_path = resolve_project_path(config, "data/manifest.json")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "preparing",
                "raw_root": str(raw_root),
                "prepared_at_utc": None,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    frames, file_records = validate_and_load_raw(config, raw_root)
    boundaries = _split_boundaries(config)

    states = {}
    features = {}
    observed = {}
    train_masks = {}
    transformed_frames = {}
    max_ffill = int(config["data"]["max_forward_fill_hours"])

    for station, raw_frame in frames.items():
        train_mask = _mask(raw_frame, **boundaries["train"])
        train_masks[station] = train_mask
        state = fit_station_imputer(station, raw_frame.loc[train_mask])
        feature_frame, target_observed = transform_station(raw_frame, state, max_ffill)
        states[station] = state
        features[station] = feature_frame
        observed[station] = target_observed
        transformed_frames[station] = raw_frame.copy()
        transformed_frames[station][TARGET_COLUMN] = feature_frame[TARGET_COLUMN].to_numpy()

    scaler = fit_global_scaler(features, frames, train_masks)
    cache_root = resolve_project_path(config, config["data"]["cache_dir"])
    cache_root.mkdir(parents=True, exist_ok=True)
    quality_rows = []
    missing_rows = []
    split_counts: dict[str, dict[str, int]] = {}
    window = int(config["data"]["window"])
    horizon = int(config["data"]["horizon"])
    stride = int(config["data"]["stride"])

    for station, raw_frame in frames.items():
        for column in MISSING_SOURCE_COLUMNS:
            if column in NONNEGATIVE_COLUMNS:
                invalid_count = int((raw_frame[column] < 0).sum())
            elif column == "wd":
                invalid_count = int((raw_frame[column].notna() & ~raw_frame[column].isin(WIND_DEGREES)).sum())
            else:
                invalid_count = 0
            missing_rows.append(
                {
                    "station": station,
                    "variable": column,
                    "rows": len(raw_frame),
                    "raw_missing": int(raw_frame[column].isna().sum()),
                    "invalid_negative": invalid_count,
                    "raw_missing_rate": float(raw_frame[column].isna().mean()),
                }
            )
        feature_scaled = scale_features(features[station], scaler)
        target_raw = transformed_frames[station][TARGET_COLUMN].to_numpy(dtype=np.float32)
        target_scaled = scale_target(target_raw, scaler)
        split_counts[station] = {}
        for split_name, interval in boundaries.items():
            split = build_window_split(
                feature_scaled,
                target_raw,
                observed[station],
                raw_frame["timestamp"],
                target_scaled,
                interval["start"],
                interval["end"],
                window,
                horizon,
                stride,
            )
            _save_split(cache_root / station / split_name, split)
            split_counts[station][split_name] = len(split.x)
            if len(split.target_timestamp_ns):
                quality_rows.append(
                    {
                        "station": station,
                        "split": split_name,
                        "samples": len(split.x),
                        "start": str(pd.Timestamp(split.target_timestamp_ns.min())),
                        "end": str(pd.Timestamp(split.target_timestamp_ns.max())),
                        "target_missing_excluded": int(
                            ((raw_frame["timestamp"] >= interval["start"])
                             & (raw_frame["timestamp"] <= interval["end"])
                             & raw_frame[TARGET_COLUMN].isna()).sum()
                        ),
                    }
                )

    metadata = {
        "schema_version": 1,
        "prepared_at_utc": datetime.now(UTC).isoformat(),
        "stations": list(frames),
        "feature_dim": len(scaler.feature_columns),
        "feature_columns": scaler.feature_columns,
        "window": window,
        "horizon": horizon,
        "splits": boundaries,
        "split_counts": split_counts,
        "scaler": asdict(scaler),
        "imputers": {station: asdict(state) for station, state in states.items()},
    }
    (cache_root / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(quality_rows).to_csv(cache_root / "data_quality_report.csv", index=False)
    pd.DataFrame(missing_rows).to_csv(cache_root / "station_missingness_report.csv", index=False)

    manifest = {
        "schema_version": 1,
        "status": "prepared",
        "prepared_at_utc": metadata["prepared_at_utc"],
        "dataset": "Beijing Multi-Site Air Quality",
        "doi": "10.24432/C5RK5G",
        "license": "CC BY 4.0",
        "raw_root": str(raw_root),
        "files": file_records,
        "total_rows": sum(record["rows"] for record in file_records),
        "station_count": len(frames),
        "cache_metadata": str(cache_root / "metadata.json"),
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest
