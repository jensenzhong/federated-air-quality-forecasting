from __future__ import annotations

import json

import numpy as np
import pytest
from conftest import write_prsa

from aqfl.data.dataset import load_cache_metadata, load_station_dataset
from aqfl.data.pipeline import prepare_dataset


@pytest.mark.integration
def test_synthetic_pipeline_writes_leakage_safe_cache(tmp_path, raw_frame, monkeypatch) -> None:
    raw_root = tmp_path / "raw"
    raw_root.mkdir()
    write_prsa(raw_root / "PRSA_Data_A.csv", raw_frame)
    monkeypatch.setenv("SYNTHETIC_AQ_ROOT", str(raw_root))
    config_path = tmp_path / "configs" / "base.yaml"
    config_path.parent.mkdir()
    config_path.touch()
    config = {
        "_config_path": str(config_path),
        "data": {
            "root_env": "SYNTHETIC_AQ_ROOT",
            "file_glob": "PRSA_Data_*.csv",
            "expected_files": 1,
            "expected_rows_per_station": 72,
            "expected_total_rows": 72,
            "expected_start": "2013-03-01 00:00:00",
            "expected_end": "2013-03-03 23:00:00",
            "window": 24,
            "horizon": 1,
            "stride": 1,
            "max_forward_fill_hours": 6,
            "cache_dir": "data/cache",
            "train": {"start": "2013-03-01 00:00:00", "end": "2013-03-02 23:00:00"},
            "val": {"start": "2013-03-03 00:00:00", "end": "2013-03-03 11:00:00"},
            "test": {"start": "2013-03-03 12:00:00", "end": "2013-03-03 23:00:00"},
        },
    }
    manifest = prepare_dataset(config)
    assert manifest["status"] == "prepared"
    assert manifest["total_rows"] == 72
    metadata = load_cache_metadata(config)
    assert metadata["feature_dim"] == 31
    train = load_station_dataset(config, "Aotizhongxin", "train")
    val = load_station_dataset(config, "Aotizhongxin", "val")
    test = load_station_dataset(config, "Aotizhongxin", "test")
    assert train.x.shape[1:] == (24, 31)
    assert set(np.asarray(train.target_timestamp_ns)).isdisjoint(np.asarray(val.target_timestamp_ns))
    assert set(np.asarray(val.target_timestamp_ns)).isdisjoint(np.asarray(test.target_timestamp_ns))
    assert train[0][0].shape == (24, 31)
    quality = tmp_path / "data" / "cache" / "data_quality_report.csv"
    missingness = tmp_path / "data" / "cache" / "station_missingness_report.csv"
    assert quality.is_file()
    assert missingness.is_file()
    disk_manifest = json.loads((tmp_path / "data" / "manifest.json").read_text(encoding="utf-8"))
    assert disk_manifest["station_count"] == 1
