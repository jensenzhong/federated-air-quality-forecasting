from __future__ import annotations

import pytest
from conftest import write_prsa

from aqfl.data.manifest import discover_raw_files, inspect_raw_file, sha256_file


def test_selector_ignores_stock_csv(tmp_path, raw_frame) -> None:
    nested = tmp_path / "official"
    nested.mkdir()
    official = nested / "PRSA_Data_A.csv"
    write_prsa(official, raw_frame)
    (tmp_path / "data.csv").write_text("date,close\n2020-01-01,1\n", encoding="utf-8")
    (tmp_path / "test.csv").write_text("date,close\n2020-01-02,2\n", encoding="utf-8")
    assert discover_raw_files(tmp_path) == [official.resolve()]


def test_inspect_validates_station_time_and_hash(tmp_path, raw_frame) -> None:
    path = tmp_path / "PRSA_Data_A.csv"
    write_prsa(path, raw_frame)
    loaded, record = inspect_raw_file(path, 72)
    assert record.station == "Aotizhongxin"
    assert len(record.sha256) == 64
    assert record.sha256 == sha256_file(path)
    assert loaded["timestamp"].is_monotonic_increasing


def test_inspect_rejects_duplicate_hour(tmp_path, raw_frame) -> None:
    broken = raw_frame.copy()
    broken.loc[1, ["year", "month", "day", "hour"]] = broken.loc[0, ["year", "month", "day", "hour"]]
    path = tmp_path / "PRSA_Data_A.csv"
    write_prsa(path, broken)
    with pytest.raises(ValueError, match="Duplicate"):
        inspect_raw_file(path)


def test_inspect_rejects_mixed_station_and_schema(tmp_path, raw_frame) -> None:
    mixed = raw_frame.copy()
    mixed.loc[0, "station"] = "Other"
    path = tmp_path / "PRSA_Data_A.csv"
    write_prsa(path, mixed)
    with pytest.raises(ValueError, match="one station"):
        inspect_raw_file(path)
    bad = tmp_path / "PRSA_Data_B.csv"
    raw_frame.drop(columns=["station", "timestamp"]).to_csv(bad, index=False)
    with pytest.raises(ValueError, match="schema"):
        inspect_raw_file(bad)


def test_inspect_rejects_wrong_rows(tmp_path, raw_frame) -> None:
    path = tmp_path / "PRSA_Data_A.csv"
    write_prsa(path, raw_frame)
    with pytest.raises(ValueError, match="Expected 71"):
        inspect_raw_file(path, 71)
