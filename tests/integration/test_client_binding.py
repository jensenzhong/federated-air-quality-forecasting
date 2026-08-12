from __future__ import annotations

import json

import pytest
from flwr.app import Context, RecordDict

from aqfl.federated.client_app import _station


def config_with_metadata(tmp_path, stations: list[str]) -> dict:
    cache = tmp_path / "data" / "cache"
    cache.mkdir(parents=True)
    (cache / "metadata.json").write_text(json.dumps({"stations": stations}), encoding="utf-8")
    config_path = tmp_path / "configs" / "base.yaml"
    config_path.parent.mkdir()
    config_path.write_text("project: {}\ndata: {}\n", encoding="utf-8")
    return {"_config_path": str(config_path), "data": {"cache_dir": "data/cache"}}


@pytest.mark.integration
def test_partition_is_bound_to_exact_station(tmp_path) -> None:
    config = config_with_metadata(tmp_path, ["a", "b"])
    context = Context(1, 1, {"partition-id": 1, "num-partitions": 2, "station": "b"}, RecordDict(), {})
    assert _station(context, config) == "b"


@pytest.mark.integration
def test_duplicate_missing_or_wrong_station_fails(tmp_path) -> None:
    config = config_with_metadata(tmp_path, ["a", "b"])
    wrong_count = Context(1, 1, {"partition-id": 0, "num-partitions": 3}, RecordDict(), {})
    with pytest.raises(ValueError, match="partitions"):
        _station(wrong_count, config)
    wrong_station = Context(1, 1, {"partition-id": 0, "num-partitions": 2, "station": "b"}, RecordDict(), {})
    with pytest.raises(ValueError, match="binding mismatch"):
        _station(wrong_station, config)
