from __future__ import annotations

import json

import pytest

from aqfl.config import canonical_json, load_config, resolve_data_root, set_seed


def test_load_config_and_stable_json(tmp_path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("a: 1\nb: two\n", encoding="utf-8")
    config = load_config(path)
    assert config["a"] == 1
    assert canonical_json({"b": 1, "a": 2}) == '{"a":2,"b":1}'
    assert json.loads(canonical_json(config))["b"] == "two"


def test_load_config_rejects_non_mapping(tmp_path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("- one\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(path)


def test_resolve_data_root_requires_environment(monkeypatch) -> None:
    monkeypatch.delenv("TEST_AQ_ROOT", raising=False)
    with pytest.raises(RuntimeError):
        resolve_data_root({"data": {"root_env": "TEST_AQ_ROOT"}})


def test_set_seed_repeats_numpy() -> None:
    import numpy as np

    set_seed(7)
    left = np.random.random(3)
    set_seed(7)
    assert np.array_equal(left, np.random.random(3))
