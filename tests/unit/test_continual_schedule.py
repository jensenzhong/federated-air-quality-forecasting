from __future__ import annotations

import pytest

from aqfl.data.continual_schedule import (
    BENCHMARK_BASE_END,
    BENCHMARK_BASE_START,
    benchmark_task_schedule,
    task_key,
    validate_task_schedule,
)


def test_supplied_benchmark_schedule_is_exactly_11_disjoint_windows() -> None:
    tasks = benchmark_task_schedule()
    assert len(tasks) == 11
    assert tasks[0].start.isoformat() == "2014-05-05T00:00:00"
    assert tasks[-1].end.isoformat() == "2017-02-02T23:00:00"
    assert BENCHMARK_BASE_START.isoformat() == "2013-05-01T00:00:00"
    assert BENCHMARK_BASE_END < tasks[0].start


def test_task_schedule_and_keys_fail_closed() -> None:
    tasks = benchmark_task_schedule()
    with pytest.raises(ValueError, match="exactly 11"):
        validate_task_schedule(tasks[:-1])
    assert task_key(11, "test") == "task_11_test"
    with pytest.raises(ValueError, match="Unknown continual task"):
        task_key(12, "test")
