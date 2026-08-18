from __future__ import annotations

import pytest

from aqfl.data.continual_schedule import (
    BENCHMARK_BASE_END,
    BENCHMARK_BASE_START,
    BENCHMARK_PHASE_COUNT,
    benchmark_evaluation_window,
    benchmark_phase_window,
    benchmark_task_schedule,
    continual_task_id_for_round,
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
    assert BENCHMARK_PHASE_COUNT == 12
    assert benchmark_phase_window(0) == (BENCHMARK_BASE_START, BENCHMARK_BASE_END)
    assert benchmark_evaluation_window(0)[0].isoformat() == "2014-05-04T00:00:00"
    assert task_key(0, "test") == "base_test"
    assert task_key(11, "test") == "task_11_test"
    with pytest.raises(ValueError, match="Unknown continual task"):
        task_key(-1, "test")
    with pytest.raises(ValueError, match="Unknown continual task"):
        task_key(12, "test")


def test_continual_round_mapping_is_deterministic_and_bounded() -> None:
    assert [
        continual_task_id_for_round(round_number)
        for round_number in range(1, 13)
    ] == list(range(12))
    assert continual_task_id_for_round(4, base_rounds=2, rounds_per_task=2) == 1
    with pytest.raises(ValueError, match="exceeds"):
        continual_task_id_for_round(13)
