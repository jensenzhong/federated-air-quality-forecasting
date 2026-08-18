"""Frozen task schedule extracted from the supplied continual-FL benchmark source.

This module contains only public temporal boundaries.  It does not load raw rows,
retain task samples, or define a server-side client protocol.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class ContinualTaskWindow:
    task_id: int
    start: datetime
    end: datetime

    def __post_init__(self) -> None:
        if self.task_id < 1:
            raise ValueError("Continual task IDs start at one")
        if self.start >= self.end:
            raise ValueError("Task window start must precede its end")


BENCHMARK_BASE_START = datetime.fromisoformat("2013-05-01T00:00:00")
BENCHMARK_BASE_END = datetime.fromisoformat("2014-05-03T23:00:00")
BENCHMARK_BASE_TEST_START = datetime.fromisoformat("2014-05-04T00:00:00")
BENCHMARK_BASE_TEST_END = datetime.fromisoformat("2014-05-04T23:00:00")
BENCHMARK_TASK_COUNT = 11
BENCHMARK_PHASE_COUNT = BENCHMARK_TASK_COUNT + 1

BENCHMARK_TASK_BOUNDS: tuple[tuple[str, str], ...] = (
    ("2014-05-05", "2014-08-06"),
    ("2014-08-07", "2014-11-06"),
    ("2014-11-07", "2015-02-02"),
    ("2015-02-03", "2015-05-04"),
    ("2015-05-05", "2015-08-06"),
    ("2015-08-07", "2015-11-06"),
    ("2015-11-07", "2016-02-02"),
    ("2016-02-03", "2016-05-04"),
    ("2016-05-05", "2016-08-06"),
    ("2016-08-07", "2016-11-06"),
    ("2016-11-07", "2017-02-02"),
)


def _inclusive_day_end(value: str) -> datetime:
    return datetime.fromisoformat(f"{value}T23:00:00")


def benchmark_task_schedule() -> tuple[ContinualTaskWindow, ...]:
    tasks = tuple(
        ContinualTaskWindow(
            task_id=index,
            start=datetime.fromisoformat(f"{start}T00:00:00"),
            end=_inclusive_day_end(end),
        )
        for index, (start, end) in enumerate(BENCHMARK_TASK_BOUNDS, start=1)
    )
    validate_task_schedule(tasks)
    return tasks


def benchmark_phase_window(task_id: int) -> tuple[datetime, datetime]:
    """Return the inclusive public time window for base (0) or task 1..11."""
    if task_id == 0:
        return BENCHMARK_BASE_START, BENCHMARK_BASE_END
    tasks = benchmark_task_schedule()
    if not 1 <= task_id <= len(tasks):
        raise ValueError("Unknown continual task ID")
    task = tasks[task_id - 1]
    return task.start, task.end


def benchmark_evaluation_window(task_id: int) -> tuple[datetime, datetime]:
    """Return the base-test or task evaluation window without client state."""
    if task_id == 0:
        return BENCHMARK_BASE_TEST_START, BENCHMARK_BASE_TEST_END
    return benchmark_phase_window(task_id)


def continual_task_id_for_round(
    round_number: int,
    *,
    base_rounds: int = 1,
    rounds_per_task: int = 1,
    task_count: int = BENCHMARK_TASK_COUNT,
) -> int:
    """Map a continual communication round to T0 or one of T1..TN.

    The mapping is deterministic and public.  It is used only to select local
    windows; no client identity or private trajectory is encoded in the task ID.
    """
    if round_number < 1:
        raise ValueError("Communication rounds start at one")
    if base_rounds < 1 or rounds_per_task < 1:
        raise ValueError("Continual phase lengths must be positive")
    if task_count < 1 or task_count > BENCHMARK_TASK_COUNT:
        raise ValueError("Continual task_count is outside the frozen schedule")
    if round_number <= base_rounds:
        return 0
    task_id = (round_number - base_rounds - 1) // rounds_per_task + 1
    if task_id > task_count:
        raise ValueError("Round exceeds the configured continual task schedule")
    return int(task_id)


def validate_task_schedule(tasks: tuple[ContinualTaskWindow, ...]) -> None:
    if len(tasks) != 11:
        raise ValueError("The supplied benchmark schedule must contain exactly 11 tasks")
    if tuple(task.task_id for task in tasks) != tuple(range(1, 12)):
        raise ValueError("Continual task IDs must be contiguous and chronological")
    if tasks[0].start <= BENCHMARK_BASE_END or tasks[0].start <= BENCHMARK_BASE_TEST_END:
        raise ValueError("Continual tasks overlap the benchmark base phase")
    for previous, current in zip(tasks[:-1], tasks[1:], strict=True):
        if current.start <= previous.end:
            raise ValueError("Continual task windows must be disjoint and ordered")


def task_key(task_id: int, split: str) -> str:
    if task_id < 0 or task_id > len(BENCHMARK_TASK_BOUNDS):
        raise ValueError("Unknown continual task ID")
    if split not in {"train", "test"}:
        raise ValueError("Continual task split must be train or test")
    return f"base_{split}" if task_id == 0 else f"task_{task_id}_{split}"
