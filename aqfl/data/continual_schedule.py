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
    if task_id < 1 or task_id > len(BENCHMARK_TASK_BOUNDS):
        raise ValueError("Unknown continual task ID")
    if split not in {"train", "test"}:
        raise ValueError("Continual task split must be train or test")
    return f"task_{task_id}_{split}"
