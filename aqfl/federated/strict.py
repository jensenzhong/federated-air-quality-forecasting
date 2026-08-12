"""Failure-intolerant Flower strategies for formal experiments."""

from __future__ import annotations

from collections.abc import Iterable

from flwr.app import ArrayRecord, Message, MetricRecord
from flwr.serverapp.strategy import FedAdam, FedAvg, FedProx, QFedAvg


class StrictRepliesMixin:
    expected_clients: int
    latest_arrays: ArrayRecord | None
    best_arrays: ArrayRecord | None
    best_macro_mae: float
    best_round: int | None

    def _init_tracking(self) -> None:
        self.latest_arrays = None
        self.best_arrays = None
        self.best_macro_mae = float("inf")
        self.best_round = None

    def _require_all(self, replies: Iterable[Message], is_train: bool) -> list[Message]:
        replies_list = list(replies)
        valid, errors = self._check_and_log_replies(replies_list, is_train=is_train)  # type: ignore[attr-defined]
        if errors or len(valid) != self.expected_clients:
            raise RuntimeError(
                f"Formal round invalid: expected {self.expected_clients} successful clients, "
                f"received {len(valid)} successes and {len(errors)} failures"
            )
        partition_ids = []
        for reply in valid:
            metrics = next(iter(reply.content.metric_records.values()))
            if "partition-id" in metrics:
                partition_ids.append(int(metrics["partition-id"]))
        if partition_ids and sorted(partition_ids) != list(range(self.expected_clients)):
            raise RuntimeError(
                "Formal round invalid: partition IDs must be unique and complete; "
                f"received {sorted(partition_ids)}"
            )
        return valid

    def aggregate_train(self, server_round: int, replies: Iterable[Message]) -> tuple[ArrayRecord | None, MetricRecord | None]:
        valid = self._require_all(replies, is_train=True)
        result = super().aggregate_train(server_round, valid)  # type: ignore[misc]
        self.latest_arrays = result[0]
        return result

    def aggregate_evaluate(self, server_round: int, replies: Iterable[Message]) -> MetricRecord | None:
        valid = self._require_all(replies, is_train=False)
        metrics = super().aggregate_evaluate(server_round, valid)  # type: ignore[misc]
        if metrics is not None and self.latest_arrays is not None:
            macro = float(metrics["macro_mae"])
            if macro < self.best_macro_mae:
                self.best_macro_mae = macro
                self.best_arrays = self.latest_arrays
                self.best_round = server_round
        return metrics


class StrictFedAvg(StrictRepliesMixin, FedAvg):
    def __init__(self, *, expected_clients: int, **kwargs: object) -> None:
        self.expected_clients = expected_clients
        self._init_tracking()
        super().__init__(**kwargs)


class StrictFedProx(StrictRepliesMixin, FedProx):
    def __init__(self, *, expected_clients: int, **kwargs: object) -> None:
        self.expected_clients = expected_clients
        self._init_tracking()
        super().__init__(**kwargs)


class StrictQFedAvg(StrictRepliesMixin, QFedAvg):
    def __init__(self, *, expected_clients: int, **kwargs: object) -> None:
        self.expected_clients = expected_clients
        self._init_tracking()
        super().__init__(**kwargs)


class StrictFedAdam(StrictRepliesMixin, FedAdam):
    def __init__(self, *, expected_clients: int, **kwargs: object) -> None:
        self.expected_clients = expected_clients
        self._init_tracking()
        super().__init__(**kwargs)
