"""Failure-intolerant Flower strategies for formal experiments."""

from __future__ import annotations

from collections.abc import Iterable

from flwr.app import ArrayRecord, ConfigRecord, Message, MessageType, MetricRecord, RecordDict
from flwr.serverapp.strategy import FedAdam, FedAvg, FedProx, QFedAvg


class DeterministicSchedulingMixin:
    """Schedule every connected node in a stable order for each round."""

    expected_clients: int

    def _deterministic_messages(
        self,
        arrays: ArrayRecord,
        config: ConfigRecord,
        grid: object,
        message_type: str,
    ) -> Iterable[Message]:
        node_ids = sorted(int(node_id) for node_id in grid.get_node_ids())  # type: ignore[attr-defined]
        if len(node_ids) != self.expected_clients:
            raise RuntimeError(
                f"Formal round invalid: expected {self.expected_clients} nodes, "
                f"found {len(node_ids)}"
            )
        record = RecordDict({self.arrayrecord_key: arrays, self.configrecord_key: config})  # type: ignore[attr-defined]
        return list(self._construct_messages(record, node_ids, message_type))  # type: ignore[attr-defined]

    def configure_train(
        self,
        server_round: int,
        arrays: ArrayRecord,
        config: ConfigRecord,
        grid: object,
    ) -> Iterable[Message]:
        if self.fraction_train == 0.0:  # type: ignore[attr-defined]
            return []
        config["server-round"] = server_round
        return self._deterministic_messages(arrays, config, grid, MessageType.TRAIN)

    def configure_evaluate(
        self,
        server_round: int,
        arrays: ArrayRecord,
        config: ConfigRecord,
        grid: object,
    ) -> Iterable[Message]:
        if self.fraction_evaluate == 0.0:  # type: ignore[attr-defined]
            return []
        config["server-round"] = server_round
        return self._deterministic_messages(arrays, config, grid, MessageType.EVALUATE)


class StrictRepliesMixin(DeterministicSchedulingMixin):
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

    def configure_train(
        self,
        server_round: int,
        arrays: ArrayRecord,
        config: ConfigRecord,
        grid: object,
    ) -> Iterable[Message]:
        config["proximal-mu"] = self.proximal_mu
        return super().configure_train(server_round, arrays, config, grid)


class StrictQFedAvg(StrictRepliesMixin, QFedAvg):
    def __init__(self, *, expected_clients: int, **kwargs: object) -> None:
        self.expected_clients = expected_clients
        self._init_tracking()
        super().__init__(**kwargs)

    def configure_train(
        self,
        server_round: int,
        arrays: ArrayRecord,
        config: ConfigRecord,
        grid: object,
    ) -> Iterable[Message]:
        self.current_arrays = arrays.copy()
        return super().configure_train(server_round, arrays, config, grid)


class StrictFedAdam(StrictRepliesMixin, FedAdam):
    def __init__(self, *, expected_clients: int, **kwargs: object) -> None:
        self.expected_clients = expected_clients
        self._init_tracking()
        super().__init__(**kwargs)

    def configure_train(
        self,
        server_round: int,
        arrays: ArrayRecord,
        config: ConfigRecord,
        grid: object,
    ) -> Iterable[Message]:
        self.current_arrays = {key: array.numpy() for key, array in arrays.items()}
        return super().configure_train(server_round, arrays, config, grid)
