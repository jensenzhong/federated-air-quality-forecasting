"""In-process sequential Grid for low-memory Flower execution.

The runner deliberately keeps Flower's ClientApp and Strategy APIs intact while
dispatching one client message at a time in a deterministic station order. This
changes scheduling only; it does not change the number of clients, aggregation,
or round semantics.
"""

from __future__ import annotations

import copy
import gc
import time
from collections.abc import Iterable
from typing import Any

import psutil
from flwr.app import Context, Message, Metadata, RecordDict
from flwr.clientapp import ClientApp
from flwr.serverapp import Grid

from aqfl.federated.resources import resource_snapshot


class SequentialGrid(Grid):
    """Execute all configured ClientApps serially in node-id order."""

    def __init__(
        self,
        client_app: ClientApp,
        *,
        run_id: int,
        node_configs: dict[int, dict[str, Any]],
        run_config: dict[str, Any],
    ) -> None:
        self._client_app = client_app
        self._run_id = run_id
        self._node_configs = {
            int(node_id): dict(config) for node_id, config in node_configs.items()
        }
        self._run_config = dict(run_config)
        if str(self._run_config.get("method", "")).startswith("pafa_"):
            partition_ids = [
                int(config.get("partition-id", -1))
                for config in self._node_configs.values()
            ]
            stations = [config.get("station") for config in self._node_configs.values()]
            if (
                sorted(partition_ids) != list(range(len(self._node_configs)))
                or any(station is None for station in stations)
                or len(set(stations)) != len(stations)
            ):
                raise RuntimeError(
                    "Sequential PAFA requires unique, complete private station bindings"
                )
        self._states = {node_id: RecordDict() for node_id in self._node_configs}
        self._pending: dict[str, Message] = {}
        self._message_counter = 0
        self._telemetry: list[dict[str, Any]] = []
        self._process = psutil.Process()

    @property
    def telemetry(self) -> list[dict[str, Any]]:
        return list(self._telemetry)

    def set_run(self, run_id: int) -> None:
        if (
            str(self._run_config.get("method", "")).startswith("pafa_")
            and int(run_id) != self._run_id
        ):
            raise RuntimeError("Sequential PAFA client state cannot be reused across runs")
        self._run_id = int(run_id)

    @property
    def run(self) -> Any:
        """This local Grid has no SuperLink Run object."""
        raise RuntimeError("SequentialGrid does not expose a SuperLink Run")

    def create_message(
        self,
        content: RecordDict,
        message_type: str,
        dst_node_id: int,
        group_id: str,
        ttl: float | None = None,
    ) -> Message:
        if int(dst_node_id) not in self._node_configs:
            raise ValueError(f"Unknown sequential node id: {dst_node_id}")
        return Message(
            content=content,
            message_type=message_type,
            dst_node_id=int(dst_node_id),
            group_id=group_id,
            ttl=ttl,
        )

    def get_node_ids(self) -> Iterable[int]:
        return tuple(sorted(self._node_configs))

    def push_messages(self, messages: Iterable[Message]) -> Iterable[str]:
        message_ids = []
        for message in messages:
            self._message_counter += 1
            message_id = f"sequential-{self._message_counter}"
            self._pending[message_id] = message
            message_ids.append(message_id)
        return message_ids

    def pull_messages(self, message_ids: Iterable[str]) -> Iterable[Message]:
        return tuple(self._pending.pop(message_id) for message_id in message_ids)

    def send_and_receive(
        self,
        messages: Iterable[Message],
        *,
        timeout: float | None = None,
    ) -> Iterable[Message]:
        del timeout
        message_list = list(messages)
        by_node = {message.metadata.dst_node_id: message for message in message_list}
        expected = set(self._node_configs)
        received = set(by_node)
        private_run = str(self._run_config.get("method", "")).startswith("pafa_")
        if received != expected or len(message_list) != len(received):
            if private_run:
                raise RuntimeError(
                    "Sequential secure cohort must address every client exactly once"
                )
            missing = sorted(expected - received)
            extra = sorted(received - expected)
            raise RuntimeError(
                "Sequential formal round must address every node exactly once; "
                f"missing={missing}, extra={extra}"
            )

        replies: list[Message] = []
        cohort_telemetry: list[dict[str, float]] = []
        for node_id in sorted(expected):
            # Real Flower transport serializes each message. Copying prevents one
            # in-process SecAgg+ client from mutating another client's stage config.
            message = copy.deepcopy(by_node[node_id])
            self._message_counter += 1
            original = message.metadata
            message = Message(
                content=message.content,
                metadata=Metadata(
                    run_id=self._run_id,
                    message_id=f"sequential-instruction-{self._message_counter}",
                    src_node_id=original.src_node_id,
                    dst_node_id=original.dst_node_id,
                    reply_to_message_id=original.reply_to_message_id,
                    group_id=original.group_id,
                    created_at=original.created_at,
                    ttl=original.ttl,
                    message_type=original.message_type,
                ),
            )
            started = time.perf_counter()
            before = resource_snapshot()
            context = Context(
                run_id=self._run_id,
                node_id=node_id,
                node_config=self._node_configs[node_id],
                state=self._states[node_id],
                run_config=self._run_config,
            )
            try:
                reply = self._client_app(message, context)
                if (
                    reply.metadata.run_id != self._run_id
                    or reply.metadata.src_node_id != node_id
                    or reply.metadata.dst_node_id != message.metadata.src_node_id
                    or reply.metadata.group_id != message.metadata.group_id
                    or reply.metadata.message_type != message.metadata.message_type
                    or reply.metadata.reply_to_message_id != message.metadata.message_id
                ):
                    raise RuntimeError("Sequential ClientApp reply identity validation failed")
                replies.append(reply)
            except Exception as exc:
                if private_run:
                    raise RuntimeError(
                        "Sequential PAFA ClientApp failed during secure cohort dispatch"
                    ) from exc
                raise RuntimeError(
                    f"Sequential ClientApp failed for node {node_id} "
                    f"(message_type={message.metadata.message_type})"
                ) from exc
            finally:
                after = resource_snapshot()
                row = {
                    "elapsed_seconds": time.perf_counter() - started,
                    "rss_gb": float(self._process.memory_info().rss / 1024**3),
                    "available_memory_before_gb": before["available_memory_gb"],
                    "available_memory_after_gb": after["available_memory_gb"],
                }
                if private_run:
                    cohort_telemetry.append(row)
                else:
                    self._telemetry.append(
                        {
                            "event": "sequential_client",
                            "round": int(message.content["config"].get("server-round", 0)),
                            "message_type": message.metadata.message_type,
                            "node_id": node_id,
                            **row,
                        }
                    )
                # Explicitly release per-client temporary tensors/datasets before
                # constructing the next client's context.
                del context
                gc.collect()
        if private_run and cohort_telemetry:
            self._telemetry.append(
                {
                    "event": "secure_cohort_dispatch",
                    "round": int(message_list[0].metadata.group_id or 0),
                    "message_type": message_list[0].metadata.message_type,
                    "cohort_size": len(cohort_telemetry),
                    "elapsed_seconds": float(
                        sum(item["elapsed_seconds"] for item in cohort_telemetry)
                    ),
                    "peak_rss_gb": float(max(item["rss_gb"] for item in cohort_telemetry)),
                    "minimum_available_memory_gb": float(
                        min(item["available_memory_after_gb"] for item in cohort_telemetry)
                    ),
                }
            )
        return replies
