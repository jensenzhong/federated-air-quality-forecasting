"""Bounded episodic memory and per-client state trajectory construction."""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Any

import numpy as np

from aqfl.agents.v2_contracts import ClientStateCapsule, CreditRecord, ExecutionDecision


class EpisodicMemory:
    def __init__(self, max_records_per_client: int = 20) -> None:
        if max_records_per_client < 1:
            raise ValueError("max_records_per_client must be positive")
        self.max_records_per_client = max_records_per_client
        self._capsules: dict[str, deque[ClientStateCapsule]] = defaultdict(
            lambda: deque(maxlen=max_records_per_client)
        )
        self._credits: dict[str, deque[CreditRecord]] = defaultdict(
            lambda: deque(maxlen=max_records_per_client)
        )
        self._executions: dict[tuple[str, int], ExecutionDecision] = {}

    def add_capsule(self, capsule: ClientStateCapsule) -> None:
        history = self._capsules[capsule.client_id]
        if history and capsule.round_number <= history[-1].round_number:
            raise ValueError("Capsule rounds must increase strictly for each client")
        history.append(capsule)

    def add_execution(self, round_number: int, decision: ExecutionDecision) -> None:
        key = (decision.client_id, round_number)
        if key in self._executions:
            raise ValueError(f"Duplicate execution record: {key}")
        self._executions[key] = decision

    def add_credit(self, credit: CreditRecord) -> None:
        self._credits[credit.client_id].append(credit)

    def capsules(self, client_id: str) -> tuple[ClientStateCapsule, ...]:
        return tuple(self._capsules.get(client_id, ()))

    def credits(self, client_id: str) -> tuple[CreditRecord, ...]:
        return tuple(self._credits.get(client_id, ()))

    def execution(self, client_id: str, round_number: int) -> ExecutionDecision | None:
        return self._executions.get((client_id, round_number))

    def recent_successes(self, client_id: str, limit: int = 3) -> list[dict[str, Any]]:
        positive = [item for item in self.credits(client_id) if item.realized_gain > 0]
        return [item.to_dict() for item in positive[-limit:]]

    def to_dict(self) -> dict[str, Any]:
        client_ids = sorted(set(self._capsules) | set(self._credits))
        return {
            "max_records_per_client": self.max_records_per_client,
            "capsules": {
                client_id: [item.to_dict() for item in self.capsules(client_id)]
                for client_id in client_ids
            },
            "credits": {
                client_id: [item.to_dict() for item in self.credits(client_id)]
                for client_id in client_ids
            },
            "executions": [
                {"round_number": round_number, "decision": decision.to_dict()}
                for (client_id, round_number), decision in sorted(self._executions.items())
                if client_id in client_ids
            ],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EpisodicMemory:
        memory = cls(int(data.get("max_records_per_client", 20)))
        for records in data.get("capsules", {}).values():
            for item in records:
                memory.add_capsule(ClientStateCapsule.from_dict(item))
        for records in data.get("credits", {}).values():
            for item in records:
                memory.add_credit(CreditRecord.from_dict(item))
        for item in data.get("executions", []):
            memory.add_execution(
                int(item["round_number"]),
                ExecutionDecision.from_dict(item["decision"]),
            )
        return memory

    def build_capsule(
        self,
        *,
        client_id: str,
        round_number: int,
        val_mae: float,
        val_rmse: float,
        high_pollution_mae: float,
        train_loss: float,
        update_norm: float,
        update_cosine: float,
        train_seconds: float,
        local_epochs: int,
    ) -> ClientStateCapsule:
        previous = self.capsules(client_id)
        maes = [capsule.val_mae for capsule in previous[-3:]] + [float(val_mae)]
        ema = float(val_mae) if not previous else 0.4 * float(val_mae) + 0.6 * previous[-1].mae_ema
        slope = 0.0 if not previous else float(val_mae) - previous[-1].val_mae
        oscillation = float(np.std(np.diff(maes))) if len(maes) >= 3 else 0.0
        scale = max(ema, 1e-8)
        drift_score = abs(slope) / scale + oscillation / scale + max(-float(update_cosine), 0.0)
        prior_credit = self.credits(client_id)
        previous_gain = prior_credit[-1].realized_gain if prior_credit else 0.0
        previous_action = prior_credit[-1].action_id if prior_credit else "none"
        return ClientStateCapsule(
            client_id=client_id,
            round_number=round_number,
            val_mae=float(val_mae),
            val_rmse=float(val_rmse),
            high_pollution_mae=float(high_pollution_mae),
            train_loss=float(train_loss),
            update_norm=float(update_norm),
            update_cosine=float(np.clip(update_cosine, -1.0, 1.0)),
            mae_ema=ema,
            mae_slope=slope,
            mae_oscillation=oscillation,
            drift_score=drift_score,
            previous_action_id=previous_action,
            previous_realized_gain=previous_gain,
            train_seconds=float(train_seconds),
            local_epochs=int(local_epochs),
        )
