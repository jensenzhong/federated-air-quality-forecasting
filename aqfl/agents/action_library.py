"""Frozen, auditable PAFA action library."""

from __future__ import annotations

from aqfl.agents.v2_contracts import ClientAction


def build_action_library() -> dict[str, ClientAction]:
    actions = (
        ClientAction("safe_default", 1.0, 1, 0.01, "normal", "frozen FedProx fallback"),
        ClientAction("cautious", 0.5, 1, 0.01, "downweight_conflict", "reduce unstable local movement"),
        ClientAction("adapt_fast", 1.5, 1, 0.001, "normal", "accelerate a stable but underfitting client"),
        ClientAction("tail_focus", 1.0, 2, 0.01, "protect_tail", "allocate one extra epoch to tail risk"),
    )
    return {action.action_id: action for action in actions}


def resolve_action_ids(action_ids: list[str] | tuple[str, ...]) -> tuple[ClientAction, ...]:
    library = build_action_library()
    if not action_ids:
        raise ValueError("At least one candidate action is required")
    if len(action_ids) > 3:
        raise ValueError("At most three candidate actions are allowed")
    unknown = sorted(set(action_ids) - library.keys())
    if unknown:
        raise ValueError(f"Unknown action IDs: {unknown}")
    if len(action_ids) != len(set(action_ids)):
        raise ValueError("Candidate action IDs must be unique")
    return tuple(library[action_id] for action_id in action_ids)
