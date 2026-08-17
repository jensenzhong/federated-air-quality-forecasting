"""Rule, contextual-bandit, and LLM proposers sharing one PAFA action space."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol, cast

import numpy as np
from openai import OpenAI

from aqfl.agents.action_library import build_action_library, resolve_action_ids
from aqfl.agents.memory import EpisodicMemory
from aqfl.agents.v2_contracts import (
    ActionProposal,
    ClientStateCapsule,
    CreditRecord,
    DiagnosisTag,
    ProposalSource,
)
from aqfl.config import canonical_json
from aqfl.privacy import assert_prompt_keys_are_private_safe, enforce_client_level_llm_policy


class ActionProposer(Protocol):
    def propose(
        self,
        round_number: int,
        capsules: list[ClientStateCapsule],
        memory: EpisodicMemory,
    ) -> dict[str, ActionProposal]: ...

    def observe(self, credit: CreditRecord) -> None: ...


def _diagnose(capsule: ClientStateCapsule, median_mae: float) -> tuple[DiagnosisTag, tuple[str, ...]]:
    if capsule.update_cosine < -0.1:
        return "conflict", ("update cosine is negative", "protect global consensus")
    if capsule.drift_score > 0.12:
        return "drift", ("normalized trajectory drift exceeds 0.12", "reduce unstable movement")
    if capsule.val_mae > 1.15 * max(median_mae, 1e-8) or capsule.high_pollution_mae > 1.25 * max(
        capsule.val_mae, 1e-8
    ):
        return "tail_risk", ("client or high-pollution error is in the tail", "protect tail utility")
    if capsule.mae_slope < -0.01 * max(capsule.mae_ema, 1e-8):
        return "underfit", ("validation MAE is still improving", "controlled acceleration may help")
    if capsule.mae_slope > 0.01 * max(capsule.mae_ema, 1e-8):
        return "overfit", ("validation MAE worsened", "avoid aggressive local fitting")
    return "stable", ("no drift, conflict, or tail trigger",)


class RuleActionProposer:
    def propose(
        self,
        round_number: int,
        capsules: list[ClientStateCapsule],
        memory: EpisodicMemory,
    ) -> dict[str, ActionProposal]:
        del round_number, memory
        median = float(np.median([item.val_mae for item in capsules])) if capsules else 0.0
        mapping = {
            "stable": ["safe_default", "adapt_fast"],
            "underfit": ["adapt_fast", "safe_default", "tail_focus"],
            "overfit": ["cautious", "safe_default"],
            "drift": ["cautious", "safe_default", "tail_focus"],
            "conflict": ["cautious", "safe_default"],
            "tail_risk": ["tail_focus", "safe_default", "cautious"],
        }
        proposals = {}
        for capsule in capsules:
            diagnosis, evidence = _diagnose(capsule, median)
            proposals[capsule.client_id] = ActionProposal(
                capsule.client_id,
                diagnosis,
                evidence,
                resolve_action_ids(mapping[diagnosis]),
                "rule",
            )
        return proposals

    def observe(self, credit: CreditRecord) -> None:
        del credit


def _feature(capsule: ClientStateCapsule, action_id: str) -> np.ndarray:
    library = build_action_library()
    action = library[action_id]
    client = np.asarray(
        [
            1.0,
            np.tanh(capsule.mae_slope / max(capsule.mae_ema, 1e-8)),
            np.tanh(capsule.mae_oscillation / max(capsule.mae_ema, 1e-8)),
            np.tanh(capsule.drift_score),
            capsule.update_cosine,
            np.tanh(capsule.previous_realized_gain),
        ],
        dtype=np.float64,
    )
    action_features = np.asarray(
        [
            action.lr_scale - 1.0,
            float(action.local_epochs - 1),
            action.proximal_mu / 0.01,
            float(action.aggregation_gate == "downweight_conflict"),
            float(action.aggregation_gate == "protect_tail"),
            float(action.action_id == "safe_default"),
        ],
        dtype=np.float64,
    )
    return np.concatenate([client, action_features])


class ContextualBanditProposer:
    """Deterministic LinUCB proposer used as the same-action-space classical control."""

    def __init__(self, alpha: float = 0.5, ridge: float = 1.0) -> None:
        if alpha < 0 or ridge <= 0:
            raise ValueError("LinUCB alpha must be non-negative and ridge positive")
        self.alpha = float(alpha)
        self.dimension = 12
        self.a = ridge * np.eye(self.dimension, dtype=np.float64)
        self.b = np.zeros(self.dimension, dtype=np.float64)
        self._features: dict[tuple[str, int, str], np.ndarray] = {}

    def propose(
        self,
        round_number: int,
        capsules: list[ClientStateCapsule],
        memory: EpisodicMemory,
    ) -> dict[str, ActionProposal]:
        del memory
        library = build_action_library()
        inverse = np.linalg.inv(self.a)
        theta = inverse @ self.b
        median = float(np.median([item.val_mae for item in capsules])) if capsules else 0.0
        proposals = {}
        for capsule in capsules:
            scored: list[tuple[float, str]] = []
            for action_id in sorted(library):
                vector = _feature(capsule, action_id)
                score = float(theta @ vector + self.alpha * np.sqrt(vector @ inverse @ vector))
                scored.append((score, action_id))
                self._features[(capsule.client_id, round_number, action_id)] = vector
            selected = [action_id for _, action_id in sorted(scored, reverse=True)[:2]]
            if "safe_default" not in selected:
                selected.append("safe_default")
            diagnosis, evidence = _diagnose(capsule, median)
            proposals[capsule.client_id] = ActionProposal(
                capsule.client_id,
                diagnosis,
                ("LinUCB score over the frozen action library", *evidence),
                resolve_action_ids(selected),
                "bandit",
            )
        return proposals

    def observe(self, credit: CreditRecord) -> None:
        vector = self._features.get((credit.client_id, credit.round_number, credit.action_id))
        if vector is None:
            return
        self.a += np.outer(vector, vector)
        self.b += float(credit.realized_gain) * vector


class ProbeOracleProposer:
    """Mechanism upper bound: expose every non-fallback intervention to equal-cost probes."""

    def propose(
        self,
        round_number: int,
        capsules: list[ClientStateCapsule],
        memory: EpisodicMemory,
    ) -> dict[str, ActionProposal]:
        del round_number, memory
        return {
            capsule.client_id: ActionProposal(
                capsule.client_id,
                "stable",
                ("mechanism upper bound exposes all three non-fallback interventions",),
                resolve_action_ids(["cautious", "adapt_fast", "tail_focus"]),
                "rule",
            )
            for capsule in capsules
        }

    def observe(self, credit: CreditRecord) -> None:
        del credit


class LLMActionProposer:
    def __init__(
        self,
        config: dict[str, Any],
        cache_dir: Path | None,
        *,
        strict: bool = True,
    ) -> None:
        self.config = config["llm"]
        enforce_client_level_llm_policy(config)
        self.cache_dir = cache_dir
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.strict = strict
        self.rule = RuleActionProposer()
        self.last_proposals: dict[str, ActionProposal] | None = None

    @staticmethod
    def build_prompt(
        round_number: int,
        capsules: list[ClientStateCapsule],
        memory: EpisodicMemory,
    ) -> str:
        clients = []
        for capsule in capsules:
            state = capsule.to_dict()
            if any("test" in key.lower() for key in state):
                raise RuntimeError("Test information is forbidden in PAFA state capsules")
            clients.append(
                {
                    "state": state,
                    "recent_successful_actions": memory.recent_successes(capsule.client_id),
                }
            )
        payload = {
            "round": round_number,
            "clients": clients,
            "action_library": {
                key: value.to_dict() for key, value in build_action_library().items()
            },
        }
        assert_prompt_keys_are_private_safe(payload)
        return (
            "Diagnose every client and propose 1-3 candidate action IDs. The client will verify "
            "them with a local probe; do not invent parameters or weights. Allowed diagnoses: "
            "stable, underfit, overfit, drift, conflict, tail_risk. Return exactly "
            "{proposals:[{client_id,diagnosis,evidence,candidate_action_ids}]}. "
            "Evidence must cite supplied state fields. No test information is available.\n"
            + canonical_json(payload)
        )

    def propose(
        self,
        round_number: int,
        capsules: list[ClientStateCapsule],
        memory: EpisodicMemory,
    ) -> dict[str, ActionProposal]:
        interval = int(self.config.get("call_every_n_rounds", 2))
        if self.last_proposals is not None and (round_number - 1) % interval != 0:
            return {
                client_id: ActionProposal(
                    proposal.client_id,
                    proposal.diagnosis,
                    ("reused prior local decision without a new LLM call", *proposal.evidence),
                    proposal.candidates,
                    "cache",
                    proposal.prompt_hash,
                )
                for client_id, proposal in self.last_proposals.items()
            }
        prompt = self.build_prompt(round_number, capsules, memory)
        prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        cache_path = (
            self.cache_dir / f"pafa-{prompt_hash}.json" if self.cache_dir is not None else None
        )
        if cache_path is not None and cache_path.is_file():
            data = json.loads(cache_path.read_text(encoding="utf-8"))["response"]
            proposals = self._parse(data, capsules, "cache", prompt_hash)
            self.last_proposals = proposals
            return proposals
        try:
            raw = self._call(prompt)
            data = json.loads(raw)
            proposals = self._parse(data, capsules, "llm", prompt_hash)
            if cache_path is not None:
                cache_path.write_text(
                json.dumps(
                    {
                        "created_at_utc": datetime.now(UTC).isoformat(),
                        "model": self.config["model"],
                        "prompt_hash": prompt_hash,
                        "prompt": prompt,
                        "response": data,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                    encoding="utf-8",
                )
        except Exception as exc:
            if self.cache_dir is not None:
                failure_path = self.cache_dir / (
                    f"pafa-{prompt_hash}.failure."
                    f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%S%fZ')}.json"
                )
                failure_path.write_text(
                    json.dumps(
                        {
                            "created_at_utc": datetime.now(UTC).isoformat(),
                            "prompt_hash": prompt_hash,
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                        },
                        ensure_ascii=False,
                        indent=2,
                    ),
                    encoding="utf-8",
                )
            if self.strict:
                raise RuntimeError(f"Formal PAFA LLM proposal failed: {exc}") from exc
            proposals = self.rule.propose(round_number, capsules, memory)
            proposals = {
                client_id: ActionProposal(
                    proposal.client_id,
                    proposal.diagnosis,
                    (f"LLM failure fallback: {type(exc).__name__}", *proposal.evidence),
                    proposal.candidates,
                    "fallback",
                    prompt_hash,
                )
                for client_id, proposal in proposals.items()
            }
        self.last_proposals = proposals
        return proposals

    def _call(self, prompt: str) -> str:
        key_env = str(self.config.get("client_api_key_env", ""))
        key = os.getenv(key_env) if key_env else "local-client-agent"
        if not key:
            raise RuntimeError(f"Missing {key_env}")
        client = OpenAI(
            api_key=key,
            base_url=str(self.config.get("client_base_url", self.config["base_url"])),
        )
        response = client.chat.completions.create(
            model=str(self.config.get("client_model", self.config["model"])),
            temperature=float(self.config["temperature"]),
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": "Return only the requested JSON. Never invent action IDs or fields.",
                },
                {"role": "user", "content": prompt},
            ],
        )
        return response.choices[0].message.content or ""

    @staticmethod
    def _parse(
        data: dict[str, Any],
        capsules: list[ClientStateCapsule],
        source: ProposalSource,
        prompt_hash: str,
    ) -> dict[str, ActionProposal]:
        if set(data) != {"proposals"} or not isinstance(data["proposals"], list):
            raise ValueError("PAFA response must contain only a proposals list")
        expected = {capsule.client_id for capsule in capsules}
        proposals: dict[str, ActionProposal] = {}
        required = {"client_id", "diagnosis", "evidence", "candidate_action_ids"}
        for raw in data["proposals"]:
            if not isinstance(raw, dict) or set(raw) != required:
                raise ValueError("Each PAFA proposal must contain exactly the required fields")
            client_id = str(raw["client_id"])
            if client_id in proposals:
                raise ValueError(f"Duplicate client proposal: {client_id}")
            evidence = raw["evidence"]
            action_ids = raw["candidate_action_ids"]
            if not isinstance(evidence, list) or not all(isinstance(item, str) for item in evidence):
                raise ValueError("Proposal evidence must be a string list")
            if any("test" in item.lower() for item in evidence):
                raise ValueError("Test information is forbidden in PAFA proposal evidence")
            if not isinstance(action_ids, list) or not all(isinstance(item, str) for item in action_ids):
                raise ValueError("candidate_action_ids must be a string list")
            proposals[client_id] = ActionProposal(
                client_id,
                cast(DiagnosisTag, str(raw["diagnosis"])),
                tuple(evidence),
                resolve_action_ids(action_ids),
                source,
                prompt_hash,
            )
        if set(proposals) != expected:
            raise ValueError(
                f"PAFA response client set mismatch: expected={sorted(expected)}, got={sorted(proposals)}"
            )
        return proposals

    def observe(self, credit: CreditRecord) -> None:
        del credit
