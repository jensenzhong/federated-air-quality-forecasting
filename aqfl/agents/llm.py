"""DeepSeek planning agent with schema validation and hash-addressed replay cache."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from openai import OpenAI

from aqfl.agents.decision import Decision, StrategyName
from aqfl.agents.rule import RulePlanningAgent
from aqfl.config import canonical_json


class LLMPlanningAgent:
    def __init__(
        self,
        config: dict[str, Any],
        cache_dir: Path,
        strict: bool = True,
        fixed_budget: bool = False,
        include_fairness: bool = True,
    ) -> None:
        self.config = config["llm"]
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.strict = strict
        self.fixed_budget = fixed_budget
        self.include_fairness = include_fairness
        self.rule = RulePlanningAgent()
        self.last_decision: Decision | None = None

    def choose(self, round_number: int, history: list[dict[str, Any]], current: dict[str, float]) -> Decision:
        forced = ["size_only", "perf_only", "hybrid", "fairness_clip"]
        if 1 <= round_number <= 4:
            prompt_hash = hashlib.sha256(f"forced-exploration-round-{round_number}".encode()).hexdigest()
            decision = Decision(
                cast(StrategyName, forced[round_number - 1]),
                1.0,
                1,
                "pre-registered forced exploration",
                prompt_hash,
                "rule",
            )
            self.last_decision = decision
            return decision
        if round_number % int(self.config["call_every_n_rounds"]) == 0 and self.last_decision is not None:
            return self.last_decision

        prompt = self.build_prompt(round_number, history, current, self.include_fairness)
        prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        cached = self.cache_dir / f"{prompt_hash}.json"
        if cached.is_file():
            payload = json.loads(cached.read_text(encoding="utf-8"))
            decision = Decision.from_dict(payload["decision"], source="cache", prompt_hash=prompt_hash)
            self.last_decision = decision
            return decision

        try:
            decision, raw_response = self._call(prompt, prompt_hash)
            if self.fixed_budget:
                decision = Decision(
                    decision.strategy,
                    1.0,
                    1,
                    f"fixed-budget ablation; {decision.reason}",
                    decision.prompt_hash,
                    decision.source,
                )
            payload = {
                "created_at_utc": datetime.now(UTC).isoformat(),
                "model": self.config["model"],
                "prompt": prompt,
                "prompt_hash": prompt_hash,
                "response": raw_response,
                "parse_status": "valid",
                "decision": decision.to_dict(),
            }
            cached.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as exc:
            failure_payload = {
                "created_at_utc": datetime.now(UTC).isoformat(),
                "model": self.config["model"],
                "prompt": prompt,
                "prompt_hash": prompt_hash,
                "response": None,
                "parse_status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            failure_path = self.cache_dir / (
                f"{prompt_hash}.failure.{datetime.now(UTC).strftime('%Y%m%dT%H%M%S%fZ')}.json"
            )
            failure_path.write_text(
                json.dumps(failure_payload, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            if self.strict:
                raise RuntimeError(f"Formal LLM decision failed: {exc}") from exc
            fallback = self.rule.choose(round_number, history, current)
            decision = Decision(
                fallback.strategy,
                fallback.lr_scale,
                fallback.local_epochs,
                f"LLM failure fallback: {exc}; {fallback.reason}",
                prompt_hash,
                "fallback",
            )
        self.last_decision = decision
        return decision

    def _call(self, prompt: str, prompt_hash: str) -> tuple[Decision, str]:
        key = os.getenv(str(self.config["api_key_env"]))
        if not key:
            raise RuntimeError(f"Missing {self.config['api_key_env']}")
        client = OpenAI(api_key=key, base_url=str(self.config["base_url"]))
        response = client.chat.completions.create(
            model=str(self.config["model"]),
            temperature=float(self.config["temperature"]),
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": "Return only a JSON object following the supplied schema."},
                {"role": "user", "content": prompt},
            ],
        )
        raw = response.choices[0].message.content or ""
        data = json.loads(raw)
        return Decision.from_dict(data, source="llm", prompt_hash=prompt_hash), raw

    @staticmethod
    def build_prompt(
        round_number: int,
        history: list[dict[str, Any]],
        current: dict[str, float],
        include_fairness: bool = True,
    ) -> str:
        fairness_keys = {"worst_station_mae", "station_mae_std", "station_mae_cv"}
        safe_current = {
            key: value
            for key, value in current.items()
            if "test" not in key.lower() and (include_fairness or key not in fairness_keys)
        }
        safe_history = [
            {
                key: value
                for key, value in record.items()
                if "test" not in key.lower() and (include_fairness or key not in fairness_keys)
            }
            for record in history[-10:]
        ]
        payload = {"round": round_number, "validation_current": safe_current, "validation_history": safe_history}
        return (
            "Choose one aggregation strategy and local budget for PM2.5 federated validation. "
            "Allowed strategy: size_only, perf_only, hybrid, fairness_clip. "
            "Allowed lr_scale: 0.5, 1.0, 1.5. Allowed local_epochs: 1, 2. "
            "Schema: {strategy, lr_scale, local_epochs, reason}. No test information is available.\n"
            + canonical_json(payload)
        )
