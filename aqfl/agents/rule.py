"""Deterministic rule planning agent."""

from __future__ import annotations

from typing import Any, cast

from aqfl.agents.decision import Decision, StrategyName


class RulePlanningAgent:
    def choose(
        self,
        round_number: int,
        history: list[dict[str, Any]],
        current: dict[str, float],
    ) -> Decision:
        del round_number
        macro = max(float(current.get("macro_mae", 0.0)), 1e-8)
        worst_ratio = float(current.get("worst_station_mae", macro)) / macro
        mae_cv = float(current.get("station_mae_cv", 0.0))
        update_cv = float(current.get("update_norm_cv", 0.0))

        if worst_ratio > 1.25 or mae_cv > 0.20:
            strategy = "fairness_clip"
            strategy_reason = "station fairness threshold exceeded"
        elif len(history) >= 3 and self._relative_improvement(history[-3:]) < 0.005:
            strategy = "perf_only"
            strategy_reason = "three-round macro MAE improvement below 0.5%"
        elif update_cv > 0.5:
            strategy = "hybrid"
            strategy_reason = "client update-norm heterogeneity exceeded 0.5"
        else:
            strategy = "size_only"
            strategy_reason = "no fairness, stagnation, or update-drift trigger"

        if (
            len(history) >= 3
            and history[-1]["macro_mae"] > history[-2]["macro_mae"]
            and history[-2]["macro_mae"] > history[-3]["macro_mae"]
        ):
            lr_scale, local_epochs = 0.5, 1
            budget_reason = "two decision points show worsening validation MAE"
        elif len(history) >= 3 and self._relative_improvement(history[-3:]) < 0.002:
            lr_scale, local_epochs = 1.5, 2
            budget_reason = "three-round improvement below 0.2% without consecutive degradation"
        else:
            lr_scale, local_epochs = 1.0, 1
            budget_reason = "default local optimization budget"
        return Decision(
            cast(StrategyName, strategy),
            lr_scale,
            local_epochs,
            f"{strategy_reason}; {budget_reason}",
            "",
            "rule",
        )

    @staticmethod
    def _relative_improvement(records: list[dict[str, Any]]) -> float:
        if len(records) < 2:
            return float("inf")
        first = max(float(records[0]["macro_mae"]), 1e-8)
        last = float(records[-1]["macro_mae"])
        return (first - last) / first
