"""Budget accounting and conservative probe-gated action execution."""

from __future__ import annotations

from dataclasses import dataclass

from aqfl.agents.v2_contracts import (
    ActionProposal,
    ClientAction,
    ExecutionDecision,
    ProbeOutcome,
)


@dataclass
class ProbeBudget:
    max_candidates: int
    batches_per_candidate: int
    consumed_batches: int = 0

    def __post_init__(self) -> None:
        if self.max_candidates < 1 or self.batches_per_candidate < 1:
            raise ValueError("Probe budget limits must be positive")

    @property
    def maximum_batches(self) -> int:
        return self.max_candidates * self.batches_per_candidate

    def charge(self, batches: int) -> None:
        if batches < 0:
            raise ValueError("Cannot charge a negative probe cost")
        if self.consumed_batches + batches > self.maximum_batches:
            raise RuntimeError("Probe budget exceeded")
        self.consumed_batches += batches


class SafeActionExecutor:
    def __init__(
        self,
        fallback_action: ClientAction,
        *,
        minimum_gain: float = 0.0,
        uncertainty_margin: float = 0.0,
        extra_epoch_penalty: float = 0.0,
    ) -> None:
        if minimum_gain < 0 or uncertainty_margin < 0 or extra_epoch_penalty < 0:
            raise ValueError("Safety thresholds must be non-negative")
        self.fallback_action = fallback_action
        self.minimum_gain = float(minimum_gain)
        self.uncertainty_margin = float(uncertainty_margin)
        self.extra_epoch_penalty = float(extra_epoch_penalty)

    def select(
        self,
        proposal: ActionProposal,
        outcomes: tuple[ProbeOutcome, ...],
        budget: ProbeBudget,
        *,
        probe_enabled: bool = True,
    ) -> ExecutionDecision:
        if not probe_enabled:
            selected = proposal.candidates[0]
            return ExecutionDecision(
                proposal.client_id,
                selected,
                True,
                "no-probe ablation selected the first proposed candidate",
                0.0,
                (),
            )
        candidate_ids = {candidate.action_id for candidate in proposal.candidates}
        outcome_ids = {outcome.action_id for outcome in outcomes}
        if outcome_ids != candidate_ids:
            raise ValueError("Probe outcomes must exactly match proposed candidates")
        for outcome in outcomes:
            budget.charge(outcome.cost_batches)
        action_by_id = {candidate.action_id: candidate for candidate in proposal.candidates}
        ranked = sorted(
            outcomes,
            key=lambda outcome: (
                outcome.estimated_gain
                - self.extra_epoch_penalty
                * (action_by_id[outcome.action_id].local_epochs - 1),
                -outcome.probed_loss,
                outcome.action_id,
            ),
            reverse=True,
        )
        best = ranked[0]
        selected = action_by_id[best.action_id]
        conservative_gain = (
            best.estimated_gain
            - self.extra_epoch_penalty * (selected.local_epochs - 1)
            - self.uncertainty_margin
        )
        if conservative_gain < self.minimum_gain:
            return ExecutionDecision(
                proposal.client_id,
                self.fallback_action,
                False,
                "all candidates failed the conservative probe-gain gate",
                conservative_gain,
                outcomes,
            )
        return ExecutionDecision(
            proposal.client_id,
            action_by_id[best.action_id],
            True,
            "candidate passed the conservative probe-gain gate",
            conservative_gain,
            outcomes,
        )
