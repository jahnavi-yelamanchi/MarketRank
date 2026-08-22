"""Apply live constraints and marketplace health objectives after model ranking."""

from __future__ import annotations

from dataclasses import dataclass

from marketrank.ranking.baselines import RankedCandidate
from marketrank.simulation.marketplace import MarketplaceRequest, MarketplaceState


@dataclass(frozen=True)
class RerankResult:
    candidates: list[RankedCandidate]
    excluded_offer_ids: list[str]
    constraints_applied: list[str]


@dataclass(frozen=True)
class ConstraintAwareReranker:
    price_weight: float = 0.12
    utilization_weight: float = 0.10
    exposure_weight: float = 0.15
    diversity_weight: float = 0.08

    def rerank(
        self,
        request: MarketplaceRequest,
        state: MarketplaceState,
        ranked: list[RankedCandidate],
    ) -> RerankResult:
        """Guarantee live feasibility and rebalance a model-ranked candidate list."""
        feasible = state.feasible_candidates(request)
        feasible_offer_ids = {candidate.offer.offer_id for candidate in feasible}
        excluded = [
            candidate.row.offer_id
            for candidate in ranked
            if candidate.row.offer_id not in feasible_offer_ids
        ]
        remaining = [
            candidate for candidate in ranked if candidate.row.offer_id in feasible_offer_ids
        ]
        selected: list[RankedCandidate] = []
        provider_positions: dict[str, int] = {}

        while remaining:
            next_candidate = max(
                remaining,
                key=lambda candidate: self._adjusted_score(candidate, provider_positions),
            )
            provider_id = next_candidate.row.provider_id
            position = provider_positions.get(provider_id, 0)
            adjustment = (
                self._adjusted_score(next_candidate, provider_positions) - next_candidate.score
            )
            reasons = next_candidate.reasons + [f"Marketplace adjustment {adjustment:+.3f}"]
            selected.append(
                RankedCandidate(
                    row=next_candidate.row,
                    score=next_candidate.score + adjustment,
                    policy=f"{next_candidate.policy}+rerank",
                    reasons=reasons,
                )
            )
            provider_positions[provider_id] = position + 1
            remaining.remove(next_candidate)

        return RerankResult(
            candidates=selected,
            excluded_offer_ids=excluded,
            constraints_applied=["budget", "service_radius", "inventory", "capacity", "category"],
        )

    def _adjusted_score(
        self, candidate: RankedCandidate, provider_positions: dict[str, int]
    ) -> float:
        values = candidate.row.values
        price_bonus = self.price_weight * (1 - values["price_to_budget"])
        utilization_penalty = self.utilization_weight * values["provider_utilization"]
        exposure_penalty = self.exposure_weight * values["provider_exposure_share"]
        diversity_penalty = self.diversity_weight * provider_positions.get(
            candidate.row.provider_id, 0
        )
        return (
            candidate.score
            + price_bonus
            - utilization_penalty
            - exposure_penalty
            - diversity_penalty
        )
