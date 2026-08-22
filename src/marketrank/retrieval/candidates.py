"""Retrieve feasible supply before expensive feature computation and ranking."""

from __future__ import annotations

from dataclasses import dataclass, replace

from marketrank.simulation.marketplace import (
    FeasibleCandidate,
    MarketplaceRequest,
    MarketplaceState,
)


@dataclass(frozen=True)
class RetrievalResult:
    candidates: list[FeasibleCandidate]
    applied_radius_km: float
    fallback: str | None


@dataclass(frozen=True)
class CandidateRetriever:
    minimum_candidates: int = 3
    maximum_radius_km: float = 200.0
    expansion_factor: float = 2.0

    def retrieve(self, state: MarketplaceState, request: MarketplaceRequest) -> RetrievalResult:
        """Find live candidates, widening geography only when local supply is sparse."""
        candidates = state.feasible_candidates(request)
        at_maximum_radius = request.max_distance_km >= self.maximum_radius_km
        if len(candidates) >= self.minimum_candidates or at_maximum_radius:
            fallback = None if candidates else "no_feasible_supply"
            return RetrievalResult(candidates, request.max_distance_km, fallback)

        expanded_radius = min(
            request.max_distance_km * self.expansion_factor, self.maximum_radius_km
        )
        expanded_request = replace(request, max_distance_km=expanded_radius)
        expanded = state.feasible_candidates(expanded_request)
        return RetrievalResult(expanded, expanded_radius, "expanded_radius")
