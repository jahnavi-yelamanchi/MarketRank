"""Composable serving path for an explainable marketplace match decision."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

from marketrank.features.point_in_time import FeatureContext, compute_features
from marketrank.ranking.baselines import rank_baseline
from marketrank.reranking.policy import ConstraintAwareReranker
from marketrank.retrieval.candidates import CandidateRetriever
from marketrank.simulation.marketplace import (
    Location,
    MarketplaceRequest,
    MarketplaceState,
    SupplyOffer,
)


@dataclass(frozen=True)
class MatchInput:
    request_id: str
    user_id: str
    category: str
    latitude: float
    longitude: float
    budget: float
    quantity: int = 1
    max_distance_km: float = 50.0


@dataclass(frozen=True)
class MatchResult:
    offer_id: str
    provider_id: str
    predicted_score: float
    features: dict[str, float]
    reasons: list[str]


@dataclass(frozen=True)
class MatchResponse:
    request_id: str
    policy: str
    fallback: str | None
    constraints_applied: list[str]
    results: list[MatchResult]


class MarketplaceMatchingService:
    """Serve matching decisions without mutating supply until a match is accepted."""

    def __init__(self, state: MarketplaceState, model=None) -> None:  # noqa: ANN001
        self.state = state
        self.model = model
        self.retriever = CandidateRetriever()
        self.reranker = ConstraintAwareReranker()

    def match(self, match_input: MatchInput) -> MatchResponse:
        request = MarketplaceRequest(
            request_id=match_input.request_id,
            user_id=match_input.user_id,
            category=match_input.category,
            location=Location(match_input.latitude, match_input.longitude),
            budget=match_input.budget,
            quantity=match_input.quantity,
            created_at=datetime.now(UTC),
            max_distance_km=match_input.max_distance_km,
        )
        retrieval = self.retriever.retrieve(self.state, request)
        features = compute_features(
            request,
            retrieval.candidates,
            FeatureContext(provider_exposure=self.state.exposure),
        )
        policy = "lambdamart" if self.model else "heuristic_fallback"
        ranked = self.model.rank(features) if self.model else rank_baseline(features, "heuristic")
        reranked = self.reranker.rerank(request, self.state, ranked)
        return MatchResponse(
            request_id=request.request_id,
            policy=policy,
            fallback=retrieval.fallback,
            constraints_applied=reranked.constraints_applied,
            results=[
                MatchResult(
                    offer_id=candidate.row.offer_id,
                    provider_id=candidate.row.provider_id,
                    predicted_score=candidate.score,
                    features=candidate.row.values,
                    reasons=candidate.reasons,
                )
                for candidate in reranked.candidates
            ],
        )


def demo_marketplace_state() -> MarketplaceState:
    """Provide a deterministic no-download state for local API and UI exploration."""
    location = Location(-23.55, -46.63)
    offers = [
        SupplyOffer(
            "books-a", "paper-trail", "books", Location(-23.56, -46.64), 24, 0.92, 0.96, 8, 3
        ),
        SupplyOffer(
            "books-b",
            "neighborhood-books",
            "books",
            Location(-23.60, -46.61),
            19,
            0.81,
            0.91,
            6,
            2,
        ),
        SupplyOffer(
            "books-c", "city-readers", "books", Location(-23.66, -46.68), 17, 0.73, 0.85, 12, 4
        ),
        SupplyOffer("groceries-a", "fresh-route", "groceries", location, 35, 0.9, 0.94, 15, 5),
    ]
    return MarketplaceState({offer.offer_id: offer for offer in offers})
