"""Replay demand through competing marketplace policies for offline A/B-style analysis."""

from __future__ import annotations

from dataclasses import dataclass, replace
from time import perf_counter
from typing import Literal

from marketrank.evaluation.metrics import (
    MarketplaceHealthReport,
    OnlineMatchEvent,
    evaluate_marketplace_health,
)
from marketrank.features.point_in_time import FeatureContext, compute_features
from marketrank.ranking.baselines import BaselinePolicy, rank_baseline
from marketrank.reranking.policy import ConstraintAwareReranker
from marketrank.retrieval.candidates import CandidateRetriever
from marketrank.simulation.marketplace import MarketplaceRequest, MarketplaceState

ExperimentPolicy = Literal[
    "heuristic", "nearest", "cheapest", "highest_rated", "heuristic_reranked"
]


@dataclass(frozen=True)
class PolicyExperimentResult:
    policy: str
    health: MarketplaceHealthReport
    events: list[OnlineMatchEvent]


def compare_policies(
    initial_state: MarketplaceState,
    requests: list[MarketplaceRequest],
    policies: list[ExperimentPolicy],
) -> list[PolicyExperimentResult]:
    """Run each policy against an identical starting supply snapshot."""
    return [run_policy_experiment(initial_state, requests, policy) for policy in policies]


def run_policy_experiment(
    initial_state: MarketplaceState,
    requests: list[MarketplaceRequest],
    policy: ExperimentPolicy,
) -> PolicyExperimentResult:
    """Replay requests sequentially so price, inventory, and capacity stay operational.

    This is a simulator, not causal proof: it gives pre-launch trade-off signals
    for relevance proxies, completion, utilization, and exposure concentration.
    """
    if not requests:
        raise ValueError("At least one request is required")
    state = _clone_state(initial_state)
    retriever = CandidateRetriever()
    reranker = ConstraintAwareReranker()
    events: list[OnlineMatchEvent] = []
    started_at = perf_counter()

    for request in requests:
        decision_started_at = perf_counter()
        retrieval = retriever.retrieve(state, request)
        features = compute_features(
            request,
            retrieval.candidates,
            FeatureContext(provider_exposure=state.exposure),
        )
        base_policy: BaselinePolicy = "heuristic" if policy == "heuristic_reranked" else policy
        ranked = rank_baseline(features, base_policy)
        candidates = (
            reranker.rerank(request, state, ranked).candidates
            if policy.endswith("reranked")
            else ranked
        )
        top = candidates[0] if candidates else None
        accepted = False
        completed = False
        provider_id = None
        if top:
            attempt = state.attempt_match(request, top.row.offer_id)
            accepted = attempt.accepted
            provider_id = top.row.provider_id
            if accepted:
                completed = state.complete_match(request.request_id).completed
        events.append(
            OnlineMatchEvent(
                request_id=request.request_id,
                provider_id=provider_id,
                candidate_count=len(retrieval.candidates),
                accepted=accepted,
                completed=completed,
                latency_ms=(perf_counter() - decision_started_at) * 1_000,
            )
        )

    return PolicyExperimentResult(
        policy=policy,
        health=evaluate_marketplace_health(events, perf_counter() - started_at),
        events=events,
    )


def _clone_state(source: MarketplaceState) -> MarketplaceState:
    offers = {offer_id: replace(offer) for offer_id, offer in source.offers.items()}
    return MarketplaceState(
        offers=offers,
        random_seed=source.random_seed,
        exposure=source.exposure.copy(),
    )
