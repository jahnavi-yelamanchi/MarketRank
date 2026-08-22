"""Metrics for ranking accuracy and two-sided marketplace health."""

from __future__ import annotations

from dataclasses import dataclass
from math import log2

from marketrank.ranking.baselines import RankedCandidate


@dataclass(frozen=True)
class RankingReport:
    recall_at_k: float
    ndcg_at_k: float
    mrr: float
    map: float
    acceptance_proxy: float
    evaluated_requests: int


@dataclass(frozen=True)
class OnlineMatchEvent:
    request_id: str
    provider_id: str | None
    candidate_count: int
    accepted: bool
    completed: bool
    latency_ms: float


@dataclass(frozen=True)
class MarketplaceHealthReport:
    no_match_rate: float
    acceptance_rate: float
    completion_rate: float
    supply_exposure_hhi: float
    mean_latency_ms: float
    throughput_per_second: float


def evaluate_rankings(
    rankings: dict[str, list[RankedCandidate]], labels: dict[tuple[str, str], float], k: int = 5
) -> RankingReport:
    """Evaluate ranked request-provider lists against graded relevance labels."""
    if not rankings:
        raise ValueError("At least one ranked request is required")
    if k < 1:
        raise ValueError("k must be positive")

    recalls, ndcgs, reciprocal_ranks, average_precisions, top_labels = [], [], [], [], []
    for request_id, ranked in rankings.items():
        relevance = [labels.get((request_id, item.row.offer_id), 0.0) for item in ranked]
        positive_count = sum(value > 0 for value in relevance)
        top_relevance = relevance[:k]
        retrieved_positives = sum(value > 0 for value in top_relevance)
        recalls.append(retrieved_positives / positive_count if positive_count else 0.0)
        ndcgs.append(_ndcg(top_relevance, relevance, k))
        reciprocal_ranks.append(_reciprocal_rank(relevance))
        average_precisions.append(_average_precision(relevance))
        top_labels.append(relevance[0] if relevance else 0.0)

    count = len(rankings)
    return RankingReport(
        recall_at_k=sum(recalls) / count,
        ndcg_at_k=sum(ndcgs) / count,
        mrr=sum(reciprocal_ranks) / count,
        map=sum(average_precisions) / count,
        acceptance_proxy=sum(top_labels) / count,
        evaluated_requests=count,
    )


def evaluate_marketplace_health(
    events: list[OnlineMatchEvent], elapsed_seconds: float
) -> MarketplaceHealthReport:
    """Summarize conversion, fulfilment, latency, and supplier concentration."""
    if not events or elapsed_seconds <= 0:
        raise ValueError("Events and a positive elapsed duration are required")
    attempts = len(events)
    accepted = sum(event.accepted for event in events)
    completed = sum(event.completed for event in events)
    exposure: dict[str, int] = {}
    for event in events:
        if event.provider_id:
            exposure[event.provider_id] = exposure.get(event.provider_id, 0) + 1
    exposure_hhi = sum((count / attempts) ** 2 for count in exposure.values())
    return MarketplaceHealthReport(
        no_match_rate=sum(event.candidate_count == 0 for event in events) / attempts,
        acceptance_rate=accepted / attempts,
        completion_rate=completed / attempts,
        supply_exposure_hhi=exposure_hhi,
        mean_latency_ms=sum(event.latency_ms for event in events) / attempts,
        throughput_per_second=attempts / elapsed_seconds,
    )


def _ndcg(top_relevance: list[float], all_relevance: list[float], k: int) -> float:
    actual = sum((2**value - 1) / log2(index + 2) for index, value in enumerate(top_relevance))
    ideal = sorted(all_relevance, reverse=True)[:k]
    ideal_score = sum((2**value - 1) / log2(index + 2) for index, value in enumerate(ideal))
    return actual / ideal_score if ideal_score else 0.0


def _reciprocal_rank(relevance: list[float]) -> float:
    return next((1 / index for index, value in enumerate(relevance, start=1) if value > 0), 0.0)


def _average_precision(relevance: list[float]) -> float:
    positives = sum(value > 0 for value in relevance)
    if not positives:
        return 0.0
    hits = 0
    precision_sum = 0.0
    for index, value in enumerate(relevance, start=1):
        if value > 0:
            hits += 1
            precision_sum += hits / index
    return precision_sum / positives
