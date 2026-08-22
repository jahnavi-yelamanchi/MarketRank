"""Deterministic simulated training set for an end-to-end ranking demonstration."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from random import Random

from marketrank.evaluation.metrics import RankingReport, evaluate_rankings
from marketrank.features.point_in_time import FeatureContext, RankingFeatureRow, compute_features
from marketrank.ranking.baselines import rank_baseline
from marketrank.ranking.training import (
    TrainingExample,
    train_classifier,
    train_lambdamart,
)
from marketrank.retrieval.candidates import CandidateRetriever
from marketrank.simulation.marketplace import (
    Location,
    MarketplaceRequest,
    MarketplaceState,
    SupplyOffer,
)


@dataclass(frozen=True)
class DemoTrainingReport:
    examples: int
    train_requests: int
    test_requests: int
    heuristic: RankingReport
    classifier: RankingReport
    lambdamart: RankingReport


def run_demo_training(*, request_count: int = 120, random_seed: int = 7) -> DemoTrainingReport:
    """Train both model types on deterministic simulated marketplace impressions."""
    state, requests = build_demo_marketplace(request_count=request_count, random_seed=random_seed)
    examples = _make_examples(state, requests)
    candidate_groups = _group_rows(examples)
    viable_request_ids = {
        request_id for request_id, rows in candidate_groups.items() if len(rows) >= 2
    }
    examples = [example for example in examples if example.row.request_id in viable_request_ids]
    request_ids = sorted({example.row.request_id for example in examples})
    split = max(1, int(len(request_ids) * 0.8))
    train_ids = set(request_ids[:split])
    train_examples = [example for example in examples if example.row.request_id in train_ids]
    test_examples = [example for example in examples if example.row.request_id not in train_ids]
    if not test_examples:
        raise ValueError(
            "request_count must provide a non-empty test set with at least two candidates"
        )

    classifier = train_classifier(train_examples, random_seed=random_seed)
    lambdamart = train_lambdamart(train_examples, random_seed=random_seed)
    test_rows = _group_rows(test_examples)
    labels = {
        (example.row.request_id, example.row.offer_id): example.label for example in test_examples
    }
    return DemoTrainingReport(
        examples=len(examples),
        train_requests=len(train_ids),
        test_requests=len(request_ids) - len(train_ids),
        heuristic=evaluate_rankings(
            {request_id: rank_baseline(rows) for request_id, rows in test_rows.items()}, labels
        ),
        classifier=evaluate_rankings(
            {request_id: classifier.rank(rows) for request_id, rows in test_rows.items()}, labels
        ),
        lambdamart=evaluate_rankings(
            {request_id: lambdamart.rank(rows) for request_id, rows in test_rows.items()}, labels
        ),
    )


def build_demo_marketplace(
    *, request_count: int, random_seed: int
) -> tuple[MarketplaceState, list[MarketplaceRequest]]:
    """Generate geographically coherent buyer demand and provider supply near São Paulo."""
    if request_count < 10:
        raise ValueError("request_count must be at least 10")
    rng = Random(random_seed)
    categories = ("books", "groceries", "home")
    origin = Location(-23.5505, -46.6333)
    offers = {}
    for category_index, category in enumerate(categories):
        for provider_index in range(14):
            provider_id = f"{category}-provider-{provider_index}"
            offer = SupplyOffer(
                offer_id=f"{provider_id}-offer",
                provider_id=provider_id,
                category=category,
                location=Location(
                    origin.latitude + rng.uniform(-0.18, 0.18),
                    origin.longitude + rng.uniform(-0.18, 0.18),
                ),
                price=round(rng.uniform(10 + category_index * 4, 42 + category_index * 5), 2),
                quality=round(rng.uniform(0.45, 0.98), 3),
                completion_rate=round(rng.uniform(0.58, 0.99), 3),
                inventory=rng.randint(4, 14),
                capacity=rng.randint(1, 5),
            )
            offers[offer.offer_id] = offer
    requests = [
        MarketplaceRequest(
            request_id=f"sim-request-{index}",
            user_id=f"sim-user-{index % 24}",
            category=categories[index % len(categories)],
            location=Location(
                origin.latitude + rng.uniform(-0.1, 0.1),
                origin.longitude + rng.uniform(-0.1, 0.1),
            ),
            budget=rng.uniform(24, 55),
            quantity=1,
            created_at=datetime(2025, 1, 1, tzinfo=UTC) + timedelta(minutes=index),
            max_distance_km=50,
        )
        for index in range(request_count)
    ]
    return MarketplaceState(offers, random_seed=random_seed), requests


def _make_examples(
    state: MarketplaceState, requests: list[MarketplaceRequest]
) -> list[TrainingExample]:
    retriever = CandidateRetriever()
    examples = []
    for request in requests:
        candidates = retriever.retrieve(state, request).candidates
        rows = compute_features(request, candidates, FeatureContext())
        examples.extend(TrainingExample(row=row, label=_relevance_label(row)) for row in rows)
    return examples


def _relevance_label(row: RankingFeatureRow) -> float:
    """Observed-outcome proxy used only for this public reproducible demo."""
    values = row.values
    distance_fit = max(0.0, 1 - values["distance_km"] / 50)
    utility = (
        0.32 * values["quality"]
        + 0.28 * values["completion_rate"]
        + 0.18 * values["user_price_fit"]
        + 0.14 * distance_fit
        + 0.08 * values["freshness_score"]
    )
    return float(max(0, min(3, int((utility - 0.60) * 12))))


def _group_rows(examples: list[TrainingExample]) -> dict[str, list[RankingFeatureRow]]:
    grouped: dict[str, list[RankingFeatureRow]] = {}
    for example in examples:
        grouped.setdefault(example.row.request_id, []).append(example.row)
    return grouped
