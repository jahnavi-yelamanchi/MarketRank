"""Reproducible classifier and LambdaMART training for request-level ranking."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
from lightgbm import Booster, Dataset, train

from marketrank.features.point_in_time import RankingFeatureRow
from marketrank.ranking.baselines import RankedCandidate

FEATURE_NAMES = (
    "capacity_remaining",
    "candidate_count",
    "completion_rate",
    "delivery_time_hours",
    "distance_km",
    "freshness_score",
    "inventory",
    "is_new_provider",
    "is_new_user",
    "price",
    "price_to_budget",
    "provider_exposure_share",
    "provider_utilization",
    "quality",
    "user_category_affinity",
    "user_price_fit",
)


@dataclass(frozen=True)
class TrainingExample:
    row: RankingFeatureRow
    label: float


@dataclass
class TrainedRanker:
    model: Booster
    model_name: str
    feature_names: tuple[str, ...] = FEATURE_NAMES

    def rank(self, rows: list[RankingFeatureRow]) -> list[RankedCandidate]:
        """Score a request's candidates with the model's predicted utility."""
        if not rows:
            return []
        matrix = _feature_matrix(rows, self.feature_names)
        scores = self.model.predict(matrix)
        candidates = [
            RankedCandidate(
                row=row,
                score=float(score),
                policy=self.model_name,
                reasons=[f"{self.model_name} predicted utility {score:.3f}"],
            )
            for row, score in zip(rows, scores, strict=True)
        ]
        return sorted(candidates, key=lambda candidate: (-candidate.score, candidate.row.offer_id))


def train_classifier(examples: list[TrainingExample], random_seed: int = 7) -> TrainedRanker:
    """Train the binary conversion baseline on independent candidate impressions."""
    _validate_examples(examples)
    labels = np.array([int(example.label > 0) for example in examples])
    if len(np.unique(labels)) < 2:
        raise ValueError("Classifier training needs both positive and negative examples")
    model = train(
        {
            "objective": "binary",
            "num_leaves": 7,
            "min_data_in_leaf": 1,
            "learning_rate": 0.1,
            "seed": random_seed,
            "verbosity": -1,
            "num_threads": 1,
        },
        Dataset(
            _feature_matrix([example.row for example in examples], FEATURE_NAMES),
            label=labels,
        ),
        num_boost_round=40,
    )
    return TrainedRanker(model=model, model_name="classifier")


def train_lambdamart(examples: list[TrainingExample], random_seed: int = 7) -> TrainedRanker:
    """Train LambdaMART with request IDs as LightGBM ranking groups."""
    _validate_examples(examples)
    grouped: dict[str, list[TrainingExample]] = defaultdict(list)
    for example in examples:
        grouped[example.row.request_id].append(example)
    if any(len(group) < 2 for group in grouped.values()):
        raise ValueError("Each LambdaMART request group needs at least two candidates")

    ordered = [example for request_id in sorted(grouped) for example in grouped[request_id]]
    group_sizes = [len(grouped[request_id]) for request_id in sorted(grouped)]
    model = train(
        {
            "objective": "lambdarank",
            "metric": "ndcg",
            "num_leaves": 7,
            "min_data_in_leaf": 1,
            "learning_rate": 0.1,
            "seed": random_seed,
            "verbosity": -1,
            "num_threads": 1,
        },
        Dataset(
            _feature_matrix([example.row for example in ordered], FEATURE_NAMES),
            label=np.array([example.label for example in ordered]),
            group=group_sizes,
        ),
        num_boost_round=40,
    )
    return TrainedRanker(model=model, model_name="lambdamart")


def _feature_matrix(rows: list[RankingFeatureRow], feature_names: tuple[str, ...]) -> np.ndarray:
    missing = {
        name
        for row in rows
        for name in feature_names
        if name not in row.values
    }
    if missing:
        raise ValueError(f"Missing model features: {sorted(missing)}")
    return np.array([[row.values[name] for name in feature_names] for row in rows], dtype=float)


def _validate_examples(examples: list[TrainingExample]) -> None:
    if not examples:
        raise ValueError("At least one training example is required")
    _feature_matrix([example.row for example in examples], FEATURE_NAMES)
