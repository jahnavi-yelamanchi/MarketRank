"""Transparent ranking policies used as evaluation controls."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from marketrank.features.point_in_time import RankingFeatureRow

BaselinePolicy = Literal["heuristic", "nearest", "cheapest", "highest_rated"]


@dataclass(frozen=True)
class RankedCandidate:
    row: RankingFeatureRow
    score: float
    policy: str
    reasons: list[str]


def rank_baseline(
    rows: list[RankingFeatureRow], policy: BaselinePolicy = "heuristic"
) -> list[RankedCandidate]:
    """Rank a feasible candidate set with a comparable deterministic policy."""
    scoring = {
        "heuristic": _heuristic_score,
        "nearest": lambda values: -values["distance_km"],
        "cheapest": lambda values: -values["price"],
        "highest_rated": lambda values: values["quality"],
    }
    if policy not in scoring:
        raise ValueError(f"Unknown baseline policy: {policy}")

    ranked = [
        RankedCandidate(
            row=row,
            score=scoring[policy](row.values),
            policy=policy,
            reasons=_reasons(row, policy),
        )
        for row in rows
    ]
    return sorted(ranked, key=lambda candidate: (-candidate.score, candidate.row.offer_id))


def _heuristic_score(values: dict[str, float]) -> float:
    """Deliberately simple marketplace utility, not a learned proxy."""
    distance_fit = 1 / (1 + values["distance_km"] / 20)
    return (
        0.27 * values["quality"]
        + 0.22 * values["completion_rate"]
        + 0.18 * values["user_price_fit"]
        + 0.13 * distance_fit
        + 0.1 * values["user_category_affinity"]
        + 0.1 * values["freshness_score"]
        - 0.1 * values["provider_utilization"]
    )


def _reasons(row: RankingFeatureRow, policy: BaselinePolicy) -> list[str]:
    values = row.values
    if policy == "nearest":
        return [f"Nearest feasible provider at {values['distance_km']:.1f} km"]
    if policy == "cheapest":
        return [f"Lowest feasible price at {values['price']:.2f}"]
    if policy == "highest_rated":
        return [f"Highest provider quality at {values['quality']:.2f}"]
    return [
        f"Quality {values['quality']:.2f} and completion rate {values['completion_rate']:.2f}",
        f"Price fit {values['user_price_fit']:.2f}; distance {values['distance_km']:.1f} km",
    ]
