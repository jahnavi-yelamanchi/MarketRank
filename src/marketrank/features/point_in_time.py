"""Feature computation for feasible request-provider pairs."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from math import exp, isfinite

from marketrank.simulation.marketplace import FeasibleCandidate, MarketplaceRequest


class FeatureValidationError(ValueError):
    """Raised when feature creation would violate the point-in-time contract."""


@dataclass(frozen=True)
class UserProfile:
    category_affinity: dict[str, float] = field(default_factory=dict)
    price_sensitivity: float = 0.5
    historical_requests: int = 0


@dataclass(frozen=True)
class ProviderProfile:
    historical_matches: int = 0


@dataclass(frozen=True)
class FeatureContext:
    user_profiles: dict[str, UserProfile] = field(default_factory=dict)
    provider_profiles: dict[str, ProviderProfile] = field(default_factory=dict)
    provider_exposure: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class RankingFeatureRow:
    request_id: str
    offer_id: str
    provider_id: str
    values: dict[str, float]


def compute_features(
    request: MarketplaceRequest,
    candidates: list[FeasibleCandidate],
    context: FeatureContext | None = None,
    as_of: datetime | None = None,
) -> list[RankingFeatureRow]:
    """Create validated numerical features using information available at ``as_of``."""
    context = context or FeatureContext()
    as_of = _as_utc(as_of or request.created_at)
    _assert_no_future_availability(candidates, as_of)
    user = context.user_profiles.get(request.user_id, UserProfile())
    exposure_total = max(1, sum(context.provider_exposure.values()))
    rows = []
    for candidate in candidates:
        offer = candidate.offer
        provider = context.provider_profiles.get(offer.provider_id, ProviderProfile())
        freshness = _freshness_score(offer.availability_updated_at, as_of)
        values = {
            "distance_km": candidate.distance_km,
            "delivery_time_hours": 0.4 + candidate.distance_km / 35,
            "price": offer.price,
            "price_to_budget": offer.price / request.budget,
            "quality": offer.quality,
            "completion_rate": offer.completion_rate,
            "inventory": float(offer.inventory),
            "capacity_remaining": float(offer.capacity - offer.active_assignments),
            "provider_utilization": offer.utilization,
            "freshness_score": freshness,
            "user_category_affinity": user.category_affinity.get(request.category, 0.5),
            "user_price_fit": _price_fit(request.budget, offer.price, user.price_sensitivity),
            "provider_exposure_share": context.provider_exposure.get(offer.provider_id, 0)
            / exposure_total,
            "is_new_user": float(user.historical_requests == 0),
            "is_new_provider": float(provider.historical_matches == 0),
            "candidate_count": float(len(candidates)),
        }
        _validate_feature_values(values)
        rows.append(
            RankingFeatureRow(request.request_id, offer.offer_id, offer.provider_id, values)
        )
    return rows


def _assert_no_future_availability(candidates: list[FeasibleCandidate], as_of: datetime) -> None:
    future_offer_ids = [
        candidate.offer.offer_id
        for candidate in candidates
        if candidate.offer.availability_updated_at
        and _as_utc(candidate.offer.availability_updated_at) > as_of
    ]
    if future_offer_ids:
        raise FeatureValidationError(
            f"Availability snapshots are newer than the decision time: {future_offer_ids}"
        )


def _freshness_score(updated_at: datetime | None, as_of: datetime) -> float:
    if updated_at is None:
        return 0.5
    age_hours = max(0.0, (as_of - _as_utc(updated_at)).total_seconds() / 3600)
    return exp(-age_hours / 24)


def _price_fit(budget: float, price: float, sensitivity: float) -> float:
    headroom = max(0.0, min(1.0, (budget - price) / budget))
    return (1 - sensitivity) + sensitivity * headroom


def _validate_feature_values(values: dict[str, float]) -> None:
    invalid = [name for name, value in values.items() if not isfinite(value)]
    if invalid:
        raise FeatureValidationError(f"Non-finite feature values: {invalid}")


def _as_utc(value: datetime) -> datetime:
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)
