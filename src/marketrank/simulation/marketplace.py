"""Deterministic operational model for two-sided marketplace experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from math import asin, cos, radians, sin, sqrt
from random import Random


@dataclass(frozen=True)
class Location:
    latitude: float
    longitude: float


@dataclass(frozen=True)
class MarketplaceRequest:
    request_id: str
    user_id: str
    category: str
    location: Location
    budget: float
    quantity: int
    created_at: datetime
    max_distance_km: float = 50.0


@dataclass
class SupplyOffer:
    offer_id: str
    provider_id: str
    category: str
    location: Location
    price: float
    quality: float
    completion_rate: float
    inventory: int
    capacity: int
    active_assignments: int = 0
    availability_updated_at: datetime | None = None

    @property
    def utilization(self) -> float:
        return self.active_assignments / self.capacity if self.capacity else 1.0


@dataclass(frozen=True)
class FeasibleCandidate:
    offer: SupplyOffer
    distance_km: float


@dataclass(frozen=True)
class MatchAttempt:
    request_id: str
    offer_id: str | None
    accepted: bool
    acceptance_probability: float
    reason: str


@dataclass(frozen=True)
class MatchCompletion:
    request_id: str
    completed: bool
    completion_probability: float


@dataclass
class MarketplaceState:
    """Mutable live supply state with auditable hard-constraint enforcement."""

    offers: dict[str, SupplyOffer]
    random_seed: int = 7
    exposure: dict[str, int] = field(default_factory=dict)
    _reservations: dict[str, str] = field(default_factory=dict, init=False, repr=False)
    _rng: Random = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._rng = Random(self.random_seed)
        if len(self.offers) != len(set(self.offers)):
            raise ValueError("Offer IDs must be unique")

    def feasible_candidates(self, request: MarketplaceRequest) -> list[FeasibleCandidate]:
        """Return supply that can fulfil the request now, before any score is applied."""
        candidates = []
        for offer in self.offers.values():
            if offer.category != request.category:
                continue
            if offer.price > request.budget or offer.inventory < request.quantity:
                continue
            if offer.active_assignments >= offer.capacity:
                continue
            distance_km = haversine_km(request.location, offer.location)
            if distance_km > request.max_distance_km:
                continue
            candidates.append(FeasibleCandidate(offer=offer, distance_km=distance_km))
        return sorted(candidates, key=lambda candidate: candidate.distance_km)

    def attempt_match(self, request: MarketplaceRequest, offer_id: str) -> MatchAttempt:
        """Sample acceptance and reserve capacity only after successful acceptance."""
        candidate = next(
            (
                item
                for item in self.feasible_candidates(request)
                if item.offer.offer_id == offer_id
            ),
            None,
        )
        if candidate is None:
            return MatchAttempt(request.request_id, None, False, 0.0, "offer_not_feasible")

        probability = self._acceptance_probability(request, candidate)
        accepted = self._rng.random() < probability
        if not accepted:
            return MatchAttempt(
                request.request_id, offer_id, False, probability, "provider_declined"
            )

        offer = candidate.offer
        offer.inventory -= request.quantity
        offer.active_assignments += 1
        self.exposure[offer.provider_id] = self.exposure.get(offer.provider_id, 0) + 1
        self._reservations[request.request_id] = offer_id
        return MatchAttempt(request.request_id, offer_id, True, probability, "reserved")

    def complete_match(self, request_id: str) -> MatchCompletion:
        """Release reserved capacity and sample completion for a previously accepted match."""
        offer_id = self._reservations.pop(request_id, None)
        if offer_id is None:
            raise ValueError(f"No active reservation for request {request_id}")

        offer = self.offers[offer_id]
        probability = min(
            0.99,
            max(0.05, 0.25 + 0.4 * offer.quality + 0.35 * offer.completion_rate),
        )
        completed = self._rng.random() < probability
        offer.active_assignments -= 1
        return MatchCompletion(request_id, completed, probability)

    def _acceptance_probability(
        self, request: MarketplaceRequest, candidate: FeasibleCandidate
    ) -> float:
        offer = candidate.offer
        price_fit = min(1.0, request.budget / offer.price)
        distance_fit = max(0.0, 1.0 - candidate.distance_km / request.max_distance_km)
        raw = (
            0.05
            + 0.32 * offer.quality
            + 0.28 * offer.completion_rate
            + 0.15 * price_fit
            + 0.15 * distance_fit
            - 0.2 * offer.utilization
        )
        return min(0.98, max(0.02, raw))


def haversine_km(left: Location, right: Location) -> float:
    """Return great-circle distance using the Earth's mean radius in kilometres."""
    latitude_delta = radians(right.latitude - left.latitude)
    longitude_delta = radians(right.longitude - left.longitude)
    a = (
        sin(latitude_delta / 2) ** 2
        + cos(radians(left.latitude))
        * cos(radians(right.latitude))
        * sin(longitude_delta / 2) ** 2
    )
    return 6371.0088 * 2 * asin(sqrt(a))
