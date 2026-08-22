from datetime import datetime

import pytest

from marketrank.retrieval.candidates import CandidateRetriever
from marketrank.simulation.marketplace import (
    Location,
    MarketplaceRequest,
    MarketplaceState,
    SupplyOffer,
    haversine_km,
)


def request() -> MarketplaceRequest:
    return MarketplaceRequest(
        request_id="r1",
        user_id="u1",
        category="books",
        location=Location(-23.55, -46.63),
        budget=50,
        quantity=1,
        created_at=datetime(2026, 1, 1),
        max_distance_km=20,
    )


def offer(**overrides: object) -> SupplyOffer:
    values: dict[str, object] = {
        "offer_id": "o1",
        "provider_id": "p1",
        "category": "books",
        "location": Location(-23.56, -46.64),
        "price": 30.0,
        "quality": 1.0,
        "completion_rate": 1.0,
        "inventory": 1,
        "capacity": 1,
    }
    values.update(overrides)
    return SupplyOffer(**values)  # type: ignore[arg-type]


def test_feasible_candidates_enforce_operational_constraints() -> None:
    live = offer()
    full = offer(offer_id="o2", provider_id="p2", active_assignments=1)
    expensive = offer(offer_id="o3", provider_id="p3", price=60.0)
    wrong_category = offer(offer_id="o4", provider_id="p4", category="toys")
    offers = [live, full, expensive, wrong_category]
    state = MarketplaceState({item.offer_id: item for item in offers})

    candidates = state.feasible_candidates(request())

    assert [candidate.offer.offer_id for candidate in candidates] == ["o1"]


def test_acceptance_reserves_inventory_and_capacity_until_completion() -> None:
    live = offer()
    state = MarketplaceState({live.offer_id: live}, random_seed=1)

    attempt = state.attempt_match(request(), "o1")

    assert attempt.accepted
    assert live.inventory == 0
    assert live.active_assignments == 1
    assert state.feasible_candidates(request()) == []

    completion = state.complete_match("r1")

    assert completion.completed
    assert live.active_assignments == 0


def test_distance_is_approximately_one_degree_of_latitude() -> None:
    assert haversine_km(Location(0, 0), Location(1, 0)) == pytest.approx(111.2, abs=0.2)


def test_retrieval_expands_radius_only_for_sparse_supply() -> None:
    distant = offer(location=Location(-23.85, -46.63))
    state = MarketplaceState({distant.offer_id: distant})

    retriever = CandidateRetriever(minimum_candidates=1, maximum_radius_km=50)
    result = retriever.retrieve(request=request(), state=state)

    assert result.fallback == "expanded_radius"
    assert result.applied_radius_km == 40
    assert [candidate.offer.offer_id for candidate in result.candidates] == ["o1"]
