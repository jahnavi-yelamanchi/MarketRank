from datetime import datetime, timedelta

import pytest
from test_marketplace import offer, request

from marketrank.features.point_in_time import (
    FeatureContext,
    FeatureValidationError,
    ProviderProfile,
    UserProfile,
    compute_features,
)
from marketrank.simulation.marketplace import FeasibleCandidate


def test_features_capture_marketplace_and_cold_start_signals() -> None:
    match = offer(availability_updated_at=datetime(2025, 12, 31))
    candidate = FeasibleCandidate(match, distance_km=4.0)
    context = FeatureContext(
        user_profiles={
            "u1": UserProfile({"books": 0.9}, price_sensitivity=1, historical_requests=4)
        },
        provider_profiles={"p1": ProviderProfile(historical_matches=10)},
        provider_exposure={"p1": 3, "p2": 1},
    )

    row = compute_features(request(), [candidate], context)[0]

    assert row.values["user_category_affinity"] == 0.9
    assert row.values["is_new_user"] == 0
    assert row.values["is_new_provider"] == 0
    assert row.values["provider_exposure_share"] == 0.75
    assert row.values["delivery_time_hours"] > 0


def test_future_availability_is_rejected_to_prevent_offline_leakage() -> None:
    future = request().created_at + timedelta(minutes=1)
    candidate = FeasibleCandidate(offer(availability_updated_at=future), distance_km=1.0)

    with pytest.raises(FeatureValidationError, match="newer than the decision time"):
        compute_features(request(), [candidate])
