from datetime import UTC, datetime

from marketrank.evaluation.experiments import compare_policies
from marketrank.serving.service import demo_marketplace_state
from marketrank.simulation.marketplace import Location, MarketplaceRequest


def test_compare_policies_replays_marketplace_constraints() -> None:
    requests = [
        MarketplaceRequest(
            request_id=f"request-{index}",
            user_id=f"user-{index % 2}",
            category="books",
            location=Location(-23.5505, -46.6333),
            budget=30,
            quantity=1,
            created_at=datetime.now(UTC),
        )
        for index in range(8)
    ]

    results = compare_policies(
        demo_marketplace_state(),
        requests,
        ["nearest", "cheapest", "heuristic", "heuristic_reranked"],
    )

    assert [result.policy for result in results] == [
        "nearest",
        "cheapest",
        "heuristic",
        "heuristic_reranked",
    ]
    assert all(len(result.events) == len(requests) for result in results)
    assert all(0 <= result.health.completion_rate <= 1 for result in results)
