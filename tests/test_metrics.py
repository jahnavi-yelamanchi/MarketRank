import pytest

from marketrank.evaluation.metrics import (
    OnlineMatchEvent,
    evaluate_marketplace_health,
    evaluate_rankings,
)
from marketrank.features.point_in_time import RankingFeatureRow
from marketrank.ranking.baselines import RankedCandidate


def ranked(offer_id: str) -> RankedCandidate:
    row = RankingFeatureRow("r1", offer_id, offer_id, {})
    return RankedCandidate(row=row, score=1.0, policy="test", reasons=[])


def test_ranking_metrics_reward_relevant_items_near_the_top() -> None:
    report = evaluate_rankings(
        {"r1": [ranked("good"), ranked("bad"), ranked("also-good")]},
        {("r1", "good"): 1.0, ("r1", "also-good"): 1.0},
        k=2,
    )

    assert report.recall_at_k == 0.5
    assert report.ndcg_at_k == pytest.approx(0.613, abs=0.001)
    assert report.mrr == 1
    assert report.map == pytest.approx(0.833, abs=0.001)
    assert report.acceptance_proxy == 1


def test_marketplace_health_exposes_supply_concentration() -> None:
    events = [
        OnlineMatchEvent("r1", "p1", 3, True, True, 10),
        OnlineMatchEvent("r2", "p1", 2, True, False, 20),
        OnlineMatchEvent("r3", None, 0, False, False, 30),
    ]

    report = evaluate_marketplace_health(events, elapsed_seconds=1.5)

    assert report.no_match_rate == pytest.approx(1 / 3)
    assert report.acceptance_rate == pytest.approx(2 / 3)
    assert report.completion_rate == pytest.approx(1 / 3)
    assert report.supply_exposure_hhi == pytest.approx(4 / 9)
    assert report.mean_supply_utilization == 0
    assert report.throughput_per_second == 2
