from test_marketplace import offer, request

from marketrank.features.point_in_time import RankingFeatureRow
from marketrank.ranking.baselines import RankedCandidate
from marketrank.reranking.policy import ConstraintAwareReranker
from marketrank.simulation.marketplace import MarketplaceState


def ranked(offer_id: str, provider_id: str, score: float, exposure: float) -> RankedCandidate:
    values = {
        "price_to_budget": 0.8,
        "provider_utilization": 0.1,
        "provider_exposure_share": exposure,
    }
    row = RankingFeatureRow("r1", offer_id, provider_id, values)
    return RankedCandidate(row=row, score=score, policy="model", reasons=[])


def test_reranker_filters_stale_capacity_and_penalizes_overexposed_supply() -> None:
    overexposed = offer(offer_id="o1", provider_id="p1")
    underexposed = offer(offer_id="o2", provider_id="p2")
    full = offer(offer_id="o3", provider_id="p3", active_assignments=1)
    state = MarketplaceState({item.offer_id: item for item in [overexposed, underexposed, full]})
    candidates = [
        ranked("o1", "p1", score=0.8, exposure=1.0),
        ranked("o2", "p2", score=0.8, exposure=0.0),
        ranked("o3", "p3", score=1.0, exposure=0.0),
    ]

    result = ConstraintAwareReranker().rerank(request(), state, candidates)

    assert result.excluded_offer_ids == ["o3"]
    assert result.candidates[0].row.offer_id == "o2"
    assert "capacity" in result.constraints_applied
