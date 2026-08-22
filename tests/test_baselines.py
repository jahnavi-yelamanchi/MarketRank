from marketrank.features.point_in_time import RankingFeatureRow
from marketrank.ranking.baselines import rank_baseline


def rows() -> list[RankingFeatureRow]:
    shared = {
        "completion_rate": 0.8,
        "freshness_score": 1.0,
        "user_category_affinity": 0.5,
        "user_price_fit": 0.5,
        "provider_utilization": 0.1,
    }
    return [
        RankingFeatureRow(
            "r1",
            "near-expensive",
            "p1",
            shared | {"distance_km": 1, "price": 40, "quality": 0.7},
        ),
        RankingFeatureRow(
            "r1",
            "far-cheap",
            "p2",
            shared | {"distance_km": 8, "price": 20, "quality": 0.9},
        ),
    ]


def test_baselines_preserve_their_declared_objective() -> None:
    assert rank_baseline(rows(), "nearest")[0].row.offer_id == "near-expensive"
    assert rank_baseline(rows(), "cheapest")[0].row.offer_id == "far-cheap"
    assert rank_baseline(rows(), "highest_rated")[0].row.offer_id == "far-cheap"


def test_heuristic_returns_reasons_for_an_explainable_control() -> None:
    result = rank_baseline(rows(), "heuristic")

    assert len(result) == 2
    assert result[0].reasons
