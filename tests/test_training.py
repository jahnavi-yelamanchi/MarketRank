from marketrank.features.point_in_time import RankingFeatureRow
from marketrank.ranking.training import TrainingExample, train_classifier, train_lambdamart


def examples() -> list[TrainingExample]:
    output = []
    for request_number in range(4):
        for candidate_number in range(3):
            quality = 0.9 - candidate_number * 0.3
            values = {
                "capacity_remaining": 2.0,
                "candidate_count": 3.0,
                "completion_rate": quality,
                "delivery_time_hours": 1.0 + candidate_number,
                "distance_km": float(candidate_number + 1),
                "freshness_score": 1.0,
                "inventory": 3.0,
                "is_new_provider": 0.0,
                "is_new_user": 0.0,
                "price": 20.0 + candidate_number,
                "price_to_budget": 0.5,
                "provider_exposure_share": 0.1,
                "provider_utilization": 0.2,
                "quality": quality,
                "user_category_affinity": 0.8,
                "user_price_fit": 0.9 - candidate_number * 0.1,
            }
            row = RankingFeatureRow(f"r{request_number}", f"o{candidate_number}", "p", values)
            output.append(TrainingExample(row=row, label=float(candidate_number == 0)))
    return output


def test_classifier_and_lambdamart_rank_feature_rows() -> None:
    classifier = train_classifier(examples())
    lambdamart = train_lambdamart(examples())
    request_rows = [example.row for example in examples()[:3]]

    assert len(classifier.rank(request_rows)) == 3
    assert len(lambdamart.rank(request_rows)) == 3
    assert lambdamart.rank(request_rows)[0].row.offer_id == "o0"
