from marketrank.pipelines.demo import run_demo_training


def test_demo_training_is_reproducible_and_compares_rankers() -> None:
    report = run_demo_training(request_count=30, random_seed=9)

    assert report.examples > 30
    assert 0 < report.train_requests + report.test_requests <= 30
    assert 0 <= report.heuristic.ndcg_at_k <= 1
    assert 0 <= report.classifier.ndcg_at_k <= 1
    assert 0 <= report.lambdamart.ndcg_at_k <= 1
