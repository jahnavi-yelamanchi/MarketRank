from marketrank.monitoring.observability import (
    DecisionLog,
    decision_event,
    detect_feature_drift,
    summarize_decision_quality,
)


def test_decision_log_is_bounded_and_summarizes_serving_health() -> None:
    log = DecisionLog(max_events=2)
    log.record(
        decision_event(
            request_id="one",
            policy="heuristic",
            fallback=None,
            candidate_count=3,
            selected_provider_id="provider-a",
            selected_score=0.8,
            latency_ms=8.0,
        )
    )
    log.record(
        decision_event(
            request_id="two",
            policy="heuristic",
            fallback="expanded_radius",
            candidate_count=0,
            selected_provider_id=None,
            selected_score=None,
            latency_ms=12.0,
        )
    )
    log.record(
        decision_event(
            request_id="three",
            policy="heuristic",
            fallback=None,
            candidate_count=2,
            selected_provider_id="provider-a",
            selected_score=0.7,
            latency_ms=10.0,
        )
    )

    report = summarize_decision_quality(log.snapshot())

    assert report.decisions == 2
    assert report.no_match_rate == 0.5
    assert report.top_provider_exposure_share == 0.5
    assert report.mean_latency_ms == 11.0


def test_feature_drift_detects_distribution_shift() -> None:
    stable = detect_feature_drift("price_fit", [0.1, 0.2, 0.3, 0.4] * 20, [0.1, 0.2, 0.3, 0.4] * 20)
    shifted = detect_feature_drift("price_fit", [0.1, 0.2, 0.3, 0.4] * 20, [0.99] * 80)

    assert stable.population_stability_index < 0.01
    assert not stable.drift_detected
    assert shifted.population_stability_index > 0.2
    assert shifted.drift_detected
