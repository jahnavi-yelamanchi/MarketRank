from marketrank.benchmarking.serving import run_match_benchmark
from marketrank.serving.service import (
    MarketplaceMatchingService,
    MatchInput,
    demo_marketplace_state,
)


def test_serving_benchmark_reports_latency_and_throughput() -> None:
    service = MarketplaceMatchingService(demo_marketplace_state())
    report = run_match_benchmark(
        service,
        lambda index: MatchInput(
            request_id=f"request-{index}",
            user_id="user",
            category="books",
            latitude=-23.5505,
            longitude=-46.6333,
            budget=30,
        ),
        iterations=6,
        workers=2,
    )

    assert report.requests == 6
    assert report.errors == 0
    assert report.throughput_per_second > 0
    assert report.p95_ms >= report.p50_ms
