"""Local load benchmark for the candidate-to-rerank serving path."""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from time import perf_counter

from marketrank.serving.service import MarketplaceMatchingService, MatchInput


@dataclass(frozen=True)
class LatencyBenchmark:
    requests: int
    workers: int
    errors: int
    elapsed_seconds: float
    throughput_per_second: float
    p50_ms: float
    p95_ms: float
    p99_ms: float

    def as_dict(self) -> dict[str, float | int]:
        return asdict(self)


def run_match_benchmark(
    service: MarketplaceMatchingService,
    request_factory: Callable[[int], MatchInput],
    *,
    iterations: int = 100,
    workers: int = 4,
) -> LatencyBenchmark:
    """Measure concurrent in-process match latency without network overhead."""
    if iterations < 1:
        raise ValueError("iterations must be positive")
    if workers < 1:
        raise ValueError("workers must be positive")

    latencies: list[float] = []
    errors = 0
    started_at = perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(_timed_match, service, request_factory(index))
            for index in range(iterations)
        ]
        for future in as_completed(futures):
            try:
                latencies.append(future.result())
            except Exception:  # noqa: BLE001 - benchmark must count service failures
                errors += 1
    elapsed_seconds = perf_counter() - started_at
    return LatencyBenchmark(
        requests=iterations,
        workers=workers,
        errors=errors,
        elapsed_seconds=elapsed_seconds,
        throughput_per_second=iterations / elapsed_seconds,
        p50_ms=_percentile(latencies, 50),
        p95_ms=_percentile(latencies, 95),
        p99_ms=_percentile(latencies, 99),
    )


def _timed_match(service: MarketplaceMatchingService, match_input: MatchInput) -> float:
    started_at = perf_counter()
    service.match(match_input)
    return (perf_counter() - started_at) * 1_000


def _percentile(values: list[float], percentile: int) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = round((percentile / 100) * (len(ordered) - 1))
    return ordered[index]
