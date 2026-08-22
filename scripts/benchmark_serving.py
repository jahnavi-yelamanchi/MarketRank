"""Run a reproducible local serving-path benchmark.

Example: uv run python scripts/benchmark_serving.py --iterations 250 --workers 8
"""

from __future__ import annotations

import argparse
import json

from marketrank.benchmarking.serving import run_match_benchmark
from marketrank.serving.service import (
    MarketplaceMatchingService,
    MatchInput,
    demo_marketplace_state,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark MarketRank's in-process matching path")
    parser.add_argument("--iterations", type=int, default=250)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    service = MarketplaceMatchingService(demo_marketplace_state())
    report = run_match_benchmark(
        service,
        lambda index: MatchInput(
            request_id=f"benchmark-{index}",
            user_id=f"benchmark-user-{index % 25}",
            category="books",
            latitude=-23.5505,
            longitude=-46.6333,
            budget=30.0,
        ),
        iterations=args.iterations,
        workers=args.workers,
    )
    print(json.dumps(report.as_dict(), indent=2))


if __name__ == "__main__":
    main()
