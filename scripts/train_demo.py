"""Train and compare MarketRank models using deterministic simulated marketplace data."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from marketrank.pipelines.demo import run_demo_training


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MarketRank's reproducible training demo")
    parser.add_argument("--requests", type=int, default=120)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    report = run_demo_training(request_count=args.requests, random_seed=args.seed)
    print(json.dumps(asdict(report), indent=2))


if __name__ == "__main__":
    main()
