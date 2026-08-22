# MarketRank

MarketRank is an explainable, constraint-aware matching and ranking system for a
**two-sided marketplace**. It chooses which available provider to show a buyer now,
not which catalog item they may like later.

The difference matters: a beautiful relevance score is useless if the provider is
over capacity, out of inventory, too far away, above budget, or repeatedly absorbs
all demand. MarketRank makes those operational conditions first-class inputs to a
real-time decision.

![MarketRank's monochrome ranking inspector on liquid iridescence](src/marketrank/static/assets/liquid-iridescence.png)

## Product problem

A buyer requests a category at a location within a budget. Eligible providers differ
in quality, price, geography, availability, capacity, reliability, and historical
exposure. The system must optimize buyer utility while keeping the marketplace
healthy enough to fulfil tomorrow's requests.

```text
request → feasible candidate generation → point-in-time features → learned ranker
        → constraint-aware reranker → explainable matched results
```

Hard constraints are applied before scoring. The reranker then makes bounded tradeoffs
between conversion utility, price, provider utilization, exposure fairness, and result
diversity.

## What is implemented

- Olist public e-commerce adapter with schema, key, reference, range, and coverage validation
- Live marketplace simulator with inventory, capacity, acceptance, completion, and exposure state
- Feasible retrieval by category, budget, location/radius, inventory, and provider capacity
- Point-in-time features with explicit leakage guardrails and cold-start priors
- Transparent controls: heuristic, nearest, cheapest, and highest-rated policies
- Binary LightGBM classifier and request-grouped LightGBM LambdaMART ranker
- Constraint-aware reranking for price fit, capacity, exposure, and provider diversity
- Offline ranking metrics plus online-style marketplace policy replay
- FastAPI endpoint and an inspectable interface that exposes score, features, reasons, and constraints
- Decision logging, serving-quality summaries, and PSI feature-drift checks
- Local concurrency benchmark, Docker packaging, unit tests, and reproducible commands

## Quick start

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
make install
make test
make run
```

Open `http://localhost:8001` for the ranking inspector. It ships with a deterministic
in-memory marketplace so no data download is required for exploration.

```bash
curl http://localhost:8000/health

curl -X POST http://localhost:8000/v1/matches \
  -H 'content-type: application/json' \
  -d '{
    "request_id":"demo-1", "user_id":"buyer-1", "category":"books",
    "latitude":-23.5505, "longitude":-46.6333, "budget":30
  }'
```

## Data and simulation design

[Olist Brazilian e-commerce data](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)
provides observed orders, sellers, customers, products, reviews, price, and geography.
It does not contain the counterfactual candidate sets a ranker needs—who else was
available, could accept, or would complete. MarketRank therefore uses Olist as a
validated marketplace substrate and explicitly simulates the missing live state.

| Layer | Observed or simulated | Purpose |
| --- | --- | --- |
| Buyer requests, sellers, product category, price, geography | Olist | Marketplace realism |
| Inventory, assignment capacity, availability changes | Simulated | Real-time feasibility |
| Acceptance and completion | Simulated from quality, price, distance, utilization | Policy replay labels |
| Cold start | Smoothed provider/user priors | Safe coverage for new entities |

Raw source data remains local and is never committed. Use
`load_olist_dataset(raw_dir)` followed by `build_olist_marketplace_seed(dataset)` to
turn a downloaded Olist archive into requests and supply offers.

## Ranking formulation

The model estimates a request-provider utility proxy. LambdaMART directly optimizes
ordering within each request; the binary classifier is a lower-complexity control.

```text
utility(request, provider) ≈ P(accept × complete | point-in-time features)

final score = ranker score
            + price-fit adjustment
            - utilization penalty
            - historical exposure penalty
            - repeated-provider diversity penalty
```

The following are **not** soft score terms: category compatibility, budget, service
radius, inventory, and capacity. They are feasibility gates. This prevents a model
from recommending impossible supply merely because it has an attractive historical
conversion score.

Signals include provider quality, completion rate, price/budget fit, distance and ETA,
inventory and capacity remaining, freshness, user-category affinity, new-user and
new-provider flags, and provider exposure share.

## Reproducible experiments

```bash
# Builds deterministic simulated request groups, trains classifier + LambdaMART,
# and compares each against the heuristic control.
make train-demo

# Replays the same request stream through nearest, cheapest, heuristic, and reranked
# policies while state changes after every outcome.
uv run pytest tests/test_experiments.py
```

The training demo reports Recall@K, NDCG@K, MRR, MAP, and an acceptance proxy. The
online-style replay additionally reports no-match rate, acceptance/completion rate,
exposure HHI, latency, and throughput. It is a counterfactual simulator, not a claim
of causal lift; production A/B tests remain necessary.

## Serving and interface

`POST /v1/matches` validates the request and returns only feasible offers:

```json
{
  "policy": "heuristic_fallback",
  "fallback": null,
  "constraints_applied": ["budget", "service_radius", "inventory", "capacity", "category"],
  "matches": [{
    "provider_id": "paper-trail",
    "predicted_score": 0.81,
    "ranking_features": {"quality": 0.92, "price_to_budget": 0.8},
    "reasons": ["Quality 0.92 and completion rate 0.96"]
  }]
}
```

The interface is intentionally quiet: monochrome operational UI over one liquid
iridescent visual surface. It follows basic UX heuristics by keeping one primary
action, preserving visible system status, showing constraints and decision evidence,
and respecting reduced-motion preferences.

## Latency and load benchmark

```bash
make benchmark
```

This measures the **in-process decision path** (retrieval → features → ranking →
reranking) with a deterministic four-offer demo state. A 250-request / 8-worker local
run completed with p50 `0.04 ms`, p95 `0.05 ms`, and ~16.5k decisions/sec; those numbers
exclude HTTP, a network feature store, model loading, and distributed inventory reads.
Use the command on the target machine for an actionable number. The benchmark reports
requests, worker count, errors, elapsed time, throughput, p50, p95, and p99.

## Monitoring and data quality

Each served match creates a bounded, thread-safe `DecisionEvent` containing request
ID, policy, fallback, candidate count, selected provider/score, and latency—without
logging buyer attributes. `summarize_decision_quality` alerts on no-match, fallback,
latency, and provider-concentration changes before delayed labels arrive.

`detect_feature_drift` computes Population Stability Index against a training baseline.
PSI ≥ 0.2 is an investigation threshold, not an automatic rollback. Raw Olist input
validation rejects schema drift, duplicate keys, dangling references, invalid ranges,
and missing delivered/category coverage before feature creation.

## Failure analysis and intended behavior

| Scenario | System behavior | Production follow-up |
| --- | --- | --- |
| Low-supply market | Expands retrieval radius, then returns a truthful no-match | Drive supply acquisition or promise an alternate delivery window |
| New provider | Uses available quality/completion priors and `is_new_provider` | Add exploration quotas and Bayesian uncertainty |
| New user | Uses default category/price priors and `is_new_user` | Learn session intent without storing unnecessary identity data |
| Conflicting objectives | Reranker makes price, utilization, exposure, and diversity adjustments explicit | Tune policy weights through guarded experiments |
| Stale availability | Reranker rechecks live feasibility just before returning results | Read a strongly consistent inventory/capacity service and reconcile reservations |
| Geographic imbalance | Distance gate prevents impossible offers; sparse fallback is visible | Use region-aware supply targets and localized retrieval indexes |
| Popularity bias | Exposure penalty and diversity discourage repeated provider dominance | Monitor exposure HHI by cohort and enforce fairness guardrails |

## At Uber, DoorDash, or Airbnb scale

This repository keeps the boundaries deliberately small and inspectable. At larger
scale, candidate retrieval becomes geo/category ANN plus an availability index;
features move to offline/online feature stores with point-in-time joins; inventory and
capacity require strongly consistent reservation services; training uses sampled
impressions, delayed-label attribution, calibration, and counterfactual correction;
serving requires model registry, cache policy, circuit breakers, and shadow/canary
experiments. Marketplace health constraints become segmented guardrails, not a single
global reranking coefficient.

## Repository map

```text
src/marketrank/
  data/          Olist loading, validation, and entity transformation
  simulation/    Dynamic supply, capacity, acceptance, and completion model
  retrieval/     Feasible candidate generation and sparse-market fallback
  features/      Point-in-time request/provider/candidate features
  ranking/       Heuristics, classifier, LambdaMART
  reranking/     Operational constraints, fairness, diversity adjustments
  evaluation/    Offline metrics and online-style policy replay
  serving/       Composable matching service
  monitoring/    Decision telemetry and PSI drift detection
  benchmarking/  In-process latency/load measurement
  pipelines/     Reproducible synthetic training experiment
  static/        Explainable Marketplace interface
tests/           Unit and integration coverage
```

For the detailed product contract and architectural choices, see
[`docs/architecture.md`](docs/architecture.md).
