# MarketRank

MarketRank is an explainable matching system for a two-sided marketplace. It ranks
feasible sellers for a buyer request while respecting live supply constraints and
balancing conversion with price, quality, delivery distance, and supplier exposure.

Unlike a catalog recommender, a result is only useful when the seller can actually
fulfil it now. MarketRank therefore models the matching path as:

```text
request -> retrieval -> point-in-time features -> learned ranker -> constrained reranker
```

The project uses public Olist e-commerce records as a marketplace substrate and a
transparent simulator for counterfactual supply choices, availability, capacity, and
outcomes. Raw source data is never committed.

## Current status

The repository currently contains the production skeleton and architecture contract.
Next milestones add data validation, simulated marketplace state, candidate retrieval,
and ranking policies.

## Quick start

```bash
make install
make test
make run
curl http://localhost:8000/health
```

## Design contract

See [`docs/architecture.md`](docs/architecture.md) for product requirements, data
assumptions, ranking objective, constraints, metrics, and scale tradeoffs.

## Repository layout

```text
src/marketrank/
  data/          # source adapters and validation
  simulation/    # live marketplace state and outcome generation
  features/      # point-in-time feature computation
  retrieval/     # feasible candidate generation
  ranking/       # baselines and learning-to-rank models
  reranking/     # capacity, fairness, and utility tradeoffs
  evaluation/    # offline and online-style policy evaluation
  serving/       # request schemas and application endpoints
  monitoring/    # data, drift, and ranking-quality checks
tests/
```
