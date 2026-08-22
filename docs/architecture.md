# MarketRank architecture and product contract

## Product problem

A buyer needs a supplier for a product category, at a location and time, within a
budget. Suppliers differ in price, distance, quality, capacity, availability, and
historical reliability. Ranking must serve the buyer without exhausting a small group
of popular suppliers or recommending supply that cannot accept the request.

The product returns a ranked list of feasible matches and an explanation containing
the model score, material ranking features, and every hard constraint applied.

## Scope

The initial vertical is local e-commerce fulfilment. A request represents a buyer
seeking an item category; a supply offer represents a seller able to fulfil that
category. Olist provides the observed marketplace substrate. The simulator supplies
the missing counterfactuals: which eligible sellers were available, which would accept,
and whether a completed match would have occurred.

Not in scope: payments, order management, authentication, or true causal production
experimentation. Those are operational systems adjacent to, not part of, the ranking
decision.

## Functional requirements

1. Reject malformed requests and return only category-compatible, live candidates.
2. Enforce budget, service radius, inventory, and capacity as hard constraints.
3. Rank candidates using point-in-time data only.
4. Support anonymous/new users and new providers without missing-feature failures.
5. Expand the retrieval radius or expose an explicit no-match fallback in sparse areas.
6. Emit a decision record suitable for replay, experiments, and quality monitoring.

## Request and supply representation

| Entity | Required fields | Examples of derived signals |
| --- | --- | --- |
| Request | request ID, user ID, category, location, budget, timestamp | price sensitivity, category affinity, local supply count |
| Offer | provider ID, category, location, price, inventory, capacity | completion rate, rating, freshness, exposure, utilization |
| Candidate | request + offer at decision time | distance, delivery ETA, budget gap, user-provider affinity |

## Decision flow

```text
buyer request
    |
    v
validate and build request context
    |
    v
retrieve category-compatible supply within a service radius
    |
    v
filter unavailable, over-capacity, and over-budget supply
    |
    v
compute point-in-time request, offer, and interaction features
    |
    v
learned ranker: predicted acceptance/completion utility
    |
    v
rerank: diversity, supplier exposure, utilization, and price tradeoffs
    |
    v
explainable ranked matches + decision event
```

## Ranking formulation

The learned ranker estimates a simulated acceptance-and-completion utility proxy for
an eligible provider. LambdaMART is the target model because it directly optimizes
ordering within each request. A binary classifier is retained as a simpler baseline.

The reranker combines the learned utility with bounded marketplace adjustments:

```text
final_score = ranker_score
            + price_fit_bonus
            - utilization_penalty - exposure_penalty - diversity_penalty
```

Hard constraints are not score terms. An unavailable, full, incompatible, distant, or
over-budget offer is removed before ranking.

## Objectives and metrics

Offline ranking: Recall@K, NDCG@K, MRR, MAP, and acceptance proxy, segmented by
supply density and cold-start state.

Online-style simulation: completion rate, no-match rate, mean provider utilization,
exposure concentration, latency, and throughput.

Policy reports compare heuristic, nearest, cheapest, highest-rated, binary-ML, and
LambdaMART policies. The report makes tradeoffs explicit: conversion versus price,
user utility versus exposure fairness, and quality versus serving latency.

## Operational principles

- **Point-in-time correctness:** no feature may inspect an event after its decision time.
- **Fallbacks:** a sparse market first expands the radius, then relaxes soft preferences,
  then reports no feasible match rather than returning impossible supply.
- **Cold start:** use category/location priors for a new user and Bayesian-smoothed
  provider quality for a new provider.
- **Observability:** every decision event has a request ID, policy/model version,
  latency, candidate count, fallback, and selected result. API responses expose the
  policy, constraints, score, features, and reasons for every returned match.

## Scale path

At marketplace scale, candidate generation moves to a geo/category index, online
features move to a feature store, availability is read from a strongly consistent
inventory/capacity service, and policy changes use guarded online experiments. This
portfolio implementation keeps those boundaries visible while using in-process,
deterministic components that can run on a laptop.
