"""Transform validated Olist history into MarketRank's two-sided entities."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from math import ceil, sqrt

import pandas as pd

from marketrank.data.olist import OlistDataset
from marketrank.simulation.marketplace import Location, MarketplaceRequest, SupplyOffer


@dataclass(frozen=True)
class OlistMarketplaceSeed:
    offers: list[SupplyOffer]
    requests: list[MarketplaceRequest]


def build_olist_marketplace_seed(dataset: OlistDataset) -> OlistMarketplaceSeed:
    """Create operational supply and observed demand from validated public history.

    Olist does not log counterfactual eligible sellers. This transformation provides
    realistic geography, prices, categories, and provider reliability; the simulator
    generates the unobserved live inventory and response outcomes.
    """
    coordinates = (
        dataset.geolocation.groupby("geolocation_zip_code_prefix")
        .agg(latitude=("geolocation_lat", "median"), longitude=("geolocation_lng", "median"))
        .reset_index()
    )
    seller_locations = dataset.sellers.merge(
        coordinates,
        left_on="seller_zip_code_prefix",
        right_on="geolocation_zip_code_prefix",
        how="inner",
    )
    customer_locations = dataset.customers.merge(
        coordinates,
        left_on="customer_zip_code_prefix",
        right_on="geolocation_zip_code_prefix",
        how="inner",
    )
    order_quality = dataset.orders[
        ["order_id", "customer_id", "order_status", "order_purchase_timestamp"]
    ]
    order_quality = order_quality.merge(
        dataset.reviews.groupby("order_id", as_index=False).review_score.mean(),
        on="order_id",
        how="left",
    )
    history = dataset.order_items.merge(
        dataset.products[["product_id", "product_category_name"]], on="product_id", how="inner"
    ).merge(order_quality, on="order_id", how="inner")
    history = history.dropna(subset=["product_category_name"])

    offers = _build_offers(history, seller_locations)
    requests = _build_requests(history, customer_locations)
    if not offers or not requests:
        raise ValueError("Olist data did not yield usable geolocated supply and demand")
    return OlistMarketplaceSeed(offers=offers, requests=requests)


def _build_offers(history: pd.DataFrame, seller_locations: pd.DataFrame) -> list[SupplyOffer]:
    located_history = history.merge(
        seller_locations[["seller_id", "latitude", "longitude"]], on="seller_id", how="inner"
    )
    global_quality = (
        float(history.review_score.mean() / 5) if history.review_score.notna().any() else 0.7
    )
    offers = []
    for (provider_id, category), group in located_history.groupby(
        ["seller_id", "product_category_name"]
    ):
        order_count = group.order_id.nunique()
        quality = (
            float(group.review_score.mean() / 5)
            if group.review_score.notna().any()
            else global_quality
        )
        location = Location(float(group.latitude.iloc[0]), float(group.longitude.iloc[0]))
        offers.append(
            SupplyOffer(
                offer_id=f"{provider_id}:{category}",
                provider_id=str(provider_id),
                category=str(category),
                location=location,
                price=float(group.price.median() + group.freight_value.median()),
                quality=min(1.0, max(0.0, quality)),
                completion_rate=float((group.order_status == "delivered").mean()),
                inventory=max(1, min(20, ceil(sqrt(order_count)))),
                capacity=max(1, min(5, ceil(sqrt(order_count) / 2))),
            )
        )
    return offers


def _build_requests(
    history: pd.DataFrame, customer_locations: pd.DataFrame
) -> list[MarketplaceRequest]:
    located_history = history.merge(
        customer_locations[["customer_id", "customer_unique_id", "latitude", "longitude"]],
        on="customer_id",
        how="inner",
    )
    requests = []
    for row in located_history.itertuples(index=False):
        created_at = datetime.fromisoformat(str(row.order_purchase_timestamp))
        requests.append(
            MarketplaceRequest(
                request_id=f"{row.order_id}:{row.order_item_id}",
                user_id=str(row.customer_unique_id),
                category=str(row.product_category_name),
                location=Location(float(row.latitude), float(row.longitude)),
                budget=float((row.price + row.freight_value) * 1.2),
                quantity=1,
                created_at=created_at,
            )
        )
    return requests
