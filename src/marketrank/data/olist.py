"""Validated adapter for the public Olist Brazilian e-commerce dataset.

Raw Olist records are the observed marketplace substrate. Later simulation steps add
the counterfactual availability and acceptance outcomes that a ranking system needs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

OLIST_SOURCE_URL = "https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce"

_FILES = {
    "orders": "olist_orders_dataset.csv",
    "order_items": "olist_order_items_dataset.csv",
    "customers": "olist_customers_dataset.csv",
    "sellers": "olist_sellers_dataset.csv",
    "products": "olist_products_dataset.csv",
    "reviews": "olist_order_reviews_dataset.csv",
    "geolocation": "olist_geolocation_dataset.csv",
}

_REQUIRED_COLUMNS = {
    "orders": {"order_id", "customer_id", "order_status", "order_purchase_timestamp"},
    "order_items": {
        "order_id",
        "order_item_id",
        "product_id",
        "seller_id",
        "price",
        "freight_value",
    },
    "customers": {
        "customer_id",
        "customer_unique_id",
        "customer_zip_code_prefix",
        "customer_city",
        "customer_state",
    },
    "sellers": {"seller_id", "seller_zip_code_prefix", "seller_city", "seller_state"},
    "products": {"product_id", "product_category_name"},
    "reviews": {"review_id", "order_id", "review_score"},
    "geolocation": {"geolocation_zip_code_prefix", "geolocation_lat", "geolocation_lng"},
}

_PRIMARY_KEYS = {
    "orders": ["order_id"],
    "customers": ["customer_id"],
    "sellers": ["seller_id"],
    "products": ["product_id"],
}


class DataValidationError(ValueError):
    """Raised when raw marketplace data cannot safely enter the pipeline."""


@dataclass(frozen=True)
class OlistDataset:
    """Source tables required by MarketRank's Olist-to-marketplace adapter."""

    orders: pd.DataFrame
    order_items: pd.DataFrame
    customers: pd.DataFrame
    sellers: pd.DataFrame
    products: pd.DataFrame
    reviews: pd.DataFrame
    geolocation: pd.DataFrame

    def tables(self) -> dict[str, pd.DataFrame]:
        return {
            "orders": self.orders,
            "order_items": self.order_items,
            "customers": self.customers,
            "sellers": self.sellers,
            "products": self.products,
            "reviews": self.reviews,
            "geolocation": self.geolocation,
        }


@dataclass(frozen=True)
class OlistValidationReport:
    """Small auditable summary emitted after a successful source-data check."""

    row_counts: dict[str, int]
    delivered_orders: int
    category_coverage: int


def load_olist_dataset(raw_dir: str | Path) -> tuple[OlistDataset, OlistValidationReport]:
    """Read and validate the required Olist CSV files from a local directory."""
    root = Path(raw_dir)
    missing = [filename for filename in _FILES.values() if not (root / filename).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Missing Olist files in {root}: {', '.join(missing)}. "
            f"Download from {OLIST_SOURCE_URL}."
        )

    tables = {name: pd.read_csv(root / filename) for name, filename in _FILES.items()}
    dataset = OlistDataset(**tables)
    return dataset, validate_olist_dataset(dataset)


def validate_olist_dataset(dataset: OlistDataset) -> OlistValidationReport:
    """Validate schema, keys, references, and numeric ranges before feature creation."""
    tables = dataset.tables()
    for name, frame in tables.items():
        missing = _REQUIRED_COLUMNS[name] - set(frame.columns)
        if missing:
            raise DataValidationError(f"{name} is missing required columns: {sorted(missing)}")
        if frame.empty:
            raise DataValidationError(f"{name} must contain at least one row")

    for name, key in _PRIMARY_KEYS.items():
        frame = tables[name]
        if frame[key].isna().any().any() or frame.duplicated(key).any():
            raise DataValidationError(f"{name} has null or duplicate primary keys: {key}")

    _require_positive(dataset.order_items, "price", "order_items")
    _require_non_negative(dataset.order_items, "freight_value", "order_items")
    _require_range(dataset.geolocation, "geolocation_lat", -90, 90, "geolocation")
    _require_range(dataset.geolocation, "geolocation_lng", -180, 180, "geolocation")
    _require_range(dataset.reviews, "review_score", 1, 5, "reviews")

    _require_references(
        dataset.orders.customer_id, dataset.customers.customer_id, "orders.customer_id"
    )
    _require_references(
        dataset.order_items.order_id, dataset.orders.order_id, "order_items.order_id"
    )
    _require_references(
        dataset.order_items.seller_id, dataset.sellers.seller_id, "order_items.seller_id"
    )
    _require_references(
        dataset.order_items.product_id, dataset.products.product_id, "order_items.product_id"
    )
    _require_references(dataset.reviews.order_id, dataset.orders.order_id, "reviews.order_id")

    delivered_orders = int((dataset.orders.order_status == "delivered").sum())
    category_coverage = int(dataset.products.product_category_name.dropna().nunique())
    if delivered_orders == 0 or category_coverage == 0:
        raise DataValidationError(
            "Dataset needs delivered orders and at least one product category"
        )

    return OlistValidationReport(
        row_counts={name: len(frame) for name, frame in tables.items()},
        delivered_orders=delivered_orders,
        category_coverage=category_coverage,
    )


def _require_positive(frame: pd.DataFrame, column: str, table: str) -> None:
    if frame[column].isna().any() or (frame[column] <= 0).any():
        raise DataValidationError(f"{table}.{column} must be positive")


def _require_non_negative(frame: pd.DataFrame, column: str, table: str) -> None:
    if frame[column].isna().any() or (frame[column] < 0).any():
        raise DataValidationError(f"{table}.{column} must be non-negative")


def _require_range(frame: pd.DataFrame, column: str, low: float, high: float, table: str) -> None:
    if frame[column].isna().any() or (~frame[column].between(low, high)).any():
        raise DataValidationError(f"{table}.{column} must be between {low} and {high}")


def _require_references(child: pd.Series, parent: pd.Series, field: str) -> None:
    dangling = ~child.isin(parent)
    if dangling.any():
        raise DataValidationError(f"{field} has {int(dangling.sum())} dangling references")
