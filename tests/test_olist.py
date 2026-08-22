import pandas as pd
import pytest

from marketrank.data.marketplace import build_olist_marketplace_seed
from marketrank.data.olist import DataValidationError, OlistDataset, validate_olist_dataset


def valid_dataset() -> OlistDataset:
    return OlistDataset(
        orders=pd.DataFrame(
            {
                "order_id": ["o1"],
                "customer_id": ["c1"],
                "order_status": ["delivered"],
                "order_purchase_timestamp": ["2018-01-01 10:00:00"],
            }
        ),
        order_items=pd.DataFrame(
            {
                "order_id": ["o1"],
                "order_item_id": [1],
                "product_id": ["p1"],
                "seller_id": ["s1"],
                "price": [30.0],
                "freight_value": [4.0],
            }
        ),
        customers=pd.DataFrame(
            {
                "customer_id": ["c1"],
                "customer_unique_id": ["u1"],
                "customer_zip_code_prefix": [10000],
                "customer_city": ["sao paulo"],
                "customer_state": ["SP"],
            }
        ),
        sellers=pd.DataFrame(
            {
                "seller_id": ["s1"],
                "seller_zip_code_prefix": [10001],
                "seller_city": ["sao paulo"],
                "seller_state": ["SP"],
            }
        ),
        products=pd.DataFrame({"product_id": ["p1"], "product_category_name": ["books"]}),
        reviews=pd.DataFrame({"review_id": ["r1"], "order_id": ["o1"], "review_score": [5]}),
        geolocation=pd.DataFrame(
            {
                "geolocation_zip_code_prefix": [10000],
                "geolocation_lat": [-23.5],
                "geolocation_lng": [-46.6],
            }
        ),
    )


def test_valid_dataset_returns_auditable_summary() -> None:
    report = validate_olist_dataset(valid_dataset())

    assert report.delivered_orders == 1
    assert report.category_coverage == 1
    assert report.row_counts["sellers"] == 1


def test_dangling_provider_is_rejected() -> None:
    dataset = valid_dataset()
    dataset.order_items.loc[0, "seller_id"] = "missing"

    with pytest.raises(DataValidationError, match="dangling"):
        validate_olist_dataset(dataset)


def test_olist_history_builds_geolocated_marketplace_entities() -> None:
    dataset = valid_dataset()
    dataset.geolocation.loc[1] = [10001, -23.51, -46.61]

    seed = build_olist_marketplace_seed(dataset)

    assert seed.offers[0].provider_id == "s1"
    assert seed.requests[0].user_id == "u1"
