from fastapi.testclient import TestClient

from marketrank.api import app


def test_healthcheck() -> None:
    response = TestClient(app).get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_match_endpoint_returns_explainable_feasible_supply() -> None:
    response = TestClient(app).post(
        "/v1/matches",
        json={
            "request_id": "demo-1",
            "user_id": "new-user",
            "category": "books",
            "latitude": -23.55,
            "longitude": -46.63,
            "budget": 30,
        },
    )

    body = response.json()
    assert response.status_code == 200
    assert body["policy"] == "heuristic_fallback"
    assert len(body["matches"]) == 3
    assert "ranking_features" in body["matches"][0]


def test_interface_is_served() -> None:
    response = TestClient(app).get("/")

    assert response.status_code == 200
    assert "Marketplace request" in response.text


def test_match_endpoint_supports_makeup_demo_supply() -> None:
    response = TestClient(app).post(
        "/v1/matches",
        json={
            "request_id": "makeup-demo",
            "user_id": "new-user",
            "category": "makeup",
            "latitude": -23.55,
            "longitude": -46.63,
            "budget": 30,
        },
    )

    assert response.status_code == 200
    assert len(response.json()["matches"]) == 3
