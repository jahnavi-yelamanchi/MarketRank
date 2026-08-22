"""HTTP service entry point for explainable marketplace matching."""

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from marketrank.serving.service import (
    MarketplaceMatchingService,
    MatchInput,
    demo_marketplace_state,
)

app = FastAPI(title="MarketRank", version="0.1.0")
service = MarketplaceMatchingService(demo_marketplace_state())
static_dir = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=static_dir), name="static")


class MatchRequestBody(BaseModel):
    request_id: str = Field(min_length=1)
    user_id: str = Field(min_length=1)
    category: str = Field(min_length=1)
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    budget: float = Field(gt=0)
    quantity: int = Field(default=1, ge=1)
    max_distance_km: float = Field(default=50, gt=0, le=500)


@app.get("/health")
def health() -> dict[str, str]:
    """Provide a dependency-free container health check."""
    return {"status": "ok"}


@app.get("/", include_in_schema=False)
def interface() -> FileResponse:
    """Serve the dependency-free ranking-inspection interface."""
    return FileResponse(static_dir / "index.html")


@app.post("/v1/matches")
def match(body: MatchRequestBody) -> dict[str, object]:
    """Return feasible, ranked provider matches with score explanations."""
    response = service.match(MatchInput(**body.model_dump()))
    return {
        "request_id": response.request_id,
        "policy": response.policy,
        "fallback": response.fallback,
        "constraints_applied": response.constraints_applied,
        "matches": [
            {
                "offer_id": result.offer_id,
                "provider_id": result.provider_id,
                "predicted_score": result.predicted_score,
                "ranking_features": result.features,
                "reasons": result.reasons,
            }
            for result in response.results
        ],
    }
