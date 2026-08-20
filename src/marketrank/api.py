"""HTTP service entry point. Matching endpoints arrive in the serving milestone."""

from fastapi import FastAPI

app = FastAPI(title="MarketRank", version="0.1.0")


@app.get("/health")
def health() -> dict[str, str]:
    """Provide a dependency-free container health check."""
    return {"status": "ok"}

