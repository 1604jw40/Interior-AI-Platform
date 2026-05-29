from typing import Any

from fastapi import FastAPI

from .schemas import (
    FitQueryRequest,
    FitQueryResponse,
    FitRagQueryRequest,
    FitRagQueryResponse,
    PerceptionRagQueryRequest,
    RagQueryRequest,
    RagQueryResponse,
)
from .services import (
    fit_products,
    health_payload,
    rag_fit_query,
    rag_perception_query,
    rag_query,
)


app = FastAPI(title="Interior Matching Engine", version="0.1.0")


@app.get("/")
def root() -> dict[str, str]:
    return {
        "service": "matching-engine",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health")
def health() -> dict[str, Any]:
    return health_payload()


@app.post("/products/fit", response_model=FitQueryResponse)
def products_fit(request: FitQueryRequest) -> FitQueryResponse:
    return fit_products(request)


@app.post("/rag/query", response_model=RagQueryResponse)
def post_rag_query(request: RagQueryRequest) -> RagQueryResponse:
    return rag_query(request)


@app.post("/rag/fit-query", response_model=FitRagQueryResponse)
def post_rag_fit_query(request: FitRagQueryRequest) -> FitRagQueryResponse:
    return rag_fit_query(request)


@app.post("/rag/perception-query", response_model=FitRagQueryResponse)
def post_rag_perception_query(request: PerceptionRagQueryRequest) -> FitRagQueryResponse:
    return rag_perception_query(request)
