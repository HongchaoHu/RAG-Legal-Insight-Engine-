from __future__ import annotations

import logging
import time
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from vector_store import LegalVectorStore

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QueryRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    top_k: int = Field(default=3, ge=1, le=10)


class QueryAnswer(BaseModel):
    summary: str
    source_path: str
    relevance_score: float


class QueryResponse(BaseModel):
    question: str
    answers: list[QueryAnswer]
    latency_ms: float


def create_app(vector_store: LegalVectorStore | None = None) -> FastAPI:
    app = FastAPI(title="RAG Legal Insight Engine", version="1.0.0")
    app.state.vector_store = vector_store or LegalVectorStore(
        index_path=Path("artifacts") / "index.faiss",
        mapping_path=Path("artifacts") / "index_mapping.json",
    )

    @app.on_event("startup")
    async def _load_index() -> None:
        try:
            app.state.vector_store.load_index()
            logger.info("FAISS index loaded on startup")
        except FileNotFoundError:
            logger.warning("No FAISS index artifacts found. Build index before querying.")

    @app.post("/query", response_model=QueryResponse)
    async def query_documents(request: QueryRequest) -> QueryResponse:
        started = time.perf_counter()
        logger.info("Incoming query: %s", request.question)

        try:
            answers = app.state.vector_store.search(request.question, request.top_k)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            logger.exception("Query processing failed: %s", exc)
            raise HTTPException(status_code=500, detail="Internal query failure") from exc

        latency_ms = round((time.perf_counter() - started) * 1000, 2)
        response = QueryResponse(
            question=request.question,
            answers=[QueryAnswer(**answer) for answer in answers],
            latency_ms=latency_ms,
        )
        return response

    return app


app = create_app()
