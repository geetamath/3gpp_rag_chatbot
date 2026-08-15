"""
FastAPI backend for the 3GPP RAG chatbot web app.

Wraps the real pipeline (src/pipeline.py -- clause-aware ingestion, hybrid
vector+BM25 retrieval, evidence gate, LLM generation, claim verification) in
an HTTP API for the React frontend.

Endpoints (mirrors the shape of a typical RAG-chat frontend):
    GET  /api/health   -> service + index status
    GET  /api/sources  -> list of indexed 3GPP documents (from real metadata)
    POST /api/query    -> ask a question, get a grounded (or refused) answer

Run:
    uvicorn webapp.backend.main:app --reload --port 8000
"""

import os
import sys
import json
import time
from collections import Counter

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

# Make the project root importable regardless of where uvicorn is launched from
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

import config
from src.pipeline import RAGPipeline

app = FastAPI(title="3GPP Evidence RAG API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_pipeline = None       # lazy-loaded singleton (index load can be slow-ish)
_load_error = None


def get_pipeline():
    global _pipeline, _load_error
    if _pipeline is None and _load_error is None:
        try:
            _pipeline = RAGPipeline()
        except Exception as e:
            _load_error = str(e)
    if _load_error:
        raise HTTPException(
            status_code=503,
            detail=f"Index not built yet: {_load_error}. Run `python -m src.ingest` "
                    f"and `python -m src.build_index` first.",
        )
    return _pipeline


class Query(BaseModel):
    question: str = Field(min_length=2, max_length=1000)


@app.get("/api/")
async def root():
    return {"message": "3GPP RAG API online", "guardrail": "evidence-gated + verified"}


@app.get("/api/health")
async def health():
    try:
        pipeline = get_pipeline()
        n_chunks = len(pipeline.retriever.chunks)
        n_docs = len({c["source_file"] for c in pipeline.retriever.chunks})
        status = "ok"
    except HTTPException as e:
        n_chunks, n_docs, status = 0, 0, "index_not_built"
    return {
        "status": status,
        "documents": n_docs,
        "chunks": n_chunks,
        "embedding_backend": config.EMBEDDING_BACKEND,
        "llm_backend": config.LLM_BACKEND,
        "mode": "real-hybrid-retrieval+evidence-gate",
    }


@app.get("/api/sources")
async def sources():
    """Derives the source list from REAL ingested metadata, not a hardcoded list."""
    pipeline = get_pipeline()
    chunks = pipeline.retriever.chunks

    by_file = {}
    for c in chunks:
        key = c["source_file"]
        if key not in by_file:
            by_file[key] = {
                "id": key,
                "title": c["ts_number"],
                "subtitle": c.get("clause_title", ""),
                "release": c["release"],
                "chunks": 0,
                "status": "Indexed",
            }
        by_file[key]["chunks"] += 1

    src_list = list(by_file.values())
    return {"sources": src_list, "total": len(src_list), "chunks": len(chunks)}


@app.post("/api/query")
async def query(payload: Query):
    pipeline = get_pipeline()
    t0 = time.time()
    result = pipeline.answer(payload.question)
    elapsed_ms = int((time.time() - t0) * 1000)

    grounded = result["status"] == "ANSWERED"
    # simple confidence display derived from verification report, not invented
    confidence = 0
    if grounded and "verification_report" in result:
        vr = result["verification_report"]
        if vr["n_claims_total"] > 0:
            confidence = round(100 * vr["n_supported"] / vr["n_claims_total"])

    citations = [
        {
            "id": c["chunk_id"],
            "document": c["ts_number"],
            "release": c["release"],
            "section": c["clause"],
            "page": c["page"],
        }
        for c in result.get("citations", [])
    ]

    return {
        "answer": result["answer"],
        "grounded": grounded,
        "confidence": confidence,
        "citations": citations,
        "status": result["status"],
        "gate_reason": result.get("gate_reason"),
        "elapsed_ms": elapsed_ms,
    }


# ---- Serve the built React frontend (single-container deployment) ---------
# This mount is registered LAST so it never shadows the /api/* routes above.
# In the Docker build, the frontend's `npm run build` output is copied to
# webapp/frontend/build before the container starts. If that folder doesn't
# exist (e.g. running the backend alone in dev), this is skipped gracefully
# and only the JSON API is served.
_FRONTEND_BUILD_DIR = os.path.join(os.path.dirname(__file__), "..", "frontend", "build")
if os.path.isdir(_FRONTEND_BUILD_DIR):
    app.mount("/static", StaticFiles(directory=os.path.join(_FRONTEND_BUILD_DIR, "static")), name="static")

    @app.get("/{full_path:path}")
    async def serve_frontend(full_path: str):
        # Any non-/api path falls through to the React app's index.html so
        # client-side routing (if added later) still works.
        requested = os.path.join(_FRONTEND_BUILD_DIR, full_path)
        if full_path and os.path.isfile(requested):
            return FileResponse(requested)
        return FileResponse(os.path.join(_FRONTEND_BUILD_DIR, "index.html"))
else:
    print(f"[main] Note: frontend build not found at {_FRONTEND_BUILD_DIR}. "
          f"Serving API only. Run `npm run build` in webapp/frontend/ to enable "
          f"single-container serving.")
