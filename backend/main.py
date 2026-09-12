"""
Phase 6: FastAPI backend.

Run from the project root with:
    uvicorn backend.main:app --reload --port 8000
"""
import sys
import os
import time
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from search import ChatSearchEngine  # noqa: E402
from config import PARTICIPANTS  # noqa: E402

app = FastAPI(title="Semantic Group Chat Search")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

engine = ChatSearchEngine()

FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "..", "frontend")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "index_loaded": engine.index is not None,
        "model_loaded": engine.embedder.backend is not None,
        "messages_indexed": len(engine.messages),
        "embedding_backend": engine.embedder.backend,
    }


@app.get("/participants")
def participants():
    return {"participants": PARTICIPANTS}


@app.get("/search")
def search(
    q: str = Query(..., description="Natural language query"),
    top_k: int = Query(5, ge=1, le=20),
    participant: str | None = Query(None, description="Restrict results to this sender"),
    start_date: str | None = Query(None, description="YYYY-MM-DD lower bound"),
    end_date: str | None = Query(None, description="YYYY-MM-DD upper bound"),
):
    time_range = None
    if start_date and end_date:
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d").replace(hour=23, minute=59)
        time_range = (start, end)
    t0 = time.perf_counter()
    out = engine.search(q, top_k=top_k, participant=participant, time_range=time_range)
    out["search_time_ms"] = round((time.perf_counter() - t0) * 1000, 2)
    return out


@app.get("/")
def root():
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))


app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")
