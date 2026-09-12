"""
FastAPI backend for ChatSense.

Run from the project root:

    python -m uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
"""

import os
import sys
import time
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
FRONTEND_DIR = PROJECT_ROOT / "frontend"

sys.path.insert(0, str(SRC_DIR))


# ---------------------------------------------------------
# Imports
# ---------------------------------------------------------

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from search import ChatSearchEngine
from config import PARTICIPANTS


# ---------------------------------------------------------
# Application
# ---------------------------------------------------------

app = FastAPI(
    title="ChatSense",
    description="Semantic search over synthetic group chat conversations.",
    version="1.0.0",
)


# ---------------------------------------------------------
# CORS
# ---------------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------
# Search engine
# ---------------------------------------------------------

engine = ChatSearchEngine()


# ---------------------------------------------------------
# Static files
# ---------------------------------------------------------
#
# This makes:
#
# /static/style.css
# /static/script.js
#
# available to the browser.
#

app.mount(
    "/static",
    StaticFiles(directory=str(FRONTEND_DIR)),
    name="static",
)


# ---------------------------------------------------------
# Home page
# ---------------------------------------------------------

@app.get("/", include_in_schema=False)
def home():
    """Serve the ChatSense landing page."""

    return FileResponse(
        str(FRONTEND_DIR / "index.html")
    )


# ---------------------------------------------------------
# Search application page
# ---------------------------------------------------------

@app.get("/app", include_in_schema=False)
def search_app():
    """Serve the ChatSense search application."""

    return FileResponse(
        str(FRONTEND_DIR / "search.html")
    )


# ---------------------------------------------------------
# Health endpoint
# ---------------------------------------------------------

@app.get("/health")
def health():
    """Return backend and index status."""

    return {
        "status": "ok",
        "index_loaded": engine.index is not None,
        "model_loaded": engine.embedder.backend is not None,
        "messages_indexed": len(engine.messages),
        "embedding_backend": engine.embedder.backend,
    }


# ---------------------------------------------------------
# Participants endpoint
# ---------------------------------------------------------

@app.get("/participants")
def participants():
    """Return available chat participants."""

    return {
        "participants": PARTICIPANTS
    }


# ---------------------------------------------------------
# Search API
# ---------------------------------------------------------

@app.get("/search")
def search(
    q: str = Query(
        ...,
        description="Natural language search query",
    ),

    top_k: int = Query(
        5,
        ge=1,
        le=20,
        description="Number of results",
    ),

    participant: str | None = Query(
        None,
        description="Restrict results to a participant",
    ),

    start_date: str | None = Query(
        None,
        description="YYYY-MM-DD lower bound",
    ),

    end_date: str | None = Query(
        None,
        description="YYYY-MM-DD upper bound",
    ),
):

    # -----------------------------------------------------
    # Parse date filters
    # -----------------------------------------------------

    time_range = None

    if start_date and end_date:

        start = datetime.strptime(
            start_date,
            "%Y-%m-%d",
        )

        end = datetime.strptime(
            end_date,
            "%Y-%m-%d",
        ).replace(
            hour=23,
            minute=59,
            second=59,
        )

        time_range = (
            start,
            end,
        )


    # -----------------------------------------------------
    # Execute semantic search
    # -----------------------------------------------------

    start_time = time.perf_counter()

    output = engine.search(
        q,
        top_k=top_k,
        participant=participant,
        time_range=time_range,
    )

    elapsed_ms = (
        time.perf_counter() - start_time
    ) * 1000

    output["search_time_ms"] = round(
        elapsed_ms,
        2,
    )

    return output