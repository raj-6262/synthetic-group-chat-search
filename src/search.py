"""
Phase 4: the actual search engine.

Handles all three query shapes the assignment calls out:
  - meaning-based  -> semantic vector similarity (FAISS)
  - person-based   -> detected "what did X say" style mentions, filters
                       candidates to that sender before ranking
  - time-based     -> detected relative/absolute time phrases ("last
                       month", "this week", "in March"), filters
                       candidates to that date range before ranking

A raw top hit is not returned alone: each result comes back with a
small window of surrounding messages from the same conversation so it
can actually be read in context, not as an isolated line.
"""
import json
import re
import pickle
from datetime import datetime, timedelta

import numpy as np
import faiss

try:
    from .config import (CHAT_MESSAGES_PATH, EMBEDDINGS_PATH, FAISS_INDEX_PATH,
                         PARTICIPANTS, SIMILARITY_THRESHOLD_ST,
                         SIMILARITY_THRESHOLD_TFIDF_SVD, CONTEXT_WINDOW)
    from .embeddings import Embedder
except ImportError:  # pragma: no cover - supports python src/search.py
    from config import (CHAT_MESSAGES_PATH, EMBEDDINGS_PATH, FAISS_INDEX_PATH,
                         PARTICIPANTS, SIMILARITY_THRESHOLD_ST,
                         SIMILARITY_THRESHOLD_TFIDF_SVD, CONTEXT_WINDOW)
    from embeddings import Embedder

MONTH_NAMES = ["january", "february", "march", "april", "may", "june", "july",
               "august", "september", "october", "november", "december"]

# "Today" for resolving relative time expressions. In a real deployment
# this would be datetime.now(); here it's pinned to match the synthetic
# corpus (see generate_data.py NOW) so results are reproducible.
REFERENCE_NOW = datetime(2026, 9, 1, 12, 0)


class ChatSearchEngine:
    def __init__(self):
        with open(CHAT_MESSAGES_PATH, encoding="utf-8") as f:
            self.messages = json.load(f)
        self.by_id = {m["id"]: m for m in self.messages}
        self.timestamps = [datetime.strptime(m["timestamp"], "%Y-%m-%d %H:%M")
                            for m in self.messages]
        self.positions_by_conversation = {}
        for i, msg in enumerate(self.messages):
            self.positions_by_conversation.setdefault(msg.get("conversation_id", "general"), []).append(i)

        self.index = faiss.read_index(FAISS_INDEX_PATH)
        self.vectors = np.load(EMBEDDINGS_PATH)

        meta = Embedder.load_meta()
        self.embedder = Embedder(backend=meta["backend"] if meta else None)
        if meta and meta["backend"] == "tfidf_svd":
            with open(EMBEDDINGS_PATH + ".fallback.pkl", "rb") as f:
                fitted = pickle.load(f)
            self.embedder._svd_vectorizer = fitted["vectorizer"]
            self.embedder._svd = fitted["svd"]
            self.embedder.backend = "tfidf_svd"
            self.embedder.dim = fitted["svd"].n_components
        else:
            self.embedder.ensure_ready(fit_texts=None)  # loads real ST model

        self.similarity_threshold = (SIMILARITY_THRESHOLD_ST if self.embedder.backend == "st"
                                      else SIMILARITY_THRESHOLD_TFIDF_SVD)

    # ---- query understanding -------------------------------------------
    @staticmethod
    def detect_person(query):
        q = query.lower()
        for name in PARTICIPANTS:
            pattern = r"\b" + re.escape(name.lower()) + r"\b"
            if re.search(pattern, q):
                return name
        return None

    def detect_time_range(self, query):
        q = query.lower()
        ref = REFERENCE_NOW

        if "yesterday" in q:
            day = ref - timedelta(days=1)
            return day.replace(hour=0, minute=0), day.replace(hour=23, minute=59)

        if "last week" in q:
            end = ref - timedelta(days=ref.weekday() + 1)
            start = end - timedelta(days=6)
            return start, end

        if "this week" in q:
            start = ref - timedelta(days=ref.weekday())
            return start, ref

        if "last month" in q:
            first_of_this_month = ref.replace(day=1)
            last_month_end = first_of_this_month - timedelta(days=1)
            last_month_start = last_month_end.replace(day=1)
            return last_month_start, last_month_end.replace(hour=23, minute=59)

        if "this month" in q:
            return ref.replace(day=1), ref

        for i, name in enumerate(MONTH_NAMES, start=1):
            if name in q:
                year = ref.year
                start = datetime(year, i, 1)
                end = (datetime(year, i + 1, 1) - timedelta(minutes=1)) if i < 12 \
                    else datetime(year, 12, 31, 23, 59)
                return start, end

        return None

    # ---- core search ------------------------------------------------------
    def search(self, query, top_k=5, participant=None, time_range=None,
               use_query_understanding=True):
        query = (query or "").strip()
        if not query:
            return {
                "query": query,
                "detected_person": participant,
                "detected_time_range": None,
                "filters_relaxed": False,
                "results": [],
                "no_relevant_message": True,
            }
        top_k = max(1, min(int(top_k), 20))
        detected_person = participant
        detected_range = time_range

        if use_query_understanding:
            detected_person = detected_person or self.detect_person(query)
            detected_range = detected_range or self.detect_time_range(query)

        candidate_indices = []
        for idx, msg in enumerate(self.messages):
            if detected_person and msg["sender"] != detected_person:
                continue
            if detected_range:
                ts = self.timestamps[idx]
                if not (detected_range[0] <= ts <= detected_range[1]):
                    continue
            candidate_indices.append(idx)

        # If filters were too strict and left nothing, fall back to
        # unfiltered ranking rather than returning an empty result.
        filters_relaxed = False
        if not candidate_indices and (detected_person or detected_range):
            filters_relaxed = True
            candidate_indices = list(range(len(self.messages)))

        query_vec = self.embedder.encode([query], fit_if_needed=False)
        candidates = self._rank_candidates(query_vec, candidate_indices, top_k)
        top = candidates[:top_k]

        results = []
        for score, msg in top:
            if score < self.similarity_threshold:
                continue
            results.append({
                "message": msg,
                "score": round(score, 4),
                "message_id": msg["id"],
                "conversation_id": msg.get("conversation_id"),
                "context": self._context_window(msg["id"]),
            })

        return {
            "query": query,
            "detected_person": detected_person,
            "detected_time_range": [d.strftime("%Y-%m-%d") for d in detected_range] if detected_range else None,
            "filters_relaxed": filters_relaxed,
            "results": results if results else [],
            "no_relevant_message": len(results) == 0,
        }

    def _rank_candidates(self, query_vec, candidate_indices, top_k):
        if not candidate_indices:
            return []
        # Full FAISS search is fine for the unfiltered path. For filtered
        # searches, rank the constrained candidate vectors directly so the
        # correct sender/date result cannot be dropped by a small global top-k.
        if len(candidate_indices) == len(self.messages):
            sims, idxs = self.index.search(query_vec, min(top_k, len(self.messages)))
            return [(float(score), self.messages[int(idx)])
                    for score, idx in zip(sims[0], idxs[0]) if idx >= 0]
        idx_array = np.array(candidate_indices, dtype=np.int64)
        sims = self.vectors[idx_array] @ query_vec[0]
        order = np.argsort(-sims)[:top_k]
        return [(float(sims[i]), self.messages[int(idx_array[i])]) for i in order]

    def _context_window(self, msg_id, window=CONTEXT_WINDOW):
        """Return nearby messages from the same conversation_id."""
        idx = msg_id - 1  # ids are 1-indexed and match list order
        conversation_id = self.messages[idx].get("conversation_id", "general")
        positions = self.positions_by_conversation.get(conversation_id, [idx])
        center_pos = positions.index(idx)
        lo = max(0, center_pos - window)
        hi = min(len(positions), center_pos + window + 1)
        context = []
        for pos in positions[lo:hi]:
            context.append(self.messages[pos])
        return context


if __name__ == "__main__":
    engine = ChatSearchEngine()
    print("Semantic Group Chat Search - terminal mode. Ctrl+C to quit.\n")
    while True:
        try:
            q = input("query> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q:
            continue
        out = engine.search(q)
        if out["no_relevant_message"]:
            print("  -> No sufficiently relevant message found.")
            continue
        for r in out["results"]:
            m = r["message"]
            print(f"  [{r['score']:.2f}] {m['sender']} ({m['timestamp']}): {m['text']}")
