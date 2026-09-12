"""
Phase 5: measured evaluation, not vibes.

Runs all 40 ground-truth queries through both the semantic engine and
the TF-IDF baseline, scores top-1 accuracy overall and on the 8 hard
(zero word-overlap) queries separately, and saves a comparison chart.
Also reports p50/p95 query latency.
"""
import json
import os
import time

try:
    from .config import RESULTS_DIR
except ImportError:  # pragma: no cover - supports python src/evaluate.py
    from config import RESULTS_DIR

os.makedirs(os.path.join(RESULTS_DIR, "matplotlib_cache"), exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", os.path.join(RESULTS_DIR, "matplotlib_cache"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from .config import EVAL_QUERIES_PATH
    from .search import ChatSearchEngine
    from .tfidf_baseline import TfidfBaseline
except ImportError:  # pragma: no cover - supports python src/evaluate.py
    from config import EVAL_QUERIES_PATH
    from search import ChatSearchEngine
    from tfidf_baseline import TfidfBaseline


def top1_id(results):
    return results[0]["message"]["id"] if results else None


def rank_of_expected(results, expected_id):
    for i, result in enumerate(results, start=1):
        if result["message"]["id"] == expected_id:
            return i
    return None


def run():
    with open(EVAL_QUERIES_PATH, encoding="utf-8") as f:
        queries = json.load(f)

    engine = ChatSearchEngine()
    baseline = TfidfBaseline()

    rows = []
    sem_latencies = []
    for q in queries:
        t0 = time.perf_counter()
        sem_out = engine.search(q["query"], top_k=5)
        sem_latencies.append(time.perf_counter() - t0)
        sem_top1 = top1_id(sem_out["results"])
        sem_rank = rank_of_expected(sem_out["results"], q["expected_message_id"])

        tfidf_out = baseline.search(q["query"], top_k=5)
        tfidf_top1 = top1_id(tfidf_out)
        tfidf_rank = rank_of_expected(tfidf_out, q["expected_message_id"])
        sem_score = sem_out["results"][0]["score"] if sem_out["results"] else None

        rows.append({
            "id": q["id"],
            "query": q["query"],
            "hard": q["hard"],
            "shape": q.get("shape", "meaning"),
            "expected_message_id": q["expected_message_id"],
            "predicted_message_id": sem_top1,
            "correct": sem_top1 == q["expected_message_id"],
            "similarity_score": sem_score,
            "latency_ms": sem_latencies[-1] * 1000,
            "semantic_top1": sem_top1,
            "semantic_correct": sem_top1 == q["expected_message_id"],
            "semantic_recall_at_5": sem_rank is not None,
            "semantic_mrr": 1 / sem_rank if sem_rank else 0.0,
            "tfidf_top1": tfidf_top1,
            "tfidf_correct": tfidf_top1 == q["expected_message_id"],
            "tfidf_recall_at_5": tfidf_rank is not None,
            "tfidf_mrr": 1 / tfidf_rank if tfidf_rank else 0.0,
        })

    def accuracy(subset, key):
        if not subset:
            return None
        return sum(1 for r in subset if r[key]) / len(subset)

    def average(subset, key):
        if not subset:
            return None
        return sum(r[key] for r in subset) / len(subset)

    hard_rows = [r for r in rows if r["hard"]]

    summary = {
        "embedding_backend": engine.embedder.backend,
        "total_queries": len(rows),
        "hard_queries": len(hard_rows),
        "semantic_overall_accuracy": accuracy(rows, "semantic_correct"),
        "semantic_hard_accuracy": accuracy(hard_rows, "semantic_correct"),
        "semantic_recall_at_5": accuracy(rows, "semantic_recall_at_5"),
        "semantic_mrr": average(rows, "semantic_mrr"),
        "tfidf_overall_accuracy": accuracy(rows, "tfidf_correct"),
        "tfidf_hard_accuracy": accuracy(hard_rows, "tfidf_correct"),
        "tfidf_recall_at_5": accuracy(rows, "tfidf_recall_at_5"),
        "tfidf_mrr": average(rows, "tfidf_mrr"),
        "semantic_avg_latency_ms": sum(sem_latencies) / len(sem_latencies) * 1000,
        "semantic_latency_p50_ms": sorted(sem_latencies)[len(sem_latencies) // 2] * 1000,
        "semantic_latency_p95_ms": sorted(sem_latencies)[int(len(sem_latencies) * 0.95) - 1] * 1000,
        "rows": rows,
    }

    with open(f"{RESULTS_DIR}/evaluation.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # --- chart: overall vs hard accuracy, semantic vs keyword ---
    labels = ["Overall (40)", f"Hard / zero-overlap ({len(hard_rows)})"]
    semantic_vals = [summary["semantic_overall_accuracy"] * 100, (summary["semantic_hard_accuracy"] or 0) * 100]
    tfidf_vals = [summary["tfidf_overall_accuracy"] * 100, (summary["tfidf_hard_accuracy"] or 0) * 100]

    x = range(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar([i - width / 2 for i in x], tfidf_vals, width, label="TF-IDF (keyword)")
    ax.bar([i + width / 2 for i in x], semantic_vals, width, label="Semantic (embeddings)")
    ax.set_ylabel("Top-1 accuracy (%)")
    ax.set_title(f"Semantic vs Keyword Search Accuracy (backend: {engine.embedder.backend})")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 100)
    ax.legend()
    for i, v in enumerate(tfidf_vals):
        ax.text(i - width / 2, v + 2, f"{v:.0f}%", ha="center")
    for i, v in enumerate(semantic_vals):
        ax.text(i + width / 2, v + 2, f"{v:.0f}%", ha="center")
    fig.tight_layout()
    fig.savefig(f"{RESULTS_DIR}/accuracy.png", dpi=150)

    # --- chart: per-shape breakdown for the semantic engine ---
    shapes = sorted(set(r["shape"] for r in rows))
    shape_acc = [accuracy([r for r in rows if r["shape"] == s], "semantic_correct") * 100 for s in shapes]
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    ax2.bar(shapes, shape_acc, color="#4C72B0")
    ax2.set_ylabel("Top-1 accuracy (%)")
    ax2.set_title("Semantic search accuracy by query shape")
    ax2.set_ylim(0, 100)
    for i, v in enumerate(shape_acc):
        ax2.text(i, v + 2, f"{v:.0f}%", ha="center")
    fig2.tight_layout()
    fig2.savefig(f"{RESULTS_DIR}/hard_queries.png", dpi=150)

    print("=============================")
    print(" Evaluation Summary")
    print("=============================")
    print(f" Embedding backend : {summary['embedding_backend']}")
    print(f" Total queries     : {summary['total_queries']}")
    print(f" Semantic overall  : {summary['semantic_overall_accuracy']*100:.1f}%")
    print(f" Semantic (hard {len(hard_rows)}) : {summary['semantic_hard_accuracy']*100:.1f}%")
    print(f" TF-IDF overall    : {summary['tfidf_overall_accuracy']*100:.1f}%")
    print(f" TF-IDF (hard {len(hard_rows)})   : {summary['tfidf_hard_accuracy']*100:.1f}%")
    print(f" Latency p50 / p95 : {summary['semantic_latency_p50_ms']:.1f}ms / {summary['semantic_latency_p95_ms']:.1f}ms")
    print(f" Charts written to {RESULTS_DIR}/")


if __name__ == "__main__":
    run()
