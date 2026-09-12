"""
Phase 2+3: embed every message and build a FAISS index over the vectors.
"""
import json
import pickle
import numpy as np
import faiss

try:
    from .config import CHAT_MESSAGES_PATH, EMBEDDINGS_PATH, FAISS_INDEX_PATH
    from .embeddings import Embedder
except ImportError:  # pragma: no cover - supports python src/build_index.py
    from config import CHAT_MESSAGES_PATH, EMBEDDINGS_PATH, FAISS_INDEX_PATH
    from embeddings import Embedder


def main():
    with open(CHAT_MESSAGES_PATH, encoding="utf-8") as f:
        messages = json.load(f)
    texts = [m["text"] for m in messages]

    embedder = Embedder()
    embedder.ensure_ready(fit_texts=texts)
    print(f"[build_index] using embedding backend: {embedder.backend} (dim={embedder.dim})")

    vectors = embedder.encode(texts, fit_if_needed=False)
    np.save(EMBEDDINGS_PATH, vectors)
    embedder.save_meta()

    # Save the fitted TF-IDF/SVD objects if that's the backend in use, so
    # search.py can embed new queries with the exact same fitted transform.
    if embedder.backend == "tfidf_svd":
        with open(EMBEDDINGS_PATH + ".fallback.pkl", "wb") as f:
            pickle.dump({"vectorizer": embedder._svd_vectorizer, "svd": embedder._svd}, f)

    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)  # inner product on normalized vecs = cosine sim
    index.add(vectors)
    faiss.write_index(index, FAISS_INDEX_PATH)

    print(f"[build_index] indexed {index.ntotal} messages, dim={dim} -> {FAISS_INDEX_PATH}")


if __name__ == "__main__":
    main()
