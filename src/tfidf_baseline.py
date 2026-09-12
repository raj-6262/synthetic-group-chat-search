"""
A plain keyword baseline (word-level TF-IDF + cosine similarity), so we
can demonstrate *why* semantic search is needed rather than just assert it.
"""
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    from .config import CHAT_MESSAGES_PATH
except ImportError:  # pragma: no cover - supports python src/tfidf_baseline.py
    from config import CHAT_MESSAGES_PATH


class TfidfBaseline:
    def __init__(self):
        with open(CHAT_MESSAGES_PATH, encoding="utf-8") as f:
            self.messages = json.load(f)
        texts = [m["text"] for m in self.messages]
        self.vectorizer = TfidfVectorizer(lowercase=True, stop_words=None)
        self.matrix = self.vectorizer.fit_transform(texts)

    def search(self, query, top_k=5):
        qvec = self.vectorizer.transform([query])
        sims = cosine_similarity(qvec, self.matrix)[0]
        top_idx = sims.argsort()[::-1][:top_k]
        return [{"message": self.messages[i], "score": float(sims[i])} for i in top_idx]


if __name__ == "__main__":
    tb = TfidfBaseline()
    for q in ["What destination did everyone finally agree upon?", "Goa trip discussion"]:
        print(q)
        for r in tb.search(q, top_k=3):
            print(f"  [{r['score']:.2f}] {r['message']['text']}")
