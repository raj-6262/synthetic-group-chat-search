"""
Text -> vector embedding, with two backends:

  1. "st"        Real sentence-transformers multilingual MiniLM model.
                 This is what you should use for the actual submission -
                 it genuinely understands meaning across English/Hindi/
                 Hinglish and is what the write-up promises.

  2. "tfidf_svd" A local, dependency-light fallback (scikit-learn only,
                 no model download) used automatically when
                 sentence-transformers or its model weights are not
                 available (e.g. this sandbox has no access to
                 huggingface.co to download weights). It is a *weaker*
                 semantic proxy - it captures term co-occurrence
                 structure via SVD, not real meaning - but it lets the
                 full pipeline run end-to-end offline.

Set EMBEDDING_BACKEND=st once you have internet access on your own
machine to get the real model; "auto" (default) picks the best
available option and tells you which one it picked.
"""
import json
import os
import numpy as np

try:
    from .config import EMBEDDING_BACKEND, ST_MODEL_NAME, SVD_DIMS, EMBEDDINGS_META_PATH
except ImportError:  # pragma: no cover - supports python src/embeddings.py
    from config import EMBEDDING_BACKEND, ST_MODEL_NAME, SVD_DIMS, EMBEDDINGS_META_PATH


class Embedder:
    def __init__(self, backend=None):
        self.backend_requested = backend or EMBEDDING_BACKEND
        self.backend = None
        self._st_model = None
        self._svd_vectorizer = None
        self._svd = None
        self.dim = None

    # -- backend selection -------------------------------------------------
    def _try_load_sentence_transformers(self):
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer(ST_MODEL_NAME)
            self._st_model = model
            self.backend = "st"
            self.dim = model.get_sentence_embedding_dimension()
            return True
        except Exception as e:
            print(f"[embeddings] sentence-transformers unavailable ({e.__class__.__name__}: {e}). "
                  f"Falling back to local TF-IDF+SVD backend.")
            return False

    def fit_tfidf_svd(self, texts):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.decomposition import TruncatedSVD

        vectorizer = TfidfVectorizer(
            analyzer="char_wb", ngram_range=(2, 4), max_features=40000, sublinear_tf=True
        )
        tfidf = vectorizer.fit_transform(texts)
        n_components = min(SVD_DIMS, tfidf.shape[1] - 1, tfidf.shape[0] - 1)
        svd = TruncatedSVD(n_components=n_components, random_state=42)
        svd.fit(tfidf)
        self._svd_vectorizer = vectorizer
        self._svd = svd
        self.backend = "tfidf_svd"
        self.dim = n_components

    def ensure_ready(self, fit_texts=None):
        """Resolve which backend to actually use. `fit_texts` is required
        the first time if the fallback backend ends up being used, since
        TF-IDF+SVD must be fit on the corpus."""
        if self.backend is not None:
            return
        if self.backend_requested in ("auto", "st"):
            ok = self._try_load_sentence_transformers()
            if ok:
                return
            if self.backend_requested == "st":
                raise RuntimeError("sentence-transformers backend was forced but is unavailable.")
        # fallback
        if fit_texts is None:
            raise RuntimeError("TF-IDF+SVD fallback needs corpus texts to fit on.")
        self.fit_tfidf_svd(fit_texts)

    # -- encoding ------------------------------------------------------------
    def encode(self, texts, fit_if_needed=True):
        if isinstance(texts, str):
            texts = [texts]
        if self.backend is None:
            self.ensure_ready(fit_texts=texts if fit_if_needed else None)

        if self.backend == "st":
            vecs = self._st_model.encode(texts, show_progress_bar=False,
                                          convert_to_numpy=True, normalize_embeddings=True)
            return vecs.astype("float32")

        # tfidf_svd path
        tfidf = self._svd_vectorizer.transform(texts)
        vecs = self._svd.transform(tfidf).astype("float32")
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vecs / norms

    def save_meta(self):
        with open(EMBEDDINGS_META_PATH, "w") as f:
            json.dump({"backend": self.backend, "dim": self.dim,
                       "model_name": ST_MODEL_NAME if self.backend == "st" else "tfidf_svd_char_ngrams"}, f, indent=2)

    @staticmethod
    def load_meta():
        if not os.path.exists(EMBEDDINGS_META_PATH):
            return None
        with open(EMBEDDINGS_META_PATH) as f:
            return json.load(f)
