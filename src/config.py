"""
Central configuration for the Semantic Group Chat Search project.
"""
import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(ROOT_DIR, "data")
MODELS_DIR = os.path.join(ROOT_DIR, "models")
INDEX_DIR = os.path.join(ROOT_DIR, "index")
RESULTS_DIR = os.path.join(ROOT_DIR, "results")

CHAT_MESSAGES_PATH = os.path.join(DATA_DIR, "chat_messages.json")
EVAL_QUERIES_PATH = os.path.join(DATA_DIR, "evaluation_queries.json")

EMBEDDINGS_PATH = os.path.join(MODELS_DIR, "embeddings.npy")
EMBEDDINGS_META_PATH = os.path.join(MODELS_DIR, "embed_meta.json")
FAISS_INDEX_PATH = os.path.join(INDEX_DIR, "chat.index")

# Participants of the synthetic group chat ("Hostel Squad 🏠")
PARTICIPANTS = [
    "Shiv", "Aman", "Priya", "Rahul",
    "Neha", "Kabir", "Ananya", "Vikram",
]

# --- Embedding backend -------------------------------------------------
# "auto"      -> try sentence-transformers (real MiniLM multilingual model);
#                if the model/weights can't be downloaded (no internet,
#                offline sandbox, etc.) fall back automatically to a local
#                TF-IDF + SVD embedding so the whole pipeline still runs.
# "st"        -> force sentence-transformers (raises if unavailable)
# "tfidf_svd" -> force the lightweight local fallback
EMBEDDING_BACKEND = os.environ.get("EMBEDDING_BACKEND", "auto")
ST_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
SVD_DIMS = 256

# Minimum cosine similarity to treat a result as relevant at all.
# NOTE: this is backend-dependent. The real sentence-transformers model
# produces well-separated scores where ~0.35 is a good cutoff (see the
# assignment's own example). The offline TF-IDF+SVD fallback compresses
# everything into a narrower, noisier range, so it needs a higher
# cutoff to avoid confidently returning garbage for unrelated queries.
SIMILARITY_THRESHOLD_ST = 0.35
SIMILARITY_THRESHOLD_TFIDF_SVD = 0.42

# How many messages of surrounding context to show around a hit.
CONTEXT_WINDOW = 2

# Number of messages to generate.
NUM_MESSAGES = 4800
MONTHS_OF_HISTORY = 6
