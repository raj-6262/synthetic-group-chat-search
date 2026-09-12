# Semantic Group Chat Search

## Problem

Keyword search breaks when you remember the meaning of a group-chat decision but not the exact words. A query like "What destination did everyone finally agree upon?" should retrieve the decision message even if the message itself says only "Done, Goa in December."

## Solution

Each chat message is embedded into a vector, stored in a FAISS index, and searched by cosine similarity. Results include a small context window from the same `conversation_id`, so the hit can be read with the surrounding discussion.

```
User query
   |
   v
Sentence Transformer embedding
   |
   v
normalized vector
   |
   v
FAISS similarity search
   |
   v
message metadata + same-conversation context
   |
   v
FastAPI + browser demo
```

The code also keeps a TF-IDF baseline for comparison. It is not used as the semantic search implementation.

## Architecture

```
data/chat_messages.json
        |
        v
src/build_index.py -> models/embeddings.npy -> index/chat.index
        |
        v
src/search.py -> backend/main.py -> frontend/index.html
        |
        v
src/evaluate.py -> results/evaluation.json + charts
```

## Tech Stack

Python, Sentence Transformers, FAISS, NumPy, scikit-learn, FastAPI, Uvicorn, vanilla HTML/CSS/JavaScript, matplotlib, pytest.

## Dataset

- 4,800 synthetic messages
- 8 synthetic participants: Shiv, Aman, Priya, Rahul, Neha, Kabir, Ananya, Vikram
- Date range: 2026-03-03 to 2026-09-01, roughly six months
- Messy chat style: Hinglish, typos, abbreviations, one-word replies, forwarded text, photos/videos/voice placeholders
- Every message has `conversation_id`
- Decision threads include trip destination, restaurant/venue, project technology, trip budget, and hostel fest planning

## Search

The intended final semantic path is:

```
query -> paraphrase-multilingual-MiniLM-L12-v2 -> normalized embedding -> FAISS -> ranked messages
```

Sender and date filters are applied before ranking when provided, so filtered answers are not lost by taking a tiny global FAISS top-k first. For this dataset size, direct ranking inside the filtered candidate set is simple and reliable.

## Evaluation

`data/evaluation_queries.json` contains 40 authored natural-language queries. Exactly 8 are hard zero-word-overlap queries, verified programmatically by comparing normalized query words with the expected answer message words.

Metrics reported:

- Overall Top-1 accuracy
- Hard zero-word-overlap Top-1 accuracy
- Recall@5
- MRR
- Average, p50, and p95 semantic search latency
- TF-IDF baseline comparison

## Results

Actual results from the current local run:

| Method | Overall Top-1 | Hard 8 Top-1 |
|---|---:|---:|
| TF-IDF baseline | 27.5% | 0.0% |
| Semantic pipeline, current backend | 42.5% | 0.0% |

Current embedding backend in `results/evaluation.json`: `tfidf_svd`.

Important: this machine did not have `sentence-transformers`/`torch` installed, and installing them timed out. The generated artifacts therefore use the disclosed local TF-IDF+SVD fallback, not the real Sentence Transformer model. The code is ready to use `paraphrase-multilingual-MiniLM-L12-v2` automatically when the dependency and model weights are available. Do not present fallback numbers as real Sentence Transformer numbers.

To force the real model:

```bash
$env:EMBEDDING_BACKEND="st"
python -m src.build_index
python -m src.evaluate
```

On macOS/Linux:

```bash
EMBEDDING_BACKEND=st python -m src.build_index
EMBEDDING_BACKEND=st python -m src.evaluate
```

## Baseline

`src/tfidf_baseline.py` uses word-level TF-IDF and cosine similarity. It is included only to show how keyword-style retrieval behaves against the same 40 ground-truth queries.

## How To Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Rebuild everything:

```bash
python -m src.generate_data
python -m src.generate_eval_queries
python -m src.build_index
python -m src.evaluate
```

Start the app:

```bash
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

Open:

```text
http://127.0.0.1:8000
```

Run tests:

```bash
python -m pytest tests -v
```

## What Is Mocked/Synthetic

Synthetic:

- Chat corpus
- Participant names
- Conversation threads
- Evaluation ground truth

Real:

- Embedding/index/search pipeline
- FAISS similarity retrieval
- Metadata filtering
- Context retrieval by `conversation_id`
- FastAPI backend and browser UI
- Evaluation calculations

## API

```text
GET /health
GET /participants
GET /search?q=<query>&top_k=5&participant=Priya&start_date=2026-05-01&end_date=2026-05-31
```

Search responses include query metadata, score, sender, timestamp, message text, `conversation_id`, context, and `search_time_ms`.

## Limitations

- The committed local index was built with the fallback backend because the real Sentence Transformer dependency could not be installed in this environment within the time limit.
- The fallback is useful for development but does not solve true zero-word-overlap semantics; hard-query accuracy is 0.0% in the current local run.
- Date/person understanding is lightweight rule-based parsing.
- JSON is the metadata store; SQLite or a production database would be the next step for larger corpora.

## Future Improvements

- Run and publish final numbers with the real multilingual MiniLM model.
- Add a production metadata database.
- Add a reranker for top candidates.
- Ingest real exported chats.
- Move from flat FAISS to an approximate index for much larger corpora v1542

