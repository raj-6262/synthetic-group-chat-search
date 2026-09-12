import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from config import CHAT_MESSAGES_PATH, EVAL_QUERIES_PATH, PARTICIPANTS  # noqa: E402


def load():
    with open(CHAT_MESSAGES_PATH, encoding="utf-8") as f:
        return json.load(f)


def test_message_count_at_least_4000():
    assert len(load()) >= 4000


def test_ids_are_unique_and_sequential():
    messages = load()
    ids = [m["id"] for m in messages]
    assert len(ids) == len(set(ids))
    assert ids == sorted(ids)


def test_eight_participants():
    messages = load()
    senders = set(m["sender"] for m in messages)
    assert senders == set(PARTICIPANTS)
    assert len(PARTICIPANTS) == 8


def test_spans_roughly_six_months():
    from datetime import datetime
    messages = load()
    first = datetime.strptime(messages[0]["timestamp"], "%Y-%m-%d %H:%M")
    last = datetime.strptime(messages[-1]["timestamp"], "%Y-%m-%d %H:%M")
    span_days = (last - first).days
    assert 150 <= span_days <= 200


def test_has_media_and_forwarded_and_typo_style_messages():
    messages = load()
    types = set(m["type"] for m in messages)
    assert "media" in types
    assert "forwarded" in types


def test_every_message_has_conversation_id():
    messages = load()
    assert all(m.get("conversation_id") for m in messages)
    assert len(set(m["conversation_id"] for m in messages)) > 100


def test_decision_threads_present():
    texts = " ".join(m["text"] for m in load())
    assert "Goa in December" in texts
    assert "Cafe Mocha" in texts
    assert "Python then" in texts


def test_evaluation_queries_and_hard_overlap():
    import re
    messages = load()
    by_id = {m["id"]: m for m in messages}
    with open(EVAL_QUERIES_PATH, encoding="utf-8") as f:
        queries = json.load(f)
    words = lambda text: set(re.findall(r"[a-zA-Z']+", text.lower()))
    hard = [q for q in queries if q["hard"]]
    assert len(queries) == 40
    assert len(hard) >= 8
    for q in hard:
        assert words(q["query"]) & words(by_id[q["expected_message_id"]]["text"]) == set()
