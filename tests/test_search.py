import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from search import ChatSearchEngine  # noqa: E402

engine = ChatSearchEngine()


def test_engine_loads_all_messages():
    assert len(engine.messages) >= 4000


def test_search_returns_results_for_normal_query():
    out = engine.search("Goa trip discussion", top_k=5)
    assert len(out["results"]) > 0


def test_top_k_is_respected():
    out = engine.search("trip discussion", top_k=2)
    assert len(out["results"]) <= 2


def test_person_filter_is_detected():
    out = engine.search("What did Priya say about the budget?", top_k=5)
    assert out["detected_person"] == "Priya"


def test_time_filter_is_detected_for_last_month():
    out = engine.search("What did we discuss last month?", top_k=5)
    assert out["detected_time_range"] is not None


def test_nonsense_query_returns_no_relevant_message():
    out = engine.search("zqxvwk jklpqz xzvbnm09482 qwrtyp", top_k=5)
    assert out["no_relevant_message"] is True


def test_empty_query_does_not_crash():
    out = engine.search("", top_k=5)
    assert out["no_relevant_message"] is True


def test_results_include_context_window():
    out = engine.search("Goa trip discussion", top_k=3)
    for r in out["results"]:
        assert "context" in r
        assert len(r["context"]) >= 1
        assert all(c.get("conversation_id") == r["message"].get("conversation_id") for c in r["context"])
