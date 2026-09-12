"""
Phase 1b: builds the 40-query ground-truth evaluation set.

- At least 8 queries are "hard": the answer message shares NONE of the
  query's words (checked programmatically below, not just by eye).
- The rest are "warm-up" queries with some lexical overlap.
- Queries are split across the three shapes the assignment calls out:
  meaning-based, person-based, time-based.
"""
import json
import re

try:
    from .config import CHAT_MESSAGES_PATH, EVAL_QUERIES_PATH
except ImportError:  # pragma: no cover - supports python src/generate_eval_queries.py
    from config import CHAT_MESSAGES_PATH, EVAL_QUERIES_PATH


def find_id(messages, substring):
    for m in messages:
        if substring in m["text"]:
            return m["id"]
    raise ValueError(f"Could not find a message containing: {substring!r}")


def words(text):
    return set(re.findall(r"[a-zA-Z']+", text.lower()))


def main():
    with open(CHAT_MESSAGES_PATH, encoding="utf-8") as f:
        messages = json.load(f)
    by_id = {m["id"]: m for m in messages}

    goa_id = find_id(messages, "Done, Goa in December.")
    mocha_id = find_id(messages, "okay let's do Cafe Mocha")
    python_id = find_id(messages, "okay Python then")
    budget_id = find_id(messages, "toh final, 8000 per head budget rakhte hai")
    fest_id = find_id(messages, "hostel ka annual fest is month mein hai na")
    fest_meeting_id = find_id(messages, "fest committee ki meeting Friday ko hai")

    manali_id = find_id(messages, "Manali chalein kya?")
    shiv_goa_id = find_id(messages, "Goa better rahega winters mein")
    rahul_cafe_id = find_id(messages, "What about that new cafe near gate 2?")
    cafe_reviews_id = find_id(messages, "Reviews are decent I heard")
    priyamocha_id = find_id(messages, "I'm in for that")
    python_fast_id = find_id(messages, "Python would be faster to build honestly")
    python_lib_id = find_id(messages, "agreed, python has better libraries too")
    fest_stalls_id = find_id(messages, "haan next week se stalls lagenge")
    dance_id = find_id(messages, "dance competition ke liye team bana lo")
    sound_id = find_id(messages, "main sound system dekh lunga")

    queries = [
        # ---- HARD: zero word-overlap, meaning-based (decision threads) ----
        {"query": "What destination did everyone finally agree upon?", "expected_message_id": goa_id, "hard": True, "shape": "meaning"},
        {"query": "Where are we spending the winter holidays this year?", "expected_message_id": goa_id, "hard": True, "shape": "meaning"},
        {"query": "Which restaurant did everyone settle on for Saturday?", "expected_message_id": mocha_id, "hard": True, "shape": "meaning"},
        {"query": "What venue got finalized for the group dinner?", "expected_message_id": mocha_id, "hard": True, "shape": "meaning"},
        {"query": "Which programming language did the team finally choose?", "expected_message_id": python_id, "hard": True, "shape": "meaning"},
        {"query": "What technology got selected for the project?", "expected_message_id": python_id, "hard": True, "shape": "meaning"},
        {"query": "How much money was finalized for each traveller?", "expected_message_id": budget_id, "hard": True, "shape": "person"},
        {"query": "What big celebration got announced for the building?", "expected_message_id": fest_id, "hard": True, "shape": "meaning"},

        # ---- authored natural-language queries with mixed lexical overlap ----
        {"query": "Where are we going for the trip?", "expected_message_id": goa_id, "hard": False, "shape": "meaning"},
        {"query": "Goa trip discussion", "expected_message_id": goa_id, "hard": False, "shape": "meaning"},
        {"query": "Manali or Goa for the trip?", "expected_message_id": goa_id, "hard": False, "shape": "meaning"},
        {"query": "Which place did Shiv say is better in winter?", "expected_message_id": shiv_goa_id, "hard": False, "shape": "person"},
        {"query": "Who first suggested Manali?", "expected_message_id": manali_id, "hard": False, "shape": "person"},
        {"query": "restaurant for Saturday dinner", "expected_message_id": mocha_id, "hard": False, "shape": "meaning"},
        {"query": "cafe near gate 2 for dinner", "expected_message_id": mocha_id, "hard": False, "shape": "meaning"},
        {"query": "What option did Rahul suggest near gate 2?", "expected_message_id": rahul_cafe_id, "hard": False, "shape": "person"},
        {"query": "What did Shiv hear about the new cafe reviews?", "expected_message_id": cafe_reviews_id, "hard": False, "shape": "person"},
        {"query": "Was Priya okay with the cafe plan?", "expected_message_id": priyamocha_id, "hard": False, "shape": "person"},
        {"query": "programming stack for the college project", "expected_message_id": python_id, "hard": False, "shape": "meaning"},
        {"query": "Java or Python for the project?", "expected_message_id": python_id, "hard": False, "shape": "meaning"},
        {"query": "Why did Priya prefer Python?", "expected_message_id": python_fast_id, "hard": False, "shape": "person"},
        {"query": "What reason did Rahul give for Python?", "expected_message_id": python_lib_id, "hard": False, "shape": "person"},
        {"query": "trip budget per head", "expected_message_id": budget_id, "hard": False, "shape": "meaning"},
        {"query": "how much budget for the trip", "expected_message_id": budget_id, "hard": False, "shape": "meaning"},
        {"query": "What amount did Priya say was reasonable per person?", "expected_message_id": find_id(messages, "mere hisaab se 8000 per head reasonable hai"), "hard": False, "shape": "person"},
        {"query": "What all did the trip cost include?", "expected_message_id": find_id(messages, "travel + stay + food sab milake"), "hard": False, "shape": "meaning"},
        {"query": "annual fest at the hostel", "expected_message_id": fest_id, "hard": False, "shape": "meaning"},
        {"query": "fest committee meeting day", "expected_message_id": fest_meeting_id, "hard": False, "shape": "time"},
        {"query": "When were stalls going to be set up?", "expected_message_id": fest_stalls_id, "hard": False, "shape": "time"},
        {"query": "Who asked people to form a dance team?", "expected_message_id": dance_id, "hard": False, "shape": "person"},
        {"query": "What did Vikram volunteer to manage?", "expected_message_id": sound_id, "hard": False, "shape": "person"},

        # ---- person-based ----
        {"query": "What did Priya say about the budget?", "expected_message_id": budget_id, "hard": False, "shape": "person"},
        {"query": "What did Priya think about Manali?", "expected_message_id": manali_id, "hard": False, "shape": "person"},
        {"query": "What did Rahul say about the new cafe?", "expected_message_id": rahul_cafe_id, "hard": False, "shape": "person"},
        {"query": "What did Shiv suggest for winter?", "expected_message_id": shiv_goa_id, "hard": False, "shape": "person"},
        {"query": "What did Vikram offer to handle for the fest?", "expected_message_id": sound_id, "hard": False, "shape": "person"},
        {"query": "What did Ananya suggest about the dance competition?", "expected_message_id": dance_id, "hard": False, "shape": "person"},

        # ---- time-based ----
        {"query": "What did we discuss last month?", "expected_message_id": fest_id, "hard": False, "shape": "time"},
        {"query": "What was decided in August?", "expected_message_id": python_id, "hard": False, "shape": "time"},
        {"query": "What happened with the trip in April?", "expected_message_id": goa_id, "hard": False, "shape": "time"},
        {"query": "What was discussed about budget in May?", "expected_message_id": budget_id, "hard": False, "shape": "time"},
        {"query": "What got decided in June about dinner?", "expected_message_id": mocha_id, "hard": False, "shape": "time"},
        {"query": "What was the college project stack decision in August?", "expected_message_id": python_id, "hard": False, "shape": "time"},
        {"query": "Which Friday meeting came up during the August fest chat?", "expected_message_id": fest_meeting_id, "hard": False, "shape": "time"},
        {"query": "What plan was closed for Saturday dinner in June?", "expected_message_id": mocha_id, "hard": False, "shape": "time"},
        {"query": "What final trip expense number was discussed in May?", "expected_message_id": budget_id, "hard": False, "shape": "time"},
        {"query": "What travel plan was settled back in April?", "expected_message_id": goa_id, "hard": False, "shape": "time"},
    ]

    queries = queries[:40]
    if len(queries) != 40:
        raise AssertionError(f"Expected exactly 40 queries, got {len(queries)}")

    # Sanity-check the "hard" flags are actually true (zero word overlap).
    for q in queries:
        if q["hard"]:
            answer_text = by_id[q["expected_message_id"]]["text"]
            overlap = words(q["query"]) & words(answer_text)
            if overlap:
                raise AssertionError(f"Query marked hard but overlaps on {overlap}: {q['query']!r} vs {answer_text!r}")

    for i, q in enumerate(queries, start=1):
        q["id"] = i

    with open(EVAL_QUERIES_PATH, "w", encoding="utf-8") as f:
        json.dump(queries, f, ensure_ascii=False, indent=2)

    n_hard = sum(1 for q in queries if q["hard"])
    print(f"Wrote {len(queries)} eval queries ({n_hard} hard/zero-overlap) -> {EVAL_QUERIES_PATH}")


if __name__ == "__main__":
    main()
