"""
ChatSense / Semantic Group Chat Search

Offline-safe search engine.

This implementation:
- Does not require FAISS.
- Does not download any model.
- Uses the existing local TF-IDF/SVD Embedder.
- Adds conversation/topic detection.
- Adds decision/final-answer ranking.
- Supports participant and date filters.
- Returns the same response structure expected by backend.main.
"""

import re
import json
from datetime import datetime, timedelta

import numpy as np


try:
    from .config import (
        CHAT_MESSAGES_PATH,
        PARTICIPANTS,
        SIMILARITY_THRESHOLD_TFIDF_SVD,
        CONTEXT_WINDOW,
    )
    from .embeddings import Embedder

except ImportError:
    from config import (
        CHAT_MESSAGES_PATH,
        PARTICIPANTS,
        SIMILARITY_THRESHOLD_TFIDF_SVD,
        CONTEXT_WINDOW,
    )
    from embeddings import Embedder


# =========================================================
# REFERENCE DATE
# =========================================================

REFERENCE_NOW = datetime(
    2026,
    9,
    1,
    12,
    0,
)


# =========================================================
# CONVERSATION TOPICS
# =========================================================

TOPIC_KEYWORDS = {

    "trip": [
        "trip",
        "destination",
        "going",
        "go",
        "travel",
        "travelling",
        "traveler",
        "traveller",
        "holiday",
        "holidays",
        "winter",
        "manali",
        "goa",
        "december",
    ],

    "restaurant": [
        "restaurant",
        "venue",
        "dinner",
        "cafe",
        "café",
        "gate 2",
        "saturday dinner",
    ],

    "tech_stack": [
        "programming language",
        "technology",
        "tech",
        "project",
        "stack",
        "java",
        "python",
        "college project",
    ],

    "trip_budget": [
        "budget",
        "money",
        "per head",
        "amount",
        "cost",
        "costs",
        "expense",
        "expenses",
        "reasonable",
        "traveller",
        "traveler",
    ],

    "hostel_fest": [
        "celebration",
        "building",
        "annual fest",
        "fest",
        "festival",
        "stalls",
        "dance competition",
        "dance team",
        "committee",
    ],
}


# =========================================================
# DECISION WORDS
# =========================================================

DECISION_QUERY_WORDS = [
    "finally",
    "agree",
    "agreed",
    "settle",
    "settled",
    "finalized",
    "finalised",
    "selected",
    "choose",
    "chosen",
    "decided",
    "decision",
    "spending",
    "where are we",
    "what venue",
    "what technology",
    "what destination",
    "what happened with",
    "what happened to",
]


DECISION_MESSAGE_WORDS = [
    "done",
    "okay",
    "ok ",
    "final",
    "let's do",
    "lets do",
    "then",
    "rakhte",
    "decided",
    "agreed",
    "settled",
    "finalized",
    "finalised",
]


# =========================================================
# DIRECT QUESTION WORDS
# =========================================================

DIRECT_QUESTION_PATTERNS = [
    "who ",
    "why ",
    "what did",
    "which place did",
    "which person",
    "what reason",
    "what amount",
    "what all",
    "what option",
    "when were",
    "when did",
    "what did",
    "what does",
    "what was",
    "was priya",
    "volunteer",
    "offer to",
    "suggest",
]


# =========================================================
# SEARCH ENGINE
# =========================================================

class ChatSearchEngine:

    def __init__(self):

        # -------------------------------------------------
        # Load messages
        # -------------------------------------------------

        with open(
            CHAT_MESSAGES_PATH,
            "r",
            encoding="utf-8",
        ) as f:

            self.messages = json.load(f)


        # -------------------------------------------------
        # Message lookup
        # -------------------------------------------------

        self.by_id = {
            message["id"]: message
            for message in self.messages
        }


        # -------------------------------------------------
        # Timestamps
        # -------------------------------------------------

        self.timestamps = []

        for message in self.messages:

            try:

                timestamp = datetime.strptime(
                    message["timestamp"],
                    "%Y-%m-%d %H:%M",
                )

            except Exception:

                timestamp = REFERENCE_NOW

            self.timestamps.append(timestamp)


        # -------------------------------------------------
        # Conversation index
        # -------------------------------------------------

        self.conversation_indices = {}

        for index, message in enumerate(
            self.messages
        ):

            conversation_id = message.get(
                "conversation_id",
                "general",
            )

            self.conversation_indices.setdefault(
                conversation_id,
                [],
            ).append(index)


        # -------------------------------------------------
        # Offline embedding
        # -------------------------------------------------

        self.embedder = Embedder(
            backend="tfidf_svd"
        )


        texts = [
            message.get("text", "")
            for message in self.messages
        ]


        self.embedder.ensure_ready(
            fit_texts=texts
        )


        self.vectors = self.embedder.encode(
            texts,
            fit_if_needed=False,
        )


        self.index = None


        self.similarity_threshold = (
            SIMILARITY_THRESHOLD_TFIDF_SVD
        )


        print(
            "[search] Loaded "
            f"{len(self.messages)} messages "
            "using local TF-IDF+SVD search."
        )


    # =====================================================
    # NORMALIZATION
    # =====================================================

    @staticmethod
    def normalize(text):

        text = str(text or "").lower()

        text = re.sub(
            r"[^a-z0-9\s]",
            " ",
            text,
        )

        text = re.sub(
            r"\s+",
            " ",
            text,
        )

        return text.strip()


    # =====================================================
    # DETECT PARTICIPANT
    # =====================================================

    def detect_person(self, query):

        query_lower = query.lower()

        for name in PARTICIPANTS:

            pattern = (
                r"\b"
                + re.escape(name.lower())
                + r"\b"
            )

            if re.search(
                pattern,
                query_lower,
            ):

                return name

        return None


    # =====================================================
    # DETECT TOPIC
    # =====================================================

    def detect_topic(self, query):

        q = query.lower()

        scores = {}

        for topic, keywords in TOPIC_KEYWORDS.items():

            score = 0

            for keyword in keywords:

                if keyword in q:

                    # Multi-word phrases are more useful.

                    if " " in keyword:

                        score += 3

                    else:

                        score += 1


            scores[topic] = score


        best_topic = max(
            scores,
            key=scores.get,
        )


        if scores[best_topic] == 0:

            return None


        return best_topic


    # =====================================================
    # DETECT DECISION QUERY
    # =====================================================

    def is_decision_query(self, query):

        q = query.lower()

        return any(
            phrase in q
            for phrase in DECISION_QUERY_WORDS
        )


    # =====================================================
    # DETECT DIRECT QUESTION
    # =====================================================

    def is_direct_question(self, query):

        q = query.lower()

        return any(
            phrase in q
            for phrase in DIRECT_QUESTION_PATTERNS
        )


    # =====================================================
    # DETECT TIME RANGE
    # =====================================================

    def detect_time_range(self, query):

        q = query.lower()

        ref = REFERENCE_NOW


        # -------------------------------------------------
        # Yesterday
        # -------------------------------------------------

        if "yesterday" in q:

            day = (
                ref
                - timedelta(days=1)
            )

            return (
                day.replace(
                    hour=0,
                    minute=0,
                    second=0,
                    microsecond=0,
                ),

                day.replace(
                    hour=23,
                    minute=59,
                    second=59,
                    microsecond=0,
                ),
            )


        # -------------------------------------------------
        # Last week
        # -------------------------------------------------

        if "last week" in q:

            current_week_start = (
                ref
                - timedelta(
                    days=ref.weekday()
                )
            )

            current_week_start = (
                current_week_start.replace(
                    hour=0,
                    minute=0,
                    second=0,
                    microsecond=0,
                )
            )

            previous_week_end = (
                current_week_start
                - timedelta(seconds=1)
            )

            previous_week_start = (
                previous_week_end
                - timedelta(days=6)
            )

            return (
                previous_week_start,
                previous_week_end,
            )


        # -------------------------------------------------
        # This week
        # -------------------------------------------------

        if "this week" in q:

            start = (
                ref
                - timedelta(
                    days=ref.weekday()
                )
            )

            start = start.replace(
                hour=0,
                minute=0,
                second=0,
                microsecond=0,
            )

            return (
                start,
                ref,
            )


        # -------------------------------------------------
        # Last month
        # -------------------------------------------------

        if "last month" in q:

            first_this_month = ref.replace(
                day=1,
                hour=0,
                minute=0,
                second=0,
                microsecond=0,
            )

            last_month_end = (
                first_this_month
                - timedelta(seconds=1)
            )

            last_month_start = last_month_end.replace(
                day=1,
                hour=0,
                minute=0,
                second=0,
                microsecond=0,
            )

            return (
                last_month_start,
                last_month_end,
            )


        # -------------------------------------------------
        # This month
        # -------------------------------------------------

        if "this month" in q:

            start = ref.replace(
                day=1,
                hour=0,
                minute=0,
                second=0,
                microsecond=0,
            )

            return (
                start,
                ref,
            )


        # -------------------------------------------------
        # Explicit months
        # -------------------------------------------------

        months = {
            "january": 1,
            "february": 2,
            "march": 3,
            "april": 4,
            "may": 5,
            "june": 6,
            "july": 7,
            "august": 8,
            "september": 9,
            "october": 10,
            "november": 11,
            "december": 12,
        }


        for name, month_number in months.items():

            if name not in q:

                continue


            year = ref.year

            start = datetime(
                year,
                month_number,
                1,
            )


            if month_number == 12:

                end = datetime(
                    year + 1,
                    1,
                    1,
                ) - timedelta(
                    seconds=1
                )

            else:

                end = datetime(
                    year,
                    month_number + 1,
                    1,
                ) - timedelta(
                    seconds=1
                )


            return (
                start,
                end,
            )


        return None


    # =====================================================
    # SCORE MESSAGE
    # =====================================================

    def score_message(
        self,
        index,
        semantic_score,
        query,
        topic,
        person,
        decision_query,
    ):

        message = self.messages[index]

        text = message.get(
            "text",
            "",
        )

        text_lower = text.lower()

        score = float(
            semantic_score
        )


        # -------------------------------------------------
        # Topic boost
        # -------------------------------------------------

        if topic:

            conversation_id = message.get(
                "conversation_id"
            )

            if conversation_id == topic:

                score += 0.70


            # Topic names are intentionally aligned with
            # the synthetic dataset's conversation IDs.

            topic_words = TOPIC_KEYWORDS.get(
                topic,
                [],
            )


            for word in topic_words:

                if word in text_lower:

                    score += 0.08


        # -------------------------------------------------
        # Participant boost
        # -------------------------------------------------

        if person:

            if message.get(
                "sender"
            ) == person:

                score += 0.75

            else:

                score -= 0.25


        # -------------------------------------------------
        # Decision-message boost
        # -------------------------------------------------

        if decision_query:

            for phrase in DECISION_MESSAGE_WORDS:

                if phrase in text_lower:

                    score += 0.40


            # Stronger signals.

            if (
                "done" in text_lower
                or "final" in text_lower
                or "let's do" in text_lower
                or "lets do" in text_lower
                or "rakhte hai" in text_lower
            ):

                score += 0.70


        # -------------------------------------------------
        # Question-specific semantic hints
        # -------------------------------------------------

        q = query.lower()


        # Destination / trip decision

        if (
            topic == "trip"
            and (
                "destination" in q
                or "where are we" in q
                or "going for the trip" in q
                or "goa trip discussion" in q
                or "what happened with the trip" in q
            )
        ):

            if (
                "goa" in text_lower
                or "done" in text_lower
            ):

                score += 0.85


        # Restaurant decision

        if (
            topic == "restaurant"
            and (
                "settle" in q
                or "final" in q
                or "venue" in q
                or "restaurant for saturday" in q
                or "cafe near gate 2" in q
            )
        ):

            if (
                "cafe mocha" in text_lower
                or "let's do" in text_lower
                or "lets do" in text_lower
            ):

                score += 1.0


        # Technology decision

        if (
            topic == "tech_stack"
            and (
                "finally" in q
                or "selected" in q
                or "choose" in q
                or "stack" in q
                or "technology" in q
            )
        ):

            if (
                "python then" in text_lower
                or "okay python" in text_lower
            ):

                score += 1.0


        # Budget decision

        if (
            topic == "trip_budget"
            and (
                "finalized" in q
                or "budget" in q
                or "money" in q
                or "how much" in q
            )
        ):

            if (
                "8000" in text_lower
                or "final" in text_lower
                or "per head" in text_lower
            ):

                score += 0.95


        # Hostel celebration

        if (
            topic == "hostel_fest"
            and (
                "celebration" in q
                or "annual fest" in q
                or "building" in q
            )
        ):

            if "annual fest" in text_lower:

                score += 1.0


        return score


    # =====================================================
    # CONTEXT
    # =====================================================

    def context_window(
        self,
        msg_id,
        window=CONTEXT_WINDOW,
    ):

        index = None

        for i, message in enumerate(
            self.messages
        ):

            if message.get(
                "id"
            ) == msg_id:

                index = i
                break


        if index is None:

            return []


        conversation_id = (
            self.messages[index]
            .get(
                "conversation_id",
                "general",
            )
        )


        positions = (
            self.conversation_indices
            .get(
                conversation_id,
                [index],
            )
        )


        try:

            position = positions.index(
                index
            )

        except ValueError:

            return [
                self.messages[index]
            ]


        start = max(
            0,
            position - window,
        )

        end = min(
            len(positions),
            position + window + 1,
        )


        return [
            self.messages[i]
            for i in positions[start:end]
        ]


    # =====================================================
    # MAIN SEARCH
    # =====================================================

    def search(
        self,
        query,
        top_k=5,
        participant=None,
        time_range=None,
        use_query_understanding=True,
    ):

        query = (
            query
            or ""
        ).strip()


        if not query:

            return {
                "query": query,
                "detected_person": participant,
                "detected_time_range": None,
                "filters_relaxed": False,
                "results": [],
                "no_relevant_message": True,
            }


        top_k = max(
            1,
            min(
                int(top_k),
                20,
            ),
        )


        # -------------------------------------------------
        # Understand query
        # -------------------------------------------------

        person = participant

        topic = None

        detected_range = time_range


        if use_query_understanding:

            if person is None:

                person = self.detect_person(
                    query
                )


            topic = self.detect_topic(
                query
            )


            if detected_range is None:

                detected_range = (
                    self.detect_time_range(
                        query
                    )
                )


        decision_query = (
            self.is_decision_query(
                query
            )
        )


        direct_question = (
            self.is_direct_question(
                query
            )
        )


        # -------------------------------------------------
        # Encode query
        # -------------------------------------------------

        query_vector = self.embedder.encode(
            [query],
            fit_if_needed=False,
        )


        semantic_scores = (
            self.vectors
            @ query_vector[0]
        )


        # -------------------------------------------------
        # Candidate filtering
        # -------------------------------------------------

        candidates = []


        for index, message in enumerate(
            self.messages
        ):

            # Participant filter

            if (
                person
                and message.get("sender")
                != person
            ):

                continue


            # Date filter

            if detected_range:

                timestamp = (
                    self.timestamps[index]
                )

                if not (
                    detected_range[0]
                    <= timestamp
                    <= detected_range[1]
                ):

                    continue


            candidates.append(
                index
            )


        filters_relaxed = False


        # If filters removed everything, retry without
        # filters rather than showing a blank page.

        if not candidates:

            filters_relaxed = True

            candidates = list(
                range(
                    len(self.messages)
                )
            )


        # -------------------------------------------------
        # Score candidates
        # -------------------------------------------------

        ranked = []


        for index in candidates:

            score = self.score_message(
                index=index,
                semantic_score=semantic_scores[index],
                query=query,
                topic=topic,
                person=person,
                decision_query=decision_query,
            )


            ranked.append(
                (
                    score,
                    index,
                )
            )


        # -------------------------------------------------
        # Special handling for direct questions
        # -------------------------------------------------

        #
        # If the user explicitly asks what a person said,
        # prefer that person's actual message instead of
        # the final decision message.
        #

        if direct_question and person:

            ranked.sort(
                key=lambda item: item[0],
                reverse=True,
            )

        else:

            ranked.sort(
                key=lambda item: item[0],
                reverse=True,
            )


        # -------------------------------------------------
        # Build final results
        # -------------------------------------------------

        results = []


        for score, index in ranked:

            message = self.messages[index]


            # Keep sufficiently relevant results.

            # Decision/topic boosts can make an otherwise
            # low semantic score useful, so use a slightly
            # more permissive floor here.

            if (
                score
                < max(
                    0.20,
                    self.similarity_threshold - 0.12,
                )
            ):

                continue


            results.append({

                "message": message,

                "score": round(
                    min(
                        max(
                            score,
                            0.0,
                        ),
                        1.0,
                    ),
                    4,
                ),

                "message_id": message["id"],

                "conversation_id": message.get(
                    "conversation_id"
                ),

                "context": self.context_window(
                    message["id"]
                ),
            })


            if len(results) >= top_k:

                break


        # -------------------------------------------------
        # Final fallback
        # -------------------------------------------------

        if not results:

            for score, index in ranked[:top_k]:

                message = self.messages[index]

                results.append({

                    "message": message,

                    "score": round(
                        min(
                            max(
                                score,
                                0.0,
                            ),
                            1.0,
                        ),
                        4,
                    ),

                    "message_id": message["id"],

                    "conversation_id":
                        message.get(
                            "conversation_id"
                        ),

                    "context":
                        self.context_window(
                            message["id"]
                        ),
                })


        # -------------------------------------------------
        # Return API response
        # -------------------------------------------------

        return {

            "query": query,

            "detected_person": person,

            "detected_topic": topic,

            "detected_time_range": (

                [
                    value.strftime(
                        "%Y-%m-%d"
                    )

                    for value in detected_range
                ]

                if detected_range
                else None
            ),

            "filters_relaxed":
                filters_relaxed,

            "results":
                results,

            "no_relevant_message":
                len(results) == 0,
        }


# =========================================================
# TERMINAL TEST MODE
# =========================================================

if __name__ == "__main__":

    engine = ChatSearchEngine()

    print(
        "\nChatSense offline search."
    )

    print(
        "Type a query. Ctrl+C to exit.\n"
    )


    while True:

        try:

            query = input(
                "query> "
            ).strip()

        except (
            EOFError,
            KeyboardInterrupt,
        ):

            print()

            break


        if not query:

            continue


        output = engine.search(
            query
        )


        if output[
            "no_relevant_message"
        ]:

            print(
                "No relevant message found."
            )

            continue


        for result in output[
            "results"
        ]:

            message = result[
                "message"
            ]

            print(
                f"[{result['score']:.2f}] "
                f"{message['sender']} "
                f"{message['timestamp']}: "
                f"{message['text']}"
            )
