"""
Generates a synthetic, messy, realistic group chat.

Design goals (mirroring the actual assignment, not just the minimums):
  - 4000+ messages, 8 participants, 6 months
  - Hinglish, typos, one-word replies, forwarded text, media-omitted lines
  - THREE decision threads that resolve something concrete
  - Messages that let us test all three query "shapes":
        meaning-based ("when did we decide on the trip")
        person-based  ("what did Priya say about the budget")
        time-based    ("what did we discuss last month")
  - No real chat data is used anywhere - everything below is invented.
"""
import json
import random
from datetime import datetime, timedelta

try:
    from .config import PARTICIPANTS, NUM_MESSAGES, MONTHS_OF_HISTORY, CHAT_MESSAGES_PATH
except ImportError:  # pragma: no cover - supports python src/generate_data.py
    from config import PARTICIPANTS, NUM_MESSAGES, MONTHS_OF_HISTORY, CHAT_MESSAGES_PATH

random.seed(42)

# Reference "today" for the whole project - all time-based queries
# ("last month", "this week", etc.) are resolved relative to this.
NOW = datetime(2026, 9, 1, 12, 0)
START = NOW - timedelta(days=int(MONTHS_OF_HISTORY * 30.4))

ONE_WORD_REPLIES = ["okay", "haan", "nah", "done", "sure", "thik hai", "ok",
                    "hmm", "kk", "bilkul", "nope", "yup", "acha"]

MEDIA_LINES = ["[Photo]", "[Video]", "Sent an image", "Voice message (0:19)",
               "Voice message (0:41)", "<Media omitted>", "[GIF]", "[Sticker]"]

FORWARDED_TEMPLATES = [
    "Forwarded: Tomorrow's class has been shifted to 11 AM.",
    "Forwarded: Library will remain closed this Sunday for maintenance.",
    "Forwarded: Fee submission deadline extended to next Friday.",
    "Forwarded: Campus wifi will be down 2-4 PM for upgrade.",
    "Forwarded: Placement cell seminar on Thursday, attendance compulsory.",
]

# ---------------------------------------------------------------------
# Generic filler topics used to pad the corpus to 4000+ messages while
# still feeling like a real hostel/college group chat.
# ---------------------------------------------------------------------
FILLER_BURSTS = [
    [("A", "bhai kal college aa raha hai?"), ("B", "haan probably"),
     ("A", "kis time?"), ("B", "10 baje tak")],
    [("A", "mess ka khana bahut bekar hai aaj"), ("B", "haan bilkul"),
     ("C", "chalo maggi banate hai"), ("D", "+1")],
    [("A", "kal ka match dekha?"), ("B", "haan India jeet gaya!"),
     ("C", "last over mein finish hua"), ("A", "kya scene tha bhai")],
    [("A", "yaar exam ka pata nahi kya hoga"), ("B", "syllabus khatam nahi hua"),
     ("C", "same here"), ("D", "chill ho jayega")],
    [("A", "assignment submit kar diya?"), ("B", "nahi yaar deadline kal hai"),
     ("C", "kal raat ko karunga"), ("A", "same plan lol")],
    [("A", "chai peene chalte hai"), ("B", "haan chalo"), ("C", "5 min mein aata hu")],
    [("A", "ye meme dekh"), ("B", "[Photo]"), ("C", "😂😂😂"), ("D", "bilkul sahi")],
    [("A", "kal quiz hai kya econ ka?"), ("B", "haan syllabus tak hai"),
     ("C", "kitna portion?"), ("A", "unit 3 tak")],
    [("A", "wifi phir se down hai kya"), ("B", "haan mera bhi"),
     ("C", "hostel wala issue hoga"), ("D", "complain karo warden ko")],
    [("A", "movie plan karte hai weekend pe"), ("B", "kaunsi?"),
     ("C", "koi bhi chalega"), ("D", "sunday theek rahega sabke liye?"),
     ("A", "haan sunday works")],
    [("A", "lab record complete hai tera?"), ("B", "nahi bhai half hi hua"),
     ("C", "submission kal hai na"), ("A", "raat ko baith ke karte hai saath")],
    [("A", "kisi ke paas notes hai DBMS ke?"), ("B", "haan bhejta hu"),
     ("C", "mujhe bhi bhej dena"), ("A", "thanks yaar")],
    [("A", "gym chalna hai kal subah?"), ("B", "haan 6 baje"),
     ("C", "utha dena mujhe bhi"), ("A", "ok")],
    [("A", "voice message (0:23)"), ("B", "sun liya"), ("C", "haha sahi hai")],
    [("A", "kal ka schedule kya hai"), ("B", "9 to 1 classes phir break"),
     ("C", "fir lab hai 2 se 4")],
]

# ---------------------------------------------------------------------
# Fixed decision threads - each resolves something concrete and each
# final line is deliberately worded so that natural questions about it
# share almost no words with the answer (the "hard" eval queries hook
# into these).
# ---------------------------------------------------------------------
DECISION_THREADS = [
    {
        "name": "trip",
        "day_offset": 38,
        "lines": [
            ("Aman", "Guys trip ka kya scene? Semester break aa raha hai"),
            ("Priya", "Manali chalein kya?"),
            ("Rahul", "December me bahut cold hoga wahan"),
            ("Shiv", "Goa better rahega winters mein"),
            ("Aman", "Goa seems good honestly"),
            ("Priya", "+1 for Goa"),
            ("Rahul", "okay Goa then"),
            ("Aman", "Done, Goa in December."),
        ],
        "final_text": "Done, Goa in December.",
    },
    {
        "name": "restaurant",
        "day_offset": 96,
        "lines": [
            ("Shiv", "Saturday dinner kaha karein?"),
            ("Aman", "BBQ Nation chalein?"),
            ("Priya", "too expensive yaar for everyone"),
            ("Rahul", "What about that new cafe near gate 2?"),
            ("Shiv", "Reviews are decent I heard"),
            ("Priya", "I'm in for that"),
            ("Aman", "okay let's do Cafe Mocha"),
        ],
        "final_text": "okay let's do Cafe Mocha",
    },
    {
        "name": "tech_stack",
        "day_offset": 151,
        "lines": [
            ("Aman", "Which stack are we using for the college project?"),
            ("Shiv", "Java?"),
            ("Priya", "Python would be faster to build honestly"),
            ("Rahul", "agreed, python has better libraries too"),
            ("Aman", "okay Python then"),
        ],
        "final_text": "okay Python then",
    },
]

# A budget-planning thread that is heavily driven by Priya, used for the
# "what did Priya say about X" (person-based) query shape.
BUDGET_THREAD = {
    "conversation_id": "trip_budget",
    "day_offset": 60,
    "lines": [
        ("Rahul", "guys trip ka budget kitna rakhein"),
        ("Priya", "mere hisaab se 8000 per head reasonable hai"),
        ("Priya", "travel + stay + food sab milake"),
        ("Shiv", "thoda kam nahi ho sakta?"),
        ("Priya", "kar sakte hai but comfort kam ho jayega"),
        ("Aman", "8000 chalega sabko lagta hai"),
        ("Priya", "toh final, 8000 per head budget rakhte hai"),
    ],
}

# A thread anchored to a specific, distinctive calendar month so a
# "what did we discuss last month" style query has a clean ground truth.
TIME_ANCHOR_THREAD = {
    "conversation_id": "hostel_fest",
    "day_offset": 164,  # lands ~Aug 15, i.e. "last month" relative to NOW (Sep 1)
    "lines": [
        ("Neha", "hostel ka annual fest is month mein hai na"),
        ("Kabir", "haan next week se stalls lagenge"),
        ("Ananya", "dance competition ke liye team bana lo"),
        ("Vikram", "main sound system dekh lunga"),
        ("Neha", "great, fest committee ki meeting Friday ko hai"),
    ],
}


def add_typo(text: str) -> str:
    """Randomly mangle a message slightly, the way people actually type."""
    if len(text) < 6:
        return text
    i = random.randint(1, len(text) - 2)
    op = random.choice(["swap", "drop", "double"])
    if op == "swap":
        chars = list(text)
        chars[i], chars[i + 1] = chars[i + 1], chars[i]
        return "".join(chars)
    if op == "drop":
        return text[:i] + text[i + 1:]
    return text[:i] + text[i] + text[i:]


def build_timeline():
    messages = []
    msg_id = 1
    cur_day = 0
    total_days = int((NOW - START).days)
    filler_conversation = 1

    def add_message(dt, sender, text, mtype="text", conversation_id="general"):
        nonlocal msg_id
        messages.append({
            "id": msg_id,
            "conversation_id": conversation_id,
            "timestamp": dt.strftime("%Y-%m-%d %H:%M"),
            "sender": sender,
            "text": text,
            "type": mtype,
        })
        msg_id += 1

    def inject_thread(thread, extra_participants=None):
        base_dt = START + timedelta(days=thread["day_offset"],
                                     hours=random.randint(18, 21),
                                     minutes=random.randint(0, 59))
        conversation_id = thread.get("conversation_id", thread.get("name", "thread"))
        for sender, text in thread["lines"]:
            add_message(base_dt, sender, text, conversation_id=conversation_id)
            base_dt += timedelta(minutes=random.randint(1, 4))

    # Seed the fixed, important threads first.
    for thread in DECISION_THREADS:
        inject_thread(thread)
    inject_thread(BUDGET_THREAD)
    inject_thread(TIME_ANCHOR_THREAD)

    # Fill the rest of the six months with noisy, everyday chatter.
    while len(messages) < NUM_MESSAGES:
        day = random.randint(0, total_days - 1)
        dt = START + timedelta(days=day, hours=random.randint(8, 23),
                                minutes=random.randint(0, 59))
        burst = random.choice(FILLER_BURSTS)
        role_map = {}
        roles = sorted(set(r for r, _ in burst))
        chosen = random.sample(PARTICIPANTS, k=min(len(roles), len(PARTICIPANTS)))
        for role, name in zip(roles, chosen):
            role_map[role] = name

        for role, text in burst:
            sender = role_map[role]
            conversation_id = f"daily_{filler_conversation:04d}"
            roll = random.random()
            if roll < 0.08:
                text = random.choice(ONE_WORD_REPLIES)
                mtype = "text"
            elif roll < 0.14:
                text = random.choice(MEDIA_LINES)
                mtype = "media"
            elif roll < 0.17:
                text = random.choice(FORWARDED_TEMPLATES)
                mtype = "forwarded"
            elif roll < 0.30:
                text = add_typo(text)
                mtype = "text"
            else:
                mtype = "text"
            add_message(dt, sender, text, mtype, conversation_id=conversation_id)
            dt += timedelta(minutes=random.randint(1, 5))

        # occasional stray one-word reply from someone else, common in
        # real group chats
        if random.random() < 0.3 and len(messages) < NUM_MESSAGES:
            add_message(dt, random.choice(PARTICIPANTS),
                        random.choice(ONE_WORD_REPLIES),
                        conversation_id=f"daily_{filler_conversation:04d}")
        filler_conversation += 1

    # Sort by timestamp (bursts were injected out of chronological order
    # relative to random fillers) and re-number ids/timestamps cleanly.
    messages.sort(key=lambda m: m["timestamp"])
    for i, m in enumerate(messages, start=1):
        m["id"] = i

    return messages


def main():
    messages = build_timeline()
    with open(CHAT_MESSAGES_PATH, "w", encoding="utf-8") as f:
        json.dump(messages, f, ensure_ascii=False, indent=2)
    print(f"Generated {len(messages)} messages -> {CHAT_MESSAGES_PATH}")
    print(f"Date range: {messages[0]['timestamp']} to {messages[-1]['timestamp']}")


if __name__ == "__main__":
    main()
