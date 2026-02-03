import os
import random
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import streamlit as st
from supabase import create_client, Client


# ----------------------------
# Config
# ----------------------------
TOTAL_QUESTIONS = 15

# Difficulty tuning (adjust to taste)
DIVISOR_MIN = 2
DIVISOR_MAX = 12

QUOTIENT_MIN = 2
QUOTIENT_MAX = 50

# Score formula edge-case handling: accuracy=0 -> use epsilon
ACCURACY_EPSILON = 0.01  # 1% accuracy floor to avoid division by zero


# ----------------------------
# Supabase helpers
# ----------------------------
def get_supabase_client() -> Optional[Client]:
    """
    Creates a Supabase client using Streamlit secrets or environment variables.
    Returns None if credentials are missing (app still runs, but won't save/percentile).
    """
    url = None
    key = None

    # Prefer Streamlit secrets
    if hasattr(st, "secrets"):
        url = st.secrets.get("SUPABASE_URL", None)
        key = st.secrets.get("SUPABASE_SERVICE_ROLE_KEY", None)

    # Fallback to env vars
    url = url or os.getenv("SUPABASE_URL")
    key = key or os.getenv("SUPABASE_SERVICE_ROLE_KEY")

    if not url or not key:
        return None

    return create_client(url, key)


def insert_score(
    sb: Client,
    score: float,
    accuracy: float,
    time_taken: float,
) -> bool:
    """
    Inserts a completed-game row into Supabase.
    Returns True on success, False on failure.
    """
    try:
        sb.table("division_scores").insert(
            {
                "score": float(score),
                "accuracy": float(accuracy),
                "time_taken": float(time_taken),
            }
        ).execute()
        return True
    except Exception:
        return False


def fetch_all_scores(sb: Client, limit: int = 5000) -> List[float]:
    """
    Fetches scores from Supabase. (Lower is better.)
    Uses a limit to avoid pulling an infinite table.
    """
    try:
        resp = sb.table("division_scores").select("score").limit(limit).execute()
        data = resp.data or []
        return [float(row["score"]) for row in data if row.get("score") is not None]
    except Exception:
        return []


def percentile_lower_is_better(all_scores: List[float], user_score: float) -> float:
    """
    Percentile where LOWER score is BETTER.
    We define percentile as:
        percentile = 100 * (count(scores >= user_score) / N)
    - Best (smallest) score => ~100th percentile
    - Worst (largest) score => ~0-very low percentile
    Handles empty list.
    """
    if not all_scores:
        return 100.0
    n = len(all_scores)
    count_ge = sum(1 for s in all_scores if s >= user_score)
    return 100.0 * (count_ge / n)


# ----------------------------
# Game logic
# ----------------------------
@dataclass(frozen=True)
class DivisionProblem:
    dividend: int
    divisor: int
    quotient: int
    remainder: int

    @property
    def text(self) -> str:
        return f"{self.dividend} ÷ {self.divisor}"


def generate_problem() -> DivisionProblem:
    """
    Generates a division problem with a remainder:
      dividend = divisor * quotient + remainder
    Ensures divisor != 0 and remainder < divisor.
    """
    divisor = random.randint(DIVISOR_MIN, DIVISOR_MAX)
    quotient = random.randint(QUOTIENT_MIN, QUOTIENT_MAX)
    remainder = random.randint(0, divisor - 1)
    dividend = divisor * quotient + remainder
    return DivisionProblem(dividend=dividend, divisor=divisor, quotient=quotient, remainder=remainder)


def generate_game_questions(n: int) -> List[DivisionProblem]:
    return [generate_problem() for _ in range(n)]


def init_state() -> None:
    if "mode" not in st.session_state:
        st.session_state.mode = "home"  # home | playing | results

    if "questions" not in st.session_state:
        st.session_state.questions = []

    if "q_index" not in st.session_state:
        st.session_state.q_index = 0

    if "correct_count" not in st.session_state:
        st.session_state.correct_count = 0

    if "answered_count" not in st.session_state:
        st.session_state.answered_count = 0

    if "start_time" not in st.session_state:
        st.session_state.start_time = None

    if "elapsed_time" not in st.session_state:
        st.session_state.elapsed_time = 0.0

    if "last_feedback" not in st.session_state:
        st.session_state.last_feedback = None  # (is_correct, message)

    if "submitted_current" not in st.session_state:
        st.session_state.submitted_current = False

    if "final_score" not in st.session_state:
        st.session_state.final_score = None

    if "final_accuracy" not in st.session_state:
        st.session_state.final_accuracy = None

    if "final_percentile" not in st.session_state:
        st.session_state.final_percentile = None

    if "supabase_ok" not in st.session_state:
        st.session_state.supabase_ok = None  # None unknown, True/False after attempt

    # Input placeholders
    if "input_q" not in st.session_state:
        st.session_state.input_q = None
    if "input_r" not in st.session_state:
        st.session_state.input_r = None


def reset_for_new_game() -> None:
    st.session_state.questions = generate_game_questions(TOTAL_QUESTIONS)
    st.session_state.q_index = 0
    st.session_state.correct_count = 0
    st.session_state.answered_count = 0
    st.session_state.start_time = time.perf_counter()
    st.session_state.elapsed_time = 0.0
    st.session_state.last_feedback = None
    st.session_state.submitted_current = False
    st.session_state.final_score = None
    st.session_state.final_accuracy = None
    st.session_state.final_percentile = None
    st.session_state.input_q = None
    st.session_state.input_r = None
    st.session_state.mode = "playing"


def compute_accuracy(correct: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return correct / total


def compute_score(accuracy: float, time_taken: float) -> float:
    """
    Score = (1/accuracy)^2 * time_taken
    Lower is better.
    Handles accuracy=0 by using ACCURACY_EPSILON.
    """
    a = max(accuracy, ACCURACY_EPSILON)
    return ((1.0 / a) ** 2) * time_taken


def update_elapsed_time_if_playing() -> None:
    if st.session_state.mode == "playing" and st.session_state.start_time is not None:
        st.session_state.elapsed_time = time.perf_counter() - st.session_state.start_time


# ----------------------------
# UI
# ----------------------------
st.set_page_config(page_title="Division Practice (Quotient + Remainder)", page_icon="➗", layout="centered")
init_state()
update_elapsed_time_if_playing()

st.title("➗ Division Practice (Quotient + Remainder)")
st.caption("15 questions per run. Score rewards accuracy hard. Lower score = better 😭🔥")


# Supabase client (optional)
sb = get_supabase_client()
if sb is None:
    st.info(
        "Supabase not configured, so scores/percentile won't save yet.\n\n"
        "Add `SUPABASE_URL` + `SUPABASE_SERVICE_ROLE_KEY` to `.streamlit/secrets.toml` or env vars."
    )


# ----------------------------
# Home screen
# ----------------------------
if st.session_state.mode == "home":
    st.subheader("Start a new game")
    st.write(
        "- You’ll get **15** division questions.\n"
        "- Type **quotient** + **remainder**.\n"
        "- Timer runs only while you’re playing.\n"
        "- Score = **(1/accuracy)² × time_taken** (lower is better).\n"
    )

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🚀 Start game", use_container_width=True):
            reset_for_new_game()
            st.rerun()

    with col2:
        st.markdown("**Difficulty:**")
        st.write(f"Divisor: {DIVISOR_MIN}–{DIVISOR_MAX}")
        st.write(f"Quotient: {QUOTIENT_MIN}–{QUOTIENT_MAX}")

    st.divider()

    st.markdown("### Scoring intuition (cause → effect)")
    st.write(
        "- Accuracy ↑ → (1/accuracy)² ↓ → score ↓ ✅\n"
        "- Time ↑ → score ↑ ❌\n"
        "- All wrong (accuracy=0) → we clamp to 1% to avoid division-by-zero 💀\n"
    )


# ----------------------------
# Playing screen
# ----------------------------
elif st.session_state.mode == "playing":
    q_idx = st.session_state.q_index
    questions: List[DivisionProblem] = st.session_state.questions

    # Safety
    if not questions or q_idx >= len(questions):
        st.session_state.mode = "results"
        st.rerun()

    current = questions[q_idx]

    top = st.columns(3)
    top[0].metric("Question", f"{q_idx + 1} / {TOTAL_QUESTIONS}")
    top[1].metric("Correct", f"{st.session_state.correct_count}")
    top[2].metric("Time", f"{st.session_state.elapsed_time:.1f}s")

    st.divider()

    st.markdown(f"## {current.text}")
    st.write("Enter your answers:")

    # Inputs
    c1, c2 = st.columns(2)
    with c1:
        st.number_input(
            "Quotient",
            min_value=0,
            step=1,
            key="input_q",
            disabled=st.session_state.submitted_current,
        )
    with c2:
        st.number_input(
            "Remainder",
            min_value=0,
            step=1,
            key="input_r",
            disabled=st.session_state.submitted_current,
        )

    # Submit / Feedback
    submit_col, next_col = st.columns(2)

    with submit_col:
        if st.button("✅ Submit", use_container_width=True, disabled=st.session_state.submitted_current):
            user_q = st.session_state.input_q
            user_r = st.session_state.input_r
