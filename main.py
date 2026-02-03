import os
import random
import time
from dataclasses import dataclass
from typing import List, Optional

import streamlit as st
from supabase import create_client, Client


# =========================
# CONFIG
# =========================
TOTAL_QUESTIONS = 15

# Difficulty tuning
DIVISOR_MIN = 2
DIVISOR_MAX = 12
QUOTIENT_MIN = 2
QUOTIENT_MAX = 50

# Score edge-case handling (avoid division by zero)
ACCURACY_EPSILON = 0.01  # 1% floor


# =========================
# SUPABASE HELPERS
# =========================
def get_supabase_client() -> Optional[Client]:
    """
    Loads Supabase credentials from Streamlit secrets or environment variables.
    If missing, returns None (app still works, just no saving/percentile).
    """
    url = None
    key = None

    if hasattr(st, "secrets"):
        url = st.secrets.get("SUPABASE_URL", None)
        key = st.secrets.get("SUPABASE_SERVICE_ROLE_KEY", None)

    url = url or os.getenv("SUPABASE_URL")
    key = key or os.getenv("SUPABASE_SERVICE_ROLE_KEY")

    if not url or not key:
        return None

    return create_client(url, key)


def insert_score(sb: Client, score: float, accuracy: float, time_taken: float) -> bool:
    try:
        sb.table("division_scores").insert(
            {"score": float(score), "accuracy": float(accuracy), "time_taken": float(time_taken)}
        ).execute()
        return True
    except Exception:
        return False


def fetch_all_scores(sb: Client, limit: int = 5000) -> List[float]:
    try:
        resp = sb.table("division_scores").select("score").limit(limit).execute()
        data = resp.data or []
        return [float(row["score"]) for row in data if row.get("score") is not None]
    except Exception:
        return []


def percentile_lower_is_better(all_scores: List[float], user_score: float) -> float:
    """
    Higher percentile = better performance.
    Lower score is better, so:
      percentile = 100 * (count(scores >= user_score) / N)
    Best score -> ~100th percentile
    """
    if not all_scores:
        return 100.0
    n = len(all_scores)
    count_ge = sum(1 for s in all_scores if s >= user_score)
    return 100.0 * (count_ge / n)


# =========================
# GAME LOGIC
# =========================
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
    Creates dividend = divisor*quotient + remainder
    Ensures remainder < divisor and divisor != 0.
    """
    divisor = random.randint(DIVISOR_MIN, DIVISOR_MAX)
    quotient = random.randint(QUOTIENT_MIN, QUOTIENT_MAX)
    remainder = random.randint(0, divisor - 1)
    dividend = divisor * quotient + remainder
    return DivisionProblem(dividend, divisor, quotient, remainder)


def generate_questions(n: int) -> List[DivisionProblem]:
    return [generate_problem() for _ in range(n)]


def compute_accuracy(correct: int, total: int) -> float:
    return correct / total if total > 0 else 0.0


def compute_score(accuracy: float, time_taken: float) -> float:
    a = max(accuracy, ACCURACY_EPSILON)
    return ((1.0 / a) ** 2) * time_taken


# =========================
# SESSION STATE
# =========================
def init_state() -> None:
    defaults = {
        "mode": "home",  # home | playing | results
        "questions": [],
        "q_index": 0,
        "correct_count": 0,
        "answered_count": 0,
        "start_time": None,
        "elapsed_time": 0.0,
        "last_toast": None,  # store last toast msg to display once
        "final_score": None,
        "final_accuracy": None,
        "final_percentile": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def start_new_game() -> None:
    st.session_state.questions = generate_questions(TOTAL_QUESTIONS)
    st.session_state.q_index = 0
    st.session_state.correct_count = 0
    st.session_state.answered_count = 0
    st.session_state.start_time = time.perf_counter()
    st.session_state.elapsed_time = 0.0
    st.session_state.last_toast = None
    st.session_state.final_score = None
    st.session_state.final_accuracy = None
    st.session_state.final_percentile = None
    st.session_state.mode = "playing"


def update_timer() -> None:
    if st.session_state.mode == "playing" and st.session_state.start_time is not None:
        st.session_state.elapsed_time = time.perf_counter() - st.session_state.start_time


def handle_submit(user_q: int, user_r: int) -> None:
    """
    Validates the user's answer, updates stats, shows toast feedback,
    and auto-advances to the next question (Enter-friendly).
    """
    idx = st.session_state.q_index
    questions: List[DivisionProblem] = st.session_state.questions
    current = questions[idx]

    # Basic validation: remainder must be < divisor
    if user_r >= current.divisor:
        st.session_state.last_toast = (
            "Remainder must be less than the divisor "
            f"({current.divisor}). Try again 💀"
        )
        # Don't count as answered, don't advance
        return

    st.session_state.answered_count += 1

    is_correct = (user_q == current.quotient) and (user_r == current.remainder)
    if is_correct:
        st.session_state.correct_count += 1
        st.session_state.last_toast = "Correct ✅🔥"
    else:
        st.session_state.last_toast = (
            f"Incorrect 😭 Correct was {current.quotient} r {current.remainder}"
        )

    # Advance immediately
    st.session_state.q_index += 1

    # If finished, stop timer & go results
    if st.session_state.q_index >= TOTAL_QUESTIONS:
        st.session_state.elapsed_time = time.perf_counter() - st.session_state.start_time
        st.session_state.mode = "results"


# =========================
# UI
# =========================
st.set_page_config(page_title="Division Practice", page_icon="➗", layout="centered")
init_state()
update_timer()

st.title("➗ Division Practice (Quotient + Remainder)")
st.caption("Press **Enter** to submit. 15 questions. Lower score = better 🙏🔥")

sb = get_supabase_client()
if sb is None:
    st.info(
        "Supabase not configured — app works, but won’t save scores / show percentile.\n\n"
        "Set `SUPABASE_URL` and `SUPABASE_SERVICE_ROLE_KEY` in `.streamlit/secrets.toml` or env vars."
    )

# Show toast once per rerun (if exists), then clear it
if st.session_state.last_toast:
    st.toast(st.session_state.last_toast)
    st.session_state.last_toast = None


# -------------------------
# HOME
# -----------------
