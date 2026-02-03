import os
import random
import time
from dataclasses import dataclass
from typing import List, Optional

import streamlit as st

# Supabase is optional; app still runs without it
try:
    from supabase import create_client, Client
except Exception:
    create_client = None
    Client = None


# =========================
# CONFIG
# =========================
TOTAL_QUESTIONS = 15

DIVISOR_MIN = 2
DIVISOR_MAX = 12

QUOTIENT_MIN = 2
QUOTIENT_MAX = 12

ACCURACY_EPSILON = 0.01  # floor to avoid division-by-zero


# =========================
# DATA MODEL
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


# =========================
# SUPABASE HELPERS
# =========================
def get_supabase_client() -> Optional["Client"]:
    """
    Returns supabase client if configured + library installed, else None.
    """
    if create_client is None:
        return None

    url = None
    key = None

    # Streamlit secrets
    if hasattr(st, "secrets"):
        url = st.secrets.get("SUPABASE_URL", None)
        key = st.secrets.get("SUPABASE_SERVICE_ROLE_KEY", None)

    # Env vars fallback
    url = url or os.getenv("SUPABASE_URL")
    key = key or os.getenv("SUPABASE_SERVICE_ROLE_KEY")

    if not url or not key:
        return None

    return create_client(url, key)


def insert_score(sb: "Client", score: float, accuracy: float, time_taken: float) -> bool:
    try:
        sb.table("division_scores").insert(
            {"score": float(score), "accuracy": float(accuracy), "time_taken": float(time_taken)}
        ).execute()
        return True
    except Exception:
        return False


def fetch_all_scores(sb: "Client", limit: int = 5000) -> List[float]:
    try:
        resp = sb.table("division_scores").select("score").limit(limit).execute()
        data = resp.data or []
        return [float(row["score"]) for row in data if row.get("score") is not None]
    except Exception:
        return []


def percentile_lower_is_better(all_scores: List[float], user_score: float) -> float:
    """
    Higher percentile = better (because lower score is better).
    percentile = 100 * (count(scores >= user_score) / N)
    """
    if not all_scores:
        return 100.0
    n = len(all_scores)
    count_ge = sum(1 for s in all_scores if s >= user_score)
    return 100.0 * (count_ge / n)


# =========================
# GAME LOGIC
# =========================
def generate_problem() -> DivisionProblem:
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
def hard_reset_state() -> None:
    """
    Nukes the important keys so landing page always comes back.
    """
    keys = [
        "mode",
        "questions",
        "q_index",
        "correct_count",
        "answered_count",
        "start_time",
        "elapsed_time",
        "last_toast",
        "final_score",
        "final_accuracy",
        "final_percentile",
    ]
    for k in keys:
        if k in st.session_state:
            del st.session_state[k]


def init_state() -> None:
    defaults = {
        "mode": "home",          # home | playing | results
        "questions": [],
        "q_index": 0,
        "correct_count": 0,
        "answered_count": 0,
        "start_time": None,
        "elapsed_time": 0.0,
        "last_toast": None,
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
    st.session_state.final_score = None
    st.session_state.final_accuracy = None
    st.session_state.final_percentile = None
    st.session_state.mode = "playing"


def update_timer() -> None:
    if st.session_state.mode == "playing" and st.session_state.start_time is not None:
        st.session_state.elapsed_time = time.perf_counter() - st.session_state.start_time


def recover_if_broken() -> None:
    """
    If state is inconsistent, force home screen.
    Prevents blank/empty screens after refresh.
    """
    if st.session_state.mode not in {"home", "playing", "results"}:
        st.session_state.mode = "home"

    if st.session_state.mode == "playing":
        # If user refreshed and we lost questions, go home
        if not st.session_state.questions or st.session_state.q_index >= TOTAL_QUESTIONS:
            st.session_state.mode = "home"


def handle_submit(user_q_raw, user_r_raw) -> None:
    idx = st.session_state.q_index
    questions: List[DivisionProblem] = st.session_state.questions

    # Guard
    if idx < 0 or idx >= len(questions):
        st.session_state.mode = "home"
        return

    current = questions[idx]

    # Force clean ints (handles float-like values safely)
    try:
        user_q = int(user_q_raw)
        user_r = int(user_r_raw)
    except (TypeError, ValueError):
        st.session_state.last_toast = "Please enter valid whole numbers."
        return

    # remainder constraint
    if user_r < 0:
        st.session_state.last_toast = "Remainder can’t be negative."
        return

    if user_r >= current.divisor:
        st.session_state.last_toast = f"Remainder must be < {current.divisor}. Try again"
        return

    st.session_state.answered_count += 1

    is_correct = (user_q == current.quotient) and (user_r == current.remainder)

    if is_correct:
        st.session_state.correct_count += 1
        st.session_state.last_toast = "Correct"
    else:
        st.session_state.last_toast = f"Incorrect - Correct was {current.quotient} r {current.remainder}"

    # advance
    st.session_state.q_index += 1

    if st.session_state.q_index >= TOTAL_QUESTIONS:
        st.session_state.elapsed_time = time.perf_counter() - st.session_state.start_time
        st.session_state.mode = "results"

# =========================
# UI
# =========================
st.set_page_config(page_title="Division Practice", page_icon="➗", layout="centered")

init_state()
recover_if_broken()
update_timer()

# Sidebar controls (always visible)
with st.sidebar:
    st.markdown("### Controls")
    if st.button("Reset app (fix blank page)", use_container_width=True):
        hard_reset_state()
        st.rerun()

    st.markdown("---")
    st.caption("Debug")
    st.write("mode:", st.session_state.mode)
    st.write("q_index:", st.session_state.q_index)
    st.write("questions:", len(st.session_state.questions))

st.title("➗ Division Practice (Quotient + Remainder)")
st.caption("Press **Enter** to submit. Lower score = better")

# Toast once per rerun
if st.session_state.last_toast:
    st.toast(st.session_state.last_toast)
    st.session_state.last_toast = None

# Supabase
sb = get_supabase_client()
if sb is None:
    st.info(
        "Supabase not configured (or supabase library missing). Scores won’t save / percentile won’t show.\n\n"
        "To enable it:\n"
        "- `pip install supabase`\n"
        "- add `SUPABASE_URL` and `SUPABASE_SERVICE_ROLE_KEY` to `.streamlit/secrets.toml`"
    )

st.divider()


# -------------------------
# HOME (landing page)
# -------------------------
if st.session_state.mode == "home":
    st.subheader("Landing page(this should never be blank now)")
    st.write(
        "Rules:\n"
        "- 15 questions\n"
        "- Enter **quotient** and **remainder**\n"
        "- Press **Enter** to submit → auto next question\n"
        "- Score = **(1/accuracy)² × time** (lower is better)\n"
    )

    c1, c2 = st.columns(2)
    with c1:
        if st.button("Start game", use_container_width=True):
            start_new_game()
            st.rerun()
    with c2:
        st.markdown("**Difficulty**")
        st.write(f"Divisor: {DIVISOR_MIN}–{DIVISOR_MAX}")
        st.write(f"Quotient: {QUOTIENT_MIN}–{QUOTIENT_MAX}")

    st.markdown("#### Scoring (cause → effect)")
    st.write(
        "- Accuracy ↑ → (1/accuracy)² → score \n"
        "- Time ↑ → score ↑ \n"
        "- Accuracy = 0 → clamped to 1% so your score doesn’t explode to infinity\n"
    )


# -------------------------
# PLAYING
# -------------------------
elif st.session_state.mode == "playing":
    questions: List[DivisionProblem] = st.session_state.questions
    idx = st.session_state.q_index
    current = questions[idx]

    a, b, c = st.columns(3)
    a.metric("Question", f"{idx + 1}/{TOTAL_QUESTIONS}")
    b.metric("Correct", str(st.session_state.correct_count))
    c.metric("Time", f"{st.session_state.elapsed_time:.1f}s")

    st.markdown(f"## {current.text}")

with st.form(key=f"answer_form_{idx}", clear_on_submit=True):
    col1, col2 = st.columns(2)

    with col1:
        user_q = st.number_input(
            "Quotient",
            min_value=0,
            step=1,
            value=0,
            key=f"q_input_{idx}",   # IMPORTANT: unique per question
        )

    with col2:
        user_r = st.number_input(
            "Remainder",
            min_value=0,
            step=1,
            value=0,
            key=f"r_input_{idx}",   # IMPORTANT: unique per question
        )

    submitted = st.form_submit_button("Submit (Enter)")

    if submitted:
        handle_submit(user_q, user_r)
        st.rerun()


# -------------------------
# RESULTS
# -------------------------
elif st.session_state.mode == "results":
    time_taken = float(st.session_state.elapsed_time)
    correct = int(st.session_state.correct_count)
    total = TOTAL_QUESTIONS

    accuracy = compute_accuracy(correct, total)
    score = compute_score(accuracy, time_taken)

    st.subheader("Results")
    r1, r2, r3 = st.columns(3)
    r1.metric("Accuracy", f"{accuracy*100:.1f}%")
    r2.metric("Time", f"{time_taken:.2f}s")
    r3.metric("Score (lower is better)", f"{score:.4f}")

    st.divider()

    percentile = None
    if sb is not None:
        saved = insert_score(sb, score=score, accuracy=accuracy, time_taken=time_taken)
        all_scores = fetch_all_scores(sb)
        percentile = percentile_lower_is_better(all_scores, score)

        if saved:
            st.success("Saved to Supabase")
        else:
            st.warning("Couldn’t save to Supabase (RLS/policies/credentials).")

    if percentile is not None:
        st.markdown(f"### Percentile: **{percentile:.1f}th**")
        st.caption("Higher percentile = better (because lower score is better).")

    c1, c2 = st.columns(2)
    with c1:
        if st.button("Play again", use_container_width=True):
            start_new_game()
            st.rerun()
    with c2:
        if st.button("Home", use_container_width=True):
            st.session_state.mode = "home"
            st.rerun()
