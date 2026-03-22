"""
club/agents/planner.py

Creates personalized study plans and daily morning briefings.
Builds day-by-day schedules that prioritize weak areas and adapt
around exam dates, then delivers a motivating daily study brief.

Example:
    >>> from agents.planner import build_schedule, generate_briefing
    >>> schedule = build_schedule(["DSA", "OS"], {"DSA": "2026-04-15"}, [])
    >>> print(schedule["days"][0]["morning_topic"])

Author: CLUB Project
License: MIT
"""

import os
import time
from datetime import datetime, timedelta
from typing import Callable

import ollama

from core.constants import (
    OLLAMA_MODEL,
    OLLAMA_RETRY_ATTEMPTS,
    OLLAMA_RETRY_DELAY,
)

# ── Constants ─────────────────────────────────────────────

MAX_RETRIES = OLLAMA_RETRY_ATTEMPTS
RETRY_DELAY_SECONDS = OLLAMA_RETRY_DELAY

BRIEFING_OUTPUT_PATH = os.path.join(
    "folder", "output", "briefing_today.md"
)

# Default hours per study session block
MORNING_HOURS = 2
AFTERNOON_HOURS = 2
EVENING_HOURS = 1            # lighter evening review session
TOTAL_DAILY_HOURS = MORNING_HOURS + AFTERNOON_HOURS + EVENING_HOURS

# ── System Prompts ────────────────────────────────────────

BRIEFING_SYSTEM_PROMPT = """You are CLUB, a study assistant.
Generate today's morning study briefing in markdown format.

The briefing must include these sections:
## 🌅 Good Morning!
A motivating one-liner to start the day.

## 📚 Today's Focus
What the student should study today, based on their profile.

## ⚠️ Weak Areas to Review
Topics needing extra attention (from their weak areas list).

## 🧠 3 Quick Practice Questions
Three questions the student should try today.
Number them 1-3.

## 💡 Did You Know?
One interesting fact related to today's study topics.

Rules:
- Keep it concise and energizing.
- Use the student's actual subjects and weak areas.
- Questions should be exam-relevant, not trivia.
- The fact should be genuinely interesting and memorable."""


# ── Core Functions ────────────────────────────────────────

def build_schedule(
    subjects: list[str],
    exam_dates: dict[str, str],
    weak_areas: list[str],
) -> dict:
    """
    Builds a day-by-day study schedule until the nearest exam.

    Allocates morning, afternoon, and evening sessions across
    subjects. Weak areas get more frequent slots. Days closer
    to an exam shift focus to that exam's subject.

    Args:
        subjects:   List of subject names (e.g. ["DSA", "OS"]).
        exam_dates: Dict mapping subject to exam date string
                    in "YYYY-MM-DD" format (e.g. {"DSA": "2026-04-15"}).
        weak_areas: List of weak topic names that should get
                    extra study time.

    Returns:
        A dict with:
            - start_date: str, today's date
            - end_date:   str, last exam date
            - total_days: int
            - days: list[dict], each day containing:
                - date:            str
                - day_number:      int
                - morning_topic:   str
                - afternoon_topic: str
                - evening_review:  str
                - estimated_hours: int

    Example:
        >>> schedule = build_schedule(
        ...     ["DSA", "OS"],
        ...     {"DSA": "2026-04-15", "OS": "2026-04-20"},
        ...     ["recursion", "deadlocks"]
        ... )
        >>> print(schedule["days"][0]["morning_topic"])
    """
    if not subjects:
        print("CLUB Planner: no subjects provided")
        return _empty_schedule()

    today = datetime.now().date()

    # Parse exam dates and find the last exam
    parsed_exams = _parse_exam_dates(exam_dates)
    last_exam_date = _find_last_exam_date(parsed_exams, today)

    total_days = (last_exam_date - today).days + 1
    if total_days <= 0:
        print("CLUB Planner: all exams have already passed")
        return _empty_schedule()

    # Build the subject priority queue — weak areas get
    # higher weight so they appear more often in the schedule
    subject_weights = _build_subject_weights(
        subjects, weak_areas, parsed_exams, today
    )

    # Generate day-by-day schedule
    daily_plans = []
    for day_offset in range(total_days):
        current_date = today + timedelta(days=day_offset)

        # Subjects with upcoming exams get priority as the
        # exam date approaches
        day_subjects = _pick_day_subjects(
            subjects, subject_weights, parsed_exams,
            current_date, day_offset
        )

        # Evening review focuses on weak areas when available
        evening_topic = _pick_evening_review(
            weak_areas, subjects, day_offset
        )

        daily_plans.append({
            "date": current_date.isoformat(),
            "day_number": day_offset + 1,
            "morning_topic": day_subjects[0],
            "afternoon_topic": day_subjects[1],
            "evening_review": evening_topic,
            "estimated_hours": TOTAL_DAILY_HOURS,
        })

    return {
        "start_date": today.isoformat(),
        "end_date": last_exam_date.isoformat(),
        "total_days": total_days,
        "days": daily_plans,
    }


def generate_briefing(
    profile: dict,
    memory_search_fn: Callable[[str], list[str]],
) -> str:
    """
    Generates today's study briefing in markdown format.

    Combines the student's profile, weak areas, and any
    relevant content from memory to create a personalized
    morning briefing via the local LLM.

    Args:
        profile:          Student profile dict from
                          knowmyschool.profile.load_profile().
        memory_search_fn: A callable that takes a query string
                          and returns a list of relevant text
                          chunks from memory. Typically
                          core.memory.search.

    Returns:
        Markdown-formatted study briefing string.
        Returns a fallback briefing on failure.

    Example:
        >>> from knowmyschool.profile import load_profile
        >>> from core.memory import search
        >>> briefing = generate_briefing(load_profile(), search)
        >>> print(briefing)
    """
    # Gather context for the LLM
    weak_areas = profile.get("weak_areas", [])
    subjects = list(profile.get("exam_dates", {}).keys())

    # If no exam dates, use a generic subject list
    if not subjects:
        subjects = [profile.get("program", "General")]

    # Search memory for content related to weak areas
    memory_context = ""
    if weak_areas and memory_search_fn:
        search_query = f"key concepts: {', '.join(weak_areas)}"
        try:
            memory_results = memory_search_fn(search_query)
            if memory_results:
                memory_context = (
                    "\n\nRelevant study material from memory:\n"
                    + "\n---\n".join(memory_results[:3])
                )
        except Exception as error:
            # Memory search failing should not block briefing
            print(
                f"CLUB Planner: memory search failed — {error}"
            )

    # Build user prompt with all context
    user_prompt = _build_briefing_context(
        profile, subjects, weak_areas, memory_context
    )

    raw_response = _call_ollama(
        BRIEFING_SYSTEM_PROMPT, user_prompt
    )

    if not raw_response:
        return _fallback_briefing(subjects, weak_areas)

    return raw_response


def schedule_daily_briefing(
    briefing_fn: Callable[[], str],
) -> None:
    """
    Schedules a daily briefing to run every morning at 8 AM.

    Uses APScheduler to trigger briefing_fn daily. The output
    is saved to folder/output/briefing_today.md so the student
    can read it when they start studying.

    Args:
        briefing_fn: A zero-argument callable that returns
                     the briefing markdown string. Typically
                     a partial of generate_briefing with
                     profile and memory_search_fn pre-bound.

    Example:
        >>> from functools import partial
        >>> from knowmyschool.profile import load_profile
        >>> from core.memory import search
        >>> fn = partial(generate_briefing, load_profile(), search)
        >>> schedule_daily_briefing(fn)
    """
    try:
        from apscheduler.schedulers.background import (
            BackgroundScheduler,
        )
    except ImportError:
        print(
            "CLUB Planner: APScheduler not installed. "
            "Install with: pip install APScheduler"
        )
        return

    def _run_and_save() -> None:
        """Runs the briefing function and saves output to file."""
        print("CLUB Planner: generating daily briefing...")
        briefing_text = briefing_fn()

        if briefing_text:
            _save_briefing(briefing_text)
            print(
                f"CLUB Planner: briefing saved to "
                f"{BRIEFING_OUTPUT_PATH}"
            )

    scheduler = BackgroundScheduler()

    # Run at 8:00 AM every day
    scheduler.add_job(
        _run_and_save,
        trigger="cron",
        hour=8,
        minute=0,
        id="daily_briefing",
    )

    scheduler.start()
    print(
        "CLUB Planner: daily briefing scheduled for 8:00 AM. "
        "Scheduler is running in the background."
    )


# ── LLM Interface ────────────────────────────────────────

def _call_ollama(system_prompt: str, user_text: str) -> str:
    """
    Sends a prompt to the local ollama LLM with retry logic.

    Args:
        system_prompt: Instructions for the LLM.
        user_text:     Content to process.

    Returns:
        Raw response text, or empty string on failure.
    """
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = ollama.chat(
                model=OLLAMA_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt,
                    },
                    {
                        "role": "user",
                        "content": user_text,
                    },
                ],
            )
            return response["message"]["content"].strip()

        except ConnectionError:
            print(
                "CLUB Planner: ollama is not running. "
                "Start it with: ollama serve"
            )
            return ""

        except Exception as error:
            is_last_attempt = attempt == MAX_RETRIES
            if is_last_attempt:
                print(
                    f"CLUB Planner: failed after "
                    f"{MAX_RETRIES} attempts — {error}"
                )
                return ""

            print(
                f"CLUB Planner: attempt {attempt} failed, "
                f"retrying in {RETRY_DELAY_SECONDS}s..."
            )
            time.sleep(RETRY_DELAY_SECONDS)

    return ""


# ── Helpers ───────────────────────────────────────────────

def _parse_exam_dates(
    exam_dates: dict[str, str],
) -> dict[str, datetime]:
    """
    Converts exam date strings to datetime.date objects.

    Args:
        exam_dates: Dict of subject → "YYYY-MM-DD" strings.

    Returns:
        Dict of subject → datetime.date, skipping invalid dates.
    """
    parsed = {}
    for subject, date_string in exam_dates.items():
        try:
            parsed[subject] = datetime.strptime(
                date_string, "%Y-%m-%d"
            ).date()
        except (ValueError, TypeError):
            print(
                f"CLUB Planner: invalid date '{date_string}' "
                f"for {subject}, skipping"
            )
    return parsed


def _find_last_exam_date(
    parsed_exams: dict,
    today: datetime,
) -> datetime:
    """
    Finds the latest exam date, or defaults to 30 days from now.

    Args:
        parsed_exams: Dict of subject → datetime.date.
        today:        Today's date.

    Returns:
        The last exam date, or today + 30 days if no exams.
    """
    future_dates = [
        date for date in parsed_exams.values()
        if date >= today
    ]

    if future_dates:
        return max(future_dates)

    # No exam dates set — default to a 30-day study window
    default_window_days = 30
    return today + timedelta(days=default_window_days)


def _build_subject_weights(
    subjects: list[str],
    weak_areas: list[str],
    parsed_exams: dict,
    today: datetime,
) -> dict[str, float]:
    """
    Assigns priority weights to subjects.

    Subjects with upcoming exams and those related to weak
    areas get higher weights, meaning they appear more
    frequently in the schedule.

    Args:
        subjects:     All subject names.
        weak_areas:   Weak topic names.
        parsed_exams: Subject → exam date mapping.
        today:        Today's date.

    Returns:
        Dict of subject → weight (higher = more priority).
    """
    weights = {}

    for subject in subjects:
        weight = 1.0

        # Subjects with closer exams get higher weight
        if subject in parsed_exams:
            days_until = (parsed_exams[subject] - today).days
            if days_until > 0:
                # Closer exams → higher urgency
                # 7 days away = weight 4.3, 30 days = weight 2.0
                weight += 30.0 / days_until

        # Subjects related to weak areas get a boost
        subject_lower = subject.lower()
        for area in weak_areas:
            if subject_lower in area.lower():
                # Each matching weak area adds priority
                weight += 0.5

        weights[subject] = weight

    return weights


def _pick_day_subjects(
    subjects: list[str],
    weights: dict[str, float],
    parsed_exams: dict,
    current_date: datetime,
    day_offset: int,
) -> tuple[str, str]:
    """
    Picks morning and afternoon subjects for a specific day.

    Uses a weighted rotation so high-priority subjects appear
    more often. On the day before an exam, both sessions
    focus on that exam's subject.

    Args:
        subjects:     All subject names.
        weights:      Subject priority weights.
        parsed_exams: Subject → exam date mapping.
        current_date: The date being scheduled.
        day_offset:   Days from today (0 = today).

    Returns:
        Tuple of (morning_subject, afternoon_subject).
    """
    # Check if any exam is tomorrow — if so, cram that subject
    for subject, exam_date in parsed_exams.items():
        days_until_exam = (exam_date - current_date).days
        if days_until_exam == 1:
            return (subject, subject)

    if not subjects:
        return ("General review", "General review")

    # Sort subjects by weight (descending) for priority order
    sorted_subjects = sorted(
        subjects,
        key=lambda subj: weights.get(subj, 1.0),
        reverse=True,
    )

    # Rotate through subjects using day_offset so the schedule
    # doesn't assign the same subject every single day
    morning_index = day_offset % len(sorted_subjects)
    afternoon_index = (day_offset + 1) % len(sorted_subjects)

    # Avoid same subject for both if possible
    if (len(sorted_subjects) > 1
            and morning_index == afternoon_index):
        afternoon_index = (afternoon_index + 1) % len(
            sorted_subjects
        )

    return (
        sorted_subjects[morning_index],
        sorted_subjects[afternoon_index],
    )


def _pick_evening_review(
    weak_areas: list[str],
    subjects: list[str],
    day_offset: int,
) -> str:
    """
    Picks an evening review topic, prioritizing weak areas.

    Args:
        weak_areas: List of weak topics.
        subjects:   Fallback subject list.
        day_offset: Days from today for rotation.

    Returns:
        A topic string for the evening review session.
    """
    if weak_areas:
        # Rotate through weak areas across days
        index = day_offset % len(weak_areas)
        return f"Review: {weak_areas[index]}"

    if subjects:
        index = day_offset % len(subjects)
        return f"Review: {subjects[index]}"

    return "General revision"


def _build_briefing_context(
    profile: dict,
    subjects: list[str],
    weak_areas: list[str],
    memory_context: str,
) -> str:
    """
    Builds the user prompt for the briefing LLM call.

    Combines profile info, subjects, weak areas, and any
    relevant content from memory into a structured prompt.

    Args:
        profile:        Student profile dict.
        subjects:       Subject names for context.
        weak_areas:     Topics needing extra attention.
        memory_context: Relevant text from memory search.

    Returns:
        Formatted user prompt string.
    """
    today_str = datetime.now().strftime("%A, %B %d, %Y")

    context = (
        f"Today is {today_str}.\n\n"
        f"Student: {profile.get('program', 'Student')} "
        f"Semester {profile.get('semester', '?')}\n"
        f"School: {profile.get('school', 'Unknown')}\n"
        f"Subjects: {', '.join(subjects)}\n"
    )

    if weak_areas:
        context += f"Weak areas: {', '.join(weak_areas)}\n"

    exam_dates = profile.get("exam_dates", {})
    if exam_dates:
        context += "Upcoming exams:\n"
        for subject, date in exam_dates.items():
            context += f"  - {subject}: {date}\n"

    if memory_context:
        context += memory_context

    return context


def _save_briefing(briefing_text: str) -> None:
    """
    Saves the briefing markdown to the output folder.

    Creates the output directory if it doesn't exist.

    Args:
        briefing_text: The markdown briefing content.
    """
    output_dir = os.path.dirname(BRIEFING_OUTPUT_PATH)
    os.makedirs(output_dir, exist_ok=True)

    try:
        with open(
            BRIEFING_OUTPUT_PATH, "w", encoding="utf-8"
        ) as file:
            file.write(briefing_text)
    except OSError as error:
        print(
            f"CLUB Planner: failed to save briefing — {error}"
        )


def _fallback_briefing(
    subjects: list[str],
    weak_areas: list[str],
) -> str:
    """
    Returns a simple static briefing when the LLM is unavailable.

    Args:
        subjects:   Subject names.
        weak_areas: Weak topic names.

    Returns:
        A basic markdown briefing string.
    """
    today_str = datetime.now().strftime("%A, %B %d, %Y")
    weak_section = ""

    if weak_areas:
        weak_items = "\n".join(
            f"- {area}" for area in weak_areas
        )
        weak_section = (
            f"\n## ⚠️ Weak Areas to Review\n{weak_items}\n"
        )

    subjects_text = ", ".join(subjects) if subjects else "your subjects"

    return (
        f"## 🌅 Good Morning!\n"
        f"Today is {today_str}. Let's make it count!\n\n"
        f"## 📚 Today's Focus\n"
        f"Focus on: {subjects_text}\n"
        f"{weak_section}\n"
        f"## 💡 Tip\n"
        f"CLUB could not reach ollama for a full briefing. "
        f"Start it with: `ollama serve`\n"
    )


def _empty_schedule() -> dict:
    """
    Returns an empty schedule as a fallback.

    Returns:
        Dict with schedule structure but no days.
    """
    return {
        "start_date": datetime.now().date().isoformat(),
        "end_date": datetime.now().date().isoformat(),
        "total_days": 0,
        "days": [],
    }


# ── Entry Point (Test) ────────────────────────────────────

if __name__ == "__main__":
    print("CLUB Planner: running test...\n")

    # ── Test 1: Build schedule ────────────────────────────
    print("── Test 1: Build study schedule ──────────\n")

    test_subjects = ["DSA", "OS", "DBMS"]
    test_exam_dates = {
        "DSA": "2026-04-15",
        "OS": "2026-04-20",
        "DBMS": "2026-04-25",
    }
    test_weak_areas = ["recursion", "deadlocks", "normalization"]

    schedule = build_schedule(
        test_subjects, test_exam_dates, test_weak_areas
    )

    print(
        f"  Schedule: {schedule['start_date']} → "
        f"{schedule['end_date']} ({schedule['total_days']} days)"
    )

    # Show first 5 days as preview
    preview_days = min(5, len(schedule["days"]))
    for day in schedule["days"][:preview_days]:
        print(
            f"  Day {day['day_number']} ({day['date']}): "
            f"AM={day['morning_topic']} | "
            f"PM={day['afternoon_topic']} | "
            f"EVE={day['evening_review']} | "
            f"{day['estimated_hours']}h"
        )

    if len(schedule["days"]) > preview_days:
        print(f"  ... and {len(schedule['days']) - preview_days} more days")

    # ── Test 2: Generate briefing ─────────────────────────
    print("\n── Test 2: Generate morning briefing ─────\n")

    test_profile = {
        "school": "Tribhuvan University",
        "program": "BIT",
        "semester": 3,
        "exam_dates": test_exam_dates,
        "weak_areas": test_weak_areas,
    }

    # Mock memory search that returns empty (no stored docs)
    def mock_memory_search(query: str) -> list[str]:
        """Returns empty results for testing."""
        return []

    briefing = generate_briefing(test_profile, mock_memory_search)
    print(briefing)

    # Save the test briefing
    _save_briefing(briefing)
    print(
        f"\n  Briefing saved to {BRIEFING_OUTPUT_PATH}"
    )

    print("\nCLUB Planner: test complete.")
