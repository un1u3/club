"""
club/interface/app.py

Chainlit chat interface for the CLUB study assistant.
Provides the web-based UI where students interact with agents,
handles first-run setup, and manages the file watcher.

Example:
    >>> # Run from project root:
    >>> # chainlit run interface/app.py

Author: CLUB Project
License: MIT
"""

import os
import sys
import threading

import chainlit as cl

# ── Path Setup ────────────────────────────────────────────
# Chainlit runs from the file's directory by default.
# We need the project root on sys.path so imports like
# "from core.coordinator import chat" resolve correctly.
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from core.coordinator import chat as coordinator_chat
from core.memory import add_document, search as memory_search
from core.watcher import start_watching, index_existing_files
from knowmyschool.profile import (
    load_profile,
    save_profile,
    DEFAULT_PROFILE,
)

# ── Constants ─────────────────────────────────────────────

# Timeout for setup questions (seconds) — generous because
# students may need to think or look up exam dates
SETUP_QUESTION_TIMEOUT = 120

WELCOME_MESSAGE = """
## 🎓 Welcome to CLUB!

**Continuous Learning Understanding Bots** — your local AI study assistant.

📂 **CLUB is watching your `folder/` directory.**
Drop your notes, PDFs, images, or past papers there and I'll process them automatically.

💬 **What can I do?**
- **Summarize** → "Summarize binary search trees"
- **Quiz** → "Quiz me on sorting algorithms"
- **Solve** → "Solve: explain BFS step by step"
- **Plan** → "Make me a study schedule"
- **Briefing** → "Give me today's morning briefing"

🇳🇵 You can type in **English or नेपाली** — I understand both!

---
*Type anything to get started.*
"""

SETUP_INTRO = """
## 👋 First time here! Let's set up your profile.

I need a few details to personalize your experience.
This only takes a minute.
"""


# ── Chat Lifecycle ────────────────────────────────────────

@cl.on_chat_start
async def on_chat_start():
    """
    Runs when a new chat session begins.

    Checks if the student profile needs setup (first run),
    loads the profile, starts the file watcher in a background
    thread, and shows the welcome message.
    """
    # Initialize conversation history in the session
    cl.user_session.set("history", [])

    # Check if this is a first-run (profile not configured)
    profile = load_profile()
    is_first_run = _is_default_profile(profile)

    if is_first_run:
        await _run_setup_wizard()
        # Reload after setup
        profile = load_profile()

    # Start the file watcher in a background thread
    start_watching(blocking=False)

    # Index any existing files that were added while CLUB
    # was offline
    indexed_count = index_existing_files()

    # Show the welcome message
    await cl.Message(content=WELCOME_MESSAGE).send()

    # Show indexing status
    if indexed_count > 0:
        await cl.Message(
            content=(
                f"📄 **CLUB has indexed {indexed_count} "
                f"document{'s' if indexed_count != 1 else ''}** "
                "from your study folders."
            )
        ).send()

    # Show current profile summary
    profile_summary = _format_profile_summary(profile)
    await cl.Message(content=profile_summary).send()


@cl.on_message
async def on_message(message: cl.Message):
    """
    Handles incoming user messages.

    Passes the message to the coordinator for routing to the
    correct agent, then sends the response back to the chat.

    Args:
        message: The Chainlit message object from the user.
    """
    user_text = message.content

    # Retrieve conversation history from the session
    history = cl.user_session.get("history", [])

    # Show a thinking indicator while the agent works
    thinking_message = cl.Message(content="🤔 Thinking...")
    await thinking_message.send()

    # Route to the correct agent via the coordinator
    try:
        response = coordinator_chat(
            user_message=user_text, history=history
        )
    except Exception as error:
        response = (
            f"⚠️ Something went wrong: {error}\n\n"
            "Make sure ollama is running: `ollama serve`"
        )
        print(f"CLUB App: chat error — {error}")

    # Update the thinking message with the actual response
    thinking_message.content = response
    await thinking_message.update()

    # Append to conversation history
    history.append(("user", user_text))
    history.append(("assistant", response))
    cl.user_session.set("history", history)


# ── First-Run Setup Wizard ────────────────────────────────

async def _run_setup_wizard():
    """
    Asks the student 4 setup questions on first run and
    saves their answers to config.yaml via the profile module.

    Questions:
        1. University name
        2. Program and semester
        3. Next exam subject and date
        4. Known weak topics
    """
    await cl.Message(content=SETUP_INTRO).send()

    # Question 1: University
    university_response = await cl.AskUserMessage(
        content="1️⃣ **What is your university/school name?**",
        timeout=SETUP_QUESTION_TIMEOUT,
    ).send()
    university_name = (
        university_response["output"].strip()
        if university_response
        else DEFAULT_PROFILE["school"]
    )

    # Question 2: Program and semester
    program_response = await cl.AskUserMessage(
        content=(
            "2️⃣ **What program and semester are you in?**\n"
            "e.g. `BIT Semester 3` or `BCA 4th sem`"
        ),
        timeout=SETUP_QUESTION_TIMEOUT,
    ).send()
    program_text = (
        program_response["output"].strip()
        if program_response
        else ""
    )
    program_name, semester_number = _parse_program_semester(
        program_text
    )

    # Question 3: Next exam
    exam_response = await cl.AskUserMessage(
        content=(
            "3️⃣ **What is your next exam subject and date?**\n"
            "e.g. `DSA 2026-04-15` or `OS April 20`"
        ),
        timeout=SETUP_QUESTION_TIMEOUT,
    ).send()
    exam_text = (
        exam_response["output"].strip()
        if exam_response
        else ""
    )
    exam_dates = _parse_exam_input(exam_text)

    # Question 4: Weak areas
    weak_response = await cl.AskUserMessage(
        content=(
            "4️⃣ **Any topics you already know are weak?**\n"
            "e.g. `recursion, graph traversal, deadlocks`\n"
            "Type `none` if you're not sure yet."
        ),
        timeout=SETUP_QUESTION_TIMEOUT,
    ).send()
    weak_text = (
        weak_response["output"].strip()
        if weak_response
        else ""
    )
    weak_areas = _parse_weak_areas(weak_text)

    # Build and save the profile
    profile = load_profile()
    profile["school"] = university_name
    profile["program"] = program_name
    profile["semester"] = semester_number
    profile["exam_dates"] = exam_dates
    profile["weak_areas"] = weak_areas

    is_saved = save_profile(profile)

    if is_saved:
        await cl.Message(
            content="✅ **Profile saved!** Let's get studying."
        ).send()
    else:
        await cl.Message(
            content=(
                "⚠️ Could not save profile to config.yaml. "
                "You can edit it manually later."
            )
        ).send()



# ── Helpers ───────────────────────────────────────────────

def _is_default_profile(profile: dict) -> bool:
    """
    Checks if the profile is still at default values.

    Returns True if the student hasn't configured their
    profile yet (first run detection).

    Args:
        profile: The loaded profile dict.

    Returns:
        True if profile looks unconfigured.
    """
    has_no_exam_dates = not profile.get("exam_dates")
    has_no_weak_areas = not profile.get("weak_areas")
    is_default_school = (
        profile.get("school") == DEFAULT_PROFILE["school"]
    )

    # Consider it a first run if school is default AND
    # no exam dates AND no weak areas configured
    return (
        is_default_school
        and has_no_exam_dates
        and has_no_weak_areas
    )


def _parse_program_semester(text: str) -> tuple[str, int]:
    """
    Extracts program name and semester number from user input.

    Handles formats like "BIT Semester 3", "BCA 4th sem",
    "CSIT 5", etc.

    Args:
        text: Raw user input for program/semester.

    Returns:
        Tuple of (program_name, semester_number).
        Defaults to ("BIT", 1) if parsing fails.
    """
    if not text:
        return (DEFAULT_PROFILE["program"], DEFAULT_PROFILE["semester"])

    import re

    # Try to find a number for the semester
    number_match = re.search(r"(\d+)", text)
    semester = int(number_match.group(1)) if number_match else 1

    # Program is everything before the number, cleaned up
    program = re.sub(r"[0-9]", "", text)
    program = re.sub(
        r"(semester|sem|th|rd|nd|st)", "", program,
        flags=re.IGNORECASE
    )
    program = program.strip().strip("-").strip()

    if not program:
        program = DEFAULT_PROFILE["program"]

    return (program.upper(), semester)


def _parse_exam_input(text: str) -> dict:
    """
    Parses exam subject and date from user input.

    Handles formats like "DSA 2026-04-15" or just "DSA".

    Args:
        text: Raw user input for exam info.

    Returns:
        Dict of {subject: date_string}. Empty dict if
        no valid input.
    """
    if not text or text.lower() in ("none", "no", "skip", ""):
        return {}

    import re

    exam_dates = {}
    # Try to split by commas for multiple exams
    parts = [part.strip() for part in text.split(",")]

    for part in parts:
        # Look for a date pattern (YYYY-MM-DD)
        date_match = re.search(r"(\d{4}-\d{2}-\d{2})", part)

        if date_match:
            date_string = date_match.group(1)
            # Subject is everything before the date
            subject = part[:date_match.start()].strip()
            if subject:
                exam_dates[subject.upper()] = date_string
        else:
            # No date found — just store the subject with
            # an empty date for now
            subject = part.strip()
            if subject:
                exam_dates[subject.upper()] = ""

    return exam_dates


def _parse_weak_areas(text: str) -> list[str]:
    """
    Parses comma-separated weak areas from user input.

    Args:
        text: Raw user input like "recursion, graphs, deadlocks".

    Returns:
        List of cleaned weak area strings. Empty list if
        user typed "none" or similar.
    """
    if not text or text.lower() in ("none", "no", "skip"):
        return []

    areas = [area.strip().lower() for area in text.split(",")]
    return [area for area in areas if area]


def _format_profile_summary(profile: dict) -> str:
    """
    Creates a short profile summary to show on startup.

    Args:
        profile: The loaded student profile dict.

    Returns:
        Markdown-formatted profile summary string.
    """
    lines = ["### 📋 Your Profile\n"]

    lines.append(
        f"- **School:** {profile.get('school', 'Not set')}"
    )
    lines.append(
        f"- **Program:** {profile.get('program', '?')} "
        f"Semester {profile.get('semester', '?')}"
    )

    exam_dates = profile.get("exam_dates", {})
    if exam_dates:
        lines.append("- **Upcoming exams:**")
        for subject, date in exam_dates.items():
            date_display = date if date else "date not set"
            lines.append(f"  - {subject}: {date_display}")

    weak_areas = profile.get("weak_areas", [])
    if weak_areas:
        lines.append(
            f"- **Weak areas:** {', '.join(weak_areas)}"
        )

    lines.append(
        "\n*Edit `config.yaml` to update your profile anytime.*"
    )
    return "\n".join(lines)
