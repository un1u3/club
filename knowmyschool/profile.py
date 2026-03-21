"""
club/knowmyschool/profile.py

Stores and retrieves the student's academic profile from config.yaml.
Agents use this profile to tailor summaries, quizzes, and solutions
to the student's school, program, and exam pattern.

Example:
    >>> from knowmyschool.profile import load_profile
    >>> student = load_profile()
    >>> print(student["school"])
    'Tribhuvan University'

Author: CLUB Project
License: MIT
"""

import os

import yaml

# ── Constants ─────────────────────────────────────────────

CONFIG_FILE = "config.yaml"

# Default profile used when config.yaml has no profile section
# or when creating a fresh config for the first time
DEFAULT_PROFILE = {
    "school": "Tribhuvan University",
    "program": "BIT",
    "semester": 3,
    "exam_style": "theory heavy, 3hr paper",
    "marking_pattern": "steps matter more than final answer",
    "hot_topics": [],
    "senior_insights": [],
    "exam_dates": {},
    "weak_areas": [],
}


# ── Core Functions ────────────────────────────────────────

def load_profile() -> dict:
    """
    Reads the student profile from config.yaml.

    Loads the full config file and extracts the 'profile'
    section. If the file or section is missing, returns
    a copy of the default profile.

    Returns:
        A dict containing the student's school, program,
        semester, exam style, and other profile fields.

    Example:
        >>> profile = load_profile()
        >>> print(profile["program"])
        'BIT'
    """
    if not os.path.exists(CONFIG_FILE):
        print(
            f"CLUB Profile: {CONFIG_FILE} not found, "
            "using defaults"
        )
        return DEFAULT_PROFILE.copy()

    try:
        with open(CONFIG_FILE, "r", encoding="utf-8") as file:
            config = yaml.safe_load(file)
    except yaml.YAMLError as error:
        print(f"CLUB Profile: failed to parse config — {error}")
        return DEFAULT_PROFILE.copy()

    if not config or "profile" not in config:
        print(
            "CLUB Profile: no 'profile' section in config, "
            "using defaults"
        )
        return DEFAULT_PROFILE.copy()

    # Merge with defaults so new keys added in future versions
    # are always present even if the user's config is outdated
    profile = DEFAULT_PROFILE.copy()
    profile.update(config["profile"])
    return profile


def save_profile(profile: dict) -> bool:
    """
    Writes the student profile back to config.yaml.

    Loads the full config, updates only the 'profile' section,
    and writes the entire file back. This preserves all other
    config sections (llm, memory, agents, etc.).

    Args:
        profile: The profile dict to save. Should have the
                 same keys as DEFAULT_PROFILE.

    Returns:
        True if save succeeded, False otherwise.

    Example:
        >>> profile = load_profile()
        >>> profile["semester"] = 4
        >>> save_profile(profile)
        True
    """
    # Load the full config so we don't clobber other sections
    config = _load_full_config()
    config["profile"] = profile

    try:
        with open(CONFIG_FILE, "w", encoding="utf-8") as file:
            yaml.dump(
                config,
                file,
                default_flow_style=False,
                allow_unicode=True,
                sort_keys=False,
            )
        return True

    except OSError as error:
        print(f"CLUB Profile: failed to save config — {error}")
        return False


def get_school_context() -> str:
    """
    Returns a formatted string describing the school's exam
    pattern, suitable for injecting into LLM prompts.

    Agents use this to tailor their output (summaries, quizzes,
    solutions) to match the student's exam expectations.

    Returns:
        A multi-line string with school, program, exam style,
        marking pattern, and any hot topics or weak areas.

    Example:
        >>> context = get_school_context()
        >>> print(context)
        'School: Tribhuvan University (BIT, Semester 3)...'
    """
    profile = load_profile()

    # Build the context string piece by piece
    context_lines = [
        f"School: {profile['school']} "
        f"({profile['program']}, Semester {profile['semester']})",
        f"Exam style: {profile['exam_style']}",
        f"Marking: {profile['marking_pattern']}",
    ]

    # Include hot topics if the student has listed any
    if profile.get("hot_topics"):
        topics_text = ", ".join(profile["hot_topics"])
        context_lines.append(f"Hot topics: {topics_text}")

    # Include weak areas so agents can focus on them
    if profile.get("weak_areas"):
        weak_text = ", ".join(profile["weak_areas"])
        context_lines.append(f"Weak areas: {weak_text}")

    # Include senior insights for extra exam tips
    if profile.get("senior_insights"):
        for insight in profile["senior_insights"]:
            context_lines.append(f"Senior tip: {insight}")

    return "\n".join(context_lines)


def add_weak_areas(new_areas: list[str]) -> None:
    """
    Appends new weak areas to the profile without duplicates.

    Called automatically by the feedback module after analyzing
    an exam paper. Deduplicates and saves back to config.yaml.

    Args:
        new_areas: List of topic names to mark as weak areas.

    Example:
        >>> add_weak_areas(["recursion", "graph traversal"])
    """
    profile = load_profile()

    existing_areas = set(profile.get("weak_areas", []))
    for area in new_areas:
        area_cleaned = area.strip().lower()
        if area_cleaned and area_cleaned not in existing_areas:
            existing_areas.add(area_cleaned)

    profile["weak_areas"] = sorted(existing_areas)
    is_saved = save_profile(profile)

    if is_saved:
        print(
            f"CLUB Profile: weak areas updated — "
            f"{profile['weak_areas']}"
        )


# ── Helpers ───────────────────────────────────────────────

def _load_full_config() -> dict:
    """
    Loads the entire config.yaml as a dict.

    Used internally by save_profile to preserve all sections
    when updating just the profile.

    Returns:
        Full config dict, or empty dict on failure.
    """
    if not os.path.exists(CONFIG_FILE):
        return {}

    try:
        with open(CONFIG_FILE, "r", encoding="utf-8") as file:
            config = yaml.safe_load(file)
            return config if config else {}

    except yaml.YAMLError as error:
        print(f"CLUB Profile: failed to parse config — {error}")
        return {}


# ── Entry Point (Test) ────────────────────────────────────

if __name__ == "__main__":
    print("CLUB Profile: running test...\n")

    # Test 1: Load profile
    print("── Test 1: Load profile ──────────────────\n")
    current_profile = load_profile()
    for key, value in current_profile.items():
        print(f"  {key}: {value}")

    # Test 2: Get school context
    print("\n── Test 2: School context ────────────────\n")
    context = get_school_context()
    print(f"  {context}")

    # Test 3: Add weak areas
    print("\n── Test 3: Add weak areas ────────────────\n")
    add_weak_areas(["recursion", "graph traversal"])

    # Verify they were saved
    updated_profile = load_profile()
    print(f"  Weak areas: {updated_profile['weak_areas']}")

    print("\nCLUB Profile: test complete.")
