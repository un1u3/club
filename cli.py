"""
club/cli.py

Command-line interface for the CLUB study assistant.
Provides quick commands to initialize the project, start the
UI, and generate immediate study briefings.

Example:
    >>> # From project root:
    >>> python cli.py init
    >>> python cli.py start
    >>> python cli.py briefing

Author: CLUB Project
License: MIT
"""

import os
import sys
import subprocess

# ── Constants ─────────────────────────────────────────────

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Folder structure that init creates
REQUIRED_FOLDERS = [
    os.path.join("folder", "notes"),
    os.path.join("folder", "pyqs"),
    os.path.join("folder", "images"),
    os.path.join("folder", "youtube"),
    os.path.join("folder", "output"),
]

USAGE_TEXT = """
╔══════════════════════════════════════════╗
║          CLUB Study Assistant            ║
║  Continuous Learning Understanding Bots  ║
╚══════════════════════════════════════════╝

Usage:
  python cli.py <command>

Commands:
  init      Create folder structure and run first-time setup
  start     Launch the Chainlit chat interface
  briefing  Generate today's study briefing immediately
  index     Index all existing files in folder/
  help      Show this help message
"""


# ── Commands ──────────────────────────────────────────────

def cmd_init():
    """
    Creates the full folder structure and runs first-time setup.

    Creates all study folders (notes, pyqs, images, youtube,
    output) and prompts the student to configure their profile
    if config.yaml is missing or unconfigured.
    """
    print("CLUB: initializing project structure...\n")

    # Create folder structure
    for folder_path in REQUIRED_FOLDERS:
        full_path = os.path.join(PROJECT_ROOT, folder_path)
        os.makedirs(full_path, exist_ok=True)
        print(f"  ✓ {folder_path}/")

    # Ensure .gitkeep files exist in empty folders
    for folder_path in REQUIRED_FOLDERS:
        full_path = os.path.join(PROJECT_ROOT, folder_path)
        gitkeep = os.path.join(full_path, ".gitkeep")
        if not os.path.exists(gitkeep):
            with open(gitkeep, "w") as file:
                pass

    # Create config.yaml if missing
    config_path = os.path.join(PROJECT_ROOT, "config.yaml")
    if not os.path.exists(config_path):
        print("\n  ⚠ config.yaml not found — creating default...")
        _create_default_config(config_path)
    else:
        print(f"\n  ✓ config.yaml exists")

    # Interactive profile setup
    print("\n── Profile Setup ─────────────────────────\n")
    _run_cli_profile_setup()

    print("\n── Done! ─────────────────────────────────\n")
    print("  Start CLUB with: python cli.py start")
    print("  Or generate a briefing: python cli.py briefing\n")


def cmd_start():
    """
    Launches the Chainlit chat interface.

    Runs 'chainlit run interface/app.py' as a subprocess
    from the project root directory.
    """
    print("CLUB: starting Chainlit interface...\n")

    app_path = os.path.join(PROJECT_ROOT, "interface", "app.py")

    if not os.path.exists(app_path):
        print(
            f"Error: {app_path} not found.\n"
            "Run 'python cli.py init' first."
        )
        sys.exit(1)

    try:
        # Set PYTHONPATH so all Chainlit workers (including reloaded
        # ones) can find local packages like study_agents and core.
        env = os.environ.copy()
        existing_pythonpath = env.get("PYTHONPATH", "")
        if PROJECT_ROOT not in existing_pythonpath:
            env["PYTHONPATH"] = (
                PROJECT_ROOT + os.pathsep + existing_pythonpath
                if existing_pythonpath
                else PROJECT_ROOT
            )

        subprocess.run(
            [sys.executable, "-m", "chainlit", "run", app_path, "--host", "0.0.0.0"],
            cwd=PROJECT_ROOT,
            env=env,
        )
    except FileNotFoundError:
        print(
            "Error: chainlit is not installed.\n"
            "Install with: pip install chainlit"
        )
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nCLUB: stopped.")


def cmd_briefing():
    """
    Generates today's study briefing immediately and saves it.

    Loads the student profile, searches memory for relevant
    content, and calls the planner to produce a morning
    briefing. Saves to folder/output/briefing_today.md.
    """
    # Ensure project root is on sys.path for imports
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)

    print("CLUB: generating today's briefing...\n")

    try:
        from knowmyschool.profile import load_profile
        from core.memory import search as memory_search
        from study_agents.planner import (
            generate_briefing,
            _save_briefing,
        )
    except ImportError as error:
        print(f"Error: missing dependency — {error}")
        print("Run: pip install -r requirements.txt")
        sys.exit(1)

    profile = load_profile()
    briefing = generate_briefing(profile, memory_search)

    # Print to terminal
    print(briefing)

    # Save to file
    _save_briefing(briefing)
    output_path = os.path.join(
        "folder", "output", "briefing_today.md"
    )
    print(f"\n✓ Briefing saved to {output_path}")


def cmd_index():
    """
    Indexes all existing files in the study folders.

    Scans folder/notes/, folder/pyqs/, and folder/images/
    and ingests any files into the vector memory.
    """
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)

    print("CLUB: indexing existing files...\n")

    try:
        from core.watcher import index_existing_files
    except ImportError as error:
        print(f"Error: missing dependency — {error}")
        sys.exit(1)

    count = index_existing_files()
    print(f"\n✓ Indexed {count} file(s) into memory.")


# ── Helpers ───────────────────────────────────────────────

def _create_default_config(config_path: str) -> None:
    """
    Creates a default config.yaml with all sections.

    Args:
        config_path: Full path to write config.yaml.
    """
    default_config = """# ── CLUB Configuration ─────────────────────────────────────
# Central config for the CLUB study assistant.
# Edit this file to change models, paths, and behavior.

# ── LLM Settings ──────────────────────────────────────────
llm:
  provider: ollama
  model: llama3:latest
  base_url: "http://localhost:11434"

# ── Memory Settings ───────────────────────────────────────
memory:
  provider: chromadb
  persist_directory: "./chromadb_store"
  collection_name: "club_memory"

# ── Folder Paths ──────────────────────────────────────────
folders:
  notes: "folder/notes/"
  pyqs: "folder/pyqs/"
  images: "folder/images/"
  youtube: "folder/youtube/"
  output: "folder/output/"

# ── Agent Settings ────────────────────────────────────────
agents:
  summarizer:
    max_summary_ratio: 0.3
  quizzer:
    default_difficulty: "medium"
    questions_per_quiz: 10
  planner:
    default_hours_per_day: 4

# ── Student Profile ───────────────────────────────────────
profile:
  school: "Tribhuvan University"
  program: "BIT"
  semester: 3
  exam_style: "theory heavy, 3hr paper"
  marking_pattern: "steps matter more than final answer"
  hot_topics: []
  senior_insights: []
  exam_dates: {}
  weak_areas: []
"""
    with open(config_path, "w", encoding="utf-8") as file:
        file.write(default_config)
    print("  ✓ Created default config.yaml")


def _run_cli_profile_setup() -> None:
    """
    Interactive CLI profile setup (no Chainlit needed).

    Asks the student 4 questions and saves to config.yaml.
    """
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)

    from knowmyschool.profile import load_profile, save_profile

    profile = load_profile()

    print("Answer the following (press Enter to keep defaults):\n")

    # Question 1: School
    current_school = profile.get("school", "")
    school = input(
        f"  School [{current_school}]: "
    ).strip()
    if school:
        profile["school"] = school

    # Question 2: Program
    current_program = profile.get("program", "")
    current_semester = profile.get("semester", "")
    program = input(
        f"  Program [{current_program}]: "
    ).strip()
    if program:
        profile["program"] = program.upper()

    semester = input(
        f"  Semester [{current_semester}]: "
    ).strip()
    if semester.isdigit():
        profile["semester"] = int(semester)

    # Question 3: Exam dates
    exam_input = input(
        "  Next exam (e.g. DSA 2026-04-15): "
    ).strip()
    if exam_input and exam_input.lower() not in ("none", "skip"):
        parts = exam_input.split()
        if len(parts) >= 2:
            subject = parts[0].upper()
            date = parts[1]
            current_exams = profile.get("exam_dates", {})
            current_exams[subject] = date
            profile["exam_dates"] = current_exams
        elif len(parts) == 1:
            current_exams = profile.get("exam_dates", {})
            current_exams[parts[0].upper()] = ""
            profile["exam_dates"] = current_exams

    # Question 4: Weak areas
    weak_input = input(
        "  Weak areas (comma-separated): "
    ).strip()
    if weak_input and weak_input.lower() not in ("none", "skip"):
        areas = [a.strip().lower() for a in weak_input.split(",")]
        profile["weak_areas"] = [a for a in areas if a]

    is_saved = save_profile(profile)
    if is_saved:
        print("\n  ✓ Profile saved to config.yaml")
    else:
        print("\n  ✗ Failed to save profile")


# ── Entry Point ───────────────────────────────────────────

def main():
    """
    Parses the CLI command and dispatches to the handler.
    """
    if len(sys.argv) < 2:
        print(USAGE_TEXT)
        sys.exit(0)

    command = sys.argv[1].lower()

    command_map = {
        "init": cmd_init,
        "start": cmd_start,
        "briefing": cmd_briefing,
        "index": cmd_index,
        "help": lambda: print(USAGE_TEXT),
        "--help": lambda: print(USAGE_TEXT),
        "-h": lambda: print(USAGE_TEXT),
    }

    handler = command_map.get(command)
    if handler:
        handler()
    else:
        print(f"Unknown command: {command}")
        print(USAGE_TEXT)
        sys.exit(1)


if __name__ == "__main__":
    main()
