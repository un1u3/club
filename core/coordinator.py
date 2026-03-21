"""
club/core/coordinator.py

Orchestrates the multi-agent pipeline using LangGraph.
Routes user messages to the correct agent based on intent
keywords, then returns the agent's response as a string.

Example:
    >>> from core.coordinator import chat
    >>> response = chat("Summarize my OS notes", history=[])
    >>> print(response)

Author: CLUB Project
License: MIT
"""

import json
from typing import TypedDict

from langgraph.graph import StateGraph, END

# Agent and profile imports are done INSIDE functions to
# prevent circular imports. coordinator is imported by app.py,
# which also imports memory and profile — loading all agents
# at module level would trigger a chain of heavy imports
# (ollama, chromadb, etc.) before the app is ready.

# ── Constants ─────────────────────────────────────────────

# Keywords that trigger each agent — checked against the
# lowercased user message
SUMMARIZER_KEYWORDS = [
    "summarize", "summary", "explain", "what is",
    "what are", "describe", "define", "notes",
]

QUIZZER_KEYWORDS = [
    "quiz", "test me", "questions", "practice",
    "mcq", "mock", "pyq", "past year",
]

SOLVER_KEYWORDS = [
    "solve", "answer", "how to do", "how do",
    "step by step", "solution", "work out", "calculate",
]

PLANNER_KEYWORDS = [
    "schedule", "plan", "when should", "study plan",
    "timetable", "briefing", "today", "morning",
]

# Agent node names used in the LangGraph state machine
NODE_ROUTER = "router"
NODE_SUMMARIZER = "summarizer"
NODE_QUIZZER = "quizzer"
NODE_SOLVER = "solver"
NODE_PLANNER = "planner"

DEFAULT_AGENT = NODE_SUMMARIZER


# ── State Definition ──────────────────────────────────────

class ClubState(TypedDict):
    """
    Shared state that flows through the LangGraph pipeline.

    Attributes:
        user_message: The raw message from the student.
        history:      List of prior (role, content) tuples.
        agent:        Which agent the router selected.
        context:      Relevant text chunks from memory.
        response:     The final response string to display.
    """
    user_message: str
    history: list
    agent: str
    context: str
    response: str


# ── Node Functions ────────────────────────────────────────

def router_node(state: ClubState) -> dict:
    """
    Analyzes the user message and picks the best agent.

    Uses keyword matching against the lowercased message.
    Falls back to the summarizer when no keywords match,
    since general study questions benefit from summarization.

    Args:
        state: Current graph state with user_message.

    Returns:
        Dict update with the selected agent name and any
        relevant context from memory.
    """
    from core.memory import search as memory_search

    message_lower = state["user_message"].lower()

    selected_agent = DEFAULT_AGENT

    # Check each agent's keywords in priority order
    # Solver before summarizer because "explain how to solve"
    # should route to solver, not summarizer
    if _message_matches(message_lower, SOLVER_KEYWORDS):
        selected_agent = NODE_SOLVER
    elif _message_matches(message_lower, QUIZZER_KEYWORDS):
        selected_agent = NODE_QUIZZER
    elif _message_matches(message_lower, PLANNER_KEYWORDS):
        selected_agent = NODE_PLANNER
    elif _message_matches(message_lower, SUMMARIZER_KEYWORDS):
        selected_agent = NODE_SUMMARIZER

    # Pull relevant context from memory for the agent
    memory_results = memory_search(
        state["user_message"], n_results=3
    )
    context_text = "\n\n".join(memory_results) if memory_results else ""

    print(f"CLUB Router: selected → {selected_agent}")
    return {"agent": selected_agent, "context": context_text}


def summarizer_node(state: ClubState) -> dict:
    """
    Runs the summarizer agent on the user's message.

    Combines any memory context with the user message to
    produce a structured study summary.

    Args:
        state: Current graph state with user_message and context.

    Returns:
        Dict update with the summary as response.
    """
    from agents.summarizer import summarize
    from knowmyschool.profile import load_profile

    # Combine memory context with the user's message so the
    # summarizer has relevant material to work with
    input_text = state["user_message"]
    if state.get("context"):
        input_text = (
            f"{state['context']}\n\n"
            f"Student's question: {input_text}"
        )

    profile = load_profile()
    subject = profile.get("program", "")

    summary = summarize(input_text, subject=subject)

    if not summary:
        summary = (
            "I couldn't generate a summary right now. "
            "Make sure ollama is running: `ollama serve`"
        )

    return {"response": summary}


def quizzer_node(state: ClubState) -> dict:
    """
    Runs the quizzer agent to generate practice questions.

    Detects whether the student wants MCQs, short answers,
    or past-year questions from their message.

    Args:
        state: Current graph state with user_message and context.

    Returns:
        Dict update with formatted quiz as response.
    """
    from agents.quizzer import generate_quiz, generate_pyq_style
    from knowmyschool.profile import get_school_context

    message_lower = state["user_message"].lower()

    # Detect desired question style from the message
    if "pyq" in message_lower or "past year" in message_lower:
        school_context = get_school_context()
        input_text = state.get("context", "") or state["user_message"]
        questions = generate_pyq_style(
            input_text, school_pattern=school_context
        )
    else:
        style = "MCQ"
        if "short" in message_lower:
            style = "short_answer"

        input_text = state.get("context", "") or state["user_message"]
        questions = generate_quiz(
            input_text, n_questions=5, style=style
        )

    if not questions:
        return {
            "response": (
                "I couldn't generate questions right now. "
                "Make sure ollama is running: `ollama serve`"
            )
        }

    return {"response": _format_quiz(questions)}


def solver_node(state: ClubState) -> dict:
    """
    Runs the solver agent to answer a question step-by-step.

    Includes school context so the solution matches the
    student's exam expectations.

    Args:
        state: Current graph state with user_message.

    Returns:
        Dict update with formatted solution as response.
    """
    from agents.solver import solve
    from knowmyschool.profile import load_profile, get_school_context

    profile = load_profile()
    school_context = get_school_context()

    result = solve(
        question=state["user_message"],
        subject=profile.get("program", ""),
        school_context=school_context,
    )

    if not result["steps"]:
        return {
            "response": (
                "I couldn't solve this right now. "
                "Make sure ollama is running: `ollama serve`"
            )
        }

    return {"response": _format_solution(result)}


def planner_node(state: ClubState) -> dict:
    """
    Runs the planner agent for schedules or daily briefings.

    Detects whether the student wants a full schedule or
    today's briefing from their message.

    Args:
        state: Current graph state with user_message.

    Returns:
        Dict update with schedule or briefing as response.
    """
    from agents.planner import build_schedule, generate_briefing
    from core.memory import search as memory_search
    from knowmyschool.profile import load_profile

    message_lower = state["user_message"].lower()
    profile = load_profile()

    is_briefing_request = any(
        keyword in message_lower
        for keyword in ["briefing", "today", "morning"]
    )

    if is_briefing_request:
        briefing = generate_briefing(profile, memory_search)
        return {"response": briefing}

    # Build a study schedule
    subjects = list(profile.get("exam_dates", {}).keys())
    exam_dates = profile.get("exam_dates", {})
    weak_areas = profile.get("weak_areas", [])

    # If no subjects in exam_dates, use a default list
    if not subjects:
        subjects = [profile.get("program", "General")]

    schedule = build_schedule(subjects, exam_dates, weak_areas)

    if not schedule["days"]:
        return {
            "response": (
                "No upcoming exams found in your profile. "
                "Add exam dates to config.yaml under "
                "profile → exam_dates."
            )
        }

    return {"response": _format_schedule(schedule)}


# ── Graph Builder ─────────────────────────────────────────

def _build_graph() -> StateGraph:
    """
    Constructs the LangGraph StateGraph with all agent nodes.

    The graph flows:
        router → (conditional) → agent_node → END

    The router decides which single agent to invoke based
    on user message keywords.

    Returns:
        A compiled LangGraph StateGraph ready for invocation.
    """
    graph = StateGraph(ClubState)

    # Add all nodes
    graph.add_node(NODE_ROUTER, router_node)
    graph.add_node(NODE_SUMMARIZER, summarizer_node)
    graph.add_node(NODE_QUIZZER, quizzer_node)
    graph.add_node(NODE_SOLVER, solver_node)
    graph.add_node(NODE_PLANNER, planner_node)

    # Router is the entry point
    graph.set_entry_point(NODE_ROUTER)

    # Conditional edge from router to the selected agent
    graph.add_conditional_edges(
        NODE_ROUTER,
        _route_to_agent,
        {
            NODE_SUMMARIZER: NODE_SUMMARIZER,
            NODE_QUIZZER: NODE_QUIZZER,
            NODE_SOLVER: NODE_SOLVER,
            NODE_PLANNER: NODE_PLANNER,
        },
    )

    # Each agent node goes to END after producing a response
    graph.add_edge(NODE_SUMMARIZER, END)
    graph.add_edge(NODE_QUIZZER, END)
    graph.add_edge(NODE_SOLVER, END)
    graph.add_edge(NODE_PLANNER, END)

    return graph.compile()


def _route_to_agent(state: ClubState) -> str:
    """
    Returns the agent name the router selected.

    Used as the conditional routing function for
    LangGraph's add_conditional_edges.

    Args:
        state: Current graph state with agent field set.

    Returns:
        The agent node name to invoke next.
    """
    return state.get("agent", DEFAULT_AGENT)


# ── Public Interface ──────────────────────────────────────

# Lazy compilation — built on first chat() call, not at
# import time, to avoid triggering heavy imports before
# the app is ready
_compiled_graph = None


def chat(user_message: str, history: list | None = None) -> str:
    """
    Routes a user message to the correct agent and returns
    the response.

    This is the main entry point for the CLUB study assistant.
    The coordinator analyzes the message, pulls relevant
    context from memory, routes to the best agent, and
    returns its response as a string.

    Args:
        user_message: The student's message text.
        history:      Optional conversation history as a list
                      of (role, content) tuples. Defaults to
                      an empty list.

    Returns:
        The agent's response as a string (markdown formatted).

    Example:
        >>> response = chat("Summarize binary search trees")
        >>> print(response)
    """
    global _compiled_graph

    if history is None:
        history = []

    if not user_message.strip():
        return "Please type a message so I can help you study!"

    # Lazy-compile the graph on first call
    if _compiled_graph is None:
        _compiled_graph = _build_graph()

    initial_state = {
        "user_message": user_message,
        "history": history,
        "agent": "",
        "context": "",
        "response": "",
    }

    try:
        final_state = _compiled_graph.invoke(initial_state)
        return final_state.get("response", "No response generated.")

    except Exception as error:
        print(f"CLUB Coordinator: pipeline error — {error}")
        return (
            "Something went wrong in the pipeline. "
            f"Error: {error}"
        )


# ── Formatting Helpers ────────────────────────────────────

def _message_matches(
    message: str,
    keywords: list[str],
) -> bool:
    """
    Checks if the message contains any of the given keywords.

    Args:
        message:  Lowercased user message.
        keywords: List of trigger phrases.

    Returns:
        True if any keyword is found in the message.
    """
    return any(keyword in message for keyword in keywords)


def _format_quiz(questions: list[dict]) -> str:
    """
    Formats a list of question dicts into readable markdown.

    Args:
        questions: List of question dicts from the quizzer.

    Returns:
        Markdown-formatted quiz string.
    """
    lines = ["## 📝 Practice Quiz\n"]

    for index, question_data in enumerate(questions, start=1):
        lines.append(
            f"**Q{index}.** {question_data['question']}\n"
        )

        # Show options for MCQs
        if question_data.get("options"):
            for key, value in question_data["options"].items():
                lines.append(f"   {key}) {value}")
            lines.append("")

        lines.append(
            f"<details><summary>Show Answer</summary>\n\n"
            f"**Answer:** {question_data['answer']}\n\n"
            f"**Explanation:** "
            f"{question_data.get('explanation', '')}\n"
            f"</details>\n"
        )

    return "\n".join(lines)


def _format_solution(result: dict) -> str:
    """
    Formats a solver result dict into readable markdown.

    Args:
        result: Solution dict from the solver agent.

    Returns:
        Markdown-formatted solution string.
    """
    lines = [f"## 🧮 Solution\n"]

    lines.append("### Steps\n")
    for step in result["steps"]:
        lines.append(f"- {step}")
    lines.append("")

    lines.append(f"### Answer\n{result['answer']}\n")
    lines.append(f"### Core Concept\n{result['concept']}\n")

    if result["common_mistakes"]:
        lines.append("### ⚠️ Common Mistakes\n")
        for mistake in result["common_mistakes"]:
            lines.append(f"- {mistake}")
        lines.append("")

    if result["similar_questions"]:
        lines.append("### 🔄 Similar Questions\n")
        for similar in result["similar_questions"]:
            lines.append(f"- {similar}")

    return "\n".join(lines)


def _format_schedule(schedule: dict) -> str:
    """
    Formats a schedule dict into readable markdown.

    Shows the first 7 days as a preview, with a count
    of remaining days.

    Args:
        schedule: Schedule dict from the planner agent.

    Returns:
        Markdown-formatted schedule string.
    """
    lines = [
        f"## 📅 Study Schedule\n",
        f"**{schedule['start_date']}** → "
        f"**{schedule['end_date']}** "
        f"({schedule['total_days']} days)\n",
    ]

    # Show first 7 days as a table
    preview_count = min(7, len(schedule["days"]))

    lines.append(
        "| Day | Date | Morning | Afternoon | Evening | Hours |"
    )
    lines.append(
        "|-----|------|---------|-----------|---------|-------|"
    )

    for day in schedule["days"][:preview_count]:
        lines.append(
            f"| {day['day_number']} "
            f"| {day['date']} "
            f"| {day['morning_topic']} "
            f"| {day['afternoon_topic']} "
            f"| {day['evening_review']} "
            f"| {day['estimated_hours']}h |"
        )

    remaining = len(schedule["days"]) - preview_count
    if remaining > 0:
        lines.append(f"\n*...and {remaining} more days*")

    return "\n".join(lines)


# ── Entry Point (Test) ────────────────────────────────────

if __name__ == "__main__":
    print("CLUB Coordinator: running routing test...\n")

    test_messages = [
        "Summarize binary search trees",
        "Quiz me on sorting algorithms",
        "Solve this: explain BFS step by step",
        "Plan my study schedule for DSA",
        "What is a linked list?",
        "Give me past year questions on OS",
        "Generate my morning briefing",
    ]

    for message in test_messages:
        # Only test routing, not full agent execution
        message_lower = message.lower()

        if _message_matches(message_lower, SOLVER_KEYWORDS):
            routed_to = NODE_SOLVER
        elif _message_matches(message_lower, QUIZZER_KEYWORDS):
            routed_to = NODE_QUIZZER
        elif _message_matches(message_lower, PLANNER_KEYWORDS):
            routed_to = NODE_PLANNER
        elif _message_matches(message_lower, SUMMARIZER_KEYWORDS):
            routed_to = NODE_SUMMARIZER
        else:
            routed_to = DEFAULT_AGENT

        print(f"  \"{message}\"")
        print(f"    → {routed_to}\n")

    print("CLUB Coordinator: routing test complete.")
    print(
        "\nTo test the full pipeline with LLM, run:\n"
        "  python -c \"from core.coordinator import chat; "
        "print(chat('What is a binary tree?'))\""
    )
