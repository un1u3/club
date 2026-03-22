"""
club/agents/solver.py

Solves exam questions with step-by-step professor-style reasoning.
Designed for past-year questions and student-submitted problems,
providing structured solutions with common mistakes and practice tips.

Example:
    >>> from study_agents.solver import solve
    >>> result = solve("Explain BFS traversal with an example.")
    >>> print(result["steps"][0])

Author: CLUB Project
License: MIT
"""

import json
import re
import time

import ollama

from core.constants import (
    OLLAMA_MODEL,
    OLLAMA_RETRY_ATTEMPTS,
    OLLAMA_RETRY_DELAY,
)

# ── Constants ─────────────────────────────────────────────

MAX_RETRIES = OLLAMA_RETRY_ATTEMPTS
RETRY_DELAY_SECONDS = OLLAMA_RETRY_DELAY

# ── System Prompts ────────────────────────────────────────

BASE_SYSTEM_PROMPT = """You are CLUB, an AI professor who \
solves exam questions for students.

When solving a question, you MUST:
1. Break the solution into clear numbered steps.
2. Name the core concept being tested.
3. List exactly 2 common mistakes students make on this type.
4. Suggest exactly 2 similar practice questions.

Return ONLY a valid JSON object in this exact format:
{{
  "steps": [
    "Step 1: ...",
    "Step 2: ...",
    "Step 3: ..."
  ],
  "answer": "The final, complete answer in 2-5 sentences.",
  "concept": "Name of the core concept (e.g. BFS traversal)",
  "common_mistakes": [
    "First common mistake students make",
    "Second common mistake students make"
  ],
  "similar_questions": [
    "A similar practice question",
    "Another similar practice question"
  ]
}}

Rules:
- Solve step by step like a professor explaining on a board.
- Each step should be self-contained and understandable.
- The answer should be a complete, exam-ready response.
- Do NOT include markdown formatting inside the JSON values.
- Return ONLY the JSON object, nothing else."""

SUBJECT_ADDENDUM = """
This question is from the subject: {subject}.
Use terminology and depth appropriate for {subject} exams."""

SCHOOL_CONTEXT_ADDENDUM = """
Exam context: {school_context}
Tailor the depth, format, and style of your solution to
match what this exam pattern expects."""


# ── Core Function ─────────────────────────────────────────

def solve(
    question: str,
    subject: str = "",
    school_context: str = "",
) -> dict:
    """
    Solves an exam question with step-by-step reasoning.

    Sends the question to the local LLM and parses the
    structured response into a dict containing solution
    steps, final answer, core concept, common mistakes,
    and similar practice questions.

    Args:
        question:       The exam question to solve.
        subject:        Optional subject name (e.g. "DSA", "OS").
                        Tailors vocabulary and depth.
        school_context: Optional exam pattern description
                        (e.g. "TU BIT: theory heavy, diagrams").
                        Tailors the solution format.

    Returns:
        A dict with keys:
            - steps:             list[str] of solution steps
            - answer:            str, the final answer
            - concept:           str, core concept tested
            - common_mistakes:   list[str], 2 common errors
            - similar_questions: list[str], 2 practice Qs
        Returns a fallback dict with empty values on failure.

    Example:
        >>> result = solve(
        ...     "Explain BFS with an example.",
        ...     subject="DSA"
        ... )
        >>> print(result["concept"])
        'BFS traversal'
    """
    if not question.strip():
        print("CLUB Solver: received empty question, skipping")
        return _empty_result()

    system_prompt = _build_system_prompt(subject, school_context)
    raw_response = _call_ollama(system_prompt, question)

    if not raw_response:
        return _empty_result()

    # First parse attempt
    parsed = _parse_json_response(raw_response)
    if parsed is not None:
        return _validate_result(parsed)

    # JSON was invalid — retry with correction prompt
    print("CLUB Solver: bad JSON from model, retrying...")
    correction_prompt = (
        "Your previous response was not valid JSON. "
        "Return ONLY a valid JSON object with keys: "
        "steps, answer, concept, common_mistakes, "
        "similar_questions. No markdown, no explanation."
    )
    retry_response = _call_ollama(correction_prompt, raw_response)

    if not retry_response:
        return _empty_result()

    parsed = _parse_json_response(retry_response)
    if parsed is not None:
        return _validate_result(parsed)

    print("CLUB Solver: could not get valid JSON after retry")
    return _empty_result()


# ── LLM Interface ────────────────────────────────────────

def _call_ollama(system_prompt: str, user_text: str) -> str:
    """
    Sends a prompt to the local ollama LLM with retry logic.

    Tries up to MAX_RETRIES times, waiting RETRY_DELAY_SECONDS
    between attempts.

    Args:
        system_prompt: Instructions for the LLM.
        user_text:     The question or content to process.

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
                "CLUB Solver: ollama is not running. "
                "Start it with: ollama serve"
            )
            return ""

        except Exception as error:
            is_last_attempt = attempt == MAX_RETRIES
            if is_last_attempt:
                print(
                    f"CLUB Solver: failed after "
                    f"{MAX_RETRIES} attempts — {error}"
                )
                return ""

            print(
                f"CLUB Solver: attempt {attempt} failed, "
                f"retrying in {RETRY_DELAY_SECONDS}s..."
            )
            time.sleep(RETRY_DELAY_SECONDS)

    return ""


# ── Helpers ───────────────────────────────────────────────

def _build_system_prompt(
    subject: str,
    school_context: str,
) -> str:
    """
    Constructs the full system prompt with optional context.

    Args:
        subject:        Subject name to tailor depth.
        school_context: Exam style description.

    Returns:
        Complete system prompt string.
    """
    prompt = BASE_SYSTEM_PROMPT

    if subject.strip():
        prompt += SUBJECT_ADDENDUM.format(
            subject=subject.strip()
        )

    if school_context.strip():
        prompt += SCHOOL_CONTEXT_ADDENDUM.format(
            school_context=school_context.strip()
        )

    return prompt


def _parse_json_response(raw_text: str) -> dict | None:
    """
    Extracts and parses a JSON object from the LLM response.

    Handles common LLM output issues: markdown code fences,
    commentary before/after the JSON, etc.

    Args:
        raw_text: Raw LLM output that should contain JSON.

    Returns:
        Parsed dict if successful, None otherwise.
    """
    cleaned = raw_text.strip()

    # Strip markdown code fences (```json ... ``` or ``` ... ```)
    fence_pattern = r"```(?:json)?\s*\n?(.*?)\n?\s*```"
    fence_match = re.search(fence_pattern, cleaned, re.DOTALL)
    if fence_match:
        cleaned = fence_match.group(1).strip()

    # Find the JSON object between first { and last }
    brace_start = cleaned.find("{")
    brace_end = cleaned.rfind("}")

    if brace_start != -1 and brace_end != -1:
        cleaned = cleaned[brace_start:brace_end + 1]

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as error:
        print(f"CLUB Solver: JSON parse error — {error}")
        return None

    if not isinstance(parsed, dict):
        print("CLUB Solver: expected a JSON object, got other")
        return None

    return parsed


def _validate_result(parsed: dict) -> dict:
    """
    Ensures the parsed dict has all required keys with correct types.

    Fills in missing keys with sensible defaults so downstream
    code never encounters KeyError.

    Args:
        parsed: Raw dict from JSON parsing.

    Returns:
        A dict guaranteed to have all five expected keys.
    """
    # Ensure steps is a list of strings
    raw_steps = parsed.get("steps", [])
    if not isinstance(raw_steps, list):
        raw_steps = [str(raw_steps)]
    steps = [str(step) for step in raw_steps]

    # Ensure common_mistakes is a list of strings
    raw_mistakes = parsed.get("common_mistakes", [])
    if not isinstance(raw_mistakes, list):
        raw_mistakes = [str(raw_mistakes)]
    common_mistakes = [str(m) for m in raw_mistakes]

    # Ensure similar_questions is a list of strings
    raw_similar = parsed.get("similar_questions", [])
    if not isinstance(raw_similar, list):
        raw_similar = [str(raw_similar)]
    similar_questions = [str(q) for q in raw_similar]

    return {
        "steps": steps,
        "answer": str(parsed.get("answer", "")),
        "concept": str(parsed.get("concept", "")),
        "common_mistakes": common_mistakes,
        "similar_questions": similar_questions,
    }


def _empty_result() -> dict:
    """
    Returns an empty result dict as a fallback.

    Used when the LLM fails or returns unparseable output.

    Returns:
        Dict with all keys present but empty values.
    """
    return {
        "steps": [],
        "answer": "",
        "concept": "",
        "common_mistakes": [],
        "similar_questions": [],
    }


# ── Entry Point (Test) ────────────────────────────────────

if __name__ == "__main__":
    print("CLUB Solver: running test...\n")

    sample_question = (
        "Write and explain the algorithm for BFS (Breadth "
        "First Search) traversal of a graph. Trace through "
        "the algorithm using a suitable example. [10 marks]"
    )

    print(f"Question: {sample_question}\n")
    print("Sending to ollama (llama3)...\n")

    result = solve(
        question=sample_question,
        subject="DSA",
        school_context="TU BIT: theory heavy, draw diagrams",
    )

    if result["steps"]:
        print("── Steps ─────────────────────────────────\n")
        for step in result["steps"]:
            print(f"  {step}\n")

        print("── Answer ────────────────────────────────\n")
        print(f"  {result['answer']}\n")

        print("── Concept ───────────────────────────────\n")
        print(f"  {result['concept']}\n")

        print("── Common Mistakes ───────────────────────\n")
        for mistake in result["common_mistakes"]:
            print(f"  ✗ {mistake}")

        print("\n── Similar Questions ─────────────────────\n")
        for similar in result["similar_questions"]:
            print(f"  → {similar}")

        print("\nCLUB Solver: test complete.")
    else:
        print("CLUB Solver: test failed — no solution generated")
        print("Make sure ollama is running: ollama serve")
