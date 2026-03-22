"""
club/agents/quizzer.py

Generates exam-style practice questions from study material.
Uses the local ollama LLM (llama3) to produce MCQs, short-answer,
and past-year-question-style questions with answers and explanations.

Example:
    >>> from agents.quizzer import generate_quiz
    >>> quiz = generate_quiz("Binary trees are...", n_questions=3)
    >>> print(quiz[0]["question"])

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
JSON_PARSE_RETRIES = 1       # extra attempt if JSON is invalid

VALID_STYLES = {"MCQ", "short_answer", "PYQ"}

# ── System Prompts ────────────────────────────────────────

MCQ_SYSTEM_PROMPT = """You are CLUB, a quiz generator for students.
Generate multiple-choice questions from the given study material.

Rules:
- Each question must have exactly 4 options: A, B, C, D.
- Exactly one option is correct.
- Include a brief explanation for the correct answer.
- Questions should test understanding, not just recall.
- Return ONLY a valid JSON array, nothing else.

Format (strict JSON):
[
  {{
    "question": "What is...?",
    "options": {{
      "A": "First option",
      "B": "Second option",
      "C": "Third option",
      "D": "Fourth option"
    }},
    "answer": "B",
    "explanation": "Because..."
  }}
]

Generate exactly {n_questions} questions."""

SHORT_ANSWER_SYSTEM_PROMPT = """You are CLUB, a quiz generator.
Generate short-answer questions from the given study material.

Rules:
- Questions should require 2-4 sentence answers.
- Focus on definitions, comparisons, and key concepts.
- Include the expected answer and a brief explanation.
- Return ONLY a valid JSON array, nothing else.

Format (strict JSON):
[
  {{
    "question": "Define...",
    "options": null,
    "answer": "A concise answer in 2-4 sentences.",
    "explanation": "This is important because..."
  }}
]

Generate exactly {n_questions} questions."""

PYQ_SYSTEM_PROMPT = """You are CLUB, a quiz generator that mimics
real university exam questions (past-year question style).

Rules:
- Generate questions that look like actual exam papers.
- Include a mix of theory and application questions.
- Some questions should ask for diagrams or step-by-step work.
- Include marks allocation in brackets, e.g. [5 marks].
- Include the expected answer outline and explanation.
- Return ONLY a valid JSON array, nothing else.

Format (strict JSON):
[
  {{
    "question": "Explain... with a suitable diagram. [5 marks]",
    "options": null,
    "answer": "Expected answer outline...",
    "explanation": "Key points the examiner looks for..."
  }}
]

Generate exactly {n_questions} questions."""

SCHOOL_PATTERN_ADDENDUM = """
Additional context about the exam pattern:
{school_pattern}

Match this style as closely as possible in your questions."""


# ── Core Functions ────────────────────────────────────────

def generate_quiz(
    text: str,
    n_questions: int = 5,
    style: str = "MCQ",
) -> list[dict]:
    """
    Generates exam-style questions from study material.

    Sends the text to the local LLM with a style-specific
    prompt and parses the response as a JSON list of
    question dicts.

    Args:
        text:        Study material to generate questions from.
        n_questions: Number of questions to generate.
                     Defaults to 5.
        style:       Question style — "MCQ", "short_answer",
                     or "PYQ". Defaults to "MCQ".

    Returns:
        A list of dicts, each containing:
            - question:    The question text (str)
            - options:     Answer options (dict or None)
            - answer:      Correct answer (str)
            - explanation: Why it's correct (str)
        Returns an empty list if generation fails.

    Example:
        >>> quiz = generate_quiz(
        ...     "A stack is a LIFO data structure...",
        ...     n_questions=3,
        ...     style="MCQ"
        ... )
        >>> print(quiz[0]["question"])
    """
    if not text.strip():
        print("CLUB Quizzer: received empty text, skipping")
        return []

    if style not in VALID_STYLES:
        print(
            f"CLUB Quizzer: unknown style '{style}'. "
            f"Valid: {VALID_STYLES}"
        )
        return []

    system_prompt = _get_style_prompt(style, n_questions)
    return _generate_and_parse(system_prompt, text)


def generate_pyq_style(
    text: str,
    school_pattern: str = "",
) -> list[dict]:
    """
    Generates past-year-question-style questions tailored to
    a specific school or university exam pattern.

    Uses the PYQ prompt as a base and appends school-specific
    context so the LLM mimics real exam formats.

    Args:
        text:           Study material to base questions on.
        school_pattern: Description of the exam style, e.g.
                        "TU BIT: theory heavy, always ask
                        to draw diagrams, 5 and 10 mark Qs".
                        Empty means generic PYQ style.

    Returns:
        A list of question dicts (same format as generate_quiz).
        Returns an empty list if generation fails.

    Example:
        >>> questions = generate_pyq_style(
        ...     "Process scheduling algorithms...",
        ...     school_pattern="TU BIT: theory heavy, diagrams"
        ... )
        >>> print(questions[0]["question"])
    """
    if not text.strip():
        print("CLUB Quizzer: received empty text, skipping")
        return []

    # Default to 5 questions for PYQ style
    system_prompt = _get_style_prompt("PYQ", n_questions=5)

    if school_pattern.strip():
        system_prompt += SCHOOL_PATTERN_ADDENDUM.format(
            school_pattern=school_pattern.strip()
        )

    return _generate_and_parse(system_prompt, text)


# ── LLM Interface ────────────────────────────────────────

def _generate_and_parse(
    system_prompt: str,
    user_text: str,
) -> list[dict]:
    """
    Calls the LLM and parses the response as JSON.

    If the first response contains invalid JSON, retries
    once with an explicit correction prompt asking the
    model to fix its output.

    Args:
        system_prompt: Style-specific instructions.
        user_text:     The study material.

    Returns:
        Parsed list of question dicts, or empty list on
        failure.
    """
    raw_response = _call_ollama(system_prompt, user_text)
    if not raw_response:
        return []

    # First attempt to parse
    parsed = _parse_json_response(raw_response)
    if parsed is not None:
        return parsed

    # JSON was invalid — retry with a correction prompt
    print("CLUB Quizzer: bad JSON from model, retrying...")
    correction_prompt = (
        "Your previous response was not valid JSON. "
        "Return ONLY a valid JSON array of question objects. "
        "No markdown, no explanation, just the JSON array."
    )
    retry_response = _call_ollama(
        correction_prompt, raw_response
    )
    if not retry_response:
        return []

    parsed = _parse_json_response(retry_response)
    if parsed is not None:
        return parsed

    print("CLUB Quizzer: could not get valid JSON after retry")
    return []


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
                "CLUB Quizzer: ollama is not running. "
                "Start it with: ollama serve"
            )
            return ""

        except Exception as error:
            is_last_attempt = attempt == MAX_RETRIES
            if is_last_attempt:
                print(
                    f"CLUB Quizzer: failed after "
                    f"{MAX_RETRIES} attempts — {error}"
                )
                return ""

            print(
                f"CLUB Quizzer: attempt {attempt} failed, "
                f"retrying in {RETRY_DELAY_SECONDS}s..."
            )
            time.sleep(RETRY_DELAY_SECONDS)

    return ""


# ── Helpers ───────────────────────────────────────────────

def _get_style_prompt(style: str, n_questions: int) -> str:
    """
    Returns the system prompt for a given question style.

    Args:
        style:       One of "MCQ", "short_answer", "PYQ".
        n_questions: Number of questions to request.

    Returns:
        The formatted system prompt string.
    """
    prompt_map = {
        "MCQ": MCQ_SYSTEM_PROMPT,
        "short_answer": SHORT_ANSWER_SYSTEM_PROMPT,
        "PYQ": PYQ_SYSTEM_PROMPT,
    }
    template = prompt_map[style]
    return template.format(n_questions=n_questions)


def _parse_json_response(raw_text: str) -> list[dict] | None:
    """
    Extracts and parses a JSON array from the LLM response.

    LLMs sometimes wrap JSON in markdown code fences or add
    commentary before/after. This function strips that away
    and attempts to parse the core JSON.

    Args:
        raw_text: Raw LLM output that should contain JSON.

    Returns:
        A list of dicts if parsing succeeds, None otherwise.
    """
    cleaned = raw_text.strip()

    # Strip markdown code fences if present
    # Handles ```json ... ``` and ``` ... ```
    fence_pattern = r"```(?:json)?\s*\n?(.*?)\n?\s*```"
    fence_match = re.search(fence_pattern, cleaned, re.DOTALL)
    if fence_match:
        cleaned = fence_match.group(1).strip()

    # Try to find a JSON array in the text
    # Look for content between the first [ and last ]
    bracket_start = cleaned.find("[")
    bracket_end = cleaned.rfind("]")

    if bracket_start != -1 and bracket_end != -1:
        cleaned = cleaned[bracket_start:bracket_end + 1]

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as error:
        print(f"CLUB Quizzer: JSON parse error — {error}")
        return None

    if not isinstance(parsed, list):
        print("CLUB Quizzer: expected a JSON array, got other")
        return None

    # Validate each question has required keys
    validated_questions = []
    required_keys = {"question", "answer"}

    for item in parsed:
        if not isinstance(item, dict):
            continue
        if not required_keys.issubset(item.keys()):
            continue

        # Ensure all expected keys exist with defaults
        validated_questions.append({
            "question": item.get("question", ""),
            "options": item.get("options", None),
            "answer": item.get("answer", ""),
            "explanation": item.get("explanation", ""),
        })

    if not validated_questions:
        print("CLUB Quizzer: no valid questions in response")
        return None

    return validated_questions


# ── Entry Point (Test) ────────────────────────────────────

if __name__ == "__main__":
    print("CLUB Quizzer: running test...\n")

    sample_text = (
        "A stack is a linear data structure that follows "
        "the Last In First Out (LIFO) principle. The main "
        "operations are push (add to top), pop (remove from "
        "top), and peek (view top without removing). Stacks "
        "are used in function call management, expression "
        "evaluation, undo operations, and backtracking "
        "algorithms like DFS. A stack can be implemented "
        "using arrays or linked lists. Array implementation "
        "has O(1) push/pop but fixed size. Linked list "
        "implementation has O(1) operations and dynamic size."
    )

    # ── Test 1: MCQ ───────────────────────────────────────
    print("── Test 1: MCQ style ─────────────────────\n")
    mcq_questions = generate_quiz(
        sample_text, n_questions=2, style="MCQ"
    )

    if mcq_questions:
        for question_data in mcq_questions:
            print(f"Q: {question_data['question']}")
            if question_data.get("options"):
                for key, value in question_data["options"].items():
                    print(f"   {key}) {value}")
            print(f"A: {question_data['answer']}")
            print(f"E: {question_data['explanation']}\n")
    else:
        print("No MCQ questions generated.\n")

    # ── Test 2: Short Answer ──────────────────────────────
    print("── Test 2: Short answer style ─────────────\n")
    short_questions = generate_quiz(
        sample_text, n_questions=2, style="short_answer"
    )

    if short_questions:
        for question_data in short_questions:
            print(f"Q: {question_data['question']}")
            print(f"A: {question_data['answer']}\n")
    else:
        print("No short answer questions generated.\n")

    # ── Test 3: PYQ with school pattern ───────────────────
    print("── Test 3: PYQ style (TU BIT) ─────────────\n")
    pyq_questions = generate_pyq_style(
        sample_text,
        school_pattern=(
            "TU BIT: theory heavy, always ask students "
            "to draw diagrams, 5 and 10 mark questions"
        ),
    )

    if pyq_questions:
        for question_data in pyq_questions:
            print(f"Q: {question_data['question']}")
            print(f"A: {question_data['answer']}\n")
    else:
        print("No PYQ questions generated.\n")

    print("CLUB Quizzer: test complete.")
