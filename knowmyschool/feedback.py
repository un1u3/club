"""
club/knowmyschool/feedback.py

Analyzes exam papers via OCR and LLM to identify weak areas.
Auto-updates the student profile with discovered weaknesses
so future study sessions focus where they matter most.

Example:
    >>> from knowmyschool.feedback import analyze_exam_paper
    >>> result = analyze_exam_paper("folder/images/dsa_midterm.jpg", "DSA")
    >>> print(result["weak_areas"])

Author: CLUB Project
License: MIT
"""

import json
import re
import time

import ollama

from knowmyschool.profile import get_school_context, add_weak_areas

# ── Constants ─────────────────────────────────────────────

MODEL_NAME = "llama3"

MAX_RETRIES = 3              # attempts for LLM call
RETRY_DELAY_SECONDS = 2      # wait between retries

# ── System Prompt ─────────────────────────────────────────

ANALYSIS_SYSTEM_PROMPT = """You are CLUB, an exam paper analyst.
A student has shared a photo of their graded exam paper.
The OCR-extracted text from the paper is provided below.

Analyze the paper and return ONLY a valid JSON object:
{{
  "total_marks": <number or null if unclear>,
  "lost_marks": <number or null if unclear>,
  "weak_areas": [
    "topic where student lost marks",
    "another weak topic"
  ],
  "improvement_tips": [
    "Specific, actionable tip for improvement",
    "Another tip based on the paper analysis"
  ]
}}

Rules:
- Identify EVERY topic where the student lost marks.
- Be specific about weak areas (e.g. "recursion base cases"
  not just "recursion").
- Give practical tips, not generic advice.
- If marks are visible, calculate total and lost marks.
- If marks are not visible, set them to null.
- Return ONLY the JSON object, nothing else.

{school_context}"""


# ── Core Function ─────────────────────────────────────────

def analyze_exam_paper(
    image_path: str,
    subject: str = "",
) -> dict:
    """
    OCRs an exam paper image and analyzes it for weak areas.

    Extracts text from the image using pytesseract, sends it
    to the local LLM with school context, and parses the
    structured response. Automatically saves discovered weak
    areas back to the student profile in config.yaml.

    Args:
        image_path: Path to the exam paper image
                    (.jpg, .jpeg, .png).
        subject:    Subject name (e.g. "DSA"). Included in
                    the LLM prompt for context.

    Returns:
        A dict with keys:
            - total_marks:      int or None
            - lost_marks:       int or None
            - weak_areas:       list[str]
            - improvement_tips: list[str]
        Returns a fallback dict with empty values on failure.

    Example:
        >>> result = analyze_exam_paper(
        ...     "folder/images/dsa_midterm.jpg", "DSA"
        ... )
        >>> print(result["weak_areas"])
        ['recursion base cases', 'graph BFS vs DFS']
    """
    # Step 1: Extract text from the exam paper image
    extracted_text = _ocr_image(image_path)
    if not extracted_text:
        return _empty_result()

    # Step 2: Build the prompt with school context
    school_context = get_school_context()
    system_prompt = ANALYSIS_SYSTEM_PROMPT.format(
        school_context=f"School context:\n{school_context}"
    )

    # Add subject to the user message for extra context
    user_message = f"Subject: {subject}\n\n" if subject else ""
    user_message += (
        f"OCR text from exam paper:\n\n{extracted_text}"
    )

    # Step 3: Send to LLM and parse response
    raw_response = _call_ollama(system_prompt, user_message)
    if not raw_response:
        return _empty_result()

    parsed = _parse_json_response(raw_response)

    if parsed is None:
        # Retry with correction prompt
        print("CLUB Feedback: bad JSON, retrying...")
        correction_prompt = (
            "Your previous response was not valid JSON. "
            "Return ONLY a valid JSON object with keys: "
            "total_marks, lost_marks, weak_areas, "
            "improvement_tips. No markdown, just JSON."
        )
        retry_response = _call_ollama(
            correction_prompt, raw_response
        )
        if retry_response:
            parsed = _parse_json_response(retry_response)

    if parsed is None:
        print("CLUB Feedback: could not parse analysis")
        return _empty_result()

    result = _validate_result(parsed)

    # Step 4: Auto-save weak areas to student profile
    if result["weak_areas"]:
        add_weak_areas(result["weak_areas"])
        print(
            f"CLUB Feedback: saved {len(result['weak_areas'])} "
            "weak areas to profile"
        )

    return result


# ── OCR ───────────────────────────────────────────────────

def _ocr_image(image_path: str) -> str:
    """
    Extracts text from an image using pytesseract OCR.

    Args:
        image_path: Path to the image file.

    Returns:
        Extracted text string, or empty string on failure.
    """
    try:
        import pytesseract
        from PIL import Image
    except ImportError:
        print(
            "CLUB Feedback: pytesseract or Pillow not installed. "
            "Install with: pip install pytesseract pillow"
        )
        return ""

    try:
        image = Image.open(image_path)
        extracted_text = pytesseract.image_to_string(image)
        cleaned_text = extracted_text.strip()

        if not cleaned_text:
            print(
                f"CLUB Feedback: OCR returned empty text "
                f"for {image_path} — image may be unclear"
            )
            return ""

        print(
            f"CLUB Feedback: extracted {len(cleaned_text)} "
            f"chars from {image_path}"
        )
        return cleaned_text

    except FileNotFoundError:
        print(f"CLUB Feedback: file not found — {image_path}")
        return ""
    except Exception as error:
        print(
            f"CLUB Feedback: OCR failed — {error}. "
            "Make sure Tesseract is installed: "
            "sudo apt install tesseract-ocr"
        )
        return ""


# ── LLM Interface ────────────────────────────────────────

def _call_ollama(system_prompt: str, user_text: str) -> str:
    """
    Sends a prompt to the local ollama LLM with retry logic.

    Args:
        system_prompt: Instructions for the LLM.
        user_text:     Content to analyze.

    Returns:
        Raw response text, or empty string on failure.
    """
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = ollama.chat(
                model=MODEL_NAME,
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
                "CLUB Feedback: ollama is not running. "
                "Start it with: ollama serve"
            )
            return ""

        except Exception as error:
            is_last_attempt = attempt == MAX_RETRIES
            if is_last_attempt:
                print(
                    f"CLUB Feedback: failed after "
                    f"{MAX_RETRIES} attempts — {error}"
                )
                return ""

            print(
                f"CLUB Feedback: attempt {attempt} failed, "
                f"retrying in {RETRY_DELAY_SECONDS}s..."
            )
            time.sleep(RETRY_DELAY_SECONDS)

    return ""


# ── Helpers ───────────────────────────────────────────────

def _parse_json_response(raw_text: str) -> dict | None:
    """
    Extracts and parses a JSON object from the LLM response.

    Handles markdown fences and surrounding commentary.

    Args:
        raw_text: Raw LLM output that should contain JSON.

    Returns:
        Parsed dict if successful, None otherwise.
    """
    cleaned = raw_text.strip()

    # Strip markdown code fences
    fence_pattern = r"```(?:json)?\s*\n?(.*?)\n?\s*```"
    fence_match = re.search(fence_pattern, cleaned, re.DOTALL)
    if fence_match:
        cleaned = fence_match.group(1).strip()

    # Find JSON object between first { and last }
    brace_start = cleaned.find("{")
    brace_end = cleaned.rfind("}")

    if brace_start != -1 and brace_end != -1:
        cleaned = cleaned[brace_start:brace_end + 1]

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as error:
        print(f"CLUB Feedback: JSON parse error — {error}")
        return None

    if not isinstance(parsed, dict):
        print("CLUB Feedback: expected JSON object, got other")
        return None

    return parsed


def _validate_result(parsed: dict) -> dict:
    """
    Ensures the parsed dict has all expected keys with correct types.

    Args:
        parsed: Raw dict from JSON parsing.

    Returns:
        A dict with guaranteed keys and types.
    """
    # Parse marks — can be int, float, or None
    total_marks = parsed.get("total_marks")
    lost_marks = parsed.get("lost_marks")

    # Coerce to int if numeric, otherwise None
    if isinstance(total_marks, (int, float)):
        total_marks = int(total_marks)
    else:
        total_marks = None

    if isinstance(lost_marks, (int, float)):
        lost_marks = int(lost_marks)
    else:
        lost_marks = None

    # Ensure lists are lists of strings
    raw_weak = parsed.get("weak_areas", [])
    if not isinstance(raw_weak, list):
        raw_weak = [str(raw_weak)]
    weak_areas = [str(item).strip() for item in raw_weak if item]

    raw_tips = parsed.get("improvement_tips", [])
    if not isinstance(raw_tips, list):
        raw_tips = [str(raw_tips)]
    improvement_tips = [
        str(item).strip() for item in raw_tips if item
    ]

    return {
        "total_marks": total_marks,
        "lost_marks": lost_marks,
        "weak_areas": weak_areas,
        "improvement_tips": improvement_tips,
    }


def _empty_result() -> dict:
    """
    Returns an empty result dict as a fallback.

    Returns:
        Dict with all expected keys but empty/None values.
    """
    return {
        "total_marks": None,
        "lost_marks": None,
        "weak_areas": [],
        "improvement_tips": [],
    }


# ── Entry Point (Test) ────────────────────────────────────

if __name__ == "__main__":
    import sys

    print("CLUB Feedback: running test...\n")

    if len(sys.argv) < 2:
        print(
            "Usage: python -m knowmyschool.feedback "
            "<exam_image_path> [subject]\n"
            "Example: python -m knowmyschool.feedback "
            "folder/images/dsa_midterm.jpg DSA"
        )
        sys.exit(1)

    test_image_path = sys.argv[1]
    test_subject = sys.argv[2] if len(sys.argv) > 2 else ""

    print(f"Analyzing: {test_image_path}")
    if test_subject:
        print(f"Subject:   {test_subject}")
    print()

    analysis = analyze_exam_paper(test_image_path, test_subject)

    print("── Analysis Results ──────────────────────\n")
    print(f"  Total marks: {analysis['total_marks']}")
    print(f"  Lost marks:  {analysis['lost_marks']}")

    print("\n  Weak areas:")
    for area in analysis["weak_areas"]:
        print(f"    ✗ {area}")

    print("\n  Improvement tips:")
    for tip in analysis["improvement_tips"]:
        print(f"    → {tip}")

    print("\nCLUB Feedback: test complete.")
