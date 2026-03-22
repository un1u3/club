"""
club/agents/summarizer.py

Condenses raw study material into structured, exam-focused summaries.
Uses the local ollama LLM (llama3) to produce clean markdown with
headers, key concepts, definitions, and formulas.

Example:
    >>> from agents.summarizer import summarize
    >>> result = summarize("Binary trees are...", subject="DSA")
    >>> print(result[:100])

Author: CLUB Project
License: MIT
"""

import time

import ollama

from core.constants import (
    OLLAMA_MODEL,
    OLLAMA_RETRY_ATTEMPTS,
    OLLAMA_RETRY_DELAY,
)

# ── Constants ─────────────────────────────────────────────

# Chunk size tuned to stay well within llama3's 4096 token
# context window — 2000 chars is roughly 500 tokens, leaving
# room for the system prompt and generated output
SUMMARIZER_CHUNK_SIZE = 2000

# Overlap so chunks don't cut mid-sentence
SUMMARIZER_CHUNK_OVERLAP = 200

MAX_RETRIES = OLLAMA_RETRY_ATTEMPTS
RETRY_DELAY_SECONDS = OLLAMA_RETRY_DELAY

# ── System Prompts ────────────────────────────────────────

BASE_SYSTEM_PROMPT = """You are CLUB, a study assistant AI.
Your job is to summarize study material for students.

Rules:
- Create a structured summary using markdown headers (##).
- Highlight key concepts, definitions, and formulas.
- Use bullet points for lists of related items.
- Bold important terms using **term**.
- Keep the summary concise: at most 30% of the input length.
- Do NOT add information that is not in the original text.
- Do NOT include greetings or filler — jump straight in."""

SUBJECT_PROMPT_TEMPLATE = """
Additionally, this material is for the subject: {subject}.
Tailor vocabulary, emphasis, and structure to what matters
most for {subject} exams."""

COMBINE_SYSTEM_PROMPT = """You are CLUB, a study assistant AI.
You will receive multiple partial summaries of the same
document. Combine them into one clean, unified summary.

Rules:
- Merge overlapping points — do not repeat information.
- Maintain markdown headers (##) and structure.
- Keep it concise and exam-focused.
- Preserve all key concepts, definitions, and formulas.
- Do NOT add information that is not in the partial summaries."""


# ── Core Function ─────────────────────────────────────────

def summarize(text: str, subject: str = "") -> str:
    """
    Condenses raw study material into a structured summary.

    Takes messy lecture notes or textbook excerpts and returns
    a clean, exam-focused summary with headers and key concepts.
    Handles long texts by splitting into chunks, summarizing
    each chunk individually, then combining the results.

    Args:
        text:    Raw extracted text from a study document.
                 Can be messy — this function handles cleanup.
        subject: Optional subject name (e.g. "DSA", "OS").
                 When provided, tailors the summary style
                 to that subject's exam focus.

    Returns:
        A markdown-formatted string with headers, key points,
        and important definitions. Max 30% of input length.
        Returns an empty string if summarization fails.

    Raises:
        Prints a warning if ollama is not running locally.

    Example:
        >>> summary = summarize("A binary tree is...", "DSA")
        >>> print(summary[:100])
        '## Binary Trees\\n\\nA binary tree is a hierarchical...'
    """
    if not text.strip():
        print("CLUB Summarizer: received empty text, skipping")
        return ""

    system_prompt = _build_system_prompt(subject)
    chunks = _split_for_summarizer(text)

    if len(chunks) == 1:
        # Short text — single-pass summarization
        return _call_ollama(system_prompt, chunks[0])

    # Long text — summarize each chunk, then combine
    partial_summaries = []
    for chunk_index, chunk in enumerate(chunks):
        print(
            f"CLUB Summarizer: processing chunk "
            f"{chunk_index + 1}/{len(chunks)}..."
        )
        partial = _call_ollama(system_prompt, chunk)
        if partial:
            partial_summaries.append(partial)

    if not partial_summaries:
        print("CLUB Summarizer: all chunks failed")
        return ""

    # If only one chunk succeeded, no need to combine
    if len(partial_summaries) == 1:
        return partial_summaries[0]

    # Merge partial summaries into one unified summary
    combined_input = "\n\n---\n\n".join(partial_summaries)
    return _call_ollama(COMBINE_SYSTEM_PROMPT, combined_input)


# ── LLM Interface ────────────────────────────────────────

def _call_ollama(system_prompt: str, user_text: str) -> str:
    """
    Sends a prompt to the local ollama LLM with retry logic.

    Tries up to MAX_RETRIES times, waiting RETRY_DELAY_SECONDS
    between attempts. This handles brief ollama hiccups like
    the model still loading into memory.

    Args:
        system_prompt: Instructions for the LLM's behavior.
        user_text:     The study material to summarize.

    Returns:
        The LLM's response text, or empty string if all
        retries are exhausted.
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
            # ollama server is not running
            print(
                "CLUB Summarizer: ollama is not running. "
                "Start it with: ollama serve"
            )
            return ""

        except Exception as error:
            is_last_attempt = attempt == MAX_RETRIES
            if is_last_attempt:
                print(
                    f"CLUB Summarizer: failed after "
                    f"{MAX_RETRIES} attempts — {error}"
                )
                return ""

            print(
                f"CLUB Summarizer: attempt {attempt} failed, "
                f"retrying in {RETRY_DELAY_SECONDS}s — {error}"
            )
            time.sleep(RETRY_DELAY_SECONDS)

    return ""


# ── Helpers ───────────────────────────────────────────────

def _build_system_prompt(subject: str) -> str:
    """
    Constructs the system prompt, optionally tailored to a subject.

    Args:
        subject: Subject name (e.g. "DSA"). Empty string
                 means no subject-specific tailoring.

    Returns:
        The complete system prompt string.
    """
    prompt = BASE_SYSTEM_PROMPT

    if subject.strip():
        prompt += SUBJECT_PROMPT_TEMPLATE.format(
            subject=subject.strip()
        )

    return prompt


def _split_for_summarizer(
    text: str,
    chunk_size: int = SUMMARIZER_CHUNK_SIZE,
    overlap: int = SUMMARIZER_CHUNK_OVERLAP,
) -> list[str]:
    """
    Splits text into overlapping chunks for the summarizer.

    Uses a larger chunk size than the memory module because
    the LLM needs enough context to produce coherent summaries.

    Args:
        text:       Full text to split.
        chunk_size: Max characters per chunk.
        overlap:    Characters shared between consecutive chunks.

    Returns:
        A list of text chunks. Returns the full text as a
        single-item list if it fits within chunk_size.
    """
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start_position = 0
    text_length = len(text)

    while start_position < text_length:
        end_position = start_position + chunk_size
        chunk = text[start_position:end_position]
        chunks.append(chunk)

        # Advance by (chunk_size - overlap) so chunks share
        # their tail/head for context continuity
        start_position += chunk_size - overlap

    return chunks


# ── Entry Point (Test) ────────────────────────────────────

if __name__ == "__main__":
    print("CLUB Summarizer: running test...\n")

    sample_text = (
        "A binary search tree (BST) is a binary tree where "
        "each node has a key, and for every node, all keys "
        "in the left subtree are smaller and all keys in the "
        "right subtree are larger. This property allows "
        "efficient searching, insertion, and deletion "
        "operations with an average time complexity of "
        "O(log n). However, in the worst case (a skewed "
        "tree), these operations degrade to O(n). To "
        "maintain balance, self-balancing BSTs like AVL "
        "trees and Red-Black trees are used. An AVL tree "
        "ensures that the height difference between left "
        "and right subtrees (balance factor) is at most 1. "
        "Red-Black trees use color properties and rotations "
        "to maintain approximate balance. Traversal methods "
        "include inorder (left, root, right), preorder "
        "(root, left, right), and postorder (left, right, "
        "root). Inorder traversal of a BST always gives "
        "keys in sorted order."
    )

    print(f"Input length: {len(sample_text)} chars\n")
    print("Sending to ollama (llama3)...\n")

    summary_result = summarize(sample_text, subject="DSA")

    if summary_result:
        print("── Summary ──────────────────────────────\n")
        print(summary_result)
        print(f"\n── Stats ────────────────────────────────")
        print(f"Output length: {len(summary_result)} chars")
        ratio = len(summary_result) / len(sample_text)
        print(f"Compression:   {ratio:.0%} of original")
    else:
        print("CLUB Summarizer: test failed — no output")
        print("Make sure ollama is running: ollama serve")
