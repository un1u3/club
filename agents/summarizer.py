"""
club/agents/summarizer.py

Condenses raw study material into structured, exam-focused summaries.
Uses the local LLM to produce clean markdown with key concepts.

Example:
    >>> from agents.summarizer import summarize
    >>> result = summarize("Binary trees are...", subject="DSA")
    >>> print(result[:100])

Author: CLUB Project
License: MIT
"""
