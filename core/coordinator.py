"""
club/core/coordinator.py

Orchestrates the multi-agent pipeline using LangGraph.
Decides which agents to invoke based on student intent and input type.

Example:
    >>> from core.coordinator import run_pipeline
    >>> result = run_pipeline(user_message="Summarize my OS notes")

Author: CLUB Project
License: MIT
"""
