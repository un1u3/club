"""
club/core/constants.py

Central configuration constants for the CLUB study assistant.
All LLM and runtime settings live here so that changing the
model (or any other setting) requires editing exactly one file.

Usage:
    >>> from core.constants import OLLAMA_MODEL
    >>> response = ollama.chat(model=OLLAMA_MODEL, ...)

Author: CLUB Project
License: MIT
"""

# ── LLM Configuration ─────────────────────────────────────────
# Model name must match exactly what `ollama list` returns.
# Use the full tag form (e.g. "llama3:latest") to avoid
# ambiguity when multiple versions of the same model are present.
OLLAMA_MODEL = "llama3:latest"

# Base URL for the local ollama server
OLLAMA_BASE_URL = "http://localhost:11434"

# Seconds to wait for a single ollama response before giving up
OLLAMA_TIMEOUT = 30

# How many times to retry a failed ollama call
OLLAMA_RETRY_ATTEMPTS = 3

# Seconds to wait between retry attempts
OLLAMA_RETRY_DELAY = 2
