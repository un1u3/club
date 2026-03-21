"""
club/agents/reader.py

Extracts text from study materials (PDFs, images, handwritten notes).
Acts as the first stage in the pipeline — raw input to clean text.

Example:
    >>> from agents.reader import extract_text
    >>> text = extract_text("folder/notes/os_chapter5.pdf")
    >>> print(text[:200])

Author: CLUB Project
License: MIT
"""
