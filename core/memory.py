"""
club/core/memory.py

Manages the local vector memory store using ChromaDB.
Stores and retrieves study material embeddings for context-aware responses.

Example:
    >>> from core.memory import store_document, search_memory
    >>> store_document("Binary trees are hierarchical...", subject="DSA")
    >>> results = search_memory("What is a binary tree?")

Author: CLUB Project
License: MIT
"""
