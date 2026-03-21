"""
club/core/memory.py

Manages the local vector memory store using ChromaDB.
Stores and retrieves study material embeddings so agents can
provide context-aware responses based on previously seen content.

Example:
    >>> from core.memory import add_document, search
    >>> add_document("lec1", "Binary trees are...", {"subject": "DSA"})
    >>> results = search("What is a binary tree?")
    >>> print(results[0][:50])

Author: CLUB Project
License: MIT
"""

import os
import time

import chromadb

# ── Constants ─────────────────────────────────────────────

# Where chromadb persists its data on disk
MEMORY_DIRECTORY = os.path.join("folder", ".club_memory")

# Name of the single collection that holds all study material
COLLECTION_NAME = "study_materials"

CHUNK_SIZE = 500          # characters per text chunk
CHUNK_OVERLAP = 50        # overlap to preserve sentence context

# ── Setup ─────────────────────────────────────────────────

def _get_collection() -> chromadb.Collection:
    """
    Returns the study_materials ChromaDB collection.

    Creates the persistent client and collection on first
    call. Subsequent calls return the same collection
    (ChromaDB's get_or_create_collection is idempotent).

    Returns:
        A chromadb.Collection backed by persistent storage
        in folder/.club_memory/.
    """
    client = chromadb.PersistentClient(path=MEMORY_DIRECTORY)
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME
    )
    return collection


# ── Core Functions ────────────────────────────────────────

def add_document(
    doc_id: str,
    text: str,
    metadata: dict,
) -> int:
    """
    Chunks text and stores it in the vector database.

    Splits the input text into overlapping chunks of
    CHUNK_SIZE characters, then upserts each chunk into
    ChromaDB with tracking metadata.

    Args:
        doc_id:   Unique identifier for the source document
                  (e.g. "os_chapter5.pdf"). Used to group
                  and later delete related chunks.
        text:     Full extracted text from the document.
        metadata: Extra info to attach to every chunk.
                  Typically includes {"subject": "DSA"} or
                  {"source": "lecture"}.

    Returns:
        The number of chunks stored.

    Example:
        >>> count = add_document(
        ...     "lec1.pdf",
        ...     "Binary trees are hierarchical...",
        ...     {"subject": "DSA"}
        ... )
        >>> print(f"Stored {count} chunks")
    """
    if not text.strip():
        print(f"CLUB Memory: skipping empty document {doc_id}")
        return 0

    chunks = _split_into_chunks(text)
    collection = _get_collection()

    # Unix timestamp for when this document was ingested
    ingestion_timestamp = str(int(time.time()))

    chunk_ids = []
    chunk_documents = []
    chunk_metadatas = []

    for chunk_index, chunk_text in enumerate(chunks):
        # Each chunk gets a unique ID: "doc_id__chunk_0"
        chunk_id = f"{doc_id}__chunk_{chunk_index}"

        # Merge user metadata with tracking fields
        chunk_metadata = {
            **metadata,
            "doc_id": doc_id,
            "chunk_index": chunk_index,
            "total_chunks": len(chunks),
            "timestamp": ingestion_timestamp,
        }

        chunk_ids.append(chunk_id)
        chunk_documents.append(chunk_text)
        chunk_metadatas.append(chunk_metadata)

    # Upsert so re-adding the same doc overwrites old chunks
    collection.upsert(
        ids=chunk_ids,
        documents=chunk_documents,
        metadatas=chunk_metadatas,
    )

    print(
        f"CLUB Memory: stored {len(chunks)} chunks "
        f"for '{doc_id}'"
    )
    return len(chunks)


def search(query: str, n_results: int = 5) -> list[str]:
    """
    Finds the most relevant text chunks for a query.

    Uses ChromaDB's default embedding function to perform
    similarity search against all stored study material.

    Args:
        query:     The search query (e.g. "explain BFS").
        n_results: Maximum number of chunks to return.
                   Defaults to 5.

    Returns:
        A list of text strings, ordered by relevance
        (most relevant first). Returns an empty list if
        the collection is empty or the query fails.

    Example:
        >>> results = search("What is a binary tree?", n_results=3)
        >>> for chunk in results:
        ...     print(chunk[:80])
    """
    collection = _get_collection()

    # Guard against querying an empty collection
    if collection.count() == 0:
        print("CLUB Memory: no documents stored yet")
        return []

    try:
        results = collection.query(
            query_texts=[query],
            n_results=min(n_results, collection.count()),
        )
    except Exception as error:
        print(f"CLUB Memory: search failed — {error}")
        return []

    # results["documents"] is a list of lists (one per query)
    # We only send one query, so we take index 0
    matched_documents = results.get("documents", [[]])[0]
    return matched_documents


def clear_document(doc_id: str) -> int:
    """
    Removes all chunks belonging to a specific document.

    Queries ChromaDB for all chunk IDs that match the given
    doc_id and deletes them in one batch.

    Args:
        doc_id: The document identifier used when the
                document was originally added.

    Returns:
        The number of chunks deleted.

    Example:
        >>> removed = clear_document("lec1.pdf")
        >>> print(f"Removed {removed} chunks")
    """
    collection = _get_collection()

    # Find all chunks that belong to this document
    try:
        existing = collection.get(
            where={"doc_id": doc_id}
        )
    except Exception as error:
        print(
            f"CLUB Memory: failed to find chunks "
            f"for '{doc_id}' — {error}"
        )
        return 0

    chunk_ids = existing.get("ids", [])

    if not chunk_ids:
        print(f"CLUB Memory: no chunks found for '{doc_id}'")
        return 0

    collection.delete(ids=chunk_ids)

    print(
        f"CLUB Memory: removed {len(chunk_ids)} chunks "
        f"for '{doc_id}'"
    )
    return len(chunk_ids)


# ── Helpers ───────────────────────────────────────────────

def _split_into_chunks(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    """
    Splits text into overlapping chunks of fixed character length.

    Overlap prevents cutting mid-sentence — the end of one
    chunk repeats at the start of the next so the embedding
    model sees complete context around boundaries.

    Args:
        text:       The full text to split.
        chunk_size: Maximum characters per chunk.
        overlap:    Characters shared between consecutive chunks.

    Returns:
        A list of text chunk strings.

    Example:
        >>> chunks = _split_into_chunks("a" * 1050, 500, 50)
        >>> len(chunks)
        3
    """
    chunks = []
    start_position = 0
    text_length = len(text)

    while start_position < text_length:
        end_position = start_position + chunk_size
        chunk = text[start_position:end_position]
        chunks.append(chunk)

        # Move forward by (chunk_size - overlap) so the next
        # chunk overlaps with the tail of the current one
        start_position += chunk_size - overlap

    return chunks


# ── Entry Point (Test) ────────────────────────────────────

if __name__ == "__main__":
    print("CLUB Memory: running self-test...\n")

    sample_text = (
        "A binary tree is a hierarchical data structure "
        "where each node has at most two children, referred "
        "to as left child and right child. Binary trees are "
        "used in many algorithms including binary search "
        "trees, heaps, and expression trees. Traversal "
        "methods include inorder, preorder, and postorder."
    )

    # Test 1: Add a document
    stored_count = add_document(
        doc_id="test_doc",
        text=sample_text,
        metadata={"subject": "DSA", "source": "self_test"},
    )
    print(f"  ✓ Stored {stored_count} chunk(s)\n")

    # Test 2: Search for relevant content
    results = search("What is a binary tree?", n_results=2)
    print(f"  ✓ Search returned {len(results)} result(s)")
    if results:
        preview_length = min(80, len(results[0]))
        print(f"    Preview: {results[0][:preview_length]}...\n")

    # Test 3: Clear the document
    removed_count = clear_document("test_doc")
    print(f"  ✓ Removed {removed_count} chunk(s)\n")

    # Verify it's gone
    remaining = search("binary tree", n_results=1)
    is_cleared = len(remaining) == 0
    print(f"  ✓ Collection empty after clear: {is_cleared}")

    print("\nCLUB Memory: self-test complete.")
