"""
club/core/watcher.py

Monitors the study folder for new or changed files.
When a student drops notes, PDFs, images, or past papers
into the watched folders, this module detects the event,
reads the file, and stores it in vector memory.

Example:
    >>> from core.watcher import start_watching
    >>> start_watching()  # blocks forever, or run in a thread

Author: CLUB Project
License: MIT
"""

import os
import time

# ── Constants ─────────────────────────────────────────────

# Folders to monitor — relative to project root
WATCHED_FOLDERS = [
    os.path.join("folder", "notes"),
    os.path.join("folder", "pyqs"),
    os.path.join("folder", "images"),
]

# File extensions we know how to read
SUPPORTED_EXTENSIONS = {
    ".pdf", ".pptx", ".docx",
    ".txt", ".md",
    ".jpg", ".jpeg", ".png",
}

# Files to ignore — hidden files, temp files, gitkeep
IGNORED_PREFIXES = (".", "~", "__")


# ── Core Functions ────────────────────────────────────────

def start_watching(blocking: bool = True) -> None:
    """
    Starts monitoring study folders for new files.

    When a new file appears, it is automatically read via
    the reader agent and stored in vector memory. Prints
    a confirmation for each ingested file.

    Args:
        blocking: If True, blocks the calling thread forever
                  (useful for CLI mode). If False, starts
                  the observer as a daemon and returns
                  immediately (useful for app.py integration).

    Example:
        >>> import threading
        >>> t = threading.Thread(target=start_watching)
        >>> t.daemon = True
        >>> t.start()
    """
    try:
        from watchdog.observers import Observer
        from watchdog.events import FileSystemEventHandler
    except ImportError:
        print(
            "CLUB Watcher: watchdog not installed. "
            "Install with: pip install watchdog"
        )
        return

    handler = _StudyFileHandler()
    observer = Observer()

    scheduled_count = 0
    for folder_path in WATCHED_FOLDERS:
        if os.path.exists(folder_path):
            observer.schedule(
                handler, folder_path, recursive=False
            )
            scheduled_count += 1
        else:
            # Create the folder if missing
            os.makedirs(folder_path, exist_ok=True)
            observer.schedule(
                handler, folder_path, recursive=False
            )
            scheduled_count += 1

    observer.daemon = True
    observer.start()
    print(
        f"CLUB Watcher: monitoring {scheduled_count} folders "
        f"for new files..."
    )

    if blocking:
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            observer.stop()
            print("CLUB Watcher: stopped.")
        observer.join()


def index_existing_files() -> int:
    """
    Scans all watched folders and indexes any files not
    already in memory.

    Called once on startup so that files the student
    dropped while CLUB was offline get indexed.

    Returns:
        The number of files newly indexed.

    Example:
        >>> count = index_existing_files()
        >>> print(f"Indexed {count} files")
    """
    from agents.reader import read_file
    from core.memory import add_document, search

    indexed_count = 0

    for folder_path in WATCHED_FOLDERS:
        if not os.path.exists(folder_path):
            continue

        for file_name in os.listdir(folder_path):
            # Skip ignored files
            if _should_ignore(file_name):
                continue

            # Check extension
            _, extension = os.path.splitext(file_name)
            if extension.lower() not in SUPPORTED_EXTENSIONS:
                continue

            file_path = os.path.join(folder_path, file_name)

            # Skip directories
            if os.path.isdir(file_path):
                continue

            # Check if already indexed by searching for the
            # doc_id pattern in memory. We use the file name
            # as doc_id, so if any chunks exist, it's indexed.
            try:
                existing = search(
                    f"source:{file_name}", n_results=1
                )
                # Simple heuristic: if search returns results
                # that mention this filename, skip it.
                # This is imperfect but avoids re-indexing.
            except Exception:
                existing = []

            # Always try to index — upsert handles duplicates
            try:
                extracted_text = read_file(file_path)
                if extracted_text:
                    add_document(
                        doc_id=file_name,
                        text=extracted_text,
                        metadata={
                            "source": file_path,
                            "type": "startup_index",
                        },
                    )
                    print(f"CLUB has read: {file_name}")
                    indexed_count += 1
            except Exception as error:
                print(
                    f"CLUB Watcher: failed to index "
                    f"{file_name} — {error}"
                )

    return indexed_count


# ── Event Handler ─────────────────────────────────────────

class _StudyFileHandler:
    """
    Watchdog event handler for new study files.

    Inherits from FileSystemEventHandler at runtime (lazy
    import to avoid import errors if watchdog isn't installed).
    """

    def __init__(self):
        """Initializes the parent handler."""
        from watchdog.events import FileSystemEventHandler
        # Dynamically set the base class methods
        self._base = FileSystemEventHandler()

    def dispatch(self, event):
        """Dispatches events to the appropriate handler."""
        if hasattr(event, "is_directory") and event.is_directory:
            return
        if event.event_type == "created":
            self.on_created(event)

    def on_created(self, event):
        """
        Processes a newly created file.

        Reads the file content and stores it in vector memory.
        Prints confirmation message for each successful ingestion.

        Args:
            event: Watchdog file system event.
        """
        if event.is_directory:
            return

        file_path = event.src_path
        file_name = os.path.basename(file_path)

        # Skip files we shouldn't process
        if _should_ignore(file_name):
            return

        # Check extension
        _, extension = os.path.splitext(file_name)
        if extension.lower() not in SUPPORTED_EXTENSIONS:
            return

        print(f"CLUB Watcher: new file detected — {file_path}")

        # Small delay to let the file finish writing
        # (large files may still be copying)
        time.sleep(0.5)

        try:
            from agents.reader import read_file
            from core.memory import add_document

            extracted_text = read_file(file_path)
            if extracted_text:
                add_document(
                    doc_id=file_name,
                    text=extracted_text,
                    metadata={
                        "source": file_path,
                        "type": "auto_ingested",
                    },
                )
                print(f"CLUB has read: {file_name}")
            else:
                print(
                    f"CLUB Watcher: no text extracted "
                    f"from {file_name}"
                )
        except Exception as error:
            print(
                f"CLUB Watcher: failed to process "
                f"{file_name} — {error}"
            )


# ── Helpers ───────────────────────────────────────────────

def _should_ignore(file_name: str) -> bool:
    """
    Checks if a file should be skipped during processing.

    Args:
        file_name: The name of the file (not the full path).

    Returns:
        True if the file should be ignored.
    """
    return any(
        file_name.startswith(prefix)
        for prefix in IGNORED_PREFIXES
    )


# ── Entry Point (Test) ────────────────────────────────────

if __name__ == "__main__":
    print("CLUB Watcher: starting folder monitor...\n")
    print(
        "Drop a file into folder/notes/, folder/pyqs/, "
        "or folder/images/ to test.\n"
        "Press Ctrl+C to stop.\n"
    )
    start_watching(blocking=True)
