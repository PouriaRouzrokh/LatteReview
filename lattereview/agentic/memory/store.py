"""MemoryStore — file-backed per-agent memory with index management."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Dict, List, Optional

from lattereview.agentic.memory.index import MemoryIndex


class MemoryStore:
    """File-backed memory store for a single agent within a single round.

    Memories are stored as individual markdown files (mem_001.md, mem_002.md, ...)
    in a directory, with a companion _index.json tracking titles and briefs.

    Thread-safe via asyncio.Lock on write operations, shared with the
    underlying MemoryIndex.

    Args:
        memory_dir: Directory for memory files and _index.json.
        max_memories: Maximum number of memories allowed (default 50).
    """

    def __init__(self, memory_dir: Path, max_memories: int = 50) -> None:
        self._dir = memory_dir
        self._max_memories = max_memories
        self._index = MemoryIndex(memory_dir / "_index.json")
        self._lock = asyncio.Lock()

    @property
    def memory_dir(self) -> Path:
        return self._dir

    @property
    def max_memories(self) -> int:
        return self._max_memories

    @property
    def index(self) -> MemoryIndex:
        return self._index

    async def initialize(self) -> None:
        """Create directory structure and load existing index."""
        self._dir.mkdir(parents=True, exist_ok=True)
        await self._index.load()

    async def save(self, title: str, brief: str, content: str) -> str:
        """Save a new memory. Returns the assigned memory ID.

        If max_memories is reached, returns an error message string
        starting with "ERROR:" instead of saving.

        Args:
            title: Short title for the memory.
            brief: One-line summary for the index.
            content: Full memory content (markdown).

        Returns:
            Memory ID (e.g., "mem_001") on success, or error string.
        """
        async with self._lock:
            if not self._index._loaded:
                await self._index.load()

            if len(self._index) >= self._max_memories:
                return (
                    f"ERROR: Maximum memory limit ({self._max_memories}) reached. "
                    f"Delete old memories before saving new ones."
                )

            memory_id = self._index.get_next_id()

            # Write memory file atomically
            self._dir.mkdir(parents=True, exist_ok=True)
            mem_path = self._dir / f"{memory_id}.md"
            tmp_path = mem_path.with_suffix(".tmp")
            tmp_path.write_text(content, encoding="utf-8")
            tmp_path.rename(mem_path)

            # Update index (index.add acquires its own lock, but we already hold
            # _lock, so call the internal method directly)
            self._index._entries = [e for e in self._index._entries if e["id"] != memory_id]
            self._index._entries.append({"id": memory_id, "title": title, "brief": brief})
            await self._index.save()

            return memory_id

    async def load(self, memory_id: str) -> Optional[str]:
        """Load the full content of a memory by ID.

        Args:
            memory_id: The memory ID (e.g., "mem_001").

        Returns:
            Memory content string, or None if not found.
        """
        mem_path = self._dir / f"{memory_id}.md"
        if not mem_path.exists():
            return None
        return mem_path.read_text(encoding="utf-8")

    async def load_multiple(self, memory_ids: List[str]) -> Dict[str, Optional[str]]:
        """Load multiple memories by ID.

        Args:
            memory_ids: List of memory IDs to load.

        Returns:
            Dict mapping memory_id -> content (None if not found).
        """
        results = {}
        for mid in memory_ids:
            results[mid] = await self.load(mid)
        return results

    async def delete(self, memory_id: str) -> bool:
        """Delete a memory by ID. Returns True if found and deleted.

        Args:
            memory_id: The memory ID to delete.

        Returns:
            True if deleted (file removed and/or index entry removed), False if not found.
        """
        async with self._lock:
            file_deleted = False
            mem_path = self._dir / f"{memory_id}.md"
            if mem_path.exists():
                mem_path.unlink()
                file_deleted = True

            # Remove from index (call internal to avoid double-lock)
            if not self._index._loaded:
                await self._index.load()
            before = len(self._index._entries)
            self._index._entries = [e for e in self._index._entries if e["id"] != memory_id]
            index_removed = len(self._index._entries) < before
            if index_removed:
                await self._index.save()

            return file_deleted or index_removed

    async def list_memories(self) -> List[Dict[str, str]]:
        """List all memory entries (id, title, brief).

        Returns:
            List of dicts with id, title, and brief fields.
        """
        async with self._lock:
            if not self._index._loaded:
                await self._index.load()
            return self._index.entries

    async def get_summaries(self) -> List[Dict[str, str]]:
        """Get memory summaries for system prompt injection.

        Returns:
            List of {"id": ..., "brief": ...} dicts.
        """
        return await self._index.get_summaries()

    async def count(self) -> int:
        """Return the number of stored memories."""
        async with self._lock:
            if not self._index._loaded:
                await self._index.load()
            return len(self._index)
