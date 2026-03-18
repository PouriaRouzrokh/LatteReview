"""MemoryIndex — manages the _index.json file for per-agent memory summaries."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


class MemoryIndex:
    """In-memory index of memory summaries, backed by _index.json.

    Each entry has: id (str), title (str), brief (str).
    Thread-safe via asyncio.Lock on all write operations.
    """

    def __init__(self, index_path: Path) -> None:
        self._path = index_path
        self._lock = asyncio.Lock()
        self._entries: List[Dict[str, str]] = []
        self._loaded = False

    @property
    def path(self) -> Path:
        return self._path

    @property
    def entries(self) -> List[Dict[str, str]]:
        return list(self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    async def load(self) -> None:
        """Load the index from disk. Creates empty index if file doesn't exist."""
        if self._path.exists():
            data = json.loads(self._path.read_text(encoding="utf-8"))
            self._entries = data.get("memories", [])
        else:
            self._entries = []
        self._loaded = True

    async def save(self) -> None:
        """Persist the index to disk atomically (write-to-temp-then-rename)."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self._path.with_suffix(".tmp")
        data = {"memories": self._entries}
        tmp_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        tmp_path.rename(self._path)

    async def add(self, memory_id: str, title: str, brief: str) -> None:
        """Add a memory entry to the index and persist."""
        async with self._lock:
            if not self._loaded:
                await self.load()
            # Avoid duplicates
            self._entries = [e for e in self._entries if e["id"] != memory_id]
            self._entries.append({"id": memory_id, "title": title, "brief": brief})
            await self.save()

    async def remove(self, memory_id: str) -> bool:
        """Remove a memory entry by ID. Returns True if found and removed."""
        async with self._lock:
            if not self._loaded:
                await self.load()
            before = len(self._entries)
            self._entries = [e for e in self._entries if e["id"] != memory_id]
            if len(self._entries) < before:
                await self.save()
                return True
            return False

    async def get_summaries(self) -> List[Dict[str, str]]:
        """Return all memory summaries (id + brief) for system prompt injection."""
        if not self._loaded:
            await self.load()
        return [{"id": e["id"], "brief": e["brief"]} for e in self._entries]

    def get_next_id(self) -> str:
        """Generate the next memory ID (mem_001, mem_002, ...)."""
        if not self._entries:
            return "mem_001"

        max_num = 0
        for entry in self._entries:
            mid = entry["id"]
            if mid.startswith("mem_"):
                try:
                    num = int(mid.split("_", 1)[1])
                    max_num = max(max_num, num)
                except (ValueError, IndexError):
                    pass
        return f"mem_{max_num + 1:03d}"
