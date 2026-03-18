"""FlagStore — file-backed per-agent flag store for item revisiting."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Dict, List, Optional


class FlagStore:
    """File-backed flag store for a single agent within a single round.

    Flags are stored in a single ``flags.json`` file as a list of entries,
    each containing an ``item_id``, ``reason``, and ``resolved`` status.

    Thread-safe via asyncio.Lock on write operations.

    Args:
        flags_dir: Directory for the flags.json file.
    """

    def __init__(self, flags_dir: Path) -> None:
        self._dir = flags_dir
        self._path = flags_dir / "flags.json"
        self._lock = asyncio.Lock()
        self._entries: List[Dict] = []
        self._loaded = False

    @property
    def flags_dir(self) -> Path:
        return self._dir

    @property
    def path(self) -> Path:
        return self._path

    @property
    def entries(self) -> List[Dict]:
        return list(self._entries)

    async def initialize(self) -> None:
        """Create directory structure and load existing flags."""
        self._dir.mkdir(parents=True, exist_ok=True)
        await self._load()

    async def _load(self) -> None:
        """Load flags from disk. Creates empty list if file doesn't exist."""
        if self._path.exists():
            data = json.loads(self._path.read_text(encoding="utf-8"))
            self._entries = data.get("flags", [])
        else:
            self._entries = []
        self._loaded = True

    async def _save(self) -> None:
        """Persist flags to disk atomically (write-to-temp-then-rename)."""
        self._dir.mkdir(parents=True, exist_ok=True)
        tmp_path = self._path.with_suffix(".tmp")
        data = {"flags": self._entries}
        tmp_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        tmp_path.rename(self._path)

    async def add_flag(self, item_id: str, reason: str) -> str:
        """Flag an item for revisiting.

        If the item is already flagged (and unresolved), updates the reason.

        Args:
            item_id: The ID of the item to flag.
            reason: Why the item needs revisiting.

        Returns:
            Confirmation message string.
        """
        async with self._lock:
            if not self._loaded:
                await self._load()

            # Check for existing unresolved flag
            for entry in self._entries:
                if entry["item_id"] == item_id and not entry["resolved"]:
                    entry["reason"] = reason
                    await self._save()
                    return f"Updated flag for item '{item_id}'"

            self._entries.append({"item_id": item_id, "reason": reason, "resolved": False})
            await self._save()
            return f"Flagged item '{item_id}' for revisiting"

    async def resolve_flag(self, item_id: str) -> bool:
        """Mark a flag as resolved.

        Args:
            item_id: The ID of the flagged item.

        Returns:
            True if a flag was found and resolved, False if not found.
        """
        async with self._lock:
            if not self._loaded:
                await self._load()

            for entry in self._entries:
                if entry["item_id"] == item_id and not entry["resolved"]:
                    entry["resolved"] = True
                    await self._save()
                    return True
            return False

    async def get_unresolved(self) -> List[Dict]:
        """Get all unresolved flags.

        Returns:
            List of dicts with item_id and reason fields.
        """
        if not self._loaded:
            await self._load()
        return [{"item_id": e["item_id"], "reason": e["reason"]} for e in self._entries if not e["resolved"]]

    async def get_flag(self, item_id: str) -> Optional[Dict]:
        """Get the flag entry for a specific item.

        Returns the most recent unresolved flag, or None if not found.
        """
        if not self._loaded:
            await self._load()
        for entry in reversed(self._entries):
            if entry["item_id"] == item_id and not entry["resolved"]:
                return dict(entry)
        return None

    async def list_flags(self) -> List[Dict]:
        """List all flag entries (both resolved and unresolved).

        Returns:
            List of all flag dicts.
        """
        if not self._loaded:
            await self._load()
        return list(self._entries)

    async def count(self, unresolved_only: bool = False) -> int:
        """Return the number of flags.

        Args:
            unresolved_only: If True, count only unresolved flags.
        """
        if not self._loaded:
            await self._load()
        if unresolved_only:
            return sum(1 for e in self._entries if not e["resolved"])
        return len(self._entries)
