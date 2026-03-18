"""ActionLogger — per-item structured logging as JSONL."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


class ActionLogger:
    """Per-item JSONL logger for structured action tracking.

    Each item gets its own ``.jsonl`` file containing one JSON object per line.
    Entries include a timestamp, action type, and arbitrary details dict.

    Args:
        logs_dir: Directory for log files.
    """

    def __init__(self, logs_dir: Path) -> None:
        self._dir = logs_dir
        self._dir.mkdir(parents=True, exist_ok=True)

    @property
    def logs_dir(self) -> Path:
        return self._dir

    def _log_path(self, item_id: str) -> Path:
        safe_id = item_id.replace("/", "_").replace("\\", "_")
        return self._dir / f"item_{safe_id}.jsonl"

    def log(self, item_id: str, action_type: str, details: Optional[Dict[str, Any]] = None) -> None:
        """Append a log entry for an item.

        Args:
            item_id: The item identifier.
            action_type: Type of action (e.g., "review_start", "tool_call", "review_complete").
            details: Optional dict of additional details.
        """
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "action_type": action_type,
            "item_id": item_id,
        }
        if details:
            entry["details"] = details

        path = self._log_path(item_id)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, default=str) + "\n")

    def read_log(self, item_id: str) -> List[Dict[str, Any]]:
        """Read all log entries for an item.

        Returns:
            List of log entry dicts, empty if no log file exists.
        """
        path = self._log_path(item_id)
        if not path.exists():
            return []

        entries = []
        for line in path.read_text(encoding="utf-8").strip().split("\n"):
            if line:
                entries.append(json.loads(line))
        return entries

    def has_log(self, item_id: str) -> bool:
        """Check whether a log file exists for an item."""
        return self._log_path(item_id).exists()
