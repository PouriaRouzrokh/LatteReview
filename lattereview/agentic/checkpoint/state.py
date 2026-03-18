"""RunState — serializable run progress for checkpoint/resume."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any, Dict, List

from pydantic import BaseModel, Field


class RunState(BaseModel):
    """Serializable representation of a workflow run's progress.

    Tracks which items have been completed, where the run is currently
    positioned, and whether the configuration has changed since the run
    started (via ``schema_hash``).
    """

    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    status: str = "running"  # running | completed | paused | failed
    schema_hash: str = ""
    current_round_index: int = 0
    current_reviewer_index: int = 0
    completed_items: Dict[str, List[str]] = Field(default_factory=dict)
    total_cost: float = 0.0

    def touch(self) -> None:
        """Update the ``updated_at`` timestamp."""
        self.updated_at = datetime.now(timezone.utc).isoformat()

    def mark_item_completed(self, round_id: str, reviewer_name: str, item_id: str) -> None:
        """Record that a specific item has been completed."""
        key = f"{round_id}_{reviewer_name}"
        if key not in self.completed_items:
            self.completed_items[key] = []
        if item_id not in self.completed_items[key]:
            self.completed_items[key].append(item_id)
        self.touch()

    def is_item_completed(self, round_id: str, reviewer_name: str, item_id: str) -> bool:
        """Check whether a specific item has already been completed."""
        key = f"{round_id}_{reviewer_name}"
        return item_id in self.completed_items.get(key, [])

    def get_completed_items(self, round_id: str, reviewer_name: str) -> List[str]:
        """Return the list of completed item IDs for a reviewer/round pair."""
        key = f"{round_id}_{reviewer_name}"
        return list(self.completed_items.get(key, []))


def compute_schema_hash(workflow_schema: List[Dict[str, Any]]) -> str:
    """Compute a deterministic hash of the workflow schema for change detection.

    The hash captures round IDs, reviewer names, text_inputs, and output types
    so that configuration changes between runs can be detected on resume.
    """
    parts: List[str] = []
    for task in workflow_schema:
        round_id = task.get("round", "")
        reviewers = task.get("reviewers", [])
        if not isinstance(reviewers, list):
            reviewers = [reviewers]
        text_inputs = task.get("text_inputs", [])
        if isinstance(text_inputs, str):
            text_inputs = [text_inputs]

        reviewer_sigs = []
        for r in reviewers:
            sig = f"{r.name}|{r.output_type.__name__}|{r.max_iterations}"
            reviewer_sigs.append(sig)

        parts.append(f"round={round_id};reviewers={','.join(reviewer_sigs)};inputs={','.join(text_inputs)}")

    canonical = "\n".join(parts)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
