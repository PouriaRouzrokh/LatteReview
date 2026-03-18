"""CheckpointManager — atomic per-item saves, resume detection, DataFrame snapshots."""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from lattereview.agentic.checkpoint.state import RunState, compute_schema_hash


class CheckpointManager:
    """Manages checkpoint state for an AgenticWorkflow run.

    Provides atomic per-item result saves (write-to-temp-then-rename),
    DataFrame snapshots after each round, and resume detection.

    Args:
        working_dir: Root directory for all checkpoint state.
    """

    def __init__(self, working_dir: Path) -> None:
        self._dir = working_dir
        self._state: Optional[RunState] = None

    @property
    def working_dir(self) -> Path:
        return self._dir

    @property
    def state(self) -> Optional[RunState]:
        return self._state

    @property
    def metadata_path(self) -> Path:
        return self._dir / "run_metadata.json"

    def _results_dir(self, round_id: str, reviewer_name: str) -> Path:
        return self._dir / f"round_{round_id}" / f"agent_{reviewer_name}" / "results"

    def _logs_dir(self, round_id: str, reviewer_name: str) -> Path:
        return self._dir / f"round_{round_id}" / f"agent_{reviewer_name}" / "logs"

    def _output_dir(self) -> Path:
        return self._dir / "output"

    # ------------------------------------------------------------------
    # Initialization / Resume
    # ------------------------------------------------------------------

    def initialize(self, workflow_schema: List[Dict[str, Any]]) -> RunState:
        """Create a fresh RunState for a new run.

        Args:
            workflow_schema: The workflow schema to hash.

        Returns:
            New RunState instance.
        """
        self._dir.mkdir(parents=True, exist_ok=True)
        self._state = RunState(schema_hash=compute_schema_hash(workflow_schema))
        self.save_run_state()
        return self._state

    def can_resume(self) -> bool:
        """Check whether a prior run exists that can be resumed."""
        return self.metadata_path.exists()

    def load_for_resume(self, workflow_schema: List[Dict[str, Any]]) -> RunState:
        """Load existing RunState for resume, validating the schema hash.

        Warns if the schema hash differs from the prior run. Returns the
        loaded state regardless, so callers can decide whether to proceed.

        Args:
            workflow_schema: Current workflow schema to compare against.

        Returns:
            The loaded RunState.

        Raises:
            FileNotFoundError: If no prior run_metadata.json exists.
        """
        if not self.metadata_path.exists():
            raise FileNotFoundError(f"No prior run found at {self.metadata_path}")

        self._state = self._load_run_state()
        current_hash = compute_schema_hash(workflow_schema)

        if self._state.schema_hash != current_hash:
            warnings.warn(
                f"Workflow schema has changed since the prior run "
                f"(prior: {self._state.schema_hash}, current: {current_hash}). "
                f"Results from the prior run may not be compatible.",
                UserWarning,
                stacklevel=2,
            )

        self._state.status = "running"
        self._state.touch()
        self.save_run_state()
        return self._state

    # ------------------------------------------------------------------
    # Per-Item Results
    # ------------------------------------------------------------------

    def save_item_result(
        self,
        round_id: str,
        reviewer_name: str,
        item_id: str,
        result: Dict[str, Any],
        cost: float = 0.0,
    ) -> None:
        """Atomically save a single item result to disk and update RunState.

        Uses write-to-temp-then-rename for atomicity.

        Args:
            round_id: Round identifier.
            reviewer_name: Reviewer name.
            item_id: Item identifier.
            result: The result dict to save.
            cost: Cost for this item.
        """
        results_dir = self._results_dir(round_id, reviewer_name)
        results_dir.mkdir(parents=True, exist_ok=True)

        # Sanitize item_id for filename (replace slashes, etc.)
        safe_id = item_id.replace("/", "_").replace("\\", "_")
        result_path = results_dir / f"item_{safe_id}.json"
        tmp_path = result_path.with_suffix(".tmp")

        payload = {"item_id": item_id, "result": result, "cost": cost}
        tmp_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        tmp_path.rename(result_path)

        # Update state
        if self._state is not None:
            self._state.mark_item_completed(round_id, reviewer_name, item_id)
            self._state.total_cost += cost
            self.save_run_state()

    def load_item_result(self, round_id: str, reviewer_name: str, item_id: str) -> Optional[Dict[str, Any]]:
        """Load a previously saved item result.

        Returns:
            The saved payload dict (with item_id, result, cost), or None if not found.
        """
        safe_id = item_id.replace("/", "_").replace("\\", "_")
        result_path = self._results_dir(round_id, reviewer_name) / f"item_{safe_id}.json"
        if not result_path.exists():
            return None
        return json.loads(result_path.read_text(encoding="utf-8"))

    def get_completed_items(self, round_id: str, reviewer_name: str) -> List[str]:
        """Return item IDs that have been completed for a reviewer/round pair.

        Checks RunState first, then falls back to scanning result files on disk.
        """
        if self._state is not None:
            state_items = self._state.get_completed_items(round_id, reviewer_name)
            if state_items:
                return state_items

        # Fallback: scan result files
        results_dir = self._results_dir(round_id, reviewer_name)
        if not results_dir.exists():
            return []

        completed = []
        for f in results_dir.glob("item_*.json"):
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                completed.append(data["item_id"])
            except (json.JSONDecodeError, KeyError):
                continue
        return completed

    # ------------------------------------------------------------------
    # DataFrame Snapshots
    # ------------------------------------------------------------------

    def save_dataframe_snapshot(self, df: pd.DataFrame, round_id: str) -> Path:
        """Save a DataFrame snapshot after a round completes.

        Args:
            df: The current DataFrame state.
            round_id: Round identifier.

        Returns:
            Path to the saved parquet file.
        """
        output_dir = self._output_dir()
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / f"after_round_{round_id}.parquet"
        tmp_path = path.with_suffix(".tmp")
        df.to_parquet(tmp_path)
        tmp_path.rename(path)
        return path

    def save_final_dataframe(self, df: pd.DataFrame) -> Path:
        """Save the final DataFrame after the workflow completes.

        Returns:
            Path to the saved parquet file.
        """
        output_dir = self._output_dir()
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / "final.parquet"
        tmp_path = path.with_suffix(".tmp")
        df.to_parquet(tmp_path)
        tmp_path.rename(path)
        return path

    def load_dataframe_snapshot(self, round_id: str) -> Optional[pd.DataFrame]:
        """Load a previously saved DataFrame snapshot.

        Returns:
            The saved DataFrame, or None if not found.
        """
        path = self._output_dir() / f"after_round_{round_id}.parquet"
        if not path.exists():
            return None
        return pd.read_parquet(path)

    # ------------------------------------------------------------------
    # RunState Persistence
    # ------------------------------------------------------------------

    def save_run_state(self) -> None:
        """Persist the current RunState to disk atomically."""
        if self._state is None:
            return
        self._dir.mkdir(parents=True, exist_ok=True)
        tmp_path = self.metadata_path.with_suffix(".tmp")
        tmp_path.write_text(self._state.model_dump_json(indent=2), encoding="utf-8")
        tmp_path.rename(self.metadata_path)

    def _load_run_state(self) -> RunState:
        """Load RunState from disk."""
        data = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        return RunState.model_validate(data)

    def mark_completed(self) -> None:
        """Mark the run as completed."""
        if self._state is not None:
            self._state.status = "completed"
            self._state.touch()
            self.save_run_state()

    def mark_failed(self) -> None:
        """Mark the run as failed."""
        if self._state is not None:
            self._state.status = "failed"
            self._state.touch()
            self.save_run_state()
