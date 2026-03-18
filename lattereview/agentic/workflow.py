"""AgenticWorkflow — multi-round review pipeline orchestrator."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import pandas as pd
import pydantic
from pydantic import Field

from lattereview.agentic.reviewer import AgenticReviewer


class AgenticWorkflowError(Exception):
    """Base exception for AgenticWorkflow errors."""

    pass


class AgenticWorkflow(pydantic.BaseModel):
    """Orchestrates multi-round review pipelines over DataFrames.

    Uses the same schema format as v1 ReviewWorkflow for familiarity:

        workflow_schema = [
            {
                "round": "A",
                "reviewers": [reviewer1, reviewer2],
                "text_inputs": ["Title", "Abstract"],
                "filter": lambda row: ...,  # optional
            }
        ]

    Column naming follows v1 convention:
        round-{ROUND_ID}_{REVIEWER_NAME}_{KEYWORD}
    """

    model_config = {"arbitrary_types_allowed": True}

    workflow_schema: List[Dict[str, Any]]
    working_dir: Optional[Path] = None
    verbose: bool = True
    revisit_flagged: bool = True
    reviewer_costs: Dict[Tuple[str, str], float] = Field(default_factory=dict)
    total_cost: float = 0.0

    def model_post_init(self, __context: Any) -> None:
        """Validate workflow schema after initialization."""
        self._validate_schema()

    def _validate_schema(self) -> None:
        """Validate the workflow schema structure."""
        if not self.workflow_schema:
            raise AgenticWorkflowError("workflow_schema cannot be empty")

        seen_rounds = set()
        for i, task in enumerate(self.workflow_schema):
            # Round ID required
            if "round" not in task:
                raise AgenticWorkflowError(f"Schema entry {i} missing 'round' key")
            round_id = task["round"]
            if round_id in seen_rounds:
                raise AgenticWorkflowError(f"Duplicate round ID: '{round_id}'")
            seen_rounds.add(round_id)

            # Reviewers required
            if "reviewers" not in task:
                raise AgenticWorkflowError(f"Schema entry {i} (round '{round_id}') missing 'reviewers' key")
            reviewers = task["reviewers"] if isinstance(task["reviewers"], list) else [task["reviewers"]]
            for reviewer in reviewers:
                if not isinstance(reviewer, AgenticReviewer):
                    raise AgenticWorkflowError(
                        f"Round '{round_id}': reviewer must be AgenticReviewer, got {type(reviewer).__name__}"
                    )

            # text_inputs required and must be a string or list of strings
            if "text_inputs" not in task:
                raise AgenticWorkflowError(f"Schema entry {i} (round '{round_id}') missing 'text_inputs' key")
            ti = task["text_inputs"]
            if not isinstance(ti, (str, list)):
                raise AgenticWorkflowError(
                    f"Round '{round_id}': 'text_inputs' must be a string or list of strings, got {type(ti).__name__}"
                )

            # filter must be callable if provided
            if "filter" in task and not callable(task["filter"]):
                raise AgenticWorkflowError(f"Round '{round_id}': 'filter' must be callable")

    async def __call__(self, data: Union[pd.DataFrame, Dict[str, Any], str]) -> pd.DataFrame:
        """Run the workflow on input data.

        Args:
            data: DataFrame, dict (converted to DataFrame), or file path (CSV, XLSX, RIS).

        Returns:
            DataFrame with review result columns added.
        """
        df = self._load_data(data)
        return await self.run(df)

    def _load_data(self, data: Union[pd.DataFrame, Dict[str, Any], str]) -> pd.DataFrame:
        """Convert various input types to a DataFrame."""
        if isinstance(data, pd.DataFrame):
            return data.copy()
        elif isinstance(data, dict):
            return pd.DataFrame(data)
        elif isinstance(data, str):
            return self._load_file(data)
        else:
            raise AgenticWorkflowError(f"Invalid data type: {type(data)}. Must be DataFrame, dict, or file path.")

    def _load_file(self, path: str) -> pd.DataFrame:
        """Load a file into a DataFrame."""
        if not os.path.exists(path):
            raise AgenticWorkflowError(f"File not found: {path}")

        lower = path.lower()
        if lower.endswith(".csv"):
            self._log(f"Loading CSV file: {path}")
            df = pd.read_csv(path)
        elif lower.endswith((".xlsx", ".xls")):
            self._log(f"Loading Excel file: {path}")
            df = pd.read_excel(path)
        elif lower.endswith(".ris"):
            raise AgenticWorkflowError(
                "RIS file support requires lattereview.utils.data_handler. "
                "Convert to DataFrame first using ris_to_dataframe()."
            )
        else:
            raise AgenticWorkflowError(f"Unsupported file format: {path}. Supported: .csv, .xlsx, .xls")

        if df.empty:
            raise AgenticWorkflowError(f"No data found in file: {path}")
        return df

    def _format_text_input(self, row: pd.Series, text_inputs: List[str]) -> str:
        """Format input text columns into a single string.

        Follows v1 convention: === column_name ===\\nvalue
        """
        parts = []
        for col in text_inputs:
            value = str(row[col]).strip()
            parts.append(f"=== {col} ===\n{value}")
        return "\n\n".join(parts)

    def _validate_columns(self, df: pd.DataFrame, text_inputs: List[str], round_id: str) -> None:
        """Validate that required text input columns exist in the DataFrame."""
        missing = [col for col in text_inputs if col not in df.columns]
        if missing:
            # Check if any missing columns are outputs from previous rounds
            # (they should exist by the time this round runs)
            raise AgenticWorkflowError(f"Round '{round_id}': columns not found in DataFrame: {missing}")

    async def run(self, data: pd.DataFrame) -> pd.DataFrame:
        """Execute the multi-round review pipeline.

        Args:
            data: Input DataFrame with text columns to review.

        Returns:
            DataFrame with all review result columns added.
        """
        df = data.copy()
        total_rounds = len(self.workflow_schema)

        for round_num, task in enumerate(self.workflow_schema):
            round_id = task["round"]
            self._log(f"\n====== Starting review round {round_id} ({round_num + 1}/{total_rounds}) ======\n")

            reviewers = task["reviewers"] if isinstance(task["reviewers"], list) else [task["reviewers"]]
            text_inputs = task["text_inputs"] if isinstance(task["text_inputs"], list) else [task["text_inputs"]]
            filter_func: Callable = task.get("filter", lambda x: True)

            # Validate columns exist
            self._validate_columns(df, text_inputs, round_id)

            # Apply filter
            mask = df.apply(filter_func, axis=1)
            if not mask.any():
                self._log(f"Skipping round {round_id} — no eligible rows")
                continue

            eligible_indices = df[mask].index.tolist()
            self._log(f"Processing {len(eligible_indices)} eligible rows")

            # Build text inputs for eligible rows
            text_input_strings = []
            for idx in eligible_indices:
                row = df.loc[idx]
                text_input_string = self._format_text_input(row, text_inputs)
                text_input_string = f"Review Task ID: {round_id}-{idx}\n{text_input_string}"
                text_input_strings.append(text_input_string)

            # Process each reviewer
            for reviewer in reviewers:
                output_fields = list(reviewer.output_type.model_fields.keys())
                response_cols = [f"round-{round_id}_{reviewer.name}_{field}" for field in output_fields]
                output_col = f"round-{round_id}_{reviewer.name}_output"

                # Initialize columns
                if output_col not in df.columns:
                    df[output_col] = None
                for col in response_cols:
                    if col not in df.columns:
                        df[col] = None

                # Run batch review
                self._log(f"Running reviewer: {reviewer.name}")
                item_ids = [f"{round_id}-{idx}" for idx in eligible_indices]

                # Set up shared memory store for this reviewer/round if working_dir exists
                memory_store = None
                if self.working_dir is not None and reviewer.is_agentic:
                    from lattereview.agentic.memory.store import MemoryStore

                    memory_dir = self.working_dir / f"round_{round_id}" / f"agent_{reviewer.name}" / "memory"
                    memory_store = MemoryStore(memory_dir)
                    await memory_store.initialize()

                # Set up shared flag store for this reviewer/round if working_dir exists
                flag_store = None
                if self.working_dir is not None and reviewer.is_agentic:
                    from lattereview.agentic.flags.store import FlagStore

                    flags_dir = self.working_dir / f"round_{round_id}" / f"agent_{reviewer.name}" / "flags"
                    flag_store = FlagStore(flags_dir)
                    await flag_store.initialize()

                responses, review_cost = await reviewer.review_items(
                    text_inputs=text_input_strings,
                    item_ids=item_ids,
                    round_id=round_id,
                    working_dir=self.working_dir,
                    memory_store=memory_store,
                    flag_store=flag_store,
                )

                # Track costs (tuple key matches v1 convention)
                cost_key = (round_id, reviewer.name)
                self.reviewer_costs[cost_key] = review_cost
                self.total_cost += review_cost

                if len(responses) != len(eligible_indices):
                    raise AgenticWorkflowError(
                        f"Reviewer {reviewer.name} returned {len(responses)} outputs "
                        f"for {len(eligible_indices)} inputs"
                    )

                # Populate DataFrame columns
                successful = 0
                failed = 0
                for i, (resp, idx) in enumerate(zip(responses, eligible_indices)):
                    # Store full output dict (not string) for downstream filter access
                    df.at[idx, output_col] = resp

                    has_error = "_error" in resp and resp["_error"] is not None
                    if has_error:
                        failed += 1
                    else:
                        successful += 1

                    # Store individual fields
                    for field_name in output_fields:
                        col_name = f"round-{round_id}_{reviewer.name}_{field_name}"
                        df.at[idx, col_name] = resp.get(field_name)

                self._log(f"Reviewer {reviewer.name}: {successful} successful, {failed} failed")
                self._log(f"Columns after {reviewer.name}: {df.columns.tolist()}")

                # Revisit flagged items
                if flag_store is not None and self.revisit_flagged:
                    df = await self._revisit_flagged_items(
                        df=df,
                        reviewer=reviewer,
                        flag_store=flag_store,
                        memory_store=memory_store,
                        round_id=round_id,
                        text_inputs=text_inputs,
                        eligible_indices=eligible_indices,
                        output_fields=output_fields,
                        output_col=output_col,
                    )

        self._log(f"\nWorkflow complete. Total cost: ${self.total_cost:.4f}")
        return df

    async def _revisit_flagged_items(
        self,
        *,
        df: pd.DataFrame,
        reviewer: AgenticReviewer,
        flag_store: Any,
        memory_store: Any,
        round_id: str,
        text_inputs: List[str],
        eligible_indices: List[int],
        output_fields: List[str],
        output_col: str,
    ) -> pd.DataFrame:
        """Re-review items that were flagged during the initial pass.

        Flagged items are re-reviewed one at a time (not concurrently) so the
        agent can leverage memories accumulated during the initial pass.

        After the revisit pass, any items still flagged get ``_flagged`` and
        ``_flag_reason`` columns added to the DataFrame.
        """
        unresolved = await flag_store.get_unresolved()
        if not unresolved:
            return df

        self._log(f"Revisiting {len(unresolved)} flagged items for {reviewer.name}")

        # Build a map from item_id -> DataFrame index
        item_id_to_idx = {f"{round_id}-{idx}": idx for idx in eligible_indices}

        for flag in unresolved:
            flag_item_id = flag["item_id"]
            idx = item_id_to_idx.get(flag_item_id)
            if idx is None:
                continue

            row = df.loc[idx]
            text_input_string = self._format_text_input(row, text_inputs)
            text_input_string = (
                f"Review Task ID: {flag_item_id}\n"
                f"NOTE: This item was previously flagged for revisit.\n"
                f"Previous flag reason: {flag['reason']}\n"
                f"Please re-assess with your updated knowledge. If you can now "
                f"make a confident assessment, call resolve_current_flag to mark "
                f"the flag as resolved.\n\n"
                f"{text_input_string}"
            )

            try:
                resp, revisit_cost = await reviewer.review_item(
                    item_text=text_input_string,
                    item_id=flag_item_id,
                    round_id=round_id,
                    working_dir=self.working_dir,
                    memory_store=memory_store,
                    flag_store=flag_store,
                )

                # Update cost tracking
                cost_key = (round_id, reviewer.name)
                self.reviewer_costs[cost_key] = self.reviewer_costs.get(cost_key, 0.0) + revisit_cost
                self.total_cost += revisit_cost

                # Auto-resolve the flag if the revisit produced output without error
                has_error = "_error" in resp and resp["_error"] is not None
                if not has_error:
                    await flag_store.resolve_flag(flag_item_id)

                # Update DataFrame with revisit results
                df.at[idx, output_col] = resp
                for field_name in output_fields:
                    col_name = f"round-{round_id}_{reviewer.name}_{field_name}"
                    df.at[idx, col_name] = resp.get(field_name)

                self._log(f"  Revisited {flag_item_id}")
            except Exception as e:
                self._log(f"  Error revisiting {flag_item_id}: {e}")

        # After revisit, mark still-unresolved items in DataFrame
        still_unresolved = await flag_store.get_unresolved()
        if still_unresolved:
            flagged_col = f"round-{round_id}_{reviewer.name}_flagged"
            flag_reason_col = f"round-{round_id}_{reviewer.name}_flag_reason"
            if flagged_col not in df.columns:
                df[flagged_col] = False
            if flag_reason_col not in df.columns:
                df[flag_reason_col] = None

            for flag in still_unresolved:
                flag_item_id = flag["item_id"]
                idx = item_id_to_idx.get(flag_item_id)
                if idx is not None:
                    df.at[idx, flagged_col] = True
                    df.at[idx, flag_reason_col] = flag["reason"]

            self._log(f"  {len(still_unresolved)} items still flagged after revisit " f"(see {flagged_col} column)")

        return df

    def get_total_cost(self) -> float:
        """Return the total cost across all reviewers."""
        return self.total_cost

    def _log(self, message: str) -> None:
        """Log a message if verbose mode is enabled.

        Uses print() for immediate user-visible output, matching v1 behavior.
        """
        if self.verbose:
            print(message)
