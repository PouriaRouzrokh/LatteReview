import pydantic
from typing import List, Dict, Any, Union
import pandas as pd
import json
import hashlib

from .agents.scoring_reviewer import ScoringReviewer


class ReviewWorkflowError(Exception):
    """Base exception for workflow-related errors."""

    pass


class ReviewWorkflow(pydantic.BaseModel):
    workflow_schema: List[Dict[str, Any]]
    memory: List[Dict] = list()
    reviewer_costs: Dict = dict()
    total_cost: float = 0.0
    verbose: bool = True

    def __post_init__(self, __context):
        """Initialize after Pydantic model initialization."""
        try:
            for review_task in self.workflow_schema:
                round_id = review_task["round"]
                reviewers = (
                    review_task["reviewers"]
                    if isinstance(review_task["reviewers"], list)
                    else [review_task["reviewers"]]
                )
                reviewer_names = [f"round-{round_id}_{reviewer.name}" for reviewer in reviewers]
                inputs = review_task["inputs"] if isinstance(review_task["inputs"], list) else [review_task["inputs"]]
                initial_inputs = [col for col in inputs if "_output_" not in col]

                # Validate reviewers
                for reviewer in reviewers:
                    if not isinstance(reviewer, ScoringReviewer):
                        raise ReviewWorkflowError(f"Invalid reviewer: {reviewer}")

                # Validate input columns
                for input_col in initial_inputs:
                    if input_col not in __context["data"].columns:
                        if input_col.split("_output")[0] not in reviewer_names:
                            raise ReviewWorkflowError(f"Invalid input column: {input_col}")
        except Exception as e:
            raise ReviewWorkflowError(f"Error initializing Review Workflow: {e}")

    async def __call__(self, data: Union[pd.DataFrame, Dict[str, Any]]) -> pd.DataFrame:
        """Run the workflow."""
        try:
            if isinstance(data, pd.DataFrame):
                return await self.run(data)
            elif isinstance(data, dict):
                return await self.run(pd.DataFrame(data))
            else:
                raise ReviewWorkflowError(f"Invalid data type: {type(data)}")
        except Exception as e:
            raise ReviewWorkflowError(f"Error running workflow: {e}")

    def _create_content_hash(self, content: str) -> str:
        """Create a hash of the content for tracking."""
        return hashlib.md5(content.encode()).hexdigest()

    def _format_input_text(self, row: pd.Series, inputs: List[str]) -> tuple:
        """Format input text with content tracking."""
        parts = []
        content_keys = []

        for input_col in inputs:
            if "_output_" not in input_col:
                value = str(row[input_col]).strip()
                parts.append(f"=== {input_col} ===\n{value}")
                content_keys.append(self._create_content_hash(value))

        return "\n\n".join(parts), "-".join(content_keys)

    async def run(self, data: pd.DataFrame) -> pd.DataFrame:
        """Run the review process with content validation."""
        try:
            df = data.copy()
            total_rounds = len(self.workflow_schema)

            for review_round, review_task in enumerate(self.workflow_schema):
                round_id = review_task["round"]
                self._log(f"\nStarting review round {round_id} ({review_round + 1}/{total_rounds})...")

                reviewers = (
                    review_task["reviewers"]
                    if isinstance(review_task["reviewers"], list)
                    else [review_task["reviewers"]]
                )
                inputs = review_task["inputs"] if isinstance(review_task["inputs"], list) else [review_task["inputs"]]
                filter_func = review_task.get("filter", lambda x: True)

                # Apply filter and get eligible rows
                mask = df.apply(filter_func, axis=1)
                if not mask.any():
                    self._log(f"Skipping review round {round_id} - no eligible rows")
                    continue

                self._log(f"Processing {mask.sum()} eligible rows")

                # Create input items with content tracking
                input_items = []
                input_hashes = []
                eligible_indices = []

                for idx in df[mask].index:
                    row = df.loc[idx]
                    input_text, content_hash = self._format_input_text(row, inputs)

                    # Add metadata header
                    input_text = (
                        f"Review Task ID: {round_id}-{idx}\n" f"Content Hash: {content_hash}\n\n" f"{input_text}"
                    )

                    input_items.append(input_text)
                    input_hashes.append(content_hash)
                    eligible_indices.append(idx)

                # Process each reviewer
                for reviewer in reviewers:
                    output_col = f"round-{round_id}_{reviewer.name}_output"
                    score_col = f"round-{round_id}_{reviewer.name}_score"
                    reasoning_col = f"round-{round_id}_{reviewer.name}_reasoning"

                    # Initialize the output column if it doesn't exist
                    if output_col not in df.columns:
                        df[output_col] = None
                    if score_col not in df.columns:
                        df[score_col] = None
                    if reasoning_col not in df.columns:
                        df[reasoning_col] = None

                    # Get reviewer outputs with metadata
                    outputs, review_cost = await reviewer.review_items(
                        input_items,
                        {
                            "round": round_id,
                            "reviewer_name": reviewer.name,
                        },
                    )
                    self.reviewer_costs[(round_id, reviewer.name)] = review_cost

                    # Verify output count
                    if len(outputs) != len(eligible_indices):
                        raise ReviewWorkflowError(
                            f"Reviewer {reviewer.name} returned {len(outputs)} outputs "
                            f"for {len(eligible_indices)} inputs"
                        )

                    # Process outputs with content validation
                    processed_outputs = []
                    processed_scores = []
                    processed_reasoning = []

                    for output, expected_hash in zip(outputs, input_hashes):
                        try:
                            if isinstance(output, dict):
                                processed_output = output
                            else:
                                processed_output = json.loads(output)

                            # Add content hash to output for validation
                            processed_output["_content_hash"] = expected_hash
                            processed_outputs.append(processed_output)

                            if "score" in processed_output:
                                processed_scores.append(processed_output["score"])

                            if "reasoning" in processed_output:
                                processed_reasoning.append(processed_output["reasoning"])

                        except Exception as e:
                            self._log(f"Warning: Error processing output: {e}")
                            processed_outputs.append({"reasoning": None, "score": None, "_content_hash": expected_hash})

                    # Update dataframe with validated outputs
                    output_dict = dict(zip(eligible_indices, processed_outputs))
                    df.loc[eligible_indices, output_col] = pd.Series(output_dict)

                    score_dict = dict(zip(eligible_indices, processed_scores))
                    df.loc[eligible_indices, score_col] = pd.Series(score_dict)

                    reasoning_dict = dict(zip(eligible_indices, processed_reasoning))
                    df.loc[eligible_indices, reasoning_col] = pd.Series(reasoning_dict)

            return df

        except Exception as e:
            raise ReviewWorkflowError(f"Error running workflow: {e}")

    def _log(self, x):
        """Log message if verbose mode is enabled."""
        if self.verbose:
            print(x)

    def get_total_cost(self) -> float:
        """Return the total cost of the review process."""
        return sum(self.reviewer_costs.values())
