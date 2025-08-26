import json
import os
import pandas as pd
import pydantic
from typing import List, Dict, Any, Union

from ..agents.scoring_reviewer import ScoringReviewer
from ..utils.data_handler import ris_to_dataframe


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
                # Validate reviewers
                reviewers = (
                    review_task["reviewers"]
                    if isinstance(review_task["reviewers"], list)
                    else [review_task["reviewers"]]
                )
                for reviewer in reviewers:
                    if not isinstance(reviewer, ScoringReviewer):
                        raise ReviewWorkflowError(f"Invalid reviewer: {reviewer}")

                # Validate text_input columns
                text_inputs = (
                    review_task["text_inputs"]
                    if isinstance(review_task["text_inputs"], list)
                    else [review_task["text_inputs"]]
                )
                for text_input in text_inputs:
                    if text_input not in __context["data"].columns:
                        reviewer_name = text_input.split("_")[1]
                        reviewer = next(reviewer for reviewer in reviewers if reviewer.name == reviewer_name)
                        assert reviewer is not None, f"Reviewer {reviewer_name} not found in provided inputs"
                        response_keywords = (
                            reviewer.response_format.keys()
                        )  # e.g., ["_output", "_score", "_reasoning", "_certainty"]
                        assert text_input.split("_")[-1] in response_keywords, f"Invalid input: {text_input}"

                # Validate image_input columns
                image_inputs = review_task.get("image_inputs", [])
                image_inputs = image_inputs if isinstance(image_inputs, list) else [image_inputs]
                for image_input in image_inputs:
                    if image_input not in __context["data"].columns:
                        raise ReviewWorkflowError(f"Invalid image input: {image_input}")

        except Exception as e:
            raise ReviewWorkflowError(f"Error initializing Review Workflow: {e}")

    async def __call__(self, data: Union[pd.DataFrame, Dict[str, Any], str]) -> pd.DataFrame:
        """Run the workflow.

        Parameters:
            data: Can be a pandas DataFrame, a dictionary, or a path to a RIS file

        Returns:
            A pandas DataFrame with review results
        """
        try:
            if isinstance(data, pd.DataFrame):
                return await self.run(data)
            elif isinstance(data, dict):
                return await self.run(pd.DataFrame(data))
            elif isinstance(data, str):
                if data.lower().endswith(".ris"):
                    # Handle RIS file input
                    self._log(f"Converting RIS file: {data}")
                    df = await ris_to_dataframe(data)
                    if df.empty:
                        raise ReviewWorkflowError(f"No data found in RIS file: {data}")
                    return await self.run(df)
                elif data.lower().endswith(".csv"):
                    # Handle CSV file input
                    self._log(f"Loading CSV file: {data}")
                    df = pd.read_csv(data)
                    if df.empty:
                        raise ReviewWorkflowError(f"No data found in CSV file: {data}")
                    return await self.run(df)
                elif data.lower().endswith((".xlsx", ".xls")):
                    # Handle Excel file input, loading first tab by default
                    self._log(f"Loading Excel file: {data}")
                    df = pd.read_excel(data)
                    if df.empty:
                        raise ReviewWorkflowError(f"No data found in Excel file: {data}")
                    return await self.run(df)
                else:
                    raise ReviewWorkflowError(
                        f"Unsupported file format: {data}. Supported formats are .ris, .csv, .xlsx, and .xls."
                    )
            else:
                raise ReviewWorkflowError(f"Invalid data type: {type(data)}. Must be DataFrame, dict, or file path.")
        except Exception as e:
            raise ReviewWorkflowError(f"Error running workflow: {e}")

    def _format_text_input(self, row: pd.Series, text_inputs: List[str]) -> tuple:
        """Format input text with content tracking."""
        parts = []
        for text_input in text_inputs:
            value = str(row[text_input]).strip()
            parts.append(f"=== {text_input} ===\n{value}")

        return "\n\n".join(parts)

    def _format_image_input(self, row: pd.Series, image_inputs: List[str]) -> list:
        """Format input images."""
        image_path_list = []
        for image_input in image_inputs:
            if isinstance(row[image_input], str):
                if (
                    row[image_input].endswith(".jpg")
                    or row[image_input].endswith(".jpeg")
                    or row[image_input].endswith(".png")
                ):
                    if os.path.exists(row[image_input]):
                        image_path_list.append(row[image_input])
                    else:
                        self._log(f"Warning: Image not found: {row[image_input]}")
                else:
                    self._log(f"Warning: Invalid image format: {row[image_input]}")
        return image_path_list

    def _safe_parse_output(self, output, idx, reviewer_name, response_keywords):
        """Safely parse reviewer output with detailed error logging."""
        try:
            if isinstance(output, dict):
                processed_output = output
            else:
                processed_output = json.loads(output)
            
            # Validate that all required keys are present
            missing_keys = []
            for key in response_keywords:
                if key not in processed_output:
                    missing_keys.append(key)
            
            if missing_keys:
                self._log(f"Warning: Row {idx} - Reviewer {reviewer_name} missing keys: {missing_keys}")
                # Fill missing keys with None
                for key in missing_keys:
                    processed_output[key] = None
            
            return processed_output, True
            
        except json.JSONDecodeError as e:
            self._log(f"Warning: Row {idx} - Reviewer {reviewer_name} - JSON decode error: {e}")
            self._log(f"         Raw output: {str(output)[:200]}...")
            # Return default structure with all required keys set to None
            return {key: None for key in response_keywords}, False
        except Exception as e:
            self._log(f"Warning: Row {idx} - Reviewer {reviewer_name} - Unexpected error: {e}")
            self._log(f"         Raw output: {str(output)[:200]}...")
            return {key: None for key in response_keywords}, False

    async def run(self, data: pd.DataFrame) -> pd.DataFrame:
        """Run the review process with robust error handling."""
        try:
            df = data.copy()
            total_rounds = len(self.workflow_schema)

            for review_round, review_task in enumerate(self.workflow_schema):
                round_id = review_task["round"]
                self._log(f"\n====== Starting review round {round_id} ({review_round + 1}/{total_rounds}) ======\n")

                reviewers = (
                    review_task["reviewers"]
                    if isinstance(review_task["reviewers"], list)
                    else [review_task["reviewers"]]
                )
                text_inputs = (
                    review_task["text_inputs"]
                    if isinstance(review_task["text_inputs"], list)
                    else [review_task["text_inputs"]]
                )
                image_inputs = review_task.get("image_inputs", [])
                image_inputs = image_inputs if isinstance(image_inputs, list) else [image_inputs]
                filter_func = review_task.get("filter", lambda x: True)

                # Apply filter and get eligible rows
                mask = df.apply(filter_func, axis=1)
                if not mask.any():
                    self._log(f"Skipping review round {round_id} - no eligible rows")
                    continue

                self._log(f"Processing {mask.sum()} eligible rows")

                # Create input items with content tracking
                text_input_strings = []
                image_path_lists = []
                eligible_indices = []

                for idx in df[mask].index:
                    row = df.loc[idx]

                    # text_input_string is a single string that is made by combining all text_input columns
                    text_input_string = self._format_text_input(row, text_inputs)
                    text_input_string = f"Review Task ID: {round_id}-{idx}\n" f"{text_input_string}"
                    text_input_strings.append(text_input_string)

                    # image_path_list is a list of valid paths to the images provided in the row item
                    image_path_list = self._format_image_input(row, image_inputs)
                    image_path_lists.append(image_path_list)

                    eligible_indices.append(idx)

                # Process each reviewer
                for reviewer in reviewers:
                    response_keywords = reviewer.response_format.keys()
                    response_cols = [f"round-{round_id}_{reviewer.name}_{keyword}" for keyword in response_keywords]
                    output_col = f"round-{round_id}_{reviewer.name}_output"

                    # Initialize the output column and all expected response columns if they don't exist
                    if output_col not in df.columns:
                        df[output_col] = None

                    for response_col in response_cols:
                        if response_col not in df.columns:
                            df[response_col] = None

                    # Get reviewer outputs with metadata
                    outputs, review_cost = await reviewer.review_items(
                        text_input_strings,
                        image_path_lists,
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

                    # Process outputs with robust error handling
                    processed_outputs = []
                    successful_parses = 0
                    failed_parses = 0

                    for i, (output, idx) in enumerate(zip(outputs, eligible_indices)):
                        processed_output, parse_success = self._safe_parse_output(
                            output, idx, reviewer.name, response_keywords
                        )
                        processed_outputs.append(processed_output)
                        
                        if parse_success:
                            successful_parses += 1
                        else:
                            failed_parses += 1

                    # Log parsing statistics
                    self._log(f"Reviewer {reviewer.name}: {successful_parses} successful, {failed_parses} failed parses")

                    # Update dataframe with validated outputs
                    output_dict = dict(zip(eligible_indices, outputs))  # Store original outputs
                    df.loc[eligible_indices, output_col] = pd.Series(output_dict)

                    # Safely update response columns
                    for response_keyword in response_keywords:
                        response_col = f"round-{round_id}_{reviewer.name}_{response_keyword}"
                        try:
                            response_values = []
                            for processed_output in processed_outputs:
                                # This should always work now since we ensure all keys exist
                                response_values.append(processed_output.get(response_keyword, None))
                            
                            response_dict = dict(zip(eligible_indices, response_values))
                            df.loc[eligible_indices, response_col] = pd.Series(response_dict)
                            
                        except Exception as e:
                            self._log(f"Error updating column {response_col}: {e}")
                            # Fill with None values as fallback
                            df.loc[eligible_indices, response_col] = None

                    self._log(
                        f"The following columns are present in the dataframe at the end of {reviewer.name}'s review in round {round_id}: {df.columns.tolist()}"
                    )
            
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