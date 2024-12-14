import pydantic
from typing import List, Dict, Any, Union
import pandas as pd
from tqdm.auto import tqdm
import json

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
                reviewers = review_task["reviewers"] if isinstance(review_task["reviewers"], list) else [review_task["reviewers"]]
                reviewer_names = [f"round-{round_id}_{reviewer.name}" for reviewer in reviewers]
                inputs = review_task["inputs"] if isinstance(review_task["inputs"], list) else [review_task["inputs"]]
                initial_inputs = [col for col in inputs if "_output_" not in col]
                
                for reviewer in reviewers:
                    if not isinstance(reviewer, ScoringReviewer):
                        raise ReviewWorkflowError(f"Invalid reviewer: {reviewer}")
                
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

    async def run(self, data: pd.DataFrame) -> pd.DataFrame:
        """Run the review process."""
        try:            
            df = data.copy()
            total_rounds = len(self.workflow_schema)
            
            for review_round, review_task in enumerate(self.workflow_schema):
                round_id = review_task["round"]
                print(f"\nStarting review round {round_id} ({review_round + 1}/{total_rounds})...")
                self._log(f"Reviewers: {[reviewer.name for reviewer in review_task['reviewers']]}")
                self._log(f"Input data: {review_task['inputs']}")
                
                reviewers = review_task["reviewers"] if isinstance(review_task["reviewers"], list) else [review_task["reviewers"]]
                inputs = review_task["inputs"] if isinstance(review_task["inputs"], list) else [review_task["inputs"]]
                filter_func = review_task.get("filter", lambda x: True)
                
                # Pre-process data
                output_cols = [col for col in inputs if "_output_" in col]
                for col in output_cols:
                    mask = df[col].notna()
                    if mask.any():
                        try:
                            df.loc[mask, col] = df.loc[mask, col].apply(lambda x: x if isinstance(x, dict) else json.loads(x))
                        except Exception as e:
                            self._log(ReviewWorkflowError(f"Error parsing output column {col}: {e}"))
                            df[col] = df[col].apply(lambda x: {"reasoning": None, "score": None})
                
                # Create input items
                input_text = []
                for _, row in tqdm(df.iterrows(), total=len(df), desc="Creating inputs", leave=False):
                    text = " ".join(f"{input_col}: {str(row[input_col])}" for input_col in inputs)
                    input_text.append(text)
                df["input_item"] = input_text
                
                # Apply filter
                try:
                    mask = []
                    for _, row in tqdm(df.iterrows(), total=len(df), desc="Filtering", leave=False):
                        try:
                            mask.append(filter_func(row))
                        except:
                            # If filter function fails, assume row is eligible
                            mask.append(True)                    
                    mask = pd.Series(mask, index=df.index)
                    eligible_rows = mask.sum()
                    self._log(f"Number of eligible rows for review: {eligible_rows}")
                    
                    if (eligible_rows == 0):
                        self._log(f"Skipping review round {round_id} - no eligible rows")
                        continue
                        
                except Exception as e:
                    raise ReviewWorkflowError(f"Error applying filter in round {round_id}: {e}")
                
                df_filtered = df[mask].copy()
                input_items = df_filtered["input_item"].tolist()
                
                # Create progress bar for reviewers
                reviewer_outputs = []
                
                for reviewer in reviewers:
                    outputs, review_cost = await reviewer.review_items(input_items, {"round":round_id, "reviewer_name": reviewer.name})
                    reviewer_outputs.append(outputs)
                    self.reviewer_costs[(round_id, reviewer.name)] = review_cost

                # Add reviewer outputs with round prefix
                for reviewer, outputs in zip(reviewers, reviewer_outputs):
                    output_col = f"round-{round_id}_{reviewer.name}_output"
                    processed_outputs = []
                    for output in outputs:
                        if type(output) == dict:
                            processed_outputs.append(output)
                        else:
                            processed_outputs.append(json.loads(output))
                    df_filtered[output_col] = processed_outputs
                
                # Merge results back
                merge_cols = list(set(inputs) - set([col for col in inputs if "_output" in col]))
                if not merge_cols:
                    df_filtered.index = df[mask].index
                    for reviewer in reviewers:
                        output_col = f"round-{round_id}_{reviewer.name}_output"
                        df.loc[mask, output_col] = df_filtered[output_col]
                else:
                    output_cols = [f"round-{round_id}_{r.name}_output" for r in reviewers]
                    df = pd.merge(
                        df,
                        df_filtered[merge_cols + output_cols],
                        how="left",
                        on=merge_cols
                    )
                
                # Clean up temporary columns
                if "input_item" in df.columns:
                    df = df.drop("input_item", axis=1)
                
            return df
        except Exception as e:
            raise ReviewWorkflowError(f"Error running workflow: {e}")
        
    def _log(self, x):
        if self.verbose:
            print(x)
        
    def get_total_cost(self) -> int:
        """Return the total cost of the review process."""
        return sum(self.reviewer_costs.values())
