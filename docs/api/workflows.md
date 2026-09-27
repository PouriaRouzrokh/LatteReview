# Workflows Module Documentation

This module provides workflow management functionality for the LatteReview package, implementing multi-agent review orchestration. The updated version includes support for image-based inputs alongside text inputs in the review workflow.

## Overview

The workflows module consists of one main class:

- `ReviewWorkflow`: A class that orchestrates multi-agent review processes with support for sequential rounds, filtering, and handling both text and image inputs.

## ReviewWorkflow

### Description

The `ReviewWorkflow` class manages review workflows where multiple agents can review items across different rounds. It handles the orchestration of reviews, manages outputs, and provides content validation and cost tracking.

### Class Definition

```python
class ReviewWorkflow(pydantic.BaseModel):
    workflow_schema: List[Dict[str, Any]]
    memory: List[Dict] = list()
    reviewer_costs: Dict = dict()
    total_cost: float = 0.0
    verbose: bool = True
```

### Key Attributes

- `workflow_schema`: List of dictionaries defining the review process structure
- `memory`: List storing the workflow's execution history
- `reviewer_costs`: Dictionary tracking costs per reviewer and round
- `total_cost`: Total accumulated cost of all reviews
- `verbose`: Flag to enable/disable logging output

### Methods

#### `__call__()`

Execute the workflow on provided data.

```python
async def __call__(self, data: Union[pd.DataFrame, Dict[str, Any], str]) -> pd.DataFrame:
    """
    Execute workflow on DataFrame, dictionary input, or directly from a file path.
    Supported file formats: .csv, .xlsx, and .ris
    """
    # Returns DataFrame with review results
```

#### `run()`

Core method to execute the review workflow.

```python
async def run(self, data: pd.DataFrame) -> pd.DataFrame:
    """Execute main review workflow."""
    # Process each round sequentially
    # Returns updated DataFrame with review results
```

#### `get_total_cost()`

Get total cost of workflow execution.

```python
def get_total_cost(self) -> float:
    """Return sum of all review costs."""
```

#### Internal Methods

- `_format_text_input()`: Format input for reviewers
- `_format_image_input()`: Validate and format image paths
- `_log()`: Handle logging based on verbose setting
- `_load_from_file()`: Load data from supported file formats (.csv, .xlsx, .ris)

### Image Input Handling

The updated workflow supports image inputs in addition to text inputs. Images are validated for file existence and format before being passed to reviewers.

### File Format Support

The workflow now supports loading data directly from files in the following formats:

- CSV (.csv): Standard comma-separated values files
- Excel (.xlsx): Microsoft Excel spreadsheets
- RIS (.ris): Research Information Systems format commonly used for bibliographic citations

When using RIS files, standard bibliographic tags (e.g., TI for title, AB for abstract) are automatically mapped to appropriate columns.

## Usage Examples

### Creating a Basic Review Workflow

```python
from lattereview.workflows import ReviewWorkflow
from lattereview.agents import ScoringReviewer
from lattereview.providers import LiteLLMProvider

# Create reviewers
reviewer1 = ScoringReviewer(
    provider=LiteLLMProvider(model="gpt-6-luna"),
    name="Initial",
    scoring_task="Initial paper screening"
)

reviewer2 = ScoringReviewer(
    provider=LiteLLMProvider(model="gpt-6-sol"),
    name="Expert",
    scoring_task="Detailed technical review"
)

# Define workflow schema
workflow_schema = [
    {
        "round": 'A',
        "reviewers": reviewer1,
        "text_inputs": ["title", "abstract"],
        "image_inputs": ["figure_path"]
    },
    {
        "round": 'B',
        "reviewers": reviewer2,
        "text_inputs": ["title", "abstract", "round-A_Initial_output"],
        "filter": lambda row: row["round-A_Initial_score"] > 3
    }
]

# Create and run workflow
workflow = ReviewWorkflow(workflow_schema=workflow_schema)

# Option 1: Pass a DataFrame
results = await workflow(input_data)

# Option 2: Pass a file path directly
results = await workflow("papers.xlsx")  # Can use .csv, .xlsx, or .ris files
```

### Understanding Workflow Construction

A workflow is defined by a list of dictionaries where each dictionary represents a review round. The rounds are executed sequentially in the order they appear in the list. Each round dictionary must contain:

Required Arguments:

- `round`: A string identifier for the round (e.g., 'A', 'B', '1', 'initial')
- `reviewers`: A single ScoringReviewer or list of ScoringReviewers
- `text_inputs`: A string or list of strings representing DataFrame column names to use

Optional Arguments:

- `image_inputs`: A list of DataFrame column names containing paths to image files
- `filter`: A lambda function that determines which rows to review in this round

### Handling Results

```python
# Execute workflow
workflow = ReviewWorkflow(workflow_schema=workflow_schema)
try:
    # Option 1: Pass a DataFrame
    results_df = await workflow(input_df)

    # Option 2: Pass a file path directly
    results_df = await workflow("references.ris")  # Also supports .csv and .xlsx

    # Access costs
    total_cost = workflow.get_total_cost()
    per_reviewer = workflow.reviewer_costs

    # Access results
    round_a_scores = results_df["round-A_Initial_score"]
    round_b_reasoning = results_df["round-B_Expert_reasoning"]

except ReviewWorkflowError as e:
    print(f"Workflow failed: {e}")
```

## Error Handling

The module uses `ReviewWorkflowError` for all workflow-related errors:

```python
class ReviewWorkflowError(Exception):
    """Base exception for workflow-related errors."""
    pass
```

Common error scenarios:

- Invalid workflow schema
- Missing input columns
- Reviewer initialization failures
- Content validation errors
- Output processing failures
- Invalid image file paths

## Best Practices

1. Schema Design

   - Use meaningful round identifiers
   - Design filter functions carefully
   - Consider round dependencies

2. Data Management

   - Validate input data structure
   - Handle missing values appropriately
   - Use appropriate column names

3. Cost Control

   - Monitor per-round costs
   - Set appropriate concurrent request limits
   - Track total workflow costs

4. Error Handling
   - Implement proper exception handling
   - Validate workflow schema
   - Monitor review outputs

## Limitations

- Sequential round execution only
- Memory grows with number of reviews
- No direct reviewer communication
- Limited to DataFrame-based workflows
- Requires async/await syntax
- Filter functions must be serializable
