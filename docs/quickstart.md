LatteReview enables you to create custom literature review workflows with multiple AI reviewers. Each reviewer can use different models and providers based on your needs. Please follow the below steps to perform a review task with LatteReview.

💡Also, please check our [tutorial notebooks](https://github.com/PouriaRouzrokh/LatteReview/tree/main/tutorials) that provide complete code examples for all main functionalities of the LatteReview package.

## Step 1: Set Up API Keys

To use LatteReview with different LLM engines (OpenAI, Anthropic, Google, etc.), you'll need to set up the API keys for the specific providers you plan to use. For example, if you're only using OpenAI models, you only need the OpenAI API key. Here are three ways to set up your API keys:

1. Using python-dotenv (Recommended):
   - Create a `.env` file in your project directory and add only the keys you need:

```text
# .env file - Example keys (add only what you need)
OPENAI_API_KEY=your-openai-key
ANTHROPIC_API_KEY=your-anthropic-key
```

- Load it in your code:

```python
from dotenv import load_dotenv
load_dotenv()  # Load before importing any providers
```

2. Setting environment variables directly:

```bash
# Example: Set only the keys for providers you're using
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-anthropic-key"
```

3. Passing API keys directly to providers (supported by some providers):

```python
from lattereview.providers import OpenAIProvider
provider = OpenAIProvider(api_key="your-openai-key")  # Optional, will use environment variable if not provided
```

Note: No API keys are needed if you're exclusively using local models through Ollama.

## Step 2: Prepare Your Data

Your input data should be in a CSV, Excel, or RIS file with appropriate content for review. For CSV and Excel files, the column names should match the `inputs` specified in your workflow:

```python
# Load your data
data = pd.read_excel("articles.xlsx")  # You can also use .csv or .ris files

# Example data structure for CSV/Excel:
data = pd.DataFrame({
    'title': ['Paper 1 Title', 'Paper 2 Title'],
    'abstract': ['Paper 1 Abstract', 'Paper 2 Abstract']
})
```

For RIS files, the standard bibliographic tags will be automatically mapped to appropriate columns (e.g., TI for title, AB for abstract).

## Step 3: Create Reviewers

Create reviewer agents by configuring `TitleAbstractReviewer` objects. Each reviewer needs:

- A provider (e.g., LiteLLMProvider, OpenAIProvider, OllamaProvider)
- A unique name
- Inclusion and exclusion criteria
- A reasoning argument that defaults to `brief` but can also be set to `cot` for detailed step-by-step reasoning. This cannot be `None`.
- Optional configuration like temperature and model parameters

```python
# Example of creating a TitleAbstractReviewer
reviewer1 = TitleAbstractReviewer(
    provider=LiteLLMProvider(model="gpt-4o-mini"),  # Choose your model provider
    name="Alice",                                    # Unique name for the reviewer
    inclusion_criteria="Must be relevant to AI in medical imaging.",
    exclusion_criteria="Exclude papers focusing only on basic sciences.",
    reasoning="brief",                               # Reasoning explanation
    model_args={"temperature": 0.1}                 # Model configuration
)

# Second Reviewer: More exploratory approach
reviewer2 = TitleAbstractReviewer(
    provider=LiteLLMProvider(model="gemini/gemini-1.5-flash"),
    name="Bob",
    backstory="a computer scientist specializing in medical AI",
    inclusion_criteria="Relevant to artificial intelligence in radiology.",
    exclusion_criteria="Exclude studies focused solely on hardware.",
    reasoning="cot",
    model_args={"temperature": 0.8}
)

# Expert Reviewer: Resolves disagreements
expert = TitleAbstractReviewer(
    provider=LiteLLMProvider(model="o3-mini"),
    name="Carol",
    backstory="a professor of AI in medical imaging",
    inclusion_criteria="Must align with at least one of Alice or Bob's recommendations.",
    exclusion_criteria="Exclude only if both Alice and Bob disagreed.",
    reasoning="brief",
    model_args={"reasoning_effort": "high"}  # o3-mini specific parameter
)
```

## Step 4: Create Review Workflow

Define your workflow by specifying review rounds, reviewers, and input columns. The workflow automatically creates output columns for each reviewer based on their name and review round. For each reviewer, two columns are created:

- `round-{ROUND}_{REVIEWER_NAME}_output`: Full output dictionary
- `round-{ROUND}_{REVIEWER_NAME}_evaluation`: Extracted evaluation score
- `round-{ROUND}_{REVIEWER_NAME}_reasoning`: Extracted reasoning explanation, which follows the reasoning style (`brief` or `cot`) defined for the agent.

These automatically generated columns can be used as inputs in subsequent rounds, allowing later reviewers to access and evaluate the outputs of previous reviewers:

```python
workflow = ReviewWorkflow(
    workflow_schema=[
        {
            "round": 'A',               # First round
            "reviewers": [reviewer1, reviewer2],
            "text_inputs": ["title", "abstract"]  # Original data columns
        },
        {
            "round": 'B',               # Second round
            "reviewers": [expert],
            # Access both original columns and previous reviewers' outputs
            "text_inputs": ["title", "abstract", "round-A_reviewer1_output", "round-A_reviewer2_output"],
            # Optional filter to review only certain cases
            "filter": lambda row: row["round-A_reviewer1_evaluation"] != row["round-A_reviewer2_evaluation"]
        }
    ]
)
```

In this example, the expert reviewer in round B can access both the original data columns and the outputs from round A's reviewers. The filter ensures the expert only reviews cases where the first two reviewers disagreed.

## Step 5: Run the Workflow

Execute the workflow and get results:

```python
# Run workflow
results = asyncio.run(workflow(data))  # Returns DataFrame with all results
# Or directly pass a file path:
results = asyncio.run(workflow("articles.xlsx"))  # Can use .csv, .xlsx, or .ris files

# Results include original columns plus new columns for each reviewer:
# - round-{ROUND}_{REVIEWER_NAME}_output: Full output dictionary
# - round-{ROUND}_{REVIEWER_NAME}_evaluation: Extracted evaluation score
# - round-{ROUND}_{REVIEWER_NAME}_reasoning: Reasoning explanation based on the defined style (`brief` or `cot`)
```

## Complete Working Example

```python
from lattereview.providers import LiteLLMProvider
from lattereview.agents import TitleAbstractReviewer
from lattereview.workflows import ReviewWorkflow
import pandas as pd
import asyncio
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# First Reviewer: Conservative approach
reviewer1 = TitleAbstractReviewer(
    provider=LiteLLMProvider(model="gpt-4o-mini"),
    name="Alice",
    backstory="a radiologist with expertise in systematic reviews",
    inclusion_criteria="Must be relevant to artificial intelligence in radiology.",
    exclusion_criteria="Exclude studies that are not peer-reviewed.",
    reasoning="brief",
    model_args={"temperature": 0.1}
)

# Second Reviewer: More exploratory approach
reviewer2 = TitleAbstractReviewer(
    provider=LiteLLMProvider(model="gemini/gemini-1.5-flash"),
    name="Bob",
    backstory="a computer scientist specializing in medical AI",
    inclusion_criteria="Relevant to artificial intelligence in radiology.",
    exclusion_criteria="Exclude studies focused solely on hardware.",
    reasoning="cot",
    model_args={"temperature": 0.8}
)

# Expert Reviewer: Resolves disagreements
expert = TitleAbstractReviewer(
    provider=LiteLLMProvider(model="o3-mini"),
    name="Carol",
    backstory="a professor of AI in medical imaging",
    inclusion_criteria="Must align with at least one of Alice or Bob's recommendations.",
    exclusion_criteria="Exclude only if both Alice and Bob disagreed.",
    reasoning="brief",
    model_args={"reasoning_effort": "high"}  # o3-mini specific parameter
)

# Define workflow
workflow = ReviewWorkflow(
    workflow_schema=[
        {
            "round": 'A',  # First round: Initial review by both reviewers
            "reviewers": [reviewer1, reviewer2],
            "text_inputs": ["title", "abstract"]
        },
        {
            "round": 'B',  # Second round: Expert reviews only disagreements
            "reviewers": [expert],
            "text_inputs": ["title", "abstract", "round-A_Alice_output", "round-A_Bob_output"],
            "filter": lambda row: row["round-A_Alice_evaluation"] != row["round-A_Bob_evaluation"]
        }
    ]
)

# Load and process your data
data = pd.read_excel("articles.xlsx")  # Must have 'title' and 'abstract' columns
results = asyncio.run(workflow(data))  # Returns a pandas DataFrame with all original and output columns

# Alternatively, pass the file path directly
# results = asyncio.run(workflow("articles.xlsx"))  # Can use .csv, .xlsx, or .ris files

# Save results
results.to_csv("review_results.csv", index=False)
```
