# Migration Guide: v1 to v2

LatteReview v2 introduces a new agentic framework built on Pydantic AI. The v1 API continues to work but emits deprecation warnings. This guide shows how to migrate your code.

## Summary of Changes

| Area | v1 | v2 |
|------|----|----|
| Import path | `lattereview.agents` | `lattereview.agentic` |
| Model config | Provider wrapper classes | Model strings (e.g., `"openai:gpt-5.4-mini"`) |
| Workflow class | `ReviewWorkflow` | `AgenticWorkflow` |
| Python version | >=3.9 | >=3.12 |
| New capabilities | -- | Skills, memory, helpers, checkpoint/resume |

## Providers to Model Strings

v1 required instantiating provider wrapper classes. v2 uses Pydantic AI model strings directly.

=== "v1 (deprecated)"

    ```python
    from lattereview.providers import LiteLLMProvider, OpenAIProvider

    # OpenAI via LiteLLM
    provider = LiteLLMProvider(model="gpt-5.4-mini")

    # OpenAI native
    provider = OpenAIProvider(model="gpt-5.4-mini")

    # Google via LiteLLM
    provider = LiteLLMProvider(model="gemini/gemini-3-flash-preview")

    # Anthropic via LiteLLM
    provider = LiteLLMProvider(model="anthropic/claude-sonnet-4-6")
    ```

=== "v2"

    ```python
    # No provider imports needed -- just use model strings

    model = "openai:gpt-5.4-mini"
    model = "google-gla:gemini-3-flash-preview"
    model = "anthropic:claude-sonnet-4-6"
    ```

## Reviewer Classes

The reviewer classes have the same names but live in a different module and accept `model=` instead of `provider=`.

=== "v1 (deprecated)"

    ```python
    from lattereview.providers import LiteLLMProvider
    from lattereview.agents import ScoringReviewer, TitleAbstractReviewer, AbstractionReviewer

    reviewer = ScoringReviewer(
        provider=LiteLLMProvider(model="gpt-5.4-mini"),
        name="Scorer",
        scoring_task="Rate relevance",
        scoring_set=[1, 2, 3, 4, 5],
        reasoning="brief",
        model_args={"temperature": 0.1},
    )
    ```

=== "v2"

    ```python
    from lattereview.agentic import ScoringReviewer

    reviewer = ScoringReviewer(
        model="openai:gpt-5.4-mini",
        name="Scorer",
        scoring_task="Rate relevance",
        scoring_set=[1, 2, 3, 4, 5],
    )
    ```

## Workflows

The workflow schema format is the same. Only the class name and import path change.

=== "v1 (deprecated)"

    ```python
    from lattereview.workflows import ReviewWorkflow

    workflow = ReviewWorkflow(
        workflow_schema=[
            {
                "round": "A",
                "reviewers": [reviewer1, reviewer2],
                "text_inputs": ["title", "abstract"],
            },
            {
                "round": "B",
                "reviewers": [expert],
                "text_inputs": ["title", "abstract", "round-A_Alice_output"],
                "filter": lambda row: row["round-A_Alice_evaluation"] != row["round-A_Bob_evaluation"],
            },
        ]
    )

    results = asyncio.run(workflow(data))
    ```

=== "v2"

    ```python
    from lattereview.agentic import AgenticWorkflow

    workflow = AgenticWorkflow(
        workflow_schema=[
            {
                "round": "A",
                "reviewers": [reviewer1, reviewer2],
                "text_inputs": ["title", "abstract"],
            },
            {
                "round": "B",
                "reviewers": [expert],
                "text_inputs": ["title", "abstract", "round-A_Alice_output"],
                "filter": lambda row: row["round-A_Alice_evaluation"] != row["round-A_Bob_evaluation"],
            },
        ]
    )

    results = asyncio.run(workflow(data))
    ```

## Complete Example: Side by Side

### v1 (deprecated)

```python
from lattereview.providers import LiteLLMProvider
from lattereview.agents import TitleAbstractReviewer
from lattereview.workflows import ReviewWorkflow
import pandas as pd
import asyncio

reviewer1 = TitleAbstractReviewer(
    provider=LiteLLMProvider(model="gpt-5.4-mini"),
    name="Alice",
    inclusion_criteria="Must involve AI in radiology",
    exclusion_criteria="Exclude non-peer-reviewed",
    reasoning="brief",
    model_args={"temperature": 0.1},
)

reviewer2 = TitleAbstractReviewer(
    provider=LiteLLMProvider(model="gpt-5.4-mini"),
    name="Bob",
    inclusion_criteria="Must involve AI in radiology",
    exclusion_criteria="Exclude non-peer-reviewed",
    reasoning="cot",
    model_args={"temperature": 0.5},
)

workflow = ReviewWorkflow(
    workflow_schema=[
        {
            "round": "A",
            "reviewers": [reviewer1, reviewer2],
            "text_inputs": ["title", "abstract"],
        }
    ]
)

data = pd.read_csv("articles.csv")
results = asyncio.run(workflow(data))
results.to_csv("results.csv", index=False)
```

### v2

```python
from lattereview.agentic import TitleAbstractReviewer, AgenticWorkflow
import pandas as pd
import asyncio

reviewer1 = TitleAbstractReviewer(
    model="openai:gpt-5.4-mini",
    name="Alice",
    inclusion_criteria="Must involve AI in radiology",
    exclusion_criteria="Exclude non-peer-reviewed",
)

reviewer2 = TitleAbstractReviewer(
    model="openai:gpt-5.4-mini",
    name="Bob",
    inclusion_criteria="Must involve AI in radiology",
    exclusion_criteria="Exclude non-peer-reviewed",
)

workflow = AgenticWorkflow(
    workflow_schema=[
        {
            "round": "A",
            "reviewers": [reviewer1, reviewer2],
            "text_inputs": ["title", "abstract"],
        }
    ]
)

data = pd.read_csv("articles.csv")
results = asyncio.run(workflow(data))
results.to_csv("results.csv", index=False)
```

## v2-Only Features

The following features are only available in the v2 agentic API:

**Skills** -- Enable search, memory, flagging, and helper-agent discussion via simple string names:

```python
from lattereview.agentic import ScoringReviewer

reviewer = ScoringReviewer(
    model="openai:gpt-5.4-mini",
    name="Researcher",
    scoring_task="Rate the methodological quality",
    skills=["searching-pubmed", "managing-memory", "flagging-items"],
)
```

**Checkpoint/Resume** -- Long-running workflows save progress atomically per item, so interrupted runs can be resumed without losing work:

```python
from lattereview.agentic import AgenticWorkflow
from pathlib import Path

workflow = AgenticWorkflow(
    workflow_schema=[...],
    working_dir=Path("./review_output"),
    resume=True,  # Resumes from last checkpoint
)
```

**Memory** -- Reviewers persist insights across items during a workflow run, building knowledge as they work through a batch.

**Helper agents** -- Attach specialist sub-agents that a reviewer can consult mid-review:

```python
from lattereview.agentic import ScoringReviewer

expert = ScoringReviewer(model="anthropic:claude-sonnet-4-6", name="Expert", ...)
reviewer = ScoringReviewer(
    model="openai:gpt-5.4-mini",
    helpers=[expert],
    skills=["discussing-with-helpers"],
    ...
)
```
