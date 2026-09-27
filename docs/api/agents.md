# Agents Module Documentation for LatteReview

This document provides a comprehensive explanation of the agents module within the LatteReview package.

## Overview

The agents module is a central part of LatteReview, implementing agent-based workflows for reviewing and processing inputs like text and images. The module includes the following primary classes:

- **`BasicReviewer`**: The abstract base class that serves as the foundation for all agent types.
- **`ScoringReviewer`**: A concrete implementation designed to review and score inputs based on defined criteria.
- **`AbstractionReviewer`**: A specialized agent for structured data extraction tasks.
- **`TitleAbstractReviewer`**: A specialized agent for reviewing and evaluating article titles and abstracts against defined criteria.
- **`DecisionReviewer`**, **`DecisionTitleAbstractReviewer`**, **`DecisionScoringReviewer`** (new in v1.3.0): Reviewers that use a System One decision model such as Jev instead of an LLM. See [Decision Models](../decision_models.md).

## BasicReviewer Class

### Overview

The `BasicReviewer` class provides essential functionality for all agent types. It manages prompts, agent states, example processing, and integration with language model providers.

### Class Definition

```python
class BasicReviewer(BaseModel):
    generic_prompt: Optional[str] = None
    prompt_path: Optional[Union[str, Path]] = None
    response_format: Dict[str, Any] = None
    provider: Optional[Any] = None
    model_args: Dict[str, Any] = Field(default_factory=dict)
    max_concurrent_requests: int = DEFAULT_CONCURRENT_REQUESTS
    name: str = "BasicReviewer"
    backstory: str = "a generic base agent"
    input_description: str = ""
    examples: Union[str, List[Union[str, Dict[str, Any]]]] = None
    reasoning: str = None
    system_prompt: Optional[str] = None
    formatted_prompt: Optional[str] = None
    cost_so_far: float = 0
    memory: List[Dict[str, Any]] = []
    identity: Dict[str, Any] = {}
    additional_context: Optional[Union[Callable, str]] = ""
    verbose: bool = True
```

### Key Attributes

- **`generic_prompt`**: Template string for creating prompts.
- **`prompt_path`**: Path to the template file for constructing prompts.
- **`response_format`**: Dictionary defining the structure of expected responses.
- **`provider`**: Language model provider instance (e.g., OpenAI, Ollama).
- **`model_args`**: Arguments passed to the language model (e.g., `{"reasoning_effort": "low"}`). Parameters a model rejects are dropped with a warning; see [Model compatibility](providers.md#model-compatibility).
- **`max_concurrent_requests`**: Limit for concurrent processing tasks.
- **`name`**: Identifier for the agent.
- **`backstory`**: Description of the agent's role.
- **`input_description`**: Description of the input format.
- **`examples`**: Examples for the model’s guidance.
- **`reasoning`**: Type of reasoning employed (e.g., `brief`).
- **`system_prompt`**: Generated system-level prompt for the model.
- **`formatted_prompt`**: Fully constructed input prompt.
- **`cost_so_far`**: Tracks cumulative API costs.
- **`memory`**: Log of agent interactions and responses.
- **`identity`**: Metadata defining the agent’s setup.
- **`additional_context`**: Additional data provided as a callable or string.
- **`verbose`**: Controls logging verbosity.

### Key Methods

#### Initialization and Setup

- **`setup(self)`**: Initializes the agent by preparing prompts and configuring the provider.
- **`model_post_init(self, __context: Any)`**: Post-initialization setup after creating the agent instance.
- **`reset_memory(self)`**: Clears memory and cost tracking.

#### Prompt Management

- **`_build_system_prompt(self)`**: Constructs the system prompt incorporating agent identity and task details.
- **`_process_prompt(self, base_prompt: str, item_dict: Dict[str, Any])`**: Substitutes variables in a base prompt.
- **`_extract_prompt_keywords(self, prompt: str)`**: Extracts keywords for dynamic variable replacement from the prompt.

#### Reasoning and Examples

- **`_process_reasoning(self, reasoning: str)`**: Maps reasoning types to corresponding text.
- **`_process_examples(self, examples: Union[str, Dict[str, Any], List[Union[str, Dict[str, Any]]]])`**: Formats examples consistently for prompt use.

#### Additional Utilities

- **`_clean_text(self, text: str)`**: Removes extra spaces and blank lines from the text.
- **`_log(self, message: str)`**: Logs messages if verbose mode is enabled.

#### Review Operations

- **`review_items(self, text_input_strings: List[str], image_path_lists: List[List[str]] = None, tqdm_keywords: dict = None)`**: Reviews multiple items asynchronously with concurrency control and a progress bar.
- **`review_item(self, text_input_string: str, image_path_list: List[str] = [])`**: Reviews a single item asynchronously with error handling.

---

## ScoringReviewer Class

### Overview

The `ScoringReviewer` extends `BasicReviewer` to provide scoring functionalities for input data. It supports structured scoring tasks, reasoning explanations, and customizable scoring rules.

### Class Definition

```python
class ScoringReviewer(BasicReviewer):
    response_format: Dict[str, Any] = {
        "reasoning": str,
        "score": int,
        "certainty": int
    }
    scoring_task: Optional[str] = None
    scoring_set: List[int] = [1, 2]
    scoring_rules: str = "Your scores should follow the defined schema."
    reasoning: str = "brief"
    max_retries: int = DEFAULT_MAX_RETRIES
```

### Key Attributes

- **`response_format`**: Structure of the scoring output, including reasoning, score, and certainty.
- **`scoring_task`**: Description of the scoring task.
- **`scoring_set`**: Valid scoring values.
- **`scoring_rules`**: Rules to guide scoring decisions.
- **`reasoning`**: Type of reasoning employed.
- **`max_retries`**: Maximum retry attempts.

### Key Methods

- **`review_items(self, text_input_strings: List[str], image_path_lists: List[List[str]] = None)`**: Processes multiple items concurrently.
- **`review_item(self, text_input_string: str, image_path_list: List[str] = [])`**: Processes a single item.

---

## AbstractionReviewer Class

### Overview

The `AbstractionReviewer` is a specialized agent for extracting structured information from inputs. It leverages predefined keys and detailed instructions for consistent extraction tasks.

### Class Definition

```python
class AbstractionReviewer(BasicReviewer):
    generic_prompt: Optional[str] = generic_prompt
    input_description: str = "article title/abstract"
    abstraction_keys: Dict
    key_descriptions: Dict
    max_retries: int = DEFAULT_MAX_RETRIES

    def model_post_init(self, __context: Any) -> None:
        """Initialize after Pydantic model initialization."""
        try:
            assert self.reasoning == None, "Reasoning type should be None for AbstractionReviewer"
            self.response_format = self.abstraction_keys
            self.setup()
        except Exception as e:
            raise AgentError(f"Error initializing agent: {str(e)}")
```

### Key Attributes

- **`generic_prompt`**: Template string for creating prompts.
- **`input_description`**: Description of the expected input.
- **`abstraction_keys`**: Specifies the keys to extract from the input.
- **`key_descriptions`**: Provides detailed descriptions or guidelines for each key.
- **`max_retries`**: Retry limit for failed tasks.

### Key Methods

- **`model_post_init(self, __context: Any)`**: Ensures the initialization matches the agent’s abstraction-specific requirements.
- **`review_items(self, text_input_strings: List[str], image_path_lists: List[List[str]] = None)`**: Handles multiple inputs.
- **`review_item(self, text_input_string: str, image_path_list: List[str] = [])`**: Processes a single input.

---

## TitleAbstractReviewer Class

### Overview

The `TitleAbstractReviewer` is a specialized agent for reviewing and evaluating article titles and abstracts. It supports the evaluation of inclusion and exclusion criteria and provides a reasoning-backed evaluation score. The returned score is either of the following:

- 1 means absolutely to exclude.
- 2 means better to exclude.
- 3 Not sure if to include or exclude.
- 4 means better to include.
- 5 means absolutely to include.

### Class Definition

```python
class TitleAbstractReviewer(BasicReviewer):
    generic_prompt: Optional[str] = generic_prompt
    inclusion_criteria: str = ""
    exclusion_criteria: str = ""
    response_format: Dict[str, Any] = {"reasoning": str, "evaluation": int}
    input_description: str = "article title/abstract"
    reasoning: str = "brief"
    max_retries: int = DEFAULT_MAX_RETRIES

    def model_post_init(self, __context: Any) -> None:
        """Initialize after Pydantic model initialization."""
        try:
            assert self.reasoning != None, "Reasoning type cannot be None for TitleAbstractReviewer"
            self.setup()
        except Exception as e:
            raise AgentError(f"Error initializing agent: {str(e)}")
```

### Key Attributes

- **`generic_prompt`**: Template string for creating prompts.
- **`inclusion_criteria`**: Criteria for including an article.
- **`exclusion_criteria`**: Criteria for excluding an article.
- **`response_format`**: Structure of the evaluation output, including reasoning and evaluation.
- **`input_description`**: Description of the expected input.
- **`reasoning`**: Type of reasoning employed.
- **`max_retries`**: Retry limit for failed tasks.

### Key Methods

- **`model_post_init(self, __context: Any)`**: Ensures the initialization matches the agent’s requirements.
- **`review_items(self, text_input_strings: List[str], image_path_lists: List[List[str]] = None)`**: Handles multiple inputs.
- **`review_item(self, text_input_string: str, image_path_list: List[str] = [])`**: Processes a single input.

---

## Decision Reviewers

Decision reviewers (new in v1.3.0) use a `SystemOneProvider` and a decision model such as TypeSafe's Jev. They send each item's text as the model's state, with all questions in a single request, and return probabilities instead of generated text. They are `BasicReviewer` subclasses, so they work in any `ReviewWorkflow` next to LLM reviewers. For the differences from LLM reviewers, evaluation results and backends, see [Decision Models](../decision_models.md).

Common behavior:

- `max_concurrent_requests` defaults to 8; the provider paces requests to its `requests_per_minute`.
- Text only: an item with images raises `AgentError`. `reasoning`, `examples`, `model_args`, `generic_prompt` and `prompt_path` raise `AgentError` at creation, because decision models use no prompts. `backstory` is accepted but ignored.
- `additional_context` (a string or an async function, as for LLM reviewers) is sent with the item as the state `{"item": ..., "additional_context": ...}`.
- The output of each item also holds `_answers`, the full normalized answers (probabilities and confidence), which appears in the workflow's `output` column.
- `memory` stores the state, the questions, the response and the cost of each item.

### DecisionReviewer

The generic decision reviewer. Each question becomes one output key (and workflow column): the probability of yes for `Noul`, the chosen option for `Choice`, and the expected 0-based level for `Score`.

```python
from lattereview.agents import DecisionReviewer
from lattereview.providers import SystemOneProvider, Noul, Choice

reviewer = DecisionReviewer(
    provider=SystemOneProvider(),
    name="Extractor",
    questions={
        "modality": Choice("What is the main imaging modality?", ["CT", "MRI", "X-ray"]),
        "deep_learning": Noul("Does the study use deep learning?"),
    },
)
```

### DecisionTitleAbstractReviewer

The counterpart of `TitleAbstractReviewer`. `inclusion_criteria` and `exclusion_criteria` may be a string (one criterion), a list, or a numbered dict. Each request asks a 5-level evaluation score, an overall include question, and one yes/no question per criterion.

| Output key | Type | Meaning |
| --- | --- | --- |
| `evaluation` | int 1-5 | the most likely level of the same 1-5 rubric as `TitleAbstractReviewer` |
| `include_probability` | float | probability of meeting all inclusion and no exclusion criteria (the recommended screening signal) |
| `confidence` | float or None | the model's confidence in the evaluation score |
| `criteria` | dict | `{"inclusion": {criterion: p(met)}, "exclusion": {criterion: p(applies)}}` |
| `reasoning` | str | a summary generated by LatteReview from the probabilities (the model writes no text) |

```python
from lattereview.agents import DecisionTitleAbstractReviewer
from lattereview.providers import SystemOneProvider

reviewer = DecisionTitleAbstractReviewer(
    provider=SystemOneProvider(),
    name="Jev",
    inclusion_criteria={1: "The study must involve CT scans.", 2: "The study must use deep learning."},
    exclusion_criteria={1: "The study must not include PET scans."},
)
results, cost = await reviewer.review_items(["Title: ...\nAbstract: ..."])
```

### DecisionScoringReviewer

The counterpart of `ScoringReviewer`. `scoring_set` lists 2-10 values from lowest to highest; `score_descriptions` (optional) maps values to the answer levels the model sees; `scoring_rules` is appended to `scoring_task`. Output keys: `score` (a value from `scoring_set`), `certainty` (confidence × 100, 0-100) and `probabilities` (`{score: probability}`).

```python
from lattereview.agents import DecisionScoringReviewer

reviewer = DecisionScoringReviewer(
    provider=SystemOneProvider(),
    name="Validation",
    scoring_task="How strong is the validation of the model in this study?",
    scoring_set=[1, 2, 3],
    score_descriptions={1: "no validation", 2: "internal validation only", 3: "external validation"},
)
```

### suggest_threshold

`lattereview.utils.suggest_threshold(df, probability_column, label_column, target_recall=0.95)` returns the highest probability cutoff that reaches the target recall on labeled data, with its `recall`, `precision`, `n_included` and `wss` (work saved over sampling). Fit it on a labeled sample of your own data: cutoffs do not carry over between datasets or backends.

---

## Error Handling

The module implements robust error handling:

- **Custom Exceptions**: Classes like `AgentError` and `ReviewWorkflowError` handle specific errors.
- **Retry Mechanisms**: Configurable retry limits for failed tasks.
- **Validation**: Uses Pydantic models for input and output validation.

---

## Usage Examples

### ScoringReviewer Example

```python
from lattereview.agents import ScoringReviewer
from lattereview.providers import OpenAIProvider

# Create a ScoringReviewer instance
reviewer = ScoringReviewer(
    provider=OpenAIProvider(model="gpt-6-sol"),
    name="ContentQualityReviewer",
    scoring_task="Assess the quality of given content",
    scoring_set=[1, 2, 3, 4, 5],
    reasoning="brief"
)

# Review content
text_items = ["Content piece A", "Content piece B"]
results, costs = await reviewer.review_items(text_items)
```

### TitleAbstractReviewer Example

```python
from lattereview.agents import TitleAbstractReviewer
from lattereview.providers import OpenAIProvider

# Create a TitleAbstractReviewer instance
reviewer = TitleAbstractReviewer(
    provider=OpenAIProvider(model="gpt-6-sol"),
    inclusion_criteria="Must be a peer-reviewed study",
    exclusion_criteria="Does not focus on AI in healthcare",
    reasoning="brief"
)

# Review titles and abstracts
text_items = [
    "Title: Advances in AI\nAbstract: A review of recent progress...",
    "Title: Basic Chemistry\nAbstract: Discussion of periodic elements..."
]
results, costs = await reviewer.review_items(text_items)
```

---

## Best Practices

1. **Error Handling**:
   - Implement retry mechanisms.
   - Log errors for debugging.
2. **Performance Optimization**:
   - Set appropriate concurrency limits.
   - Process tasks in batches.
3. **Prompt Engineering**:
   - Define clear and concise prompts.
   - Include examples for better task understanding.

---

## Future Enhancements

- Enhanced support for collaborative workflows.
- Advanced memory and cost management.
- New agent types for diverse tasks.
