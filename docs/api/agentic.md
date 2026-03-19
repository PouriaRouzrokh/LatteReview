# Agentic API Reference (v2)

This page documents the v2 agentic API. All classes are importable from `lattereview.agentic`.

```python
from lattereview.agentic import (
    AgenticReviewer,
    AgenticWorkflow,
    ScoringReviewer,
    TitleAbstractReviewer,
    AbstractionReviewer,
    SkillRegistry,
)
```

---

## Choosing Between Agentic and Non-Agentic Mode

Every reviewer in the v2 API supports both modes, controlled by `max_iterations`:

- **`max_iterations=1`** (non-agentic) -- single LLM call, no tools. Fast and cheap. Equivalent to v1.
- **`max_iterations > 1`** (agentic) -- the reviewer can loop, call skill tools, build memory, flag items, and consult helpers before producing output.

!!! warning "Agentic mode is not always better"
    Enabling skills and higher iteration counts adds cost and latency. For tasks where all the information the model needs is already in the input text, agentic mode can actually **hurt** performance by making unnecessary tool calls.

    **Examples of tasks that do NOT benefit from agentic mode:**

    - Extracting imaging modality from an abstract that explicitly states "we used cardiac MRI"
    - Identifying study design when the abstract says "randomized controlled trial"
    - Simple binary screening where the inclusion criteria can be assessed directly from the title and abstract

    In these cases, giving the agent a `searching-duckduckgo` or `searching-content` skill causes it to search for information already present in the text. This wastes tokens, increases cost, and may introduce noise from irrelevant search results -- unless you are using a very capable model that exercises good judgment about when to call tools.

    **Examples of tasks that DO benefit from agentic mode:**

    - Verifying whether a study was published in a high-impact journal (requires web search)
    - Checking if a claimed result has been replicated or retracted (requires literature search)
    - Building expertise across a large batch (memory helps the reviewer learn domain patterns)
    - Flagging borderline items for human review instead of making uncertain decisions

    **Rule of thumb:** Start with `max_iterations=1`. If you see errors caused by missing context that's not in the input text, enable agentic mode with the specific skills the reviewer needs.

See the [tutorial notebooks](https://github.com/PouriaRouzrokh/LatteReview/tree/main/tutorials_agentic) and [evaluation results](https://github.com/PouriaRouzrokh/LatteReview/tree/main/evaluation) for concrete comparisons of agentic vs non-agentic performance.

---

## AgenticReviewer

The base reviewer class for the v2 agentic framework. All preset reviewer types inherit from this class. Use it directly when you need full control over prompts and output models.

::: lattereview.agentic.reviewer.AgenticReviewer

---

## AgenticWorkflow

Orchestrates multi-round review workflows over DataFrames. Supports checkpoint/resume, per-item cost tracking, and action logging.

::: lattereview.agentic.workflow.AgenticWorkflow

---

## Preset Reviewer Types

These classes provide pre-configured reviewers for common review tasks. They set appropriate system prompts, output models, and parameters automatically.

### ScoringReviewer

A reviewer that scores items on a defined scale.

::: lattereview.agentic.reviewer_types.ScoringReviewer

### TitleAbstractReviewer

A reviewer that screens articles based on title and abstract against inclusion/exclusion criteria.

::: lattereview.agentic.reviewer_types.TitleAbstractReviewer

### AbstractionReviewer

A reviewer that extracts structured data from documents according to a defined abstraction schema.

::: lattereview.agentic.reviewer_types.AbstractionReviewer

---

## Skills

### SkillRegistry

Manages the registration and loading of skills (both configuration-based and tool-based) for agentic reviewers.

::: lattereview.agentic.skills.registry.SkillRegistry

---

## Output Models

Pydantic models used as structured output schemas for reviewer responses.

### ScoringOutput

::: lattereview.agentic.output_models.ScoringOutput

### EvaluationOutput

::: lattereview.agentic.output_models.EvaluationOutput

### AbstractionOutput

::: lattereview.agentic.output_models.AbstractionOutput
