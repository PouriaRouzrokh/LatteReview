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
