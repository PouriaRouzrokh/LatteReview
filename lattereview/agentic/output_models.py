"""Pydantic output models for structured reviewer responses."""

from __future__ import annotations

from typing import Any, Dict, Type

from pydantic import BaseModel, Field, create_model


class ScoringOutput(BaseModel):
    """Standard output for scoring reviewers."""

    reasoning: str = Field(description="Step-by-step reasoning for the score.")
    score: int = Field(description="Score assigned to the item.")
    certainty: int = Field(ge=0, le=100, description="Confidence in the score (0-100).")


class EvaluationOutput(BaseModel):
    """Standard output for binary inclusion/exclusion reviewers."""

    reasoning: str = Field(description="Step-by-step reasoning for the decision.")
    decision: str = Field(description="Inclusion decision (e.g., 'include' or 'exclude').")
    certainty: int = Field(ge=0, le=100, description="Confidence in the decision (0-100).")


class AbstractionOutput(BaseModel):
    """Standard output for abstraction/extraction reviewers."""

    reasoning: str = Field(description="Step-by-step reasoning for the abstraction.")
    extracted_data: Dict[str, Any] = Field(description="Extracted key-value data from the item.")


def build_dynamic_output_model(
    name: str,
    fields: Dict[str, Any],
) -> Type[BaseModel]:
    """Create a dynamic Pydantic model from a field specification dict.

    Args:
        name: Name for the generated model class.
        fields: Dict mapping field names to their types or (type, Field) tuples.
            Examples:
                {"score": int, "reasoning": str}
                {"score": (int, Field(ge=0, le=10)), "reasoning": (str, Field(description="..."))}

    Returns:
        A dynamically created Pydantic BaseModel subclass.
    """
    model_fields = {}
    for field_name, field_spec in fields.items():
        if isinstance(field_spec, tuple):
            model_fields[field_name] = field_spec
        else:
            model_fields[field_name] = (field_spec, ...)

    return create_model(name, **model_fields)
