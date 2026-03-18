"""Preset reviewer types — pre-configured AgenticReviewer subclasses."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Type

from pydantic import BaseModel, Field

from lattereview.agentic.reviewer import AgenticReviewer
from lattereview.agentic.output_models import (
    ScoringOutput,
    EvaluationOutput,
    build_dynamic_output_model,
)

# --- Default prompts matching v1 style ---

_SCORING_SYSTEM_PROMPT = (
    "You are an expert reviewer. Your task is to score items according to "
    "a defined scoring rubric. Be thorough and consistent in your assessments."
)

_SCORING_TASK_TEMPLATE = """\
**Review the input item below and complete the scoring task as instructed:**

---

**Input item:**
<<${item}$>>

**Scoring task:**
<<SCORING_TASK_PLACEHOLDER>>

---

**Instructions:**

1. **Score** the input item using only the values in this set: SCORING_SET_PLACEHOLDER.
2. Follow these rules when determining your score: <<SCORING_RULES_PLACEHOLDER>>.
3. After assigning a score, report your certainty level as a value between **0** (not certain at all) and **100** (completely certain).
4. Report your certainty level after you assigned a score.
5. Provide your reasoning before assigning a score."""

_TITLE_ABSTRACT_SYSTEM_PROMPT = (
    "You are an expert systematic reviewer. Your task is to evaluate whether "
    "studies should be included or excluded based on their title and abstract. "
    "Be thorough and apply the criteria consistently."
)

_TITLE_ABSTRACT_TASK_TEMPLATE = """\
**Review the title and abstract below and evaluate whether they should be included based on the following inclusion and exclusion criteria (if any).**
**Note that the study should be included only and only if it meets ALL inclusion criteria and NONE of the exclusion criteria.**

---

**Input item:**
<<${item}$>>

---

**Inclusion criteria:**
INCLUSION_CRITERIA_PLACEHOLDER

**Exclusion criteria:**
EXCLUSION_CRITERIA_PLACEHOLDER

---

**Instructions**

1. Output your evaluation as an integer between 1 and 5, where:
   - 1 means absolutely to exclude.
   - 2 means better to exclude.
   - 3 Not sure if to include or exclude.
   - 4 means better to include.
   - 5 means absolutely to include.
2. Report your certainty level as a value between **0** (not certain at all) and **100** (completely certain).
3. Provide your reasoning before assigning a decision."""

_ABSTRACTION_SYSTEM_PROMPT = (
    "You are an expert data extraction reviewer. Your task is to extract "
    "structured information from the provided text accurately and completely."
)

_ABSTRACTION_TASK_TEMPLATE = """\
**Review the input item below and extract the specified keys as instructed:**

---

**Input Item:**
<<${item}$>>

**Keys to Extract and Their Expected Formats:**
<<ABSTRACTION_KEYS_PLACEHOLDER>>

---

**Instructions:**

Follow the detailed guidelines below for extracting the specified keys:

<<KEY_DESCRIPTIONS_PLACEHOLDER>>"""


def _build_scoring_task_prompt(scoring_task: str, scoring_set: List[int], scoring_rules: str) -> str:
    """Build the task prompt for ScoringReviewer."""
    return (
        _SCORING_TASK_TEMPLATE.replace("SCORING_TASK_PLACEHOLDER", scoring_task)
        .replace("SCORING_SET_PLACEHOLDER", str(scoring_set))
        .replace("SCORING_RULES_PLACEHOLDER", scoring_rules)
    )


def _build_ta_task_prompt(inclusion_criteria: str, exclusion_criteria: str) -> str:
    """Build the task prompt for TitleAbstractReviewer."""
    return _TITLE_ABSTRACT_TASK_TEMPLATE.replace(
        "INCLUSION_CRITERIA_PLACEHOLDER", inclusion_criteria or "None specified."
    ).replace("EXCLUSION_CRITERIA_PLACEHOLDER", exclusion_criteria or "None specified.")


def _build_abstraction_task_prompt(abstraction_keys: Dict[str, Any], key_descriptions: Dict[str, str]) -> str:
    """Build the task prompt for AbstractionReviewer."""
    keys_desc = "\n".join(
        f"- **{k}**: {v.__name__ if isinstance(v, type) else str(v)}" for k, v in abstraction_keys.items()
    )
    descs_desc = (
        "\n".join(f"- **{k}**: {v}" for k, v in key_descriptions.items())
        or "Extract each key as accurately as possible from the text."
    )
    return _ABSTRACTION_TASK_TEMPLATE.replace("ABSTRACTION_KEYS_PLACEHOLDER", keys_desc).replace(
        "KEY_DESCRIPTIONS_PLACEHOLDER", descs_desc
    )


class ScoringReviewer(AgenticReviewer):
    """Pre-configured reviewer for scoring items on a defined scale.

    Uses ScoringOutput (reasoning, score, certainty) as the output type.
    Reasoning is enforced and cannot be disabled.

    Args:
        scoring_task: Description of what to score and how.
        scoring_set: List of allowed score values (default [1, 2]).
        scoring_rules: Rules for determining the score.
        **kwargs: Any AgenticReviewer parameter can be overridden.
    """

    scoring_task: str = "Score this item based on its relevance and quality."
    scoring_set: List[int] = Field(default_factory=lambda: [1, 2])
    scoring_rules: str = "Your scores should follow the defined schema."

    def model_post_init(self, __context: Any) -> None:
        if not self.system_prompt:
            self.system_prompt = _SCORING_SYSTEM_PROMPT

        if self.task_prompt == AgenticReviewer.model_fields["task_prompt"].default:
            self.task_prompt = _build_scoring_task_prompt(self.scoring_task, self.scoring_set, self.scoring_rules)

        # Enforce output type
        self.output_type = ScoringOutput

        super().model_post_init(__context)


class TitleAbstractReviewer(AgenticReviewer):
    """Pre-configured reviewer for title/abstract screening.

    Uses EvaluationOutput (reasoning, decision, certainty) as the output type.
    Evaluates on a 1-5 Likert scale (1=exclude, 5=include).
    Reasoning is enforced and cannot be disabled.

    Args:
        inclusion_criteria: Criteria for including a study.
        exclusion_criteria: Criteria for excluding a study.
        **kwargs: Any AgenticReviewer parameter can be overridden.
    """

    inclusion_criteria: str = ""
    exclusion_criteria: str = ""

    def model_post_init(self, __context: Any) -> None:
        if not self.system_prompt:
            self.system_prompt = _TITLE_ABSTRACT_SYSTEM_PROMPT

        if self.task_prompt == AgenticReviewer.model_fields["task_prompt"].default:
            self.task_prompt = _build_ta_task_prompt(self.inclusion_criteria, self.exclusion_criteria)

        # Enforce output type
        self.output_type = EvaluationOutput

        super().model_post_init(__context)


class AbstractionReviewer(AgenticReviewer):
    """Pre-configured reviewer for structured data extraction.

    Dynamically builds an output model from abstraction_keys using
    build_dynamic_output_model(). No reasoning field (matches v1 behavior).

    Args:
        abstraction_keys: Dict mapping field names to their types.
            Example: {"study_design": str, "sample_size": int}
        key_descriptions: Dict mapping field names to extraction guidelines.
            Example: {"study_design": "The type of study (RCT, cohort, etc.)"}
        **kwargs: Any AgenticReviewer parameter can be overridden.
    """

    abstraction_keys: Dict[str, Any] = Field(default_factory=lambda: {"extracted_data": str})
    key_descriptions: Dict[str, str] = Field(default_factory=dict)

    def model_post_init(self, __context: Any) -> None:
        if not self.system_prompt:
            self.system_prompt = _ABSTRACTION_SYSTEM_PROMPT

        if self.task_prompt == AgenticReviewer.model_fields["task_prompt"].default:
            self.task_prompt = _build_abstraction_task_prompt(self.abstraction_keys, self.key_descriptions)

        # Build dynamic output model from abstraction_keys
        self.output_type = build_dynamic_output_model("AbstractionResult", self.abstraction_keys)

        super().model_post_init(__context)
