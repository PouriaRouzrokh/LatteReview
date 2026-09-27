"""Title/abstract screening with a System One decision model: the counterpart of TitleAbstractReviewer."""

from typing import Any, ClassVar, Dict, List, Tuple, Union

from .basic_reviewer import AgentError
from .decision_reviewer import DecisionReviewer
from ..providers.system_one_provider import DecisionResult, Noul, Score

# The five levels of TitleAbstractReviewer's 1-5 evaluation rubric, lowest first.
EVALUATION_LEVELS = [
    "absolutely exclude",
    "better to exclude",
    "not sure whether to include or exclude",
    "better to include",
    "absolutely include",
]
SCREENING_TASK = (
    "Screen this article (title and abstract) for a systematic review. The article should be included only if it "
    "meets ALL of the inclusion criteria and NONE of the exclusion criteria."
)
REASONING_TEXT_LENGTH = 60

Criteria = Union[str, List[str], Dict[Union[int, str], str]]


class DecisionTitleAbstractReviewer(DecisionReviewer):
    """Screens titles/abstracts against inclusion and exclusion criteria with a decision model such as Jev.

    Criteria may be a string (one criterion), a list, or a dict such as {1: "...", 2: "..."}. All questions for an
    item go in one request: a 5-level `evaluation` score mirroring TitleAbstractReviewer's rubric, an overall
    `include` yes/no question, and one yes/no question per criterion.

    Output columns:
        evaluation: int 1-5, the most likely rubric level (same scale as TitleAbstractReviewer).
        include_probability: probability that the article meets all inclusion and no exclusion criteria.
        confidence: the model's confidence in the evaluation score (None if the backend does not report it).
        criteria: {"inclusion": {criterion: p(met)}, "exclusion": {criterion: p(applies)}}.
        reasoning: a summary generated from the probabilities by LatteReview. The model writes no text.
    """

    _builds_questions: ClassVar[bool] = True
    name: str = "DecisionTitleAbstractReviewer"
    inclusion_criteria: Criteria = ""
    exclusion_criteria: Criteria = ""

    @staticmethod
    def _normalize_criteria(criteria: Criteria) -> List[Tuple[str, str]]:
        """Return [(key, text), ...] for criteria given as a string, list, or dict."""
        if isinstance(criteria, str):
            items = [(1, criteria)]
        elif isinstance(criteria, dict):
            items = list(criteria.items())
        else:
            items = list(enumerate(criteria, start=1))
        return [(str(key), str(text).strip()) for key, text in items if str(text).strip()]

    @property
    def inclusion(self) -> List[Tuple[str, str]]:
        return self._normalize_criteria(self.inclusion_criteria)

    @property
    def exclusion(self) -> List[Tuple[str, str]]:
        return self._normalize_criteria(self.exclusion_criteria)

    def build_questions(self) -> Dict[str, Union[Noul, Score]]:
        inclusion, exclusion = self.inclusion, self.exclusion
        if not inclusion and not exclusion:
            raise AgentError("At least one inclusion or exclusion criterion is required")

        def listing(criteria: List[Tuple[str, str]]) -> str:
            return "\n".join(f"{key}. {text}" for key, text in criteria) if criteria else "None."

        screening = (
            f"{SCREENING_TASK}\n\n"
            f"Inclusion criteria:\n{listing(inclusion)}\n\n"
            f"Exclusion criteria:\n{listing(exclusion)}"
        )
        questions = {
            "evaluation": Score(f"{screening}\n\nHow strongly should this article be included?", EVALUATION_LEVELS),
            "include": Noul(
                f"{screening}\n\nDoes the article meet ALL of the inclusion criteria and NONE of the exclusion "
                "criteria?",
                true="include: meets every inclusion criterion and no exclusion criterion",
                false="exclude: fails an inclusion criterion or meets an exclusion criterion",
            ),
        }
        for key, text in inclusion:
            questions[f"inc_{key}"] = Noul(
                f"Inclusion criterion: {text}\n\nDoes the article satisfy this inclusion criterion?",
                true="the article satisfies this inclusion criterion",
                false="the article does not satisfy this inclusion criterion",
            )
        for key, text in exclusion:
            questions[f"exc_{key}"] = Noul(
                f"Exclusion criterion: {text}\n\nShould the article be excluded because of this criterion?",
                true="yes, this criterion excludes the article",
                false="no, this criterion does not exclude the article",
            )
        return questions

    def build_response_format(self) -> Dict[str, Any]:
        return {
            "evaluation": int,
            "include_probability": float,
            "confidence": float,
            "criteria": dict,
            "reasoning": str,
        }

    def format_response(self, result: DecisionResult) -> Dict[str, Any]:
        answers = result.answers
        evaluation = answers["evaluation"]
        criteria = {
            "inclusion": {text: answers[f"inc_{key}"].value for key, text in self.inclusion},
            "exclusion": {text: answers[f"exc_{key}"].value for key, text in self.exclusion},
        }
        response = {
            "evaluation": evaluation.level + 1,
            "include_probability": answers["include"].value,
            "confidence": evaluation.confidence,
            "criteria": criteria,
            "reasoning": self._generate_reasoning(answers["include"].value, evaluation.label, result),
        }
        response["_answers"] = super().format_response(result)["_answers"]
        return response

    def _generate_reasoning(self, include_probability: float, evaluation_label: str, result: DecisionResult) -> str:
        """Summarize the probabilities in words. This is generated by LatteReview, not written by the model."""

        def short(text: str) -> str:
            return text if len(text) <= REASONING_TEXT_LENGTH else text[: REASONING_TEXT_LENGTH - 1].rstrip() + "…"

        parts = [f"P(include)={include_probability:.2f}; rated '{evaluation_label}'."]
        if self.inclusion:
            probabilities = {key: result.answers[f"inc_{key}"].value for key, _ in self.inclusion}
            failed = [(key, text) for key, text in self.inclusion if probabilities[key] < 0.5]
            for key, text in failed:
                parts.append(f"Likely fails inclusion {key} (p={probabilities[key]:.2f}: '{short(text)}').")
            if not failed:
                parts.append(f"Likely meets all inclusion criteria (lowest p={min(probabilities.values()):.2f}).")
        if self.exclusion:
            probabilities = {key: result.answers[f"exc_{key}"].value for key, _ in self.exclusion}
            applied = [(key, text) for key, text in self.exclusion if probabilities[key] >= 0.5]
            for key, text in applied:
                parts.append(f"Likely meets exclusion {key} (p={probabilities[key]:.2f}: '{short(text)}').")
            if not applied:
                parts.append(f"No exclusion criterion likely applies (highest p={max(probabilities.values()):.2f}).")
        return " ".join(parts)
