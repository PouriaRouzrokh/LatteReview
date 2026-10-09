"""Scoring with a System One decision model: the counterpart of ScoringReviewer."""

from typing import Any, ClassVar, Dict, List, Optional

from .basic_reviewer import AgentError
from .decision_reviewer import DecisionReviewer
from ..providers.system_one_provider import DecisionResult, Score


class DecisionScoringReviewer(DecisionReviewer):
    """Scores items on an ordinal scale with a decision model such as Jev.

    `scoring_set` lists 2-10 score values from lowest to highest. `score_descriptions` optionally maps each value to
    what it means (the model sees these descriptions as the answer levels; values without one are shown as numbers).
    `scoring_rules` is appended to the task.

    Output columns:
        score: the most likely value from scoring_set.
        certainty: int 0-100, the model's confidence x 100 (same scale as ScoringReviewer; None if not reported).
        probabilities: {score value: probability}.

    All three are None if the model declined to answer.
    """

    _builds_questions: ClassVar[bool] = True
    name: str = "DecisionScoringReviewer"
    scoring_task: Optional[str] = None
    scoring_set: List[int] = [1, 2]
    score_descriptions: Optional[Dict[int, str]] = None
    scoring_rules: Optional[str] = None

    def build_questions(self) -> Dict[str, Score]:
        if not self.scoring_task or not self.scoring_task.strip():
            raise AgentError("scoring_task is required")
        if len(set(self.scoring_set)) != len(self.scoring_set):
            raise AgentError(f"scoring_set values must be unique, got {self.scoring_set}")
        descriptions = self.score_descriptions or {}
        unknown = set(descriptions) - set(self.scoring_set)
        if unknown:
            raise AgentError(f"score_descriptions has values not in scoring_set: {sorted(unknown)}")
        levels = [descriptions.get(value, str(value)) for value in self.scoring_set]
        instructions = self.scoring_task.strip()
        if self.scoring_rules:
            instructions += f"\n\nRules: {self.scoring_rules}"
        return {"score": Score(instructions, levels)}

    def build_response_format(self) -> Dict[str, Any]:
        return {"score": int, "certainty": int, "probabilities": dict}

    def format_response(self, result: DecisionResult) -> Dict[str, Any]:
        answer = result.answers["score"]
        probabilities = None
        if answer.probabilities is not None:
            probabilities = dict(zip(self.scoring_set, answer.probabilities.values()))
        response = {
            "score": self.scoring_set[answer.level] if answer.level is not None else None,
            "certainty": round(answer.confidence * 100) if answer.confidence is not None else None,
            "probabilities": probabilities,
        }
        response["_answers"] = super().format_response(result)["_answers"]
        return response
