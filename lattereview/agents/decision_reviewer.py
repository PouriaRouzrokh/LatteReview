"""Reviewer for System One decision models (e.g., TypeSafe's Jev), which answer typed questions with probabilities."""

import re
from typing import Annotated, Any, Callable, ClassVar, Dict, List, Union

import pydantic

from .basic_reviewer import BasicReviewer, AgentError
from ..providers.system_one_provider import (
    SystemOneProvider,
    SystemOneResponseError,
    DecisionResult,
    Noul,
    Choice,
    Score,
)

# Decision models answer in ~0.2 s, so a few concurrent requests already reach the provider's requests_per_minute pace.
DEFAULT_CONCURRENT_REQUESTS = 8
# ReviewWorkflow starts every item with this line to track LLM outputs. Decision models have no use for it, and it
# shifts their answers on borderline items, so it is left out of the state.
TASK_ID_LINE = re.compile(r"\AReview Task ID: [^\n]*\n")
DEFAULT_MAX_RETRIES = 3

Question = Annotated[Union[Noul, Choice, Score], pydantic.Field(discriminator="type")]
ANSWER_TYPES = {"noul": float, "choice": str, "score": float}


class DecisionReviewer(BasicReviewer):
    """Reviews items with a System One decision model instead of an LLM.

    Each item's text is sent as the model's state, together with all `questions` in a single request. The output has
    one key per question holding the answer's value (noul: probability of yes; choice: the chosen option; score: the
    expected 0-based level), plus `_answers` with the full normalized answers (probabilities and confidence).

    If the backend declines to answer a question (OpenAI may refuse single questions), its value is None and its
    `_answers` entry has `refused=True`; a warning is printed the first time. Route such items to an LLM or a human.

    `provider` must be a SystemOneProvider. Decision models return no written text, so `reasoning`, `examples`,
    `model_args`, prompt templates, and images are not supported, and `backstory` is ignored.
    """

    questions: Dict[str, Question] = {}
    max_concurrent_requests: int = DEFAULT_CONCURRENT_REQUESTS
    name: str = "DecisionReviewer"
    backstory: str = "a System One decision model"
    input_description: str = "article title/abstract"
    max_retries: int = DEFAULT_MAX_RETRIES
    # Presets build their questions from their own fields (criteria, scoring task) and reject user-given questions.
    _builds_questions: ClassVar[bool] = False
    _warned_refusal: bool = pydantic.PrivateAttr(default=False)

    def model_post_init(self, __context: Any) -> None:
        """Reject LLM-only options, then build the questions and response format."""
        try:
            self._check_unsupported_options()
            self.setup()
        except Exception as e:
            raise AgentError(f"Error initializing agent: {str(e)}")

    def _check_unsupported_options(self) -> None:
        unsupported = {
            "reasoning": self.reasoning is not None,
            "examples": bool(self.examples),
            "model_args": bool(self.model_args),
            "generic_prompt": bool(self.generic_prompt),
            "prompt_path": bool(self.prompt_path),
        }
        if self._builds_questions and self.questions:
            raise AgentError(
                f"{type(self).__name__} builds its questions from its own fields; use DecisionReviewer for custom "
                "questions"
            )
        for option, is_set in unsupported.items():
            if is_set:
                raise AgentError(
                    f"`{option}` is not supported by {type(self).__name__}: decision models answer predefined "
                    "questions with probabilities and do not use prompts or write text. Configure the model on the "
                    "SystemOneProvider, or use an LLM reviewer for free-text output."
                )

    def setup(self) -> None:
        """Build the questions, response format, and identity. No prompt template is involved."""
        try:
            if not isinstance(self.provider, SystemOneProvider):
                raise AgentError(
                    f"{type(self).__name__} needs a SystemOneProvider (e.g., SystemOneProvider(backend='typesafe')), "
                    f"got {type(self.provider).__name__}"
                )
            self.questions = self.build_questions()
            if not self.questions:
                raise AgentError("At least one question is required")
            self.response_format = self.build_response_format()
            self.system_prompt = None
            self.formatted_prompt = None
            self.identity = {
                "questions": {qid: question.to_payload() for qid, question in self.questions.items()},
                "endpoint": self.provider.endpoint,
                "model": self.provider.model,
            }
        except Exception as e:
            raise AgentError(f"Error in setup: {str(e)}")

    def build_questions(self) -> Dict[str, Union[Noul, Choice, Score]]:
        """Return the questions to ask about every item. Presets override this to build them from their fields."""
        return self.questions

    def build_response_format(self) -> Dict[str, Any]:
        """Return the output keys (one workflow column each) and their types."""
        return {qid: ANSWER_TYPES[question.type] for qid, question in self.questions.items()}

    def format_response(self, result: DecisionResult) -> Dict[str, Any]:
        """Turn a DecisionResult into the reviewer's output dict."""
        response = {qid: answer.value for qid, answer in result.answers.items()}
        response["_answers"] = {qid: answer.model_dump() for qid, answer in result.answers.items()}
        return response

    async def _build_state(self, text_input_string: str) -> Union[str, Dict[str, str]]:
        """Return the item text, or {"item", "additional_context"} when additional context is set.

        The workflow's "Review Task ID" line is left out of the state, so an item gets the same answer whatever its
        row or round. A callable additional_context still receives the full text, including that line.
        """
        item = TASK_ID_LINE.sub("", text_input_string)
        if not self.additional_context:
            return item
        if isinstance(self.additional_context, str):
            context = self.additional_context
        elif isinstance(self.additional_context, Callable):
            context = await self.additional_context(text_input_string)
        else:
            raise AgentError("Additional context must be a string or callable")
        return {"item": item, "additional_context": context} if context else item

    def _warn_on_refusal(self, result: DecisionResult) -> None:
        """Print a warning the first time the backend declines to answer a question for this reviewer."""
        refused = [qid for qid, answer in result.answers.items() if answer.refused]
        if refused and not self._warned_refusal:
            self._warned_refusal = True
            print(
                f"Warning: {self.name}: the model declined to answer {', '.join(refused)} for an item, so those "
                "outputs are None. Route items with missing answers to an LLM or a human reviewer. (Shown once.)"
            )

    async def review_item(
        self, text_input_string: str, image_path_list: List[str] = []
    ) -> tuple[Dict[str, Any], Dict[str, Any], float]:
        """Review one item with a single request carrying all questions."""
        if image_path_list:
            raise AgentError(
                f"{type(self).__name__} accepts text only; remove image_inputs from this round or use an LLM reviewer."
            )
        num_tried = 0
        last_error = None
        while num_tried < self.max_retries:
            try:
                state = await self._build_state(text_input_string)
                result = await self.provider.decide(state, self.questions)
                self._warn_on_refusal(result)
                input_prompt = {"state": state, "questions": self.identity["questions"]}
                return self.format_response(result), input_prompt, result.cost
            except Exception as e:
                last_error = e
                num_tried += 1
                # The provider already retried transient HTTP errors; retry only malformed responses (HTTP 200) and
                # errors outside the request (e.g., an additional_context function), so retries do not multiply.
                if isinstance(e, SystemOneResponseError) and e.status_code != 200:
                    break
                self._log(f"Error reviewing item: {str(e)}. Retrying {num_tried}/{self.max_retries}")
        raise AgentError(
            f"Failed to review item after {num_tried} attempt(s) "
            f"with model <{self.provider.model or self.provider.endpoint}>. Last error: {last_error}"
        )
