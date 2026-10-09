"""Provider for System One decision models, such as TypeSafe's Jev, Perplexity's pplx-decider, and OpenAI's gpt-6-luna.

Decision models do not generate text. They read a text "state" and a set of typed questions, and return a probability
for every allowed answer. Most backends (TypeSafe, OpenRouter, Perplexity, a self-hosted OpenJev server, or any other
gateway) speak the /v1/systemone protocol, so they differ only by URL, API key, and model name. OpenAI's Decisions API
asks the same kinds of questions in its own format; the provider translates it, so reviewers see one result type.
"""

import asyncio
import json
import os
import random
import time
from typing import Any, ClassVar, Dict, List, Literal, Optional, Tuple, Union

import httpx
import pydantic

from .base_provider import ProviderError, ClientCreationError, ResponseError

SYSTEMONE_PATH = "/v1/systemone"
DECISIONS_PATH = "/v1/decisions"  # Perplexity serves the System One protocol here; OpenAI serves its own format.

# List prices in USD per million input tokens. Output tokens are free on every backend.
JEV_INPUT_PRICE_PER_MILLION = 0.042
PERPLEXITY_INPUT_PRICE_PER_MILLION = 0.02
OPENAI_INPUT_PRICE_PER_MILLION = 0.10

# Client-side request pacing, a margin below each backend's limit: 1,200 requests per minute on TypeSafe and 10 requests
# per second on Perplexity. OpenAI allows 30,000 per minute, so its requests are not paced.
DEFAULT_REQUESTS_PER_MINUTE = 1000
PERPLEXITY_REQUESTS_PER_MINUTE = 500

# Named backends: endpoint, protocol, default model, the environment variables that may hold the API key (first match
# wins), the list price used when the response reports no cost, and the request pacing.
BACKENDS = {
    "typesafe": {
        "endpoint": "https://api.typesafe.ai" + SYSTEMONE_PATH,
        "protocol": "systemone",
        "model": "jev-latest",
        "env_vars": ("TYPESAFE_API_KEY",),
        "price": JEV_INPUT_PRICE_PER_MILLION,
        "requests_per_minute": DEFAULT_REQUESTS_PER_MINUTE,
    },
    "openrouter": {
        "endpoint": "https://openrouter.ai/api" + SYSTEMONE_PATH,
        "protocol": "systemone",
        "model": "~typesafe/jev-latest",
        "env_vars": ("OPENROUTER_API_KEY",),
        "price": JEV_INPUT_PRICE_PER_MILLION,  # OpenRouter reports the actual cost of every model in the response
        "requests_per_minute": DEFAULT_REQUESTS_PER_MINUTE,
    },
    "perplexity": {
        "endpoint": "https://api.perplexity.ai" + DECISIONS_PATH,
        "protocol": "systemone",
        "model": "pplx-decider-v1.1-27b",
        "env_vars": ("PERPLEXITY_API_KEY", "PERPLEXITYAI_API_KEY"),  # the second is LiteLLM's name
        "price": PERPLEXITY_INPUT_PRICE_PER_MILLION,
        "requests_per_minute": PERPLEXITY_REQUESTS_PER_MINUTE,
    },
    "openai": {
        "endpoint": "https://api.openai.com" + DECISIONS_PATH,
        "protocol": "openai",
        "model": "gpt-6-luna",
        "env_vars": ("OPENAI_API_KEY",),
        "price": OPENAI_INPUT_PRICE_PER_MILLION,
        "requests_per_minute": 0,
    },
}

# OpenAI's question and answer type names for the System One ones.
OPENAI_QUESTION_TYPES = {"noul": "predicate", "choice": "choice", "score": "score"}

# Transient HTTP statuses worth retrying: timeouts, rate limits (429), server errors, and "overloaded" (529).
RETRY_STATUS_CODES = {408, 429, 500, 502, 503, 504, 529}
MAX_RETRY_DELAY = 60.0

SCORE_MIN_LEVELS, SCORE_MAX_LEVELS = 2, 10


class SystemOneResponseError(ResponseError):
    """Raised when a /v1/systemone request fails. `retryable` is False for errors retrying cannot fix (e.g., 401)."""

    def __init__(self, message: str, status_code: Optional[int] = None, retryable: bool = True):
        super().__init__(message)
        self.status_code = status_code
        self.retryable = retryable


def _text(value: Any) -> str:
    """Return a value as text: a string as it is, a JSON structure (e.g., a structured option or level) serialized."""
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


class _Question(pydantic.BaseModel):
    """Base class for System One questions. Fields listed in `_positional` may also be passed positionally."""

    _positional: ClassVar[Tuple[str, ...]] = ("instructions",)
    instructions: Union[str, Dict[str, Any], List[Any]]

    def __init__(self, *args: Any, **data: Any) -> None:
        if len(args) > len(self._positional):
            raise TypeError(f"{type(self).__name__} takes at most {len(self._positional)} positional arguments")
        for name, value in zip(self._positional, args):
            if name in data:
                raise TypeError(f"{type(self).__name__} got multiple values for argument '{name}'")
            data[name] = value
        super().__init__(**data)

    @pydantic.field_validator("instructions")
    @classmethod
    def _check_instructions(cls, value: Any) -> Any:
        if not value or (isinstance(value, str) and not value.strip()):
            raise ValueError("instructions cannot be empty")
        return value

    def to_payload(self) -> Dict[str, Any]:
        """Return the question as the /v1/systemone request expects it."""
        raise NotImplementedError


class Noul(_Question):
    """A yes/no question. The answer is the probability that the answer is yes (true).

    Optional `true` and `false` describe what each answer means.
    """

    type: Literal["noul"] = "noul"
    true: Optional[Any] = None
    false: Optional[Any] = None

    def to_payload(self) -> Dict[str, Any]:
        payload = {"type": self.type, "instructions": self.instructions}
        criteria = {key: value for key, value in (("true", self.true), ("false", self.false)) if value is not None}
        if criteria:
            payload["criteria"] = criteria
        return payload


class Choice(_Question):
    """A multiple-choice question. `options` maps each option to its description, or is a list of option names."""

    _positional: ClassVar[Tuple[str, ...]] = ("instructions", "options")
    type: Literal["choice"] = "choice"
    options: Union[Dict[str, Any], List[str]]

    @pydantic.field_validator("options")
    @classmethod
    def _check_options(cls, value: Union[Dict[str, Any], List[str]]) -> Dict[str, Any]:
        if isinstance(value, list):
            if len(set(value)) != len(value):
                raise ValueError("choice options must be unique")
            value = {option: option for option in value}
        if len(value) < 2:
            raise ValueError("a choice question needs at least 2 options")
        return value

    def to_payload(self) -> Dict[str, Any]:
        return {"type": self.type, "instructions": self.instructions, "criteria": dict(self.options)}


class Score(_Question):
    """An ordinal question. `levels` lists 2-10 answer levels from lowest to highest."""

    _positional: ClassVar[Tuple[str, ...]] = ("instructions", "levels")
    type: Literal["score"] = "score"
    levels: List[Any]

    @pydantic.field_validator("levels")
    @classmethod
    def _check_levels(cls, value: List[Any]) -> List[Any]:
        if not SCORE_MIN_LEVELS <= len(value) <= SCORE_MAX_LEVELS:
            raise ValueError(f"a score question needs {SCORE_MIN_LEVELS}-{SCORE_MAX_LEVELS} levels, got {len(value)}")
        if len({_text(level) for level in value}) != len(value):
            raise ValueError("score levels must be unique")
        return value

    def to_payload(self) -> Dict[str, Any]:
        return {"type": self.type, "instructions": self.instructions, "criteria": list(self.levels)}


Question = Union[Noul, Choice, Score]
QUESTION_TYPES = {"noul": Noul, "choice": Choice, "score": Score}


class Answer(pydantic.BaseModel):
    """One question's answer, normalized across backends.

    - noul: `value` is the probability of "yes". There is no confidence: the probability itself shows certainty.
    - choice: `value` and `label` are the chosen option; `probabilities` maps each option to its probability.
    - score: `value` is the expected level (0-based, may be fractional); `level` and `label` are the most likely level;
      `probabilities` maps each level's label to its probability, in level order.

    `confidence` is None for noul answers and whenever the backend does not report one.

    `refused` is True when the backend declined to answer the question (OpenAI may refuse single questions); every
    other field except `type` is then None.
    """

    type: Literal["noul", "choice", "score"]
    value: Union[float, str, None]
    label: Optional[str] = None
    level: Optional[int] = None
    probabilities: Optional[Dict[str, float]] = None
    confidence: Optional[float] = None
    refused: bool = False


class DecisionResult(pydantic.BaseModel):
    """The normalized result of one decision request. `raw` is the response exactly as the backend sent it."""

    answers: Dict[str, Answer]
    model: Optional[str] = None
    input_tokens: int = 0
    cost: float = 0.0
    raw: Dict[str, Any] = {}


class SystemOneProvider(pydantic.BaseModel):
    """Client for decision models: the /v1/systemone protocol and OpenAI's Decisions API.

    Named backends:
        - "typesafe" (default): TypeSafe's Jev (`jev-latest`), key from TYPESAFE_API_KEY.
        - "openrouter": OpenRouter (`~typesafe/jev-latest`; also e.g. `perplexity/pplx-decider-v1.1-27b`), key from
          OPENROUTER_API_KEY.
        - "perplexity": Perplexity's Decisions API (`pplx-decider-v1.1-27b`), key from PERPLEXITY_API_KEY.
        - "openai": OpenAI's Decisions API (`gpt-6-luna`), key from OPENAI_API_KEY.

    Or pass `base_url` for any other server, such as a self-hosted OpenJev (`http://localhost:3000`) or another gateway;
    custom servers may not need an API key. A custom server speaks the System One protocol unless `protocol="openai"`.
    Transient failures (429, 529, 5xx, timeouts) are retried with exponential backoff, honoring Retry-After.

    Requests are paced to `requests_per_minute` (default 1,000 for TypeSafe and OpenRouter and 500 for Perplexity, a
    margin below their limits; unlimited for OpenAI and custom servers). Pass 0 to turn pacing off.
    """

    provider: str = "SystemOne"
    backend: Literal["typesafe", "openrouter", "perplexity", "openai"] = "typesafe"
    base_url: Optional[str] = None
    protocol: Optional[Literal["systemone", "openai"]] = None
    api_key: Optional[str] = pydantic.Field(default=None, repr=False)
    model: Optional[str] = None
    timeout: float = 60.0
    max_retries: int = 3
    requests_per_minute: Optional[float] = None
    input_price_per_million: Optional[float] = None
    endpoint: Optional[str] = None
    transport: Optional[Any] = pydantic.Field(default=None, repr=False)  # an httpx.AsyncBaseTransport, e.g. for tests

    _client: Optional[httpx.AsyncClient] = pydantic.PrivateAttr(default=None)
    _client_loop: Optional[asyncio.AbstractEventLoop] = pydantic.PrivateAttr(default=None)
    _next_request_at: float = pydantic.PrivateAttr(default=0.0)

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    def model_post_init(self, __context: Any) -> None:
        """Resolve the endpoint, protocol, model, API key, and price from the backend or the custom base_url."""
        if self.base_url:
            url = self.base_url.rstrip("/")
            self.protocol = self.protocol or "systemone"
            if not url.endswith((SYSTEMONE_PATH, DECISIONS_PATH)):
                url += DECISIONS_PATH if self.protocol == "openai" else SYSTEMONE_PATH
            self.endpoint = url
            if self.input_price_per_million is None:
                self.input_price_per_million = 0.0
            return

        preset = BACKENDS[self.backend]
        if self.protocol not in (None, preset["protocol"]):
            raise ClientCreationError(
                f"The '{self.backend}' backend uses the {preset['protocol']} protocol; `protocol` applies only to a "
                "custom base_url."
            )
        self.endpoint = preset["endpoint"]
        self.protocol = preset["protocol"]
        self.model = self.model or preset["model"]
        self.api_key = self.api_key or next(filter(None, map(os.getenv, preset["env_vars"])), None)
        if not self.api_key:
            raise ClientCreationError(
                f"No API key for the '{self.backend}' backend. Pass api_key or set the {preset['env_vars'][0]} "
                "environment variable."
            )
        if self.input_price_per_million is None:
            self.input_price_per_million = preset["price"]
        if self.requests_per_minute is None:
            self.requests_per_minute = preset["requests_per_minute"]

    def _get_client(self) -> httpx.AsyncClient:
        """Return the shared HTTP client, creating it for the running event loop if needed.

        A client's connections belong to the event loop that opened them, so scripts that call asyncio.run() more
        than once get a fresh client for each loop. Notebooks using nest_asyncio keep a single loop and client.
        """
        loop = asyncio.get_running_loop()
        if self._client is None or self._client.is_closed or self._client_loop is not loop:
            self._client = httpx.AsyncClient(timeout=self.timeout, transport=self.transport)
            self._client_loop = loop
        return self._client

    async def aclose(self) -> None:
        """Close the HTTP client."""
        if self._client is not None and not self._client.is_closed:
            try:
                await self._client.aclose()
            except RuntimeError:  # the client's event loop is already closed
                pass
        self._client = None
        self._client_loop = None

    async def _pace(self) -> None:
        """Wait for this request's slot so that requests stay under requests_per_minute."""
        if not self.requests_per_minute:
            return
        now = time.monotonic()
        slot = max(now, self._next_request_at)
        self._next_request_at = slot + 60.0 / self.requests_per_minute  # reserved before awaiting, so no lock needed
        if slot > now:
            await asyncio.sleep(slot - now)

    def _headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    async def decide(
        self,
        state: Union[str, Dict[str, Any], List[Any]],
        questions: Dict[str, Union[Question, Dict[str, Any]]],
    ) -> DecisionResult:
        """Ask all questions about one state in a single request and return the normalized answers.

        On System One backends the state is billed once per request and each extra question adds only a few tokens, so
        ask every question about an item together. OpenAI bills the state once per question instead. Questions may be
        Noul/Choice/Score objects or raw payload dicts.
        """
        if not state:
            raise ProviderError("The state cannot be empty")
        questions = self._to_questions(questions)
        if self.protocol == "openai":
            data = await self._post(self._openai_body(state, questions))
            return self._parse_result(data, questions, answers=self._openai_answers(data, questions))
        body = {"state": state, "questions": {qid: question.to_payload() for qid, question in questions.items()}}
        if self.model:
            body["model"] = self.model
        data = await self._post(body)
        return self._parse_result(data, questions)

    def _openai_body(self, state: Union[str, Dict[str, Any], List[Any]], questions: Dict[str, Question]) -> Dict:
        """Translate a System One request into OpenAI's Decisions format.

        OpenAI takes text input and plain-text instructions, so JSON states and instructions are serialized. Predicates
        have no answer descriptions, so a noul's `true`/`false` descriptions are added to its instructions.
        """
        openai_questions = []
        for qid, question in questions.items():
            instructions = _text(question.instructions)
            if isinstance(question, Noul):
                for answer, meaning in (("true", question.true), ("false", question.false)):
                    if meaning is not None:
                        instructions += f"\nAnswer {answer} if: {_text(meaning)}"
            payload = {"type": OPENAI_QUESTION_TYPES[question.type], "name": qid, "instructions": instructions}
            if isinstance(question, Choice):
                payload["choices"] = [
                    (
                        {"value": option}
                        if description in (None, option)
                        else {"value": option, "description": _text(description)}
                    )
                    for option, description in question.options.items()
                ]
            elif isinstance(question, Score):
                payload["levels"] = [{"label": _text(level)} for level in question.levels]
            openai_questions.append(payload)
        body = {"input": _text(state), "questions": openai_questions}
        if self.model:
            body["model"] = self.model
        return body

    def _openai_answers(self, data: Dict[str, Any], questions: Dict[str, Question]) -> Dict[str, Any]:
        """Translate OpenAI's answers (a list, matched by name or else by position) into System One answers."""
        answers = data.get("answers")
        if not isinstance(answers, list):
            raise SystemOneResponseError(f"{self._where()} response has no answers: {json.dumps(data)[:300]}", 200)
        by_name = {answer.get("name"): answer for answer in answers if isinstance(answer, dict) and answer.get("name")}
        converted = {}
        for position, qid in enumerate(questions):
            answer = by_name.get(qid)
            if answer is None and not by_name and position < len(answers):
                answer = answers[position]
            if not isinstance(answer, dict):
                continue  # reported as a missing answer by _parse_result
            try:
                kind = answer.get("type")
                if kind == "predicate":
                    converted[qid] = {"type": "noul", "noul": answer["probability"]}
                elif kind == "choice":
                    probabilities = {item["value"]: item["probability"] for item in answer.get("probabilities") or []}
                    converted[qid] = {
                        "type": "choice",
                        "choice": answer.get("choice"),
                        "probabilities": probabilities or None,
                    }
                elif kind == "score":
                    probabilities = {
                        str(item["value"]): item["probability"] for item in answer.get("probabilities") or []
                    }
                    converted[qid] = {
                        "type": "score",
                        "score": answer.get("score"),
                        "probabilities": probabilities or None,
                    }
                else:
                    converted[qid] = answer  # refusals pass through; unknown types fail in _parse_answer
                if kind in ("choice", "score"):
                    converted[qid]["confidence"] = answer.get("confidence")
            except (KeyError, TypeError) as e:
                raise SystemOneResponseError(
                    f"{self._where()} returned an unreadable answer for question '{qid}': {answer} ({e})", 200
                )
        return converted

    def _to_questions(self, questions: Dict[str, Union[Question, Dict[str, Any]]]) -> Dict[str, Question]:
        """Validate the questions, converting raw payload dicts into Noul/Choice/Score objects."""
        if not questions:
            raise ProviderError("At least one question is required")
        converted = {}
        for qid, question in questions.items():
            if not isinstance(qid, str) or not qid:
                raise ProviderError(f"Question IDs must be non-empty strings, got {qid!r}")
            if isinstance(question, dict):
                question = dict(question)
                question_type = question.pop("type", None)
                if question_type not in QUESTION_TYPES:
                    raise ProviderError(f"Question '{qid}' has an unknown type {question_type!r}")
                criteria = question.pop("criteria", None)
                if question_type == "noul" and criteria:
                    question.update(criteria)
                elif question_type == "choice":
                    question.setdefault("options", criteria)
                elif question_type == "score":
                    question.setdefault("levels", criteria)
                question = QUESTION_TYPES[question_type](**question)
            elif not isinstance(question, (Noul, Choice, Score)):
                raise ProviderError(f"Question '{qid}' must be a Noul, Choice, or Score, got {type(question)}")
            converted[qid] = question
        return converted

    async def _post(self, body: Dict[str, Any]) -> Dict[str, Any]:
        """POST the request, retrying transient failures with exponential backoff."""
        client = self._get_client()
        attempt = 0
        while True:
            retry_after = None
            await self._pace()
            try:
                response = await client.post(self.endpoint, json=body, headers=self._headers())
                if response.status_code == 200:
                    try:
                        data = response.json()
                    except ValueError:
                        raise SystemOneResponseError(
                            f"{self._where()} returned a non-JSON response: {response.text[:200]}", 200
                        )
                    if not isinstance(data, dict):
                        raise SystemOneResponseError(f"{self._where()} returned an unexpected response: {data!r}", 200)
                    return data
                retryable = response.status_code in RETRY_STATUS_CODES
                error = SystemOneResponseError(
                    f"{self._where()} returned HTTP {response.status_code}: {self._error_message(response)}",
                    response.status_code,
                    retryable,
                )
                retry_after = self._retry_after(response)
            except httpx.TransportError as e:  # timeouts, connection errors
                error = SystemOneResponseError(f"{self._where()} request failed: {type(e).__name__}: {e}")
            if not error.retryable or attempt >= self.max_retries:
                raise error
            attempt += 1
            delay = retry_after if retry_after is not None else min(2 ** (attempt - 1), 30) + random.uniform(0, 0.5)
            await asyncio.sleep(min(delay, MAX_RETRY_DELAY))

    def _where(self) -> str:
        return f"{self.endpoint}" + (f" (model <{self.model}>)" if self.model else "")

    @staticmethod
    def _retry_after(response: httpx.Response) -> Optional[float]:
        """Return the Retry-After delay in seconds, if the server sent one as a number."""
        try:
            return max(0.0, float(response.headers["retry-after"]))
        except (KeyError, ValueError):
            return None

    @staticmethod
    def _error_message(response: httpx.Response) -> str:
        """Extract a readable message from an error response (TypeSafe, FastAPI validation, OpenRouter or OpenAI)."""
        try:
            data = response.json()
        except ValueError:
            return response.text[:500] or response.reason_phrase
        if isinstance(data, dict):
            detail = data.get("detail", data.get("error"))
            if isinstance(detail, dict):
                return str(detail.get("message") or detail)
            if isinstance(detail, list):
                return "; ".join(
                    (
                        f"{'.'.join(str(part) for part in item.get('loc', []))}: {item.get('msg')}"
                        if isinstance(item, dict)
                        else str(item)
                    )
                    for item in detail
                )
            if detail:
                return str(detail)
        return json.dumps(data)[:500]

    def _parse_result(
        self, data: Dict[str, Any], questions: Dict[str, Question], answers: Optional[Dict[str, Any]] = None
    ) -> DecisionResult:
        """Normalize a response into a DecisionResult, whatever the backend's quirks.

        `answers` overrides the response's own answers, for backends whose answers were translated first.
        """
        answers = data.get("answers") if answers is None else answers
        if not isinstance(answers, dict):
            raise SystemOneResponseError(f"{self._where()} response has no answers: {json.dumps(data)[:300]}", 200)
        normalized = {}
        for qid, question in questions.items():
            if not isinstance(answers.get(qid), dict):
                raise SystemOneResponseError(f"{self._where()} response has no answer for question '{qid}'", 200)
            try:
                normalized[qid] = self._parse_answer(answers[qid], question)
            except (KeyError, TypeError, ValueError) as e:
                raise SystemOneResponseError(
                    f"{self._where()} returned an unreadable answer for question '{qid}': {answers[qid]} ({e})", 200
                )

        usage = data.get("usage") or {}
        input_tokens = int(usage.get("input_tokens") or 0)
        reported = usage.get("cost")
        if isinstance(reported, dict):  # a breakdown, as Perplexity's chat API reports it
            reported = reported.get("total_cost")
        if isinstance(reported, (int, float)) and not isinstance(reported, bool):
            cost = float(reported)
        else:
            cost = input_tokens * (self.input_price_per_million or 0.0) / 1_000_000
        return DecisionResult(
            answers=normalized, model=data.get("model"), input_tokens=input_tokens, cost=cost, raw=data
        )

    @staticmethod
    def _parse_answer(answer: Dict[str, Any], question: Question) -> Answer:
        """Normalize one answer, using the question's own options/levels as the source of truth for labels."""
        if answer.get("type") == "refusal":
            return Answer(type=question.type, value=None, refused=True)
        confidence = answer.get("confidence")
        confidence = float(confidence) if confidence is not None else None

        if isinstance(question, Noul):
            return Answer(type="noul", value=float(answer["noul"]))

        if isinstance(question, Choice):
            raw_probabilities = answer.get("probabilities")
            probabilities = None
            if isinstance(raw_probabilities, dict):
                probabilities = {option: float(raw_probabilities.get(option, 0.0)) for option in question.options}
            choice = answer.get("choice")
            if choice is None and probabilities:
                choice = max(probabilities, key=probabilities.get)
            if choice is None:
                raise ValueError("the answer has neither a choice nor probabilities")
            return Answer(type="choice", value=choice, label=choice, probabilities=probabilities, confidence=confidence)

        # Score: probabilities are keyed by level index ("0", "1", ...); re-key them by the question's level labels.
        labels = [_text(level) for level in question.levels]
        raw_probabilities = answer.get("probabilities")
        probabilities = None
        if isinstance(raw_probabilities, dict):
            probabilities = {label: float(raw_probabilities.get(str(i), 0.0)) for i, label in enumerate(labels)}
        expected = answer.get("score")
        if expected is not None:
            expected = float(expected)
        elif probabilities:
            expected = sum(i * p for i, p in enumerate(probabilities.values()))
        if probabilities:
            level = max(range(len(labels)), key=lambda i: probabilities[labels[i]])
        elif expected is not None:
            level = min(max(round(expected), 0), len(labels) - 1)
        else:
            raise ValueError("the answer has neither a score nor probabilities")
        return Answer(
            type="score",
            value=expected,
            label=labels[level],
            level=level,
            probabilities=probabilities,
            confidence=confidence,
        )
