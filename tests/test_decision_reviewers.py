"""Unit tests for the decision reviewers and their use in ReviewWorkflow (no network)."""

import json

import httpx
import pandas as pd
import pytest

from lattereview.agents import DecisionReviewer, DecisionTitleAbstractReviewer, DecisionScoringReviewer
from lattereview.agents.basic_reviewer import AgentError
from lattereview.providers import OpenAIProvider, Noul, Choice, Score
from lattereview.workflows import ReviewWorkflow

INCLUSION = {1: "The study must involve CT scans.", 2: "The study must use deep learning."}
EXCLUSION = {1: "The study must not include PET scans."}


class FakeBackend:
    """Answers every question in a request. `noul` maps question IDs to probabilities (default 0.9); `noul_for`
    can override per state text. Score answers put `score_probabilities` (or 0.6 on the top level) on the levels."""

    def __init__(self, noul=None, noul_for=None, score_probabilities=None, confidence=0.8, status=200):
        self.noul, self.noul_for = noul or {}, noul_for or (lambda state, qid: None)
        self.score_probabilities, self.confidence, self.status = score_probabilities, confidence, status
        self.bodies = []

    def __call__(self, request):
        body = json.loads(request.content)
        self.bodies.append(body)
        if self.status != 200:
            return httpx.Response(self.status, json={"detail": {"message": "rejected"}})
        state = body["state"] if isinstance(body["state"], str) else body["state"]["item"]
        answers = {}
        for qid, question in body["questions"].items():
            if question["type"] == "noul":
                p = self.noul_for(state, qid)
                answers[qid] = {"type": "noul", "noul": p if p is not None else self.noul.get(qid, 0.9)}
            elif question["type"] == "choice":
                options = list(question["criteria"])
                probabilities = {o: (0.7 if i == 0 else 0.3 / (len(options) - 1)) for i, o in enumerate(options)}
                answers[qid] = {"type": "choice", "choice": options[0], "probabilities": probabilities}
                answers[qid]["confidence"] = self.confidence
            else:
                n = len(question["criteria"])
                probabilities = self.score_probabilities or [0.4 / (n - 1)] * (n - 1) + [0.6]
                answers[qid] = {
                    "type": "score",
                    "score": sum(i * p for i, p in enumerate(probabilities)),
                    "probabilities": {str(i): p for i, p in enumerate(probabilities)},
                    "confidence": self.confidence,
                }
        return httpx.Response(200, json={"model": "fake", "answers": answers, "usage": {"input_tokens": 1000}})


@pytest.fixture
def fake(make_provider):
    """Return (provider, backend) for a FakeBackend configured with the given options."""

    def factory(**kwargs):
        backend = FakeBackend(**kwargs)
        provider, _ = make_provider(backend)
        return provider, backend

    return factory


# --- DecisionTitleAbstractReviewer -------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "inclusion, expected_keys",
    [
        ("Uses CT.", ["inc_1"]),
        (["Uses CT.", "Uses deep learning."], ["inc_1", "inc_2"]),
        ({1: "Uses CT.", 3: "Uses deep learning."}, ["inc_1", "inc_3"]),
        ({"a": "Uses CT."}, ["inc_a"]),
    ],
)
def test_criteria_forms(fake, inclusion, expected_keys):
    provider, _ = fake()
    reviewer = DecisionTitleAbstractReviewer(provider=provider, inclusion_criteria=inclusion, exclusion_criteria="")
    assert list(reviewer.questions) == ["evaluation", "include"] + expected_keys
    assert isinstance(reviewer.questions["evaluation"], Score) and len(reviewer.questions["evaluation"].levels) == 5
    assert "Exclusion criteria:\nNone." in reviewer.questions["include"].instructions


def test_title_abstract_questions_include_all_criteria(fake):
    provider, _ = fake()
    reviewer = DecisionTitleAbstractReviewer(
        provider=provider, inclusion_criteria=INCLUSION, exclusion_criteria=EXCLUSION
    )
    assert list(reviewer.questions) == ["evaluation", "include", "inc_1", "inc_2", "exc_1"]
    for qid in ("evaluation", "include"):
        assert "1. The study must involve CT scans." in reviewer.questions[qid].instructions
        assert "1. The study must not include PET scans." in reviewer.questions[qid].instructions
    assert INCLUSION[2] in reviewer.questions["inc_2"].instructions
    assert reviewer.response_format == {
        "evaluation": int,
        "include_probability": float,
        "confidence": float,
        "criteria": dict,
        "reasoning": str,
    }


async def test_title_abstract_output(fake):
    # Level probabilities peak at "better to exclude" (index 1 -> evaluation 2).
    provider, backend = fake(
        noul={"include": 0.12, "inc_2": 0.07, "exc_1": 0.03}, score_probabilities=[0.2, 0.5, 0.2, 0.05, 0.05]
    )
    reviewer = DecisionTitleAbstractReviewer(
        provider=provider, inclusion_criteria=INCLUSION, exclusion_criteria=EXCLUSION
    )
    response, input_prompt, cost = await reviewer.review_item("Review Task ID: A-0\n=== title ===\nA CT study")

    assert len(backend.bodies) == 1  # all questions in a single request
    assert backend.bodies[0]["state"] == "Review Task ID: A-0\n=== title ===\nA CT study"
    assert response["evaluation"] == 2
    assert response["include_probability"] == 0.12
    assert response["confidence"] == 0.8
    assert response["criteria"] == {
        "inclusion": {INCLUSION[1]: 0.9, INCLUSION[2]: 0.07},
        "exclusion": {EXCLUSION[1]: 0.03},
    }
    assert response["reasoning"] == (
        "P(include)=0.12; rated 'better to exclude'. "
        "Likely fails inclusion 2 (p=0.07: 'The study must use deep learning.'). "
        "No exclusion criterion likely applies (highest p=0.03)."
    )
    assert set(response["_answers"]) == {"evaluation", "include", "inc_1", "inc_2", "exc_1"}
    assert response["_answers"]["evaluation"]["label"] == "better to exclude"
    assert input_prompt["state"] == backend.bodies[0]["state"] and set(input_prompt["questions"]) == set(
        reviewer.questions
    )
    assert cost == pytest.approx(1000 * 0.042 / 1e6)


async def test_reasoning_all_met_and_long_criteria_truncated(fake):
    long_exclusion = "The study must not include PET scans or any other nuclear medicine imaging such as SPECT."
    provider, _ = fake(noul={"exc_1": 0.95})
    reviewer = DecisionTitleAbstractReviewer(
        provider=provider, inclusion_criteria=INCLUSION, exclusion_criteria=long_exclusion
    )
    response, _, _ = await reviewer.review_item("text")
    assert "Likely meets all inclusion criteria (lowest p=0.90)." in response["reasoning"]
    assert "Likely meets exclusion 1 (p=0.95: 'The study must not include PET scans or any other nuclear m…')." in (
        response["reasoning"]
    )


def test_title_abstract_requires_criteria(fake):
    provider, _ = fake()
    with pytest.raises(AgentError, match="criterion is required"):
        DecisionTitleAbstractReviewer(provider=provider)


# --- DecisionScoringReviewer -------------------------------------------------------------------------------------


async def test_scoring_output(fake):
    provider, _ = fake(score_probabilities=[0.1, 0.2, 0.7], confidence=0.734)
    reviewer = DecisionScoringReviewer(
        provider=provider,
        scoring_task="Rate the external validation.",
        scoring_set=[0, 5, 10],
        score_descriptions={0: "no validation", 10: "external validation"},
        scoring_rules="Internal splits do not count as external.",
    )
    question = reviewer.questions["score"]
    assert question.levels == ["no validation", "5", "external validation"]
    assert question.instructions.endswith("Rules: Internal splits do not count as external.")
    assert reviewer.response_format == {"score": int, "certainty": int, "probabilities": dict}
    response, _, _ = await reviewer.review_item("text")
    assert response["score"] == 10 and response["certainty"] == 73
    assert response["probabilities"] == {0: 0.1, 5: 0.2, 10: 0.7}


async def test_scoring_without_confidence(fake):
    provider, _ = fake(score_probabilities=[0.6, 0.4], confidence=None)
    reviewer = DecisionScoringReviewer(provider=provider, scoring_task="Is it good?", scoring_set=[1, 2])
    response, _, _ = await reviewer.review_item("text")
    assert response["score"] == 1 and response["certainty"] is None


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({}, "scoring_task is required"),
        ({"scoring_task": "x", "scoring_set": [1, 1]}, "must be unique"),
        ({"scoring_task": "x", "scoring_set": [1]}, "2-10 levels"),
        ({"scoring_task": "x", "scoring_set": list(range(11))}, "2-10 levels"),
        ({"scoring_task": "x", "score_descriptions": {3: "three"}}, "not in scoring_set"),
        ({"scoring_task": "x", "questions": {"q": Noul("x")}}, "builds its questions"),
    ],
)
def test_scoring_validation(fake, kwargs, message):
    provider, _ = fake()
    with pytest.raises(AgentError, match=message):
        DecisionScoringReviewer(provider=provider, **kwargs)


# --- Generic DecisionReviewer ------------------------------------------------------------------------------------


async def test_generic_reviewer(fake):
    provider, backend = fake(noul={"rct": 0.2})
    reviewer = DecisionReviewer(
        provider=provider,
        questions={
            "modality": Choice("Which modality?", ["CT", "MRI"]),
            "rct": Noul("Is it an RCT?"),
            "quality": Score("Quality?", ["low", "high"]),
            "raw": {"type": "noul", "instructions": "Given as a dict"},
        },
        additional_context="Focus on the methods.",
    )
    assert reviewer.response_format == {"modality": str, "rct": float, "quality": float, "raw": float}
    response, input_prompt, _ = await reviewer.review_item("text")
    assert backend.bodies[0]["state"] == {"item": "text", "additional_context": "Focus on the methods."}
    assert response["modality"] == "CT" and response["rct"] == 0.2 and response["quality"] == pytest.approx(0.6)
    assert response["_answers"]["modality"]["probabilities"] == {"CT": 0.7, "MRI": 0.3}


async def test_async_additional_context(fake):
    provider, backend = fake()

    async def context(item):
        return f"context for {item}"

    reviewer = DecisionReviewer(provider=provider, questions={"q": Noul("x")}, additional_context=context)
    await reviewer.review_item("item-1")
    assert backend.bodies[0]["state"] == {"item": "item-1", "additional_context": "context for item-1"}


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"reasoning": "brief"}, "`reasoning` is not supported"),
        ({"examples": ["an example"]}, "`examples` is not supported"),
        ({"model_args": {"temperature": 0.1}}, "`model_args` is not supported"),
        ({"generic_prompt": "Review ${item}$"}, "`generic_prompt` is not supported"),
        ({"questions": {}}, "At least one question"),
    ],
)
def test_unsupported_options_fail_loudly(fake, kwargs, message):
    provider, _ = fake()
    kwargs = {"questions": {"q": Noul("x")}, **kwargs}
    with pytest.raises(AgentError, match=message):
        DecisionReviewer(provider=provider, **kwargs)


def test_llm_provider_is_rejected(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    with pytest.raises(AgentError, match="needs a SystemOneProvider"):
        DecisionReviewer(provider=OpenAIProvider(), questions={"q": Noul("x")})


async def test_images_are_rejected(fake):
    provider, backend = fake()
    reviewer = DecisionReviewer(provider=provider, questions={"q": Noul("x")})
    with pytest.raises(AgentError, match="accepts text only"):
        await reviewer.review_item("text", ["image.png"])
    assert backend.bodies == []


async def test_exhausted_transient_errors_are_not_retried_again(make_provider):
    """The provider retries 529s itself; the reviewer must not multiply those retries."""
    provider, recorder = make_provider(httpx.Response(529, json={"detail": "overloaded"}), max_retries=1)
    reviewer = DecisionReviewer(provider=provider, questions={"q": Noul("x")}, verbose=False)
    with pytest.raises(AgentError, match="after 1 attempt"):
        await reviewer.review_item("text")
    assert len(recorder.requests) == 2


async def test_malformed_responses_are_retried_by_the_reviewer(make_provider):
    good = httpx.Response(200, json={"answers": {"q": {"type": "noul", "noul": 0.4}}})
    provider, recorder = make_provider(httpx.Response(200, json={"answers": {}}), good)
    reviewer = DecisionReviewer(provider=provider, questions={"q": Noul("x")}, verbose=False)
    response, _, _ = await reviewer.review_item("text")
    assert response["q"] == 0.4 and len(recorder.requests) == 2


async def test_client_errors_are_not_retried_by_the_reviewer(fake):
    provider, backend = fake(status=401)
    reviewer = DecisionReviewer(provider=provider, questions={"q": Noul("x")}, verbose=False)
    with pytest.raises(AgentError, match="after 1 attempt"):
        await reviewer.review_item("text")
    assert len(backend.bodies) == 1


# --- ReviewWorkflow integration ----------------------------------------------------------------------------------


async def test_workflow_routes_uncertain_items_to_round_b(fake):
    """Round A: Jev screens everything. Round B: only items with an uncertain include probability."""
    include_p = {"clear include": 0.97, "clear exclude": 0.02, "unsure": 0.5}
    provider, backend = fake(noul_for=lambda state, qid: next(p for k, p in include_p.items() if k in state))
    jev = DecisionTitleAbstractReviewer(
        provider=provider, name="Jev", inclusion_criteria=INCLUSION, exclusion_criteria=EXCLUSION, verbose=False
    )
    scorer = DecisionScoringReviewer(provider=provider, name="Scorer", scoring_task="Rate it.", verbose=False)
    df = pd.DataFrame({"title": list(include_p), "abstract": ["..."] * 3})

    workflow = ReviewWorkflow(
        workflow_schema=[
            {"round": "A", "reviewers": [jev], "text_inputs": ["title", "abstract"]},
            {
                "round": "B",
                "reviewers": [scorer],
                "text_inputs": ["title", "abstract"],
                "filter": lambda row: 0.1 <= row["round-A_Jev_include_probability"] <= 0.9,
            },
        ],
        verbose=False,
    )
    result = await workflow(df)

    for key in ("evaluation", "include_probability", "confidence", "criteria", "reasoning", "output"):
        assert f"round-A_Jev_{key}" in result.columns
    assert list(result["round-A_Jev_include_probability"]) == [0.97, 0.02, 0.5]
    assert "_answers" in result.loc[0, "round-A_Jev_output"]
    assert result["round-B_Scorer_score"].notna().tolist() == [False, False, True]
    assert len(backend.bodies) == 4
    states = sorted(body["state"] for body in backend.bodies)
    assert states[0] == "Review Task ID: A-0\n=== title ===\nclear include\n\n=== abstract ===\n..."
    assert workflow.get_total_cost() == pytest.approx(4 * 1000 * 0.042 / 1e6)
    assert len(jev.memory) == 3 and jev.memory[0]["cost"] > 0


async def test_workflow_rejects_image_inputs(fake, tmp_path):
    image = tmp_path / "scan.png"
    image.write_bytes(b"fake")
    provider, _ = fake()
    reviewer = DecisionReviewer(provider=provider, questions={"q": Noul("x")}, verbose=False)
    workflow = ReviewWorkflow(
        workflow_schema=[{"round": "A", "reviewers": [reviewer], "text_inputs": ["title"], "image_inputs": ["image"]}],
        verbose=False,
    )
    with pytest.raises(Exception, match="accepts text only"):
        await workflow(pd.DataFrame({"title": ["t"], "image": [str(image)]}))


# --- OpenAI's Decisions API and refusals ---------------------------------------------------------------------------


class FakeOpenAIBackend:
    """Answers in OpenAI's Decisions format: predicates get `probability` (default 0.9), and any question whose name is
    in `refuse` (or whose input contains `refuse_text`) is refused."""

    def __init__(self, probability=None, refuse=(), refuse_text=None):
        self.probability, self.refuse, self.refuse_text = probability or {}, set(refuse), refuse_text
        self.bodies = []

    def __call__(self, request):
        body = json.loads(request.content)
        self.bodies.append(body)
        answers = []
        for question in body["questions"]:
            name = question["name"]
            if name in self.refuse or (self.refuse_text and self.refuse_text in body["input"]):
                answers.append({"type": "refusal", "name": name})
            elif question["type"] == "predicate":
                answers.append({"type": "predicate", "name": name, "probability": self.probability.get(name, 0.9)})
            elif question["type"] == "choice":
                values = [choice["value"] for choice in question["choices"]]
                probabilities = [{"value": v, "probability": 1.0 if i == 0 else 0.0} for i, v in enumerate(values)]
                answers.append({"type": "choice", "name": name, "choice": values[0], "probabilities": probabilities})
                answers[-1]["confidence"] = 1.0
            else:
                labels = [level["label"] for level in question["levels"]]
                probabilities = [
                    {"value": i, "label": label, "probability": 1.0 if i == len(labels) - 1 else 0.0}
                    for i, label in enumerate(labels)
                ]
                answers.append(
                    {"type": "score", "name": name, "score": len(labels) - 1.0, "probabilities": probabilities}
                )
                answers[-1]["confidence"] = 0.9
        return httpx.Response(200, json={"model": "gpt-6-luna", "answers": answers, "usage": {"input_tokens": 500}})


@pytest.fixture
def fake_openai(make_provider):
    def factory(**kwargs):
        backend = FakeOpenAIBackend(**kwargs)
        provider, _ = make_provider(backend, backend="openai")
        return provider, backend

    return factory


async def test_title_abstract_reviewer_on_openai(fake_openai):
    provider, backend = fake_openai(probability={"include": 0.2, "inc_2": 0.1})
    reviewer = DecisionTitleAbstractReviewer(
        provider=provider, inclusion_criteria=INCLUSION, exclusion_criteria=EXCLUSION
    )
    response, input_prompt, cost = await reviewer.review_item("A CT study")
    names = [question["name"] for question in backend.bodies[0]["questions"]]
    assert names == ["evaluation", "include", "inc_1", "inc_2", "exc_1"]
    assert (
        "Answer true if: include: meets every inclusion criterion" in backend.bodies[0]["questions"][1]["instructions"]
    )
    assert (response["evaluation"], response["include_probability"], response["confidence"]) == (5, 0.2, 0.9)
    assert response["criteria"]["inclusion"][INCLUSION[2]] == 0.1
    assert "Likely fails inclusion 2" in response["reasoning"] and "declined" not in response["reasoning"]
    assert cost == pytest.approx(500 * 0.10 / 1e6)


async def test_title_abstract_refusals_become_none(fake_openai, capsys):
    provider, _ = fake_openai(refuse={"include", "evaluation", "inc_2"}, probability={"inc_1": 0.95, "exc_1": 0.02})
    reviewer = DecisionTitleAbstractReviewer(
        provider=provider, inclusion_criteria=INCLUSION, exclusion_criteria=EXCLUSION
    )
    response, _, _ = await reviewer.review_item("A CT study")
    assert (response["evaluation"], response["include_probability"], response["confidence"]) == (None, None, None)
    assert response["criteria"]["inclusion"] == {INCLUSION[1]: 0.95, INCLUSION[2]: None}
    assert response["_answers"]["include"]["refused"] is True
    assert response["reasoning"] == (
        "P(include) unknown. Likely meets the answered inclusion criteria (lowest p=0.95). No exclusion criterion "
        "likely applies (highest p=0.02). The model declined to answer: the evaluation score, the overall include "
        "question, inclusion 2."
    )
    await reviewer.review_item("Another study")
    assert capsys.readouterr().out.count("declined to answer") == 1  # warned once per reviewer


async def test_scoring_and_generic_refusals(fake_openai):
    provider, _ = fake_openai(refuse={"score", "kind"})
    scorer = DecisionScoringReviewer(provider=provider, scoring_task="Rate it.", scoring_set=[1, 2, 3])
    response, _, _ = await scorer.review_item("text")
    assert (response["score"], response["certainty"], response["probabilities"]) == (None, None, None)
    generic = DecisionReviewer(provider=provider, questions={"kind": Choice("Kind?", ["a", "b"]), "ok": Noul("OK?")})
    response, _, _ = await generic.review_item("text")
    assert (response["kind"], response["ok"]) == (None, 0.9)


async def test_workflow_keeps_refused_items(fake_openai):
    """A refusal must not stop the run; refused items can be routed on to another reviewer."""
    provider, backend = fake_openai(refuse_text="sensitive", probability={"include": 0.97})
    reviewer = DecisionTitleAbstractReviewer(
        provider=provider, name="Luna", inclusion_criteria=INCLUSION, exclusion_criteria=EXCLUSION, verbose=False
    )
    scorer = DecisionScoringReviewer(provider=provider, name="Scorer", scoring_task="Rate it.", verbose=False)
    workflow = ReviewWorkflow(
        workflow_schema=[
            {"round": "A", "reviewers": [reviewer], "text_inputs": ["title"]},
            {
                "round": "B",
                "reviewers": [scorer],
                "text_inputs": ["title"],
                "filter": lambda row: pd.isna(row["round-A_Luna_include_probability"]),
            },
        ],
        verbose=False,
    )
    result = await workflow(pd.DataFrame({"title": ["a CT study", "a sensitive study"]}))
    assert result["round-A_Luna_include_probability"].isna().tolist() == [False, True]
    assert result["round-B_Scorer_score"].notna().tolist() == [False, False]  # refused in round B too
    assert len(backend.bodies) == 3
