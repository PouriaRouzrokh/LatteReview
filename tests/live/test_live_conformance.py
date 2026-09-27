"""Conformance suite: the same noul/choice/score requests against every available /v1/systemone backend."""

import pytest

from lattereview.providers import Noul, Choice, Score

pytestmark = pytest.mark.live

STATE = (
    "Title: Deep learning for pneumonia detection on chest radiographs.\n"
    "Abstract: We trained a convolutional neural network on 100,000 chest X-rays from one hospital and validated it "
    "on an external cohort of 5,000 patients from another hospital. The model reached an AUC of 0.93."
)
QUESTIONS = {
    "uses_dl": Noul("Does the study use deep learning?", true="uses deep learning", false="does not use deep learning"),
    "is_rct": Noul("Is this a randomized controlled trial?"),
    "modality": Choice(
        "Which imaging modality is studied?",
        {"CT": "computed tomography", "MRI": "magnetic resonance imaging", "XR": "radiography", "US": "ultrasound"},
    ),
    "organ": Choice("Which organ is studied?", ["brain", "lung", "liver", "heart"]),
    "validation": Score("How strong is the validation?", ["none", "internal only", "external"]),
}


async def test_answers_are_well_formed(live_provider):
    result = await live_provider.decide(STATE, QUESTIONS)
    answers = result.answers
    assert set(answers) == set(QUESTIONS)

    assert answers["uses_dl"].value > 0.5
    assert answers["is_rct"].value < 0.5
    for qid in ("uses_dl", "is_rct"):
        assert 0.0 <= answers[qid].value <= 1.0 and answers[qid].confidence is None

    for qid, expected in (("modality", "XR"), ("organ", "lung")):
        answer = answers[qid]
        assert answer.value == expected and answer.label in QUESTIONS[qid].options
        assert sum(answer.probabilities.values()) == pytest.approx(1.0, abs=0.02)

    validation = answers["validation"]
    assert validation.label == "external" and validation.level == 2
    assert 0.0 <= validation.value <= 2.0
    assert list(validation.probabilities) == ["none", "internal only", "external"]
    assert sum(validation.probabilities.values()) == pytest.approx(1.0, abs=0.02)

    assert result.input_tokens > 0
    assert result.cost >= 0.0


async def test_same_request_gives_same_answers(live_provider):
    first = await live_provider.decide(STATE, QUESTIONS)
    second = await live_provider.decide(STATE, QUESTIONS)
    for qid in QUESTIONS:
        assert first.answers[qid].label == second.answers[qid].label
        if first.answers[qid].type == "noul":
            assert first.answers[qid].value == pytest.approx(second.answers[qid].value, abs=0.02)
