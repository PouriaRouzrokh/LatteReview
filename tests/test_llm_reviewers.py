"""Unit tests for LLM reviewers, their retries, the workflow's task-ID line and the Ollama provider (no network)."""

import json
from types import SimpleNamespace
from typing import Any, List

import pandas as pd
import pydantic
import pytest

from lattereview.agents import TitleAbstractReviewer
from lattereview.agents.basic_reviewer import AgentError
from lattereview.providers import OllamaProvider
from lattereview.providers.base_provider import BaseProvider
from lattereview.workflows import ReviewWorkflow

FREE = {"input_cost": 0.0, "output_cost": 0.0, "total_cost": 0.0}


class FakeLLMProvider(BaseProvider):
    """Records every prompt and answers with a valid review, or raises on every call when `fail` is set."""

    fail: bool = False
    calls: int = 0
    prompts: List[str] = pydantic.Field(default_factory=list)

    def set_response_format(self, response_format: Any) -> None:
        self.response_format = response_format

    async def get_json_response(self, input_prompt, image_path_list=[], message_list=None, **kwargs):
        self.calls += 1
        if self.fail:
            raise ValueError("Could not parse the model's JSON")
        self.prompts.append(input_prompt)
        return json.dumps({"reasoning": "Fits the criteria.", "evaluation": 1}), FREE


def make_reviewer(provider, **kwargs):
    return TitleAbstractReviewer(
        provider=provider,
        name="Alice",
        inclusion_criteria="The study must use deep learning.",
        exclusion_criteria="The study must not be a review.",
        verbose=False,
        **kwargs,
    )


# --- Retries (issue #9) --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("max_retries", [1, 3])
async def test_failing_item_stops_after_max_retries(max_retries):
    provider = FakeLLMProvider(fail=True)
    reviewer = make_reviewer(provider, max_retries=max_retries)
    reviewer.setup()
    with pytest.raises(AgentError, match=f"after {max_retries} attempt"):
        await reviewer.review_item("Title: a study")
    assert provider.calls == max_retries


# --- The workflow's task-ID line -----------------------------------------------------------------------------------


def screening_workflow(reviewer, **kwargs):
    return ReviewWorkflow(
        workflow_schema=[{"round": "A", "reviewers": [reviewer], "text_inputs": ["title", "abstract"]}],
        verbose=False,
        **kwargs,
    )


async def test_task_id_line_is_added_by_default():
    provider = FakeLLMProvider()
    df = pd.DataFrame({"title": ["first", "second"], "abstract": ["..."] * 2})
    await screening_workflow(make_reviewer(provider))(df)
    assert sorted(p for p in provider.prompts if "Review Task ID: A-" in p) == sorted(provider.prompts)
    assert len(provider.prompts) == 2


async def test_without_task_id_a_record_gets_the_same_prompt_whatever_its_row():
    """With include_task_id=False, an item's prompt does not depend on its row or on the other rows in the round."""
    first, second = FakeLLMProvider(), FakeLLMProvider()
    alone = pd.DataFrame({"title": ["target"], "abstract": ["..."]})
    shuffled = pd.DataFrame({"title": ["other", "another", "target"], "abstract": ["..."] * 3})

    result = await screening_workflow(make_reviewer(first), include_task_id=False)(alone)
    await screening_workflow(make_reviewer(second), include_task_id=False)(shuffled)

    assert not any("Review Task ID" in p for p in first.prompts + second.prompts)
    target_prompts = [p for p in second.prompts if "target" in p]
    assert target_prompts == first.prompts
    assert result.loc[0, "round-A_Alice_evaluation"] == 1


# --- Ollama structured outputs (issue #5) --------------------------------------------------------------------------


async def test_ollama_constrains_output_with_the_schema_and_keeps_the_system_prompt():
    """The schema goes in Ollama's `format` argument only; the system prompt is not rewritten to repeat it."""
    calls = []

    class FakeClient:
        async def chat(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(message=SimpleNamespace(content='{"reasoning": "ok", "evaluation": 2}'))

    provider = OllamaProvider(model="llama3.1", system_prompt="You are a reviewer.")
    provider.client = FakeClient()
    provider.set_response_format({"reasoning": str, "evaluation": int})

    content, cost = await provider.get_json_response("Review this.", temperature=0.2, top_p=0.9)

    assert json.loads(content) == {"reasoning": "ok", "evaluation": 2}
    assert cost["total_cost"] == 0
    (call,) = calls
    assert call["format"]["properties"].keys() == {"reasoning", "evaluation"}
    assert call["messages"][0] == {"role": "system", "content": "You are a reviewer."}
    assert call["options"] == {"top_p": 0.9} and "temperature" not in call
