"""Perplexity's Sonar LLMs as LLM reviewers through LiteLLMProvider: directly (PERPLEXITY_API_KEY) and via OpenRouter."""

import os

import pandas as pd
import pytest

from lattereview.agents import AbstractionReviewer, ScoringReviewer, TitleAbstractReviewer
from lattereview.providers import LiteLLMProvider
from lattereview.workflows import ReviewWorkflow

from test_live_reviewers import ITEMS

pytestmark = pytest.mark.live

MODELS = {
    "perplexity/sonar": "PERPLEXITY_API_KEY",
    "openrouter/perplexity/sonar": "OPENROUTER_API_KEY",
}


@pytest.fixture(params=list(MODELS))
def sonar_model(request):
    env_var = MODELS[request.param]
    if not os.getenv(env_var):
        pytest.skip(f"{env_var} is not set")
    return request.param


async def test_sonar_reviewers_in_a_workflow(sonar_model):
    """TitleAbstractReviewer, ScoringReviewer and AbstractionReviewer return valid structured output from Sonar."""
    provider = LiteLLMProvider(model=sonar_model)
    reviewers = [
        TitleAbstractReviewer(
            provider=provider,
            name="Screen",
            inclusion_criteria="The study must involve CT scans and use deep learning.",
            exclusion_criteria="The study must not include PET scans.",
            verbose=False,
        ),
        ScoringReviewer(
            provider=provider,
            name="Validation",
            scoring_task="How strong is the validation reported in this study?",
            scoring_set=[1, 2, 3],
            scoring_rules="1 = no validation, 2 = internal validation only, 3 = external validation.",
            verbose=False,
        ),
        AbstractionReviewer(
            provider=provider,
            name="Extract",
            abstraction_keys={"modality": str, "sample_size": int},
            key_descriptions={
                "modality": "The main imaging modality",
                "sample_size": "The number of scans or patients",
            },
            verbose=False,
        ),
    ]
    workflow = ReviewWorkflow(
        workflow_schema=[{"round": "A", "reviewers": reviewers, "text_inputs": ["title", "abstract"]}], verbose=False
    )
    df = await workflow(ITEMS)

    evaluations = df["round-A_Screen_evaluation"].tolist()
    assert all(1 <= e <= 5 for e in evaluations) and evaluations[0] >= 4 and evaluations[2] <= 2
    assert df["round-A_Screen_reasoning"].str.len().gt(0).all()
    assert set(df["round-A_Validation_score"]) <= {1, 2, 3}
    assert df["round-A_Extract_sample_size"].tolist() == [2000, 500, 120]
    assert workflow.get_total_cost() > 0
