"""A 3-item ReviewWorkflow per decision-reviewer preset against every available backend."""

import pandas as pd
import pytest

from lattereview.agents import DecisionReviewer, DecisionTitleAbstractReviewer, DecisionScoringReviewer
from lattereview.providers import Choice, Noul
from lattereview.workflows import ReviewWorkflow

pytestmark = pytest.mark.live

ITEMS = pd.DataFrame(
    {
        "title": [
            "Deep learning detection of pulmonary nodules on low-dose chest CT",
            "Knee MRI cartilage segmentation with a U-Net",
            "PET/CT radiomics for lymphoma staging",
        ],
        "abstract": [
            "We trained a 3D convolutional network on 2,000 low-dose CT scans and tested it on an external cohort.",
            "A U-Net segmented knee cartilage on 500 MRI scans with internal cross-validation.",
            "Radiomic features from FDG PET/CT predicted lymphoma stage in 120 patients using logistic regression.",
        ],
    }
)


async def run(reviewer):
    workflow = ReviewWorkflow(
        workflow_schema=[{"round": "A", "reviewers": [reviewer], "text_inputs": ["title", "abstract"]}], verbose=False
    )
    return await workflow(ITEMS), workflow.get_total_cost()


async def test_title_abstract_workflow(live_provider):
    reviewer = DecisionTitleAbstractReviewer(
        provider=live_provider,
        name="Jev",
        inclusion_criteria={1: "The study must involve CT scans.", 2: "The study must use deep learning."},
        exclusion_criteria={1: "The study must not include PET scans."},
        verbose=False,
    )
    df, cost = await run(reviewer)
    probabilities = df["round-A_Jev_include_probability"].tolist()
    assert probabilities[0] > 0.5 and probabilities[1] < 0.5 and probabilities[2] < 0.5
    assert df["round-A_Jev_evaluation"].tolist()[0] >= 4
    assert all(1 <= e <= 5 for e in df["round-A_Jev_evaluation"])
    assert df["round-A_Jev_reasoning"].str.startswith("P(include)=").all()
    assert cost >= 0


async def test_scoring_workflow(live_provider):
    reviewer = DecisionScoringReviewer(
        provider=live_provider,
        name="Validation",
        scoring_task="How strong is the validation reported in this study?",
        scoring_set=[1, 2, 3],
        score_descriptions={1: "no validation reported", 2: "internal validation only", 3: "external validation"},
        verbose=False,
    )
    df, _ = await run(reviewer)
    assert df["round-A_Validation_score"].tolist()[:2] == [3, 2]
    for probabilities in df["round-A_Validation_probabilities"]:
        assert list(probabilities) == [1, 2, 3] and sum(probabilities.values()) == pytest.approx(1.0, abs=0.02)


async def test_generic_workflow(live_provider):
    reviewer = DecisionReviewer(
        provider=live_provider,
        name="Extract",
        questions={
            "modality": Choice("What is the main imaging modality?", ["CT", "MRI", "PET/CT", "X-ray"]),
            "deep_learning": Noul("Does the study use deep learning?"),
        },
        verbose=False,
    )
    df, _ = await run(reviewer)
    assert df["round-A_Extract_modality"].tolist() == ["CT", "MRI", "PET/CT"]
    assert df["round-A_Extract_deep_learning"].tolist()[0] > 0.5 > df["round-A_Extract_deep_learning"].tolist()[2]
