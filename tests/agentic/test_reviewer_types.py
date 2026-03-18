"""Tests for reviewer_types.py — Preset reviewer types."""

import os

import pytest
import pandas as pd

from pydantic import BaseModel

from lattereview.agentic.reviewer_types import (
    ScoringReviewer,
    TitleAbstractReviewer,
    AbstractionReviewer,
)
from lattereview.agentic.output_models import ScoringOutput, EvaluationOutput
from lattereview.agentic.reviewer import AgenticReviewer

# ---------------------------------------------------------------------------
# ScoringReviewer
# ---------------------------------------------------------------------------


class TestScoringReviewerInit:
    def test_default_values(self):
        r = ScoringReviewer()
        assert r.output_type is ScoringOutput
        assert r.scoring_set == [1, 2]
        assert "scoring task" in r.task_prompt.lower() or "Score" in r.task_prompt
        assert r.system_prompt != ""

    def test_is_subclass(self):
        r = ScoringReviewer()
        assert isinstance(r, AgenticReviewer)

    def test_custom_scoring_params(self):
        r = ScoringReviewer(
            scoring_task="Rate relevance to oncology.",
            scoring_set=[1, 2, 3, 4, 5],
            scoring_rules="1=irrelevant, 5=highly relevant",
        )
        assert "oncology" in r.task_prompt
        assert "[1, 2, 3, 4, 5]" in r.task_prompt
        assert "1=irrelevant" in r.task_prompt

    def test_output_type_enforced(self):
        """Output type is always ScoringOutput regardless of what user passes."""
        r = ScoringReviewer(output_type=EvaluationOutput)
        assert r.output_type is ScoringOutput

    def test_custom_system_prompt_preserved(self):
        r = ScoringReviewer(system_prompt="Custom instructions here.")
        assert r.system_prompt == "Custom instructions here."

    def test_custom_task_prompt_preserved(self):
        custom = "My custom task: ${item}$"
        r = ScoringReviewer(task_prompt=custom)
        assert r.task_prompt == custom

    def test_override_agentic_params(self):
        r = ScoringReviewer(
            name="MyScorer",
            model="anthropic:claude-haiku-4-5-20251001",
            max_iterations=1,
            backstory="An expert scorer.",
        )
        assert r.name == "MyScorer"
        assert r.model == "anthropic:claude-haiku-4-5-20251001"
        assert r.max_iterations == 1
        assert r.is_agentic is False

    def test_non_agentic_mode(self):
        r = ScoringReviewer(max_iterations=1)
        assert r.is_agentic is False

    @pytest.mark.asyncio
    async def test_review_item_with_test_model(self):
        from pydantic_ai.models.test import TestModel

        r = ScoringReviewer(
            model=TestModel(),
            max_iterations=1,
            scoring_task="Rate quality 1-10.",
            scoring_set=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        )
        response, cost = await r.review_item(
            item_text="A study on deep learning for radiology.",
            item_id="test_001",
        )
        assert "reasoning" in response
        assert "score" in response
        assert "certainty" in response

    @pytest.mark.asyncio
    async def test_review_items_with_test_model(self):
        from pydantic_ai.models.test import TestModel

        r = ScoringReviewer(
            model=TestModel(),
            max_iterations=1,
            scoring_task="Rate quality.",
        )
        responses, cost = await r.review_items(
            text_inputs=["Paper A abstract.", "Paper B abstract."],
            item_ids=["0", "1"],
        )
        assert len(responses) == 2
        for resp in responses:
            assert "reasoning" in resp
            assert "score" in resp


# ---------------------------------------------------------------------------
# TitleAbstractReviewer
# ---------------------------------------------------------------------------


class TestTitleAbstractReviewerInit:
    def test_default_values(self):
        r = TitleAbstractReviewer()
        assert r.output_type is EvaluationOutput
        assert r.system_prompt != ""
        assert "inclusion" in r.task_prompt.lower() or "include" in r.task_prompt.lower()

    def test_is_subclass(self):
        r = TitleAbstractReviewer()
        assert isinstance(r, AgenticReviewer)

    def test_custom_criteria(self):
        r = TitleAbstractReviewer(
            inclusion_criteria="Studies on humans aged 18+",
            exclusion_criteria="Animal studies, case reports",
        )
        assert "humans aged 18+" in r.task_prompt
        assert "Animal studies" in r.task_prompt

    def test_output_type_enforced(self):
        r = TitleAbstractReviewer(output_type=ScoringOutput)
        assert r.output_type is EvaluationOutput

    def test_custom_system_prompt_preserved(self):
        r = TitleAbstractReviewer(system_prompt="Custom reviewer instructions.")
        assert r.system_prompt == "Custom reviewer instructions."

    def test_custom_task_prompt_preserved(self):
        custom = "Evaluate: ${item}$"
        r = TitleAbstractReviewer(task_prompt=custom)
        assert r.task_prompt == custom

    def test_override_agentic_params(self):
        r = TitleAbstractReviewer(
            name="Screener",
            max_iterations=1,
            model_settings={"temperature": 0.0},
        )
        assert r.name == "Screener"
        assert r.max_iterations == 1
        assert r.model_settings == {"temperature": 0.0}

    def test_likert_scale_in_prompt(self):
        r = TitleAbstractReviewer()
        assert "1" in r.task_prompt and "5" in r.task_prompt
        assert "exclude" in r.task_prompt.lower()
        assert "include" in r.task_prompt.lower()

    @pytest.mark.asyncio
    async def test_review_item_with_test_model(self):
        from pydantic_ai.models.test import TestModel

        r = TitleAbstractReviewer(
            model=TestModel(),
            max_iterations=1,
            inclusion_criteria="RCTs on diabetes",
            exclusion_criteria="Animal studies",
        )
        response, cost = await r.review_item(
            item_text="A randomized trial of metformin in type 2 diabetes patients.",
            item_id="test_001",
        )
        assert "reasoning" in response
        assert "decision" in response
        assert "certainty" in response

    @pytest.mark.asyncio
    async def test_review_items_with_test_model(self):
        from pydantic_ai.models.test import TestModel

        r = TitleAbstractReviewer(
            model=TestModel(),
            max_iterations=1,
        )
        responses, cost = await r.review_items(
            text_inputs=["Study A", "Study B", "Study C"],
        )
        assert len(responses) == 3
        for resp in responses:
            assert "decision" in resp


# ---------------------------------------------------------------------------
# AbstractionReviewer
# ---------------------------------------------------------------------------


class TestAbstractionReviewerInit:
    def test_default_values(self):
        r = AbstractionReviewer()
        assert r.system_prompt != ""
        # Output type should be dynamically built
        assert r.output_type is not ScoringOutput
        assert r.output_type is not EvaluationOutput

    def test_is_subclass(self):
        r = AbstractionReviewer()
        assert isinstance(r, AgenticReviewer)

    def test_custom_abstraction_keys(self):
        r = AbstractionReviewer(
            abstraction_keys={"study_design": str, "sample_size": int, "outcome": str},
            key_descriptions={
                "study_design": "Type of study (RCT, cohort, etc.)",
                "sample_size": "Number of participants",
                "outcome": "Primary outcome measure",
            },
        )
        # Output model should have these fields
        fields = r.output_type.model_fields
        assert "study_design" in fields
        assert "sample_size" in fields
        assert "outcome" in fields
        # No reasoning field
        assert "reasoning" not in fields

    def test_no_reasoning_field(self):
        """AbstractionReviewer should not have a reasoning field in output."""
        r = AbstractionReviewer(
            abstraction_keys={"title": str, "year": int},
        )
        fields = r.output_type.model_fields
        assert "reasoning" not in fields

    def test_key_descriptions_in_prompt(self):
        r = AbstractionReviewer(
            abstraction_keys={"drug_name": str},
            key_descriptions={"drug_name": "The name of the drug being studied."},
        )
        assert "drug_name" in r.task_prompt
        assert "name of the drug" in r.task_prompt

    def test_custom_system_prompt_preserved(self):
        r = AbstractionReviewer(system_prompt="Extract data carefully.")
        assert r.system_prompt == "Extract data carefully."

    def test_custom_task_prompt_preserved(self):
        custom = "Extract from: ${item}$"
        r = AbstractionReviewer(task_prompt=custom)
        assert r.task_prompt == custom

    def test_override_agentic_params(self):
        r = AbstractionReviewer(
            name="Extractor",
            max_iterations=5,
            agentic_effort="high",
            abstraction_keys={"data": str},
        )
        assert r.name == "Extractor"
        assert r.max_iterations == 5
        assert r.is_agentic is True

    @pytest.mark.asyncio
    async def test_review_item_with_test_model(self):
        from pydantic_ai.models.test import TestModel

        r = AbstractionReviewer(
            model=TestModel(),
            max_iterations=1,
            abstraction_keys={"study_type": str, "sample_size": int},
            key_descriptions={
                "study_type": "Type of study design",
                "sample_size": "Number of participants",
            },
        )
        response, cost = await r.review_item(
            item_text="A randomized controlled trial of 200 patients.",
            item_id="test_001",
        )
        assert "study_type" in response
        assert "sample_size" in response
        assert "reasoning" not in response

    @pytest.mark.asyncio
    async def test_review_items_with_test_model(self):
        from pydantic_ai.models.test import TestModel

        r = AbstractionReviewer(
            model=TestModel(),
            max_iterations=1,
            abstraction_keys={"finding": str},
        )
        responses, cost = await r.review_items(
            text_inputs=["Study 1 findings.", "Study 2 findings."],
        )
        assert len(responses) == 2
        for resp in responses:
            assert "finding" in resp


# ---------------------------------------------------------------------------
# Cross-cutting: Preset reviewers work with AgenticWorkflow
# ---------------------------------------------------------------------------


class TestPresetsWithWorkflow:
    @pytest.mark.asyncio
    async def test_scoring_reviewer_in_workflow(self):
        from pydantic_ai.models.test import TestModel
        from lattereview.agentic.workflow import AgenticWorkflow

        reviewer = ScoringReviewer(
            name="Scorer",
            model=TestModel(),
            max_iterations=1,
            scoring_task="Rate quality 1-5.",
            scoring_set=[1, 2, 3, 4, 5],
        )

        df = pd.DataFrame(
            {
                "Title": ["Paper A", "Paper B"],
                "Abstract": ["Abstract of A.", "Abstract of B."],
            }
        )

        schema = [
            {
                "round": "A",
                "reviewers": [reviewer],
                "text_inputs": ["Title", "Abstract"],
            }
        ]

        wf = AgenticWorkflow(workflow_schema=schema)
        result = await wf.run(df)

        assert f"round-A_Scorer_score" in result.columns
        assert f"round-A_Scorer_reasoning" in result.columns
        assert f"round-A_Scorer_certainty" in result.columns
        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_title_abstract_reviewer_in_workflow(self):
        from pydantic_ai.models.test import TestModel
        from lattereview.agentic.workflow import AgenticWorkflow

        reviewer = TitleAbstractReviewer(
            name="Screener",
            model=TestModel(),
            max_iterations=1,
            inclusion_criteria="Human studies",
            exclusion_criteria="Animal studies",
        )

        df = pd.DataFrame(
            {
                "Title": ["Human RCT", "Mouse Study"],
                "Abstract": ["A human trial.", "A mouse experiment."],
            }
        )

        schema = [
            {
                "round": "A",
                "reviewers": [reviewer],
                "text_inputs": ["Title", "Abstract"],
            }
        ]

        wf = AgenticWorkflow(workflow_schema=schema)
        result = await wf.run(df)

        assert f"round-A_Screener_decision" in result.columns
        assert f"round-A_Screener_reasoning" in result.columns
        assert f"round-A_Screener_certainty" in result.columns

    @pytest.mark.asyncio
    async def test_abstraction_reviewer_in_workflow(self):
        from pydantic_ai.models.test import TestModel
        from lattereview.agentic.workflow import AgenticWorkflow

        reviewer = AbstractionReviewer(
            name="Extractor",
            model=TestModel(),
            max_iterations=1,
            abstraction_keys={"drug": str, "dose": str},
            key_descriptions={
                "drug": "Name of the drug",
                "dose": "Dosage amount",
            },
        )

        df = pd.DataFrame(
            {
                "Title": ["Drug X Trial"],
                "Abstract": ["Patients received Drug X 100mg daily."],
            }
        )

        schema = [
            {
                "round": "A",
                "reviewers": [reviewer],
                "text_inputs": ["Title", "Abstract"],
            }
        ]

        wf = AgenticWorkflow(workflow_schema=schema)
        result = await wf.run(df)

        assert f"round-A_Extractor_drug" in result.columns
        assert f"round-A_Extractor_dose" in result.columns
        # No reasoning column for abstraction
        assert f"round-A_Extractor_reasoning" not in result.columns

    @pytest.mark.asyncio
    async def test_mixed_presets_multi_round(self):
        """Multiple preset types in a multi-round workflow."""
        from pydantic_ai.models.test import TestModel
        from lattereview.agentic.workflow import AgenticWorkflow

        scorer = ScoringReviewer(
            name="Scorer",
            model=TestModel(),
            max_iterations=1,
        )
        screener = TitleAbstractReviewer(
            name="Screener",
            model=TestModel(),
            max_iterations=1,
        )

        df = pd.DataFrame(
            {
                "Title": ["Study 1", "Study 2", "Study 3"],
                "Abstract": ["Abstract 1", "Abstract 2", "Abstract 3"],
            }
        )

        schema = [
            {
                "round": "A",
                "reviewers": [scorer],
                "text_inputs": ["Title", "Abstract"],
            },
            {
                "round": "B",
                "reviewers": [screener],
                "text_inputs": ["Title", "Abstract"],
            },
        ]

        wf = AgenticWorkflow(workflow_schema=schema)
        result = await wf.run(df)

        assert "round-A_Scorer_score" in result.columns
        assert "round-B_Screener_decision" in result.columns
        assert len(result) == 3


# ---------------------------------------------------------------------------
# Import test
# ---------------------------------------------------------------------------


class TestImports:
    def test_import_from_agentic_package(self):
        from lattereview.agentic import (
            ScoringReviewer,
            TitleAbstractReviewer,
            AbstractionReviewer,
        )

        assert ScoringReviewer is not None
        assert TitleAbstractReviewer is not None
        assert AbstractionReviewer is not None


# ---------------------------------------------------------------------------
# Live tests
# ---------------------------------------------------------------------------


class TestLive:
    @pytest.mark.live
    @pytest.mark.asyncio
    async def test_scoring_reviewer_openai(self, env_keys):
        """Live: ScoringReviewer produces valid scored output."""
        if not os.environ.get("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set")

        r = ScoringReviewer(
            name="LiveScorer",
            model="openai:gpt-5.4-mini",
            max_iterations=1,
            scoring_task="Rate the methodological quality of the study.",
            scoring_set=[1, 2, 3, 4, 5],
            scoring_rules="1=very low quality, 5=very high quality",
            model_settings={"temperature": 0.0},
        )

        response, cost = await r.review_item(
            item_text=(
                "Title: Efficacy of Drug X for Type 2 Diabetes\n"
                "Abstract: A randomized controlled trial of 500 patients "
                "comparing Drug X to placebo over 12 months showed significant "
                "HbA1c reduction (p<0.001)."
            ),
            item_id="live_001",
        )

        assert isinstance(response["reasoning"], str)
        assert len(response["reasoning"]) > 0
        assert isinstance(response["score"], int)
        assert response["score"] in [1, 2, 3, 4, 5]
        assert 0 <= response["certainty"] <= 100

    @pytest.mark.live
    @pytest.mark.asyncio
    async def test_title_abstract_reviewer_openai(self, env_keys):
        """Live: TitleAbstractReviewer produces valid evaluation output."""
        if not os.environ.get("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set")

        r = TitleAbstractReviewer(
            name="LiveScreener",
            model="openai:gpt-5.4-mini",
            max_iterations=1,
            inclusion_criteria="Randomized controlled trials on diabetes in adults",
            exclusion_criteria="Animal studies, pediatric populations, reviews",
            model_settings={"temperature": 0.0},
        )

        response, cost = await r.review_item(
            item_text=(
                "Title: Mouse Model of Insulin Resistance\n"
                "Abstract: We developed a novel mouse model to study "
                "insulin resistance mechanisms using C57BL/6J mice."
            ),
            item_id="live_002",
        )

        assert isinstance(response["reasoning"], str)
        assert len(response["reasoning"]) > 0
        assert isinstance(response["decision"], str)
        assert 0 <= response["certainty"] <= 100

    @pytest.mark.live
    @pytest.mark.asyncio
    async def test_abstraction_reviewer_openai(self, env_keys):
        """Live: AbstractionReviewer extracts structured data."""
        if not os.environ.get("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set")

        r = AbstractionReviewer(
            name="LiveExtractor",
            model="openai:gpt-5.4-mini",
            max_iterations=1,
            abstraction_keys={"drug_name": str, "sample_size": int, "study_design": str},
            key_descriptions={
                "drug_name": "The name of the drug being studied.",
                "sample_size": "Total number of participants enrolled.",
                "study_design": "Type of study (RCT, cohort, case-control, etc.)",
            },
            model_settings={"temperature": 0.0},
        )

        response, cost = await r.review_item(
            item_text=(
                "Title: Metformin for Type 2 Diabetes\n"
                "Abstract: This randomized controlled trial enrolled 350 patients "
                "with type 2 diabetes to evaluate metformin 500mg twice daily."
            ),
            item_id="live_003",
        )

        assert isinstance(response["drug_name"], str)
        assert isinstance(response["sample_size"], int)
        assert isinstance(response["study_design"], str)
        assert "reasoning" not in response

    @pytest.mark.live
    @pytest.mark.asyncio
    async def test_presets_in_workflow_openai(self, env_keys):
        """Live: All three presets run through AgenticWorkflow."""
        if not os.environ.get("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set")

        from lattereview.agentic.workflow import AgenticWorkflow

        scorer = ScoringReviewer(
            name="Scorer",
            model="openai:gpt-5.4-mini",
            max_iterations=1,
            scoring_task="Rate quality 1-5.",
            scoring_set=[1, 2, 3, 4, 5],
            model_settings={"temperature": 0.0},
        )

        df = pd.DataFrame(
            {
                "Title": [
                    "RCT on Drug X for Diabetes",
                    "Case Report of Adverse Event",
                ],
                "Abstract": [
                    "A randomized controlled trial of 500 patients.",
                    "A single case of hepatotoxicity in a 45-year-old.",
                ],
            }
        )

        schema = [
            {
                "round": "A",
                "reviewers": [scorer],
                "text_inputs": ["Title", "Abstract"],
            }
        ]

        wf = AgenticWorkflow(workflow_schema=schema)
        result = await wf.run(df)

        assert "round-A_Scorer_score" in result.columns
        assert "round-A_Scorer_reasoning" in result.columns
        assert len(result) == 2
        # Scores should be valid integers
        for score in result["round-A_Scorer_score"]:
            assert isinstance(score, int)
