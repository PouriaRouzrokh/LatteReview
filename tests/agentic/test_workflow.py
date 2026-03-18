"""Tests for workflow.py — AgenticWorkflow orchestration."""

import os
import tempfile

import pandas as pd
import pytest
from pydantic import BaseModel, Field
from pydantic_ai.models.test import TestModel

from lattereview.agentic.reviewer import AgenticReviewer
from lattereview.agentic.workflow import AgenticWorkflow, AgenticWorkflowError
from lattereview.agentic.output_models import ScoringOutput, EvaluationOutput

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_df():
    return pd.DataFrame(
        {
            "Title": ["Paper A", "Paper B", "Paper C", "Paper D"],
            "Abstract": ["Abstract of A", "Abstract of B", "Abstract of C", "Abstract of D"],
        }
    )


@pytest.fixture
def test_reviewer():
    return AgenticReviewer(
        name="TestRev",
        model=TestModel(),
        max_iterations=1,
        system_prompt="Review papers.",
        output_type=ScoringOutput,
    )


@pytest.fixture
def test_reviewer_eval():
    return AgenticReviewer(
        name="EvalRev",
        model=TestModel(),
        max_iterations=1,
        output_type=EvaluationOutput,
    )


# ---------------------------------------------------------------------------
# Schema Validation
# ---------------------------------------------------------------------------


class TestSchemaValidation:
    def test_empty_schema_raises(self):
        with pytest.raises(AgenticWorkflowError, match="cannot be empty"):
            AgenticWorkflow(workflow_schema=[])

    def test_missing_round_key(self, test_reviewer):
        with pytest.raises(AgenticWorkflowError, match="missing 'round'"):
            AgenticWorkflow(workflow_schema=[{"reviewers": [test_reviewer], "text_inputs": ["Title"]}])

    def test_missing_reviewers_key(self):
        with pytest.raises(AgenticWorkflowError, match="missing 'reviewers'"):
            AgenticWorkflow(workflow_schema=[{"round": "A", "text_inputs": ["Title"]}])

    def test_missing_text_inputs_key(self, test_reviewer):
        with pytest.raises(AgenticWorkflowError, match="missing 'text_inputs'"):
            AgenticWorkflow(workflow_schema=[{"round": "A", "reviewers": [test_reviewer]}])

    def test_duplicate_round_ids(self, test_reviewer):
        with pytest.raises(AgenticWorkflowError, match="Duplicate round ID"):
            AgenticWorkflow(
                workflow_schema=[
                    {"round": "A", "reviewers": [test_reviewer], "text_inputs": ["Title"]},
                    {"round": "A", "reviewers": [test_reviewer], "text_inputs": ["Title"]},
                ]
            )

    def test_invalid_reviewer_type(self):
        with pytest.raises(AgenticWorkflowError, match="must be AgenticReviewer"):
            AgenticWorkflow(workflow_schema=[{"round": "A", "reviewers": ["not_a_reviewer"], "text_inputs": ["Title"]}])

    def test_invalid_filter_type(self, test_reviewer):
        with pytest.raises(AgenticWorkflowError, match="must be callable"):
            AgenticWorkflow(
                workflow_schema=[
                    {
                        "round": "A",
                        "reviewers": [test_reviewer],
                        "text_inputs": ["Title"],
                        "filter": "not_callable",
                    }
                ]
            )

    def test_valid_schema(self, test_reviewer):
        wf = AgenticWorkflow(
            workflow_schema=[
                {"round": "A", "reviewers": [test_reviewer], "text_inputs": ["Title"]},
            ]
        )
        assert len(wf.workflow_schema) == 1

    def test_single_reviewer_not_in_list(self, test_reviewer):
        """Reviewers can be a single instance, not wrapped in a list."""
        wf = AgenticWorkflow(
            workflow_schema=[
                {"round": "A", "reviewers": test_reviewer, "text_inputs": ["Title"]},
            ]
        )
        assert wf is not None


# ---------------------------------------------------------------------------
# Text Input Formatting
# ---------------------------------------------------------------------------


class TestTextFormatting:
    def test_format_text_input_single_col(self, test_reviewer, sample_df):
        wf = AgenticWorkflow(workflow_schema=[{"round": "A", "reviewers": [test_reviewer], "text_inputs": ["Title"]}])
        text = wf._format_text_input(sample_df.iloc[0], ["Title"])
        assert "=== Title ===" in text
        assert "Paper A" in text

    def test_format_text_input_multi_col(self, test_reviewer, sample_df):
        wf = AgenticWorkflow(
            workflow_schema=[{"round": "A", "reviewers": [test_reviewer], "text_inputs": ["Title", "Abstract"]}]
        )
        text = wf._format_text_input(sample_df.iloc[0], ["Title", "Abstract"])
        assert "=== Title ===" in text
        assert "=== Abstract ===" in text
        assert "Paper A" in text
        assert "Abstract of A" in text


# ---------------------------------------------------------------------------
# Column Naming
# ---------------------------------------------------------------------------


class TestColumnNaming:
    @pytest.mark.asyncio
    async def test_output_columns_created(self, test_reviewer, sample_df):
        wf = AgenticWorkflow(
            workflow_schema=[
                {"round": "A", "reviewers": [test_reviewer], "text_inputs": ["Title"]},
            ]
        )
        result = await wf.run(sample_df)

        # Output column
        assert "round-A_TestRev_output" in result.columns
        # Per-field columns from ScoringOutput
        assert "round-A_TestRev_reasoning" in result.columns
        assert "round-A_TestRev_score" in result.columns
        assert "round-A_TestRev_certainty" in result.columns

    @pytest.mark.asyncio
    async def test_column_naming_convention(self, test_reviewer, sample_df):
        """Verify column naming matches v1: round-{ID}_{NAME}_{FIELD}."""
        wf = AgenticWorkflow(
            workflow_schema=[
                {"round": "X", "reviewers": [test_reviewer], "text_inputs": ["Title"]},
            ]
        )
        result = await wf.run(sample_df)
        expected_prefix = "round-X_TestRev_"
        matching = [c for c in result.columns if c.startswith(expected_prefix)]
        # Should have: output + reasoning + score + certainty = 4 columns
        assert len(matching) == 4

    @pytest.mark.asyncio
    async def test_eval_output_columns(self, test_reviewer_eval, sample_df):
        wf = AgenticWorkflow(
            workflow_schema=[
                {"round": "A", "reviewers": [test_reviewer_eval], "text_inputs": ["Title"]},
            ]
        )
        result = await wf.run(sample_df)
        assert "round-A_EvalRev_reasoning" in result.columns
        assert "round-A_EvalRev_decision" in result.columns
        assert "round-A_EvalRev_certainty" in result.columns


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------


class TestFiltering:
    @pytest.mark.asyncio
    async def test_filter_applied(self, test_reviewer):
        df = pd.DataFrame(
            {
                "Title": ["Good", "Bad", "Good2"],
                "keep": [True, False, True],
            }
        )
        wf = AgenticWorkflow(
            workflow_schema=[
                {
                    "round": "A",
                    "reviewers": [test_reviewer],
                    "text_inputs": ["Title"],
                    "filter": lambda row: row["keep"],
                },
            ]
        )
        result = await wf.run(df)
        # Filtered-out row should have None
        assert result.at[1, "round-A_TestRev_reasoning"] is None
        # Included rows should have values
        assert result.at[0, "round-A_TestRev_reasoning"] is not None
        assert result.at[2, "round-A_TestRev_reasoning"] is not None

    @pytest.mark.asyncio
    async def test_no_eligible_rows_skips_round(self, test_reviewer):
        df = pd.DataFrame({"Title": ["A", "B"], "keep": [False, False]})
        wf = AgenticWorkflow(
            workflow_schema=[
                {
                    "round": "A",
                    "reviewers": [test_reviewer],
                    "text_inputs": ["Title"],
                    "filter": lambda row: row["keep"],
                },
            ]
        )
        result = await wf.run(df)
        # No output columns should be created since the round was skipped
        assert "round-A_TestRev_output" not in result.columns


# ---------------------------------------------------------------------------
# Multi-Round & Multi-Reviewer
# ---------------------------------------------------------------------------


class TestMultiRound:
    @pytest.mark.asyncio
    async def test_two_rounds(self, test_reviewer, sample_df):
        rev2 = AgenticReviewer(name="Rev2", model=TestModel(), max_iterations=1, output_type=ScoringOutput)
        wf = AgenticWorkflow(
            workflow_schema=[
                {"round": "A", "reviewers": [test_reviewer], "text_inputs": ["Title"]},
                {"round": "B", "reviewers": [rev2], "text_inputs": ["Abstract"]},
            ]
        )
        result = await wf.run(sample_df)
        assert "round-A_TestRev_score" in result.columns
        assert "round-B_Rev2_score" in result.columns

    @pytest.mark.asyncio
    async def test_two_reviewers_same_round(self, test_reviewer, test_reviewer_eval, sample_df):
        wf = AgenticWorkflow(
            workflow_schema=[
                {
                    "round": "A",
                    "reviewers": [test_reviewer, test_reviewer_eval],
                    "text_inputs": ["Title"],
                },
            ]
        )
        result = await wf.run(sample_df)
        assert "round-A_TestRev_score" in result.columns
        assert "round-A_EvalRev_decision" in result.columns


# ---------------------------------------------------------------------------
# Cost Tracking
# ---------------------------------------------------------------------------


class TestCostTracking:
    @pytest.mark.asyncio
    async def test_cost_tracking(self, test_reviewer, sample_df):
        wf = AgenticWorkflow(
            workflow_schema=[
                {"round": "A", "reviewers": [test_reviewer], "text_inputs": ["Title"]},
            ]
        )
        await wf.run(sample_df)
        # TestModel returns 0.0 cost, but the tracking structure should exist
        assert ("A", "TestRev") in wf.reviewer_costs
        assert isinstance(wf.get_total_cost(), float)


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------


class TestDataLoading:
    @pytest.mark.asyncio
    async def test_dict_input(self, test_reviewer):
        wf = AgenticWorkflow(
            workflow_schema=[
                {"round": "A", "reviewers": [test_reviewer], "text_inputs": ["Title"]},
            ]
        )
        result = await wf({"Title": ["Paper 1", "Paper 2"]})
        assert len(result) == 2
        assert "round-A_TestRev_score" in result.columns

    @pytest.mark.asyncio
    async def test_csv_input(self, test_reviewer):
        wf = AgenticWorkflow(
            workflow_schema=[
                {"round": "A", "reviewers": [test_reviewer], "text_inputs": ["Title"]},
            ]
        )
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("Title\nPaper 1\nPaper 2\n")
            f.flush()
            result = await wf(f.name)
        os.unlink(f.name)
        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_invalid_data_type(self, test_reviewer):
        wf = AgenticWorkflow(
            workflow_schema=[
                {"round": "A", "reviewers": [test_reviewer], "text_inputs": ["Title"]},
            ]
        )
        with pytest.raises(AgenticWorkflowError, match="Invalid data type"):
            await wf(12345)

    @pytest.mark.asyncio
    async def test_missing_file(self, test_reviewer):
        wf = AgenticWorkflow(
            workflow_schema=[
                {"round": "A", "reviewers": [test_reviewer], "text_inputs": ["Title"]},
            ]
        )
        with pytest.raises(AgenticWorkflowError, match="File not found"):
            await wf("/nonexistent/file.csv")

    @pytest.mark.asyncio
    async def test_missing_column_raises(self, test_reviewer, sample_df):
        wf = AgenticWorkflow(
            workflow_schema=[
                {
                    "round": "A",
                    "reviewers": [test_reviewer],
                    "text_inputs": ["NonExistentColumn"],
                },
            ]
        )
        with pytest.raises(AgenticWorkflowError, match="columns not found"):
            await wf.run(sample_df)


# ---------------------------------------------------------------------------
# DataFrame Immutability
# ---------------------------------------------------------------------------


class TestImmutability:
    @pytest.mark.asyncio
    async def test_original_df_unchanged(self, test_reviewer, sample_df):
        original_cols = list(sample_df.columns)
        wf = AgenticWorkflow(
            workflow_schema=[
                {"round": "A", "reviewers": [test_reviewer], "text_inputs": ["Title"]},
            ]
        )
        result = await wf.run(sample_df)
        # Original should be unchanged
        assert list(sample_df.columns) == original_cols
        # Result should have new columns
        assert len(result.columns) > len(original_cols)


# ---------------------------------------------------------------------------
# Callable Interface
# ---------------------------------------------------------------------------


class TestCallable:
    @pytest.mark.asyncio
    async def test_call_with_dataframe(self, test_reviewer, sample_df):
        wf = AgenticWorkflow(
            workflow_schema=[
                {"round": "A", "reviewers": [test_reviewer], "text_inputs": ["Title"]},
            ]
        )
        result = await wf(sample_df)
        assert "round-A_TestRev_score" in result.columns


# ---------------------------------------------------------------------------
# Output Column Format
# ---------------------------------------------------------------------------


class TestOutputColumn:
    @pytest.mark.asyncio
    async def test_output_column_is_dict(self, test_reviewer, sample_df):
        """Output column stores a dict (not string) for downstream filter access.

        This matches v1 behavior where filter lambdas access fields like:
            row["round-A_Agent1_output"]["score"]
        """
        wf = AgenticWorkflow(
            workflow_schema=[
                {"round": "A", "reviewers": [test_reviewer], "text_inputs": ["Title"]},
            ]
        )
        result = await wf.run(sample_df)
        output_val = result.at[0, "round-A_TestRev_output"]
        assert isinstance(output_val, dict)
        assert "score" in output_val
        assert "reasoning" in output_val


# ---------------------------------------------------------------------------
# Cross-Round Column References
# ---------------------------------------------------------------------------


class TestCrossRound:
    @pytest.mark.asyncio
    async def test_round_b_uses_round_a_output(self, test_reviewer):
        """Round B can reference output columns from round A as text inputs."""
        rev2 = AgenticReviewer(name="Rev2", model=TestModel(), max_iterations=1, output_type=ScoringOutput)
        wf = AgenticWorkflow(
            workflow_schema=[
                {"round": "A", "reviewers": [test_reviewer], "text_inputs": ["Title"]},
                {
                    "round": "B",
                    "reviewers": [rev2],
                    "text_inputs": ["Title", "round-A_TestRev_reasoning"],
                },
            ]
        )
        df = pd.DataFrame({"Title": ["Paper X", "Paper Y"]})
        result = await wf.run(df)
        assert "round-B_Rev2_score" in result.columns

    @pytest.mark.asyncio
    async def test_filter_on_output_dict(self, test_reviewer):
        """v1 pattern: filter lambda accesses output dict fields from prior round."""
        rev2 = AgenticReviewer(name="Rev2", model=TestModel(), max_iterations=1, output_type=ScoringOutput)
        wf = AgenticWorkflow(
            workflow_schema=[
                {"round": "A", "reviewers": [test_reviewer], "text_inputs": ["Title"]},
                {
                    "round": "B",
                    "reviewers": [rev2],
                    "text_inputs": ["Title"],
                    # v1 pattern: filter accesses output dict from round A
                    "filter": lambda row: isinstance(row.get("round-A_TestRev_output"), dict),
                },
            ]
        )
        df = pd.DataFrame({"Title": ["Paper X", "Paper Y"]})
        result = await wf.run(df)
        # All rows should pass the filter since output is always a dict
        assert "round-B_Rev2_score" in result.columns


# ---------------------------------------------------------------------------
# Live Integration Tests
# ---------------------------------------------------------------------------


class TestLiveWorkflow:
    """Live tests — require API keys in .env."""

    @pytest.mark.live
    @pytest.mark.asyncio
    async def test_mini_workflow_openai(self, env_keys):
        """Run a small workflow with real OpenAI API."""
        if not os.environ.get("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set")

        reviewer = AgenticReviewer(
            name="Scorer",
            model="openai:gpt-5.4-mini",
            max_iterations=1,
            system_prompt="Rate each paper abstract on quality 1-10.",
            output_type=ScoringOutput,
            model_settings={"temperature": 0.0},
        )

        df = pd.DataFrame(
            {
                "Title": [
                    "RCT on Drug X for Diabetes",
                    "Case Report of Adverse Event",
                    "Meta-Analysis on Exercise",
                ],
                "Abstract": [
                    "A randomized controlled trial of 500 patients testing drug X for type 2 diabetes over 12 months.",
                    "We report a single case of severe hepatotoxicity associated with drug Y in a 45-year-old male.",
                    "A meta-analysis of 30 RCTs examining the effect of aerobic exercise on depression symptoms.",
                ],
            }
        )

        wf = AgenticWorkflow(
            workflow_schema=[
                {
                    "round": "A",
                    "reviewers": [reviewer],
                    "text_inputs": ["Title", "Abstract"],
                },
            ],
            verbose=False,
        )

        result = await wf.run(df)

        # Verify columns
        assert "round-A_Scorer_reasoning" in result.columns
        assert "round-A_Scorer_score" in result.columns
        assert "round-A_Scorer_certainty" in result.columns

        # Verify all rows got results
        for idx in range(len(df)):
            assert isinstance(result.at[idx, "round-A_Scorer_reasoning"], str)
            assert isinstance(result.at[idx, "round-A_Scorer_score"], int)

    @pytest.mark.live
    @pytest.mark.asyncio
    async def test_multi_round_workflow_openai(self, env_keys):
        """Two-round workflow: scoring then evaluation."""
        if not os.environ.get("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set")

        scorer = AgenticReviewer(
            name="Scorer",
            model="openai:gpt-5.4-mini",
            max_iterations=1,
            system_prompt="Rate paper quality 1-10.",
            output_type=ScoringOutput,
            model_settings={"temperature": 0.0},
        )

        evaluator = AgenticReviewer(
            name="Evaluator",
            model="openai:gpt-5.4-mini",
            max_iterations=1,
            system_prompt="Decide include/exclude for systematic review.",
            output_type=EvaluationOutput,
            model_settings={"temperature": 0.0},
        )

        df = pd.DataFrame(
            {
                "Title": ["RCT on Drug X", "Case Report Y"],
                "Abstract": [
                    "Large RCT testing drug X for diabetes.",
                    "Single case report of adverse event.",
                ],
            }
        )

        wf = AgenticWorkflow(
            workflow_schema=[
                {"round": "A", "reviewers": [scorer], "text_inputs": ["Title", "Abstract"]},
                {"round": "B", "reviewers": [evaluator], "text_inputs": ["Title", "Abstract"]},
            ],
            verbose=False,
        )

        result = await wf.run(df)

        assert "round-A_Scorer_score" in result.columns
        assert "round-B_Evaluator_decision" in result.columns
        assert len(result) == 2
