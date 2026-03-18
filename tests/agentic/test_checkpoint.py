"""Tests for checkpoint/resume system — RunState, CheckpointManager, and workflow integration."""

import json
import os
from pathlib import Path

import pandas as pd
import pytest
from pydantic_ai.models.test import TestModel

from lattereview.agentic.checkpoint.state import RunState, compute_schema_hash
from lattereview.agentic.checkpoint.manager import CheckpointManager
from lattereview.agentic.reviewer import AgenticReviewer
from lattereview.agentic.workflow import AgenticWorkflow
from lattereview.agentic.output_models import ScoringOutput, EvaluationOutput

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def working_dir(tmp_path):
    return tmp_path / "run_dir"


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
def test_schema(test_reviewer):
    return [
        {"round": "A", "reviewers": [test_reviewer], "text_inputs": ["Title"]},
    ]


# ---------------------------------------------------------------------------
# RunState
# ---------------------------------------------------------------------------


class TestRunState:
    def test_default_values(self):
        state = RunState()
        assert state.status == "running"
        assert state.current_round_index == 0
        assert state.current_reviewer_index == 0
        assert state.completed_items == {}
        assert state.total_cost == 0.0

    def test_mark_item_completed(self):
        state = RunState()
        state.mark_item_completed("A", "Rev1", "A-0")
        assert state.is_item_completed("A", "Rev1", "A-0")
        assert not state.is_item_completed("A", "Rev1", "A-1")

    def test_mark_item_completed_idempotent(self):
        state = RunState()
        state.mark_item_completed("A", "Rev1", "A-0")
        state.mark_item_completed("A", "Rev1", "A-0")
        assert state.get_completed_items("A", "Rev1") == ["A-0"]

    def test_get_completed_items_empty(self):
        state = RunState()
        assert state.get_completed_items("A", "Rev1") == []

    def test_multiple_rounds(self):
        state = RunState()
        state.mark_item_completed("A", "Rev1", "A-0")
        state.mark_item_completed("B", "Rev2", "B-0")
        assert state.get_completed_items("A", "Rev1") == ["A-0"]
        assert state.get_completed_items("B", "Rev2") == ["B-0"]

    def test_touch_updates_timestamp(self):
        state = RunState()
        old = state.updated_at
        state.touch()
        # Updated should be >= old (may be same if very fast)
        assert state.updated_at >= old

    def test_serialization(self):
        state = RunState(schema_hash="abc123", status="completed", total_cost=1.5)
        state.mark_item_completed("A", "Rev1", "A-0")
        data = state.model_dump_json()
        restored = RunState.model_validate_json(data)
        assert restored.schema_hash == "abc123"
        assert restored.status == "completed"
        assert restored.total_cost == 1.5
        assert restored.is_item_completed("A", "Rev1", "A-0")


# ---------------------------------------------------------------------------
# compute_schema_hash
# ---------------------------------------------------------------------------


class TestSchemaHash:
    def test_deterministic(self, test_schema):
        h1 = compute_schema_hash(test_schema)
        h2 = compute_schema_hash(test_schema)
        assert h1 == h2

    def test_different_for_different_rounds(self, test_reviewer):
        schema1 = [{"round": "A", "reviewers": [test_reviewer], "text_inputs": ["Title"]}]
        schema2 = [{"round": "B", "reviewers": [test_reviewer], "text_inputs": ["Title"]}]
        assert compute_schema_hash(schema1) != compute_schema_hash(schema2)

    def test_different_for_different_reviewers(self):
        rev1 = AgenticReviewer(name="Rev1", model=TestModel(), max_iterations=1, output_type=ScoringOutput)
        rev2 = AgenticReviewer(name="Rev2", model=TestModel(), max_iterations=1, output_type=ScoringOutput)
        schema1 = [{"round": "A", "reviewers": [rev1], "text_inputs": ["Title"]}]
        schema2 = [{"round": "A", "reviewers": [rev2], "text_inputs": ["Title"]}]
        assert compute_schema_hash(schema1) != compute_schema_hash(schema2)

    def test_different_for_different_inputs(self, test_reviewer):
        schema1 = [{"round": "A", "reviewers": [test_reviewer], "text_inputs": ["Title"]}]
        schema2 = [{"round": "A", "reviewers": [test_reviewer], "text_inputs": ["Abstract"]}]
        assert compute_schema_hash(schema1) != compute_schema_hash(schema2)

    def test_different_for_different_output_types(self):
        rev1 = AgenticReviewer(name="Rev", model=TestModel(), max_iterations=1, output_type=ScoringOutput)
        rev2 = AgenticReviewer(name="Rev", model=TestModel(), max_iterations=1, output_type=EvaluationOutput)
        schema1 = [{"round": "A", "reviewers": [rev1], "text_inputs": ["Title"]}]
        schema2 = [{"round": "A", "reviewers": [rev2], "text_inputs": ["Title"]}]
        assert compute_schema_hash(schema1) != compute_schema_hash(schema2)

    def test_hash_length(self, test_schema):
        h = compute_schema_hash(test_schema)
        assert len(h) == 16  # Truncated SHA-256


# ---------------------------------------------------------------------------
# CheckpointManager — initialization
# ---------------------------------------------------------------------------


class TestCheckpointManagerInit:
    def test_initialize_creates_dir(self, working_dir, test_schema):
        mgr = CheckpointManager(working_dir)
        mgr.initialize(test_schema)
        assert working_dir.exists()
        assert mgr.metadata_path.exists()

    def test_initialize_creates_run_state(self, working_dir, test_schema):
        mgr = CheckpointManager(working_dir)
        state = mgr.initialize(test_schema)
        assert state.status == "running"
        assert state.schema_hash == compute_schema_hash(test_schema)

    def test_can_resume_false_initially(self, working_dir):
        mgr = CheckpointManager(working_dir)
        assert not mgr.can_resume()

    def test_can_resume_true_after_init(self, working_dir, test_schema):
        mgr = CheckpointManager(working_dir)
        mgr.initialize(test_schema)
        mgr2 = CheckpointManager(working_dir)
        assert mgr2.can_resume()


# ---------------------------------------------------------------------------
# CheckpointManager — per-item results
# ---------------------------------------------------------------------------


class TestCheckpointManagerResults:
    def test_save_and_load_item(self, working_dir, test_schema):
        mgr = CheckpointManager(working_dir)
        mgr.initialize(test_schema)
        result = {"reasoning": "good", "score": 8, "certainty": 0.9}
        mgr.save_item_result("A", "Rev1", "A-0", result, cost=0.01)

        loaded = mgr.load_item_result("A", "Rev1", "A-0")
        assert loaded is not None
        assert loaded["result"] == result
        assert loaded["cost"] == 0.01
        assert loaded["item_id"] == "A-0"

    def test_load_nonexistent(self, working_dir, test_schema):
        mgr = CheckpointManager(working_dir)
        mgr.initialize(test_schema)
        assert mgr.load_item_result("A", "Rev1", "A-99") is None

    def test_save_updates_run_state(self, working_dir, test_schema):
        mgr = CheckpointManager(working_dir)
        mgr.initialize(test_schema)
        mgr.save_item_result("A", "TestRev", "A-0", {"score": 5}, cost=0.01)
        assert mgr.state.is_item_completed("A", "TestRev", "A-0")
        assert mgr.state.total_cost == 0.01

    def test_get_completed_items_from_state(self, working_dir, test_schema):
        mgr = CheckpointManager(working_dir)
        mgr.initialize(test_schema)
        mgr.save_item_result("A", "TestRev", "A-0", {"score": 5})
        mgr.save_item_result("A", "TestRev", "A-1", {"score": 7})
        completed = mgr.get_completed_items("A", "TestRev")
        assert set(completed) == {"A-0", "A-1"}

    def test_get_completed_items_fallback_to_disk(self, working_dir, test_schema):
        """If RunState is empty, scan result files on disk."""
        mgr = CheckpointManager(working_dir)
        mgr.initialize(test_schema)

        # Write a result file directly (bypassing state update)
        results_dir = working_dir / "round_A" / "agent_TestRev" / "results"
        results_dir.mkdir(parents=True)
        (results_dir / "item_A-0.json").write_text(json.dumps({"item_id": "A-0", "result": {"score": 5}, "cost": 0.0}))

        # Create a fresh manager (no state tracking for this item)
        mgr2 = CheckpointManager(working_dir)
        mgr2._state = RunState()  # Fresh state with no completed items
        completed = mgr2.get_completed_items("A", "TestRev")
        assert "A-0" in completed

    def test_item_id_with_special_chars(self, working_dir, test_schema):
        mgr = CheckpointManager(working_dir)
        mgr.initialize(test_schema)
        mgr.save_item_result("A", "Rev", "A/0", {"score": 5})
        loaded = mgr.load_item_result("A", "Rev", "A/0")
        assert loaded is not None
        assert loaded["item_id"] == "A/0"


# ---------------------------------------------------------------------------
# CheckpointManager — DataFrame snapshots
# ---------------------------------------------------------------------------


class TestCheckpointManagerSnapshots:
    def test_save_and_load_snapshot(self, working_dir, test_schema, sample_df):
        mgr = CheckpointManager(working_dir)
        mgr.initialize(test_schema)
        path = mgr.save_dataframe_snapshot(sample_df, "A")
        assert path.exists()

        loaded = mgr.load_dataframe_snapshot("A")
        assert loaded is not None
        pd.testing.assert_frame_equal(loaded, sample_df)

    def test_load_nonexistent_snapshot(self, working_dir, test_schema):
        mgr = CheckpointManager(working_dir)
        mgr.initialize(test_schema)
        assert mgr.load_dataframe_snapshot("X") is None

    def test_save_final_dataframe(self, working_dir, test_schema, sample_df):
        mgr = CheckpointManager(working_dir)
        mgr.initialize(test_schema)
        path = mgr.save_final_dataframe(sample_df)
        assert path.exists()
        assert path.name == "final.parquet"
        loaded = pd.read_parquet(path)
        pd.testing.assert_frame_equal(loaded, sample_df)


# ---------------------------------------------------------------------------
# CheckpointManager — run state persistence
# ---------------------------------------------------------------------------


class TestCheckpointManagerPersistence:
    def test_save_and_reload_state(self, working_dir, test_schema):
        mgr = CheckpointManager(working_dir)
        state = mgr.initialize(test_schema)
        state.mark_item_completed("A", "TestRev", "A-0")
        state.total_cost = 0.05
        mgr.save_run_state()

        # Load from disk in a new manager
        mgr2 = CheckpointManager(working_dir)
        loaded = mgr2.load_for_resume(test_schema)
        assert loaded.is_item_completed("A", "TestRev", "A-0")
        assert loaded.total_cost == 0.05

    def test_mark_completed(self, working_dir, test_schema):
        mgr = CheckpointManager(working_dir)
        mgr.initialize(test_schema)
        mgr.mark_completed()
        assert mgr.state.status == "completed"

    def test_mark_failed(self, working_dir, test_schema):
        mgr = CheckpointManager(working_dir)
        mgr.initialize(test_schema)
        mgr.mark_failed()
        assert mgr.state.status == "failed"


# ---------------------------------------------------------------------------
# CheckpointManager — resume with schema change
# ---------------------------------------------------------------------------


class TestCheckpointManagerResume:
    def test_resume_warns_on_schema_change(self, working_dir):
        rev1 = AgenticReviewer(name="Rev1", model=TestModel(), max_iterations=1, output_type=ScoringOutput)
        schema1 = [{"round": "A", "reviewers": [rev1], "text_inputs": ["Title"]}]

        mgr = CheckpointManager(working_dir)
        mgr.initialize(schema1)

        # Change schema
        rev2 = AgenticReviewer(name="Rev2", model=TestModel(), max_iterations=1, output_type=ScoringOutput)
        schema2 = [{"round": "A", "reviewers": [rev2], "text_inputs": ["Title"]}]

        mgr2 = CheckpointManager(working_dir)
        with pytest.warns(UserWarning, match="schema has changed"):
            mgr2.load_for_resume(schema2)

    def test_resume_no_warning_same_schema(self, working_dir, test_schema):
        mgr = CheckpointManager(working_dir)
        mgr.initialize(test_schema)

        mgr2 = CheckpointManager(working_dir)
        # No warning expected
        state = mgr2.load_for_resume(test_schema)
        assert state.status == "running"

    def test_resume_file_not_found(self, working_dir, test_schema):
        mgr = CheckpointManager(working_dir)
        with pytest.raises(FileNotFoundError):
            mgr.load_for_resume(test_schema)


# ---------------------------------------------------------------------------
# Workflow + Checkpoint Integration (unit tests)
# ---------------------------------------------------------------------------


class TestWorkflowCheckpointIntegration:
    @pytest.mark.asyncio
    async def test_workflow_with_working_dir_creates_checkpoint(self, test_reviewer, sample_df, working_dir):
        wf = AgenticWorkflow(
            workflow_schema=[
                {"round": "A", "reviewers": [test_reviewer], "text_inputs": ["Title"]},
            ],
            working_dir=working_dir,
            verbose=False,
        )
        result = await wf.run(sample_df)

        # Checkpoint files should exist
        assert (working_dir / "run_metadata.json").exists()
        assert (working_dir / "output" / "after_round_A.parquet").exists()
        assert (working_dir / "output" / "final.parquet").exists()

        # Results should be populated
        assert "round-A_TestRev_score" in result.columns
        for idx in range(len(sample_df)):
            assert result.at[idx, "round-A_TestRev_score"] is not None

    @pytest.mark.asyncio
    async def test_workflow_per_item_results_saved(self, test_reviewer, sample_df, working_dir):
        wf = AgenticWorkflow(
            workflow_schema=[
                {"round": "A", "reviewers": [test_reviewer], "text_inputs": ["Title"]},
            ],
            working_dir=working_dir,
            verbose=False,
        )
        await wf.run(sample_df)

        # Per-item result files
        results_dir = working_dir / "round_A" / "agent_TestRev" / "results"
        assert results_dir.exists()
        result_files = list(results_dir.glob("item_*.json"))
        assert len(result_files) == len(sample_df)

    @pytest.mark.asyncio
    async def test_workflow_action_logs_created(self, test_reviewer, sample_df, working_dir):
        wf = AgenticWorkflow(
            workflow_schema=[
                {"round": "A", "reviewers": [test_reviewer], "text_inputs": ["Title"]},
            ],
            working_dir=working_dir,
            verbose=False,
        )
        await wf.run(sample_df)

        # Action log files
        logs_dir = working_dir / "round_A" / "agent_TestRev" / "logs"
        assert logs_dir.exists()
        log_files = list(logs_dir.glob("item_*.jsonl"))
        assert len(log_files) == len(sample_df)

    @pytest.mark.asyncio
    async def test_workflow_run_state_completed(self, test_reviewer, sample_df, working_dir):
        wf = AgenticWorkflow(
            workflow_schema=[
                {"round": "A", "reviewers": [test_reviewer], "text_inputs": ["Title"]},
            ],
            working_dir=working_dir,
            verbose=False,
        )
        await wf.run(sample_df)

        state = RunState.model_validate_json((working_dir / "run_metadata.json").read_text())
        assert state.status == "completed"

    @pytest.mark.asyncio
    async def test_resume_skips_completed_items(self, test_reviewer, sample_df, working_dir):
        """Run workflow, then resume — all items should already be completed."""
        wf = AgenticWorkflow(
            workflow_schema=[
                {"round": "A", "reviewers": [test_reviewer], "text_inputs": ["Title"]},
            ],
            working_dir=working_dir,
            verbose=False,
        )
        result1 = await wf.run(sample_df)

        # Resume — should skip all items
        wf2 = AgenticWorkflow(
            workflow_schema=[
                {"round": "A", "reviewers": [test_reviewer], "text_inputs": ["Title"]},
            ],
            working_dir=working_dir,
            resume=True,
            verbose=False,
        )
        result2 = await wf2.run(sample_df)

        # Results should match
        assert "round-A_TestRev_score" in result2.columns
        for idx in range(len(sample_df)):
            assert result2.at[idx, "round-A_TestRev_score"] is not None

    @pytest.mark.asyncio
    async def test_resume_partial_completion(self, working_dir):
        """Simulate a partial run by pre-saving some results, then resume."""
        rev = AgenticReviewer(name="Rev", model=TestModel(), max_iterations=1, output_type=ScoringOutput)
        schema = [{"round": "A", "reviewers": [rev], "text_inputs": ["Title"]}]

        df = pd.DataFrame({"Title": ["Paper A", "Paper B", "Paper C"]})

        # Pre-save results for first 2 items
        mgr = CheckpointManager(working_dir)
        mgr.initialize(schema)
        mgr.save_item_result("A", "Rev", "A-0", {"reasoning": "pre-saved", "score": 5, "certainty": 0.8})
        mgr.save_item_result("A", "Rev", "A-1", {"reasoning": "pre-saved", "score": 7, "certainty": 0.9})

        # Resume — should only run item A-2
        wf = AgenticWorkflow(
            workflow_schema=schema,
            working_dir=working_dir,
            resume=True,
            verbose=False,
        )
        result = await wf.run(df)

        assert result.at[0, "round-A_Rev_reasoning"] == "pre-saved"
        assert result.at[1, "round-A_Rev_reasoning"] == "pre-saved"
        # Item 2 should have been run by TestModel
        assert result.at[2, "round-A_Rev_score"] is not None

    @pytest.mark.asyncio
    async def test_workflow_without_working_dir_no_checkpoint(self, test_reviewer, sample_df):
        """Without working_dir, no checkpoint files created."""
        wf = AgenticWorkflow(
            workflow_schema=[
                {"round": "A", "reviewers": [test_reviewer], "text_inputs": ["Title"]},
            ],
            verbose=False,
        )
        result = await wf.run(sample_df)
        assert "round-A_TestRev_score" in result.columns

    @pytest.mark.asyncio
    async def test_multi_round_checkpoint(self, working_dir, sample_df):
        rev1 = AgenticReviewer(name="Rev1", model=TestModel(), max_iterations=1, output_type=ScoringOutput)
        rev2 = AgenticReviewer(name="Rev2", model=TestModel(), max_iterations=1, output_type=ScoringOutput)

        wf = AgenticWorkflow(
            workflow_schema=[
                {"round": "A", "reviewers": [rev1], "text_inputs": ["Title"]},
                {"round": "B", "reviewers": [rev2], "text_inputs": ["Abstract"]},
            ],
            working_dir=working_dir,
            verbose=False,
        )
        result = await wf.run(sample_df)

        # Both round snapshots should exist
        assert (working_dir / "output" / "after_round_A.parquet").exists()
        assert (working_dir / "output" / "after_round_B.parquet").exists()
        assert (working_dir / "output" / "final.parquet").exists()

    @pytest.mark.asyncio
    async def test_resume_warns_on_schema_change(self, working_dir, sample_df):
        rev1 = AgenticReviewer(name="Rev1", model=TestModel(), max_iterations=1, output_type=ScoringOutput)
        schema1 = [{"round": "A", "reviewers": [rev1], "text_inputs": ["Title"]}]

        wf1 = AgenticWorkflow(workflow_schema=schema1, working_dir=working_dir, verbose=False)
        await wf1.run(sample_df)

        # Change schema and resume
        rev2 = AgenticReviewer(name="Rev2", model=TestModel(), max_iterations=1, output_type=ScoringOutput)
        schema2 = [{"round": "A", "reviewers": [rev2], "text_inputs": ["Title"]}]

        wf2 = AgenticWorkflow(workflow_schema=schema2, working_dir=working_dir, resume=True, verbose=False)
        with pytest.warns(UserWarning, match="schema has changed"):
            await wf2.run(sample_df)

    @pytest.mark.asyncio
    async def test_workflow_failure_marks_state(self, working_dir):
        """If the workflow raises, state should be marked failed."""
        rev = AgenticReviewer(name="Rev", model=TestModel(), max_iterations=1, output_type=ScoringOutput)
        wf = AgenticWorkflow(
            workflow_schema=[
                {"round": "A", "reviewers": [rev], "text_inputs": ["NonExistent"]},
            ],
            working_dir=working_dir,
            verbose=False,
        )
        df = pd.DataFrame({"Title": ["A"]})
        with pytest.raises(Exception):
            await wf.run(df)

        state = RunState.model_validate_json((working_dir / "run_metadata.json").read_text())
        assert state.status == "failed"


# ---------------------------------------------------------------------------
# Live Integration Tests
# ---------------------------------------------------------------------------


class TestLiveCheckpoint:
    """Live tests — require API keys in .env."""

    @pytest.mark.live
    @pytest.mark.asyncio
    async def test_checkpoint_resume_openai(self, env_keys, tmp_path):
        """Run workflow with checkpoint, then resume — verify completion."""
        if not os.environ.get("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set")

        working_dir = tmp_path / "live_checkpoint"

        reviewer = AgenticReviewer(
            name="Scorer",
            model="openai:gpt-5.4-mini",
            max_iterations=1,
            system_prompt="Rate paper quality 1-10.",
            output_type=ScoringOutput,
            model_settings={"temperature": 0.0},
        )

        df = pd.DataFrame(
            {
                "Title": ["RCT on Drug X", "Case Report Y", "Meta-Analysis Z"],
                "Abstract": [
                    "A large RCT testing drug X for type 2 diabetes.",
                    "Single case of adverse event with drug Y.",
                    "Meta-analysis of 30 RCTs on exercise and depression.",
                ],
            }
        )

        schema = [{"round": "A", "reviewers": [reviewer], "text_inputs": ["Title", "Abstract"]}]

        # First run
        wf1 = AgenticWorkflow(
            workflow_schema=schema,
            working_dir=working_dir,
            verbose=False,
        )
        result1 = await wf1.run(df)

        # Verify checkpoint state
        state = RunState.model_validate_json((working_dir / "run_metadata.json").read_text())
        assert state.status == "completed"

        # Per-item results saved
        results_dir = working_dir / "round_A" / "agent_Scorer" / "results"
        assert len(list(results_dir.glob("item_*.json"))) == 3

        # Resume — should skip all items
        wf2 = AgenticWorkflow(
            workflow_schema=schema,
            working_dir=working_dir,
            resume=True,
            verbose=False,
        )
        result2 = await wf2.run(df)

        # Results should be identical (loaded from checkpoint)
        for idx in range(len(df)):
            assert result2.at[idx, "round-A_Scorer_score"] is not None
            assert isinstance(result2.at[idx, "round-A_Scorer_reasoning"], str)

    @pytest.mark.live
    @pytest.mark.asyncio
    async def test_partial_resume_openai(self, env_keys, tmp_path):
        """Pre-save some results, then resume to complete remaining items."""
        if not os.environ.get("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set")

        working_dir = tmp_path / "partial_resume"

        reviewer = AgenticReviewer(
            name="Scorer",
            model="openai:gpt-5.4-mini",
            max_iterations=1,
            system_prompt="Rate paper quality 1-10.",
            output_type=ScoringOutput,
            model_settings={"temperature": 0.0},
        )

        df = pd.DataFrame(
            {
                "Title": ["Paper A", "Paper B", "Paper C"],
                "Abstract": ["Abstract A", "Abstract B", "Abstract C"],
            }
        )

        schema = [{"round": "A", "reviewers": [reviewer], "text_inputs": ["Title", "Abstract"]}]

        # Pre-save result for first item only
        mgr = CheckpointManager(working_dir)
        mgr.initialize(schema)
        mgr.save_item_result(
            "A",
            "Scorer",
            "A-0",
            {"reasoning": "Pre-saved result", "score": 8, "certainty": 0.95},
            cost=0.001,
        )

        # Resume — should only call API for items A-1 and A-2
        wf = AgenticWorkflow(
            workflow_schema=schema,
            working_dir=working_dir,
            resume=True,
            verbose=False,
        )
        result = await wf.run(df)

        # First item should have pre-saved result
        assert result.at[0, "round-A_Scorer_reasoning"] == "Pre-saved result"
        assert result.at[0, "round-A_Scorer_score"] == 8

        # Other items should have real results
        assert isinstance(result.at[1, "round-A_Scorer_reasoning"], str)
        assert isinstance(result.at[2, "round-A_Scorer_reasoning"], str)
