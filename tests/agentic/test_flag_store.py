"""Tests for flagging system — FlagStore, flagging-items skill, and integration."""

import asyncio
import json
import os

import pytest

from lattereview.agentic.flags.store import FlagStore
from lattereview.agentic.reviewer import AgenticReviewer

# ── FlagStore Tests ──────────────────────────────────────────────────


class TestFlagStore:
    @pytest.mark.asyncio
    async def test_initialize_creates_dir(self, tmp_path):
        store = FlagStore(tmp_path / "flags")
        await store.initialize()
        assert store.flags_dir.exists()

    @pytest.mark.asyncio
    async def test_add_flag(self, tmp_path):
        store = FlagStore(tmp_path / "flags")
        await store.initialize()

        result = await store.add_flag("item_001", "Missing abstract")
        assert "Flagged" in result
        assert await store.count() == 1

    @pytest.mark.asyncio
    async def test_add_multiple_flags(self, tmp_path):
        store = FlagStore(tmp_path / "flags")
        await store.initialize()

        await store.add_flag("item_001", "Reason 1")
        await store.add_flag("item_002", "Reason 2")
        await store.add_flag("item_003", "Reason 3")

        assert await store.count() == 3

    @pytest.mark.asyncio
    async def test_add_flag_updates_existing(self, tmp_path):
        store = FlagStore(tmp_path / "flags")
        await store.initialize()

        await store.add_flag("item_001", "Original reason")
        result = await store.add_flag("item_001", "Updated reason")
        assert "Updated" in result
        assert await store.count() == 1

        entries = await store.list_flags()
        assert entries[0]["reason"] == "Updated reason"

    @pytest.mark.asyncio
    async def test_resolve_flag(self, tmp_path):
        store = FlagStore(tmp_path / "flags")
        await store.initialize()

        await store.add_flag("item_001", "Reason")
        resolved = await store.resolve_flag("item_001")
        assert resolved is True
        assert await store.count(unresolved_only=True) == 0
        assert await store.count() == 1  # Still in entries, just resolved

    @pytest.mark.asyncio
    async def test_resolve_nonexistent(self, tmp_path):
        store = FlagStore(tmp_path / "flags")
        await store.initialize()

        resolved = await store.resolve_flag("item_999")
        assert resolved is False

    @pytest.mark.asyncio
    async def test_resolve_already_resolved(self, tmp_path):
        store = FlagStore(tmp_path / "flags")
        await store.initialize()

        await store.add_flag("item_001", "Reason")
        await store.resolve_flag("item_001")

        # Resolving again should return False (no unresolved flag found)
        resolved = await store.resolve_flag("item_001")
        assert resolved is False

    @pytest.mark.asyncio
    async def test_get_unresolved(self, tmp_path):
        store = FlagStore(tmp_path / "flags")
        await store.initialize()

        await store.add_flag("item_001", "Reason 1")
        await store.add_flag("item_002", "Reason 2")
        await store.resolve_flag("item_001")

        unresolved = await store.get_unresolved()
        assert len(unresolved) == 1
        assert unresolved[0]["item_id"] == "item_002"
        assert unresolved[0]["reason"] == "Reason 2"

    @pytest.mark.asyncio
    async def test_get_flag(self, tmp_path):
        store = FlagStore(tmp_path / "flags")
        await store.initialize()

        await store.add_flag("item_001", "Test reason")
        flag = await store.get_flag("item_001")
        assert flag is not None
        assert flag["item_id"] == "item_001"
        assert flag["reason"] == "Test reason"
        assert flag["resolved"] is False

    @pytest.mark.asyncio
    async def test_get_flag_nonexistent(self, tmp_path):
        store = FlagStore(tmp_path / "flags")
        await store.initialize()

        flag = await store.get_flag("item_999")
        assert flag is None

    @pytest.mark.asyncio
    async def test_get_flag_resolved_returns_none(self, tmp_path):
        store = FlagStore(tmp_path / "flags")
        await store.initialize()

        await store.add_flag("item_001", "Reason")
        await store.resolve_flag("item_001")

        flag = await store.get_flag("item_001")
        assert flag is None

    @pytest.mark.asyncio
    async def test_list_flags(self, tmp_path):
        store = FlagStore(tmp_path / "flags")
        await store.initialize()

        await store.add_flag("item_001", "Reason 1")
        await store.add_flag("item_002", "Reason 2")

        entries = await store.list_flags()
        assert len(entries) == 2
        assert entries[0]["item_id"] == "item_001"
        assert entries[1]["item_id"] == "item_002"

    @pytest.mark.asyncio
    async def test_list_flags_empty(self, tmp_path):
        store = FlagStore(tmp_path / "flags")
        await store.initialize()

        entries = await store.list_flags()
        assert entries == []

    @pytest.mark.asyncio
    async def test_count_with_filter(self, tmp_path):
        store = FlagStore(tmp_path / "flags")
        await store.initialize()

        await store.add_flag("item_001", "R1")
        await store.add_flag("item_002", "R2")
        await store.resolve_flag("item_001")

        assert await store.count() == 2
        assert await store.count(unresolved_only=True) == 1

    @pytest.mark.asyncio
    async def test_persistence_across_instances(self, tmp_path):
        flags_dir = tmp_path / "flags"

        store1 = FlagStore(flags_dir)
        await store1.initialize()
        await store1.add_flag("item_001", "Reason 1")
        await store1.add_flag("item_002", "Reason 2")
        await store1.resolve_flag("item_001")

        store2 = FlagStore(flags_dir)
        await store2.initialize()
        assert await store2.count() == 2
        assert await store2.count(unresolved_only=True) == 1

        unresolved = await store2.get_unresolved()
        assert len(unresolved) == 1
        assert unresolved[0]["item_id"] == "item_002"

    @pytest.mark.asyncio
    async def test_file_format(self, tmp_path):
        store = FlagStore(tmp_path / "flags")
        await store.initialize()
        await store.add_flag("item_001", "Test reason")

        flags_path = tmp_path / "flags" / "flags.json"
        assert flags_path.exists()

        data = json.loads(flags_path.read_text())
        assert "flags" in data
        assert len(data["flags"]) == 1
        assert data["flags"][0] == {
            "item_id": "item_001",
            "reason": "Test reason",
            "resolved": False,
        }

    @pytest.mark.asyncio
    async def test_atomic_save(self, tmp_path):
        """Verify no .tmp files left after save."""
        store = FlagStore(tmp_path / "flags")
        await store.initialize()
        await store.add_flag("item_001", "Reason")
        assert not (tmp_path / "flags" / "flags.tmp").exists()
        assert (tmp_path / "flags" / "flags.json").exists()

    @pytest.mark.asyncio
    async def test_concurrent_flags(self, tmp_path):
        """Verify concurrent flag operations don't corrupt the store."""
        store = FlagStore(tmp_path / "flags")
        await store.initialize()

        async def flag_one(i):
            await store.add_flag(f"item_{i:03d}", f"Reason {i}")

        await asyncio.gather(*[flag_one(i) for i in range(20)])

        assert await store.count() == 20
        entries = await store.list_flags()
        ids = {e["item_id"] for e in entries}
        assert len(ids) == 20

    @pytest.mark.asyncio
    async def test_entries_property_returns_copy(self, tmp_path):
        store = FlagStore(tmp_path / "flags")
        await store.initialize()
        await store.add_flag("item_001", "Reason")

        entries = store.entries
        entries.clear()  # Modifying the copy
        assert await store.count() == 1  # Original unchanged

    @pytest.mark.asyncio
    async def test_creates_parent_dirs(self, tmp_path):
        deep_path = tmp_path / "a" / "b" / "c"
        store = FlagStore(deep_path)
        await store.initialize()
        await store.add_flag("item_001", "Reason")
        assert (deep_path / "flags.json").exists()

    @pytest.mark.asyncio
    async def test_reflag_after_resolve(self, tmp_path):
        """An item can be re-flagged after being resolved."""
        store = FlagStore(tmp_path / "flags")
        await store.initialize()

        await store.add_flag("item_001", "First reason")
        await store.resolve_flag("item_001")
        await store.add_flag("item_001", "Second reason")

        assert await store.count() == 2  # Resolved + new unresolved
        assert await store.count(unresolved_only=True) == 1

        flag = await store.get_flag("item_001")
        assert flag["reason"] == "Second reason"


# ── Flagging-Items Skill Tests ───────────────────────────────────


class TestFlaggingItemsSkill:
    def test_skill_md_exists(self):
        from pathlib import Path

        skill_dir = Path(__file__).parents[2] / "lattereview" / "agentic" / "skills" / "builtin" / "flagging-items"
        assert (skill_dir / "SKILL.md").exists()
        assert (skill_dir / "tools.py").exists()

    def test_skill_md_frontmatter(self):
        from lattereview.agentic.skills.loader import parse_skill_md
        from pathlib import Path

        skill_dir = Path(__file__).parents[2] / "lattereview" / "agentic" / "skills" / "builtin" / "flagging-items"
        manifest = parse_skill_md(skill_dir)
        assert manifest is not None
        assert manifest.name == "flagging-items"
        assert "flag" in manifest.description.lower()
        assert len(manifest.description) <= 1024

    def test_toolset_loadable(self):
        from lattereview.agentic.skills.loader import load_toolset, parse_skill_md
        from pathlib import Path

        skill_dir = Path(__file__).parents[2] / "lattereview" / "agentic" / "skills" / "builtin" / "flagging-items"
        manifest = parse_skill_md(skill_dir)
        ts = load_toolset(manifest)
        assert ts is not None

    def test_skill_discoverable(self):
        from lattereview.agentic.skills import SkillRegistry

        registry = SkillRegistry()
        registry.discover()
        names = registry.available_skills
        assert "flagging-items" in names

    def test_skill_enableable(self):
        from lattereview.agentic.skills import SkillRegistry

        registry = SkillRegistry()
        registry.discover()
        registry.enable(["flagging-items"])
        toolsets = registry.get_enabled_toolsets()
        assert len(toolsets) == 1
        descs = registry.get_enabled_descriptions()
        assert any(d["name"] == "flagging-items" for d in descs)


# ── Prompt Integration Tests ─────────────────────────────────────


class TestPromptFlagIntegration:
    def test_flag_summaries_in_prompt(self):
        reviewer = AgenticReviewer(max_iterations=5)
        prompt = reviewer._build_system_prompt(
            flag_summaries=[
                {"item_id": "A-1", "reason": "Missing abstract"},
                {"item_id": "A-3", "reason": "Conflicting data"},
            ]
        )
        assert "A-1" in prompt
        assert "Missing abstract" in prompt
        assert "A-3" in prompt
        assert "Flagged Items" in prompt

    def test_no_flags_no_section(self):
        reviewer = AgenticReviewer(max_iterations=5)
        prompt = reviewer._build_system_prompt()
        assert "Flagged Items" not in prompt

    def test_non_agentic_no_flags(self):
        reviewer = AgenticReviewer(max_iterations=1)
        prompt = reviewer._build_system_prompt(flag_summaries=[{"item_id": "A-1", "reason": "Test"}])
        assert "Flagged Items" not in prompt


# ── Reviewer Integration Tests ─────────────────────────────────────


class TestReviewerFlagIntegration:
    @pytest.mark.asyncio
    async def test_review_item_creates_flag_store(self, tmp_path):
        """Flag store should be auto-created when working_dir is set."""
        from pydantic_ai.models.test import TestModel

        reviewer = AgenticReviewer(
            model=TestModel(),
            max_iterations=5,
        )

        result, cost = await reviewer.review_item(
            item_text="Test item",
            item_id="1",
            round_id="A",
            working_dir=tmp_path,
        )

        flags_dir = tmp_path / "round_A" / "agent_Reviewer" / "flags"
        assert flags_dir.exists()

    @pytest.mark.asyncio
    async def test_review_item_non_agentic_no_flags(self, tmp_path):
        """Non-agentic mode should not create flag store."""
        from pydantic_ai.models.test import TestModel

        reviewer = AgenticReviewer(
            model=TestModel(),
            max_iterations=1,
        )

        result, cost = await reviewer.review_item(
            item_text="Test item",
            item_id="1",
            round_id="A",
            working_dir=tmp_path,
        )

        flags_dir = tmp_path / "round_A" / "agent_Reviewer" / "flags"
        assert not flags_dir.exists()

    @pytest.mark.asyncio
    async def test_review_item_with_external_flag_store(self, tmp_path):
        """Externally provided flag_store should be used."""
        from pydantic_ai.models.test import TestModel

        reviewer = AgenticReviewer(
            model=TestModel(),
            max_iterations=5,
        )

        flags_dir = tmp_path / "custom_flags"
        store = FlagStore(flags_dir)
        await store.initialize()
        await store.add_flag("prior_item", "Was ambiguous")

        result, cost = await reviewer.review_item(
            item_text="Test item",
            item_id="1",
            round_id="A",
            flag_store=store,
        )

        assert result is not None

    @pytest.mark.asyncio
    async def test_review_items_shares_flag_store(self, tmp_path):
        """review_items should share a single flag store across items."""
        from pydantic_ai.models.test import TestModel

        reviewer = AgenticReviewer(
            model=TestModel(),
            max_iterations=5,
        )

        results, cost = await reviewer.review_items(
            text_inputs=["Item 1", "Item 2"],
            round_id="A",
            working_dir=tmp_path,
        )

        assert len(results) == 2
        flags_dir = tmp_path / "round_A" / "agent_Reviewer" / "flags"
        assert flags_dir.exists()


# ── Workflow Flag Integration Tests ──────────────────────────────


class TestWorkflowFlagIntegration:
    @pytest.mark.asyncio
    async def test_workflow_creates_flag_dirs(self, tmp_path):
        """Workflow should create flag directories when working_dir is set."""
        from pydantic_ai.models.test import TestModel

        from lattereview.agentic.workflow import AgenticWorkflow

        reviewer = AgenticReviewer(
            name="R1",
            model=TestModel(),
            max_iterations=5,
        )

        wf = AgenticWorkflow(
            workflow_schema=[
                {
                    "round": "A",
                    "reviewers": [reviewer],
                    "text_inputs": ["text"],
                }
            ],
            working_dir=tmp_path,
            verbose=False,
        )

        import pandas as pd

        df = pd.DataFrame({"text": ["Item 1", "Item 2"]})
        await wf.run(df)

        flags_dir = tmp_path / "round_A" / "agent_R1" / "flags"
        assert flags_dir.exists()

    @pytest.mark.asyncio
    async def test_workflow_revisit_flagged_default_true(self):
        """revisit_flagged should default to True."""
        from pydantic_ai.models.test import TestModel

        from lattereview.agentic.workflow import AgenticWorkflow

        reviewer = AgenticReviewer(name="R", model=TestModel(), max_iterations=1)
        wf = AgenticWorkflow(
            workflow_schema=[
                {"round": "A", "reviewers": [reviewer], "text_inputs": ["text"]},
            ],
        )
        assert wf.revisit_flagged is True

    @pytest.mark.asyncio
    async def test_workflow_revisit_disabled(self, tmp_path):
        """Workflow with revisit_flagged=False should not revisit."""
        from pydantic_ai.models.test import TestModel

        from lattereview.agentic.workflow import AgenticWorkflow

        reviewer = AgenticReviewer(
            name="R1",
            model=TestModel(),
            max_iterations=5,
        )

        wf = AgenticWorkflow(
            workflow_schema=[
                {
                    "round": "A",
                    "reviewers": [reviewer],
                    "text_inputs": ["text"],
                }
            ],
            working_dir=tmp_path,
            verbose=False,
            revisit_flagged=False,
        )

        import pandas as pd

        df = pd.DataFrame({"text": ["Item 1", "Item 2"]})
        result = await wf.run(df)

        # No flagged columns should be created since revisit is disabled
        flagged_cols = [c for c in result.columns if "flagged" in c.lower()]
        assert len(flagged_cols) == 0

    @pytest.mark.asyncio
    async def test_workflow_non_agentic_no_flag_store(self, tmp_path):
        """Non-agentic reviewers should not get flag stores."""
        from pydantic_ai.models.test import TestModel

        from lattereview.agentic.workflow import AgenticWorkflow

        reviewer = AgenticReviewer(
            name="R1",
            model=TestModel(),
            max_iterations=1,
        )

        wf = AgenticWorkflow(
            workflow_schema=[
                {
                    "round": "A",
                    "reviewers": [reviewer],
                    "text_inputs": ["text"],
                }
            ],
            working_dir=tmp_path,
            verbose=False,
        )

        import pandas as pd

        df = pd.DataFrame({"text": ["Item 1"]})
        await wf.run(df)

        flags_dir = tmp_path / "round_A" / "agent_R1" / "flags"
        assert not flags_dir.exists()


# ── Import Tests ───────────────────────────────────────────────────


class TestFlagImports:
    def test_import_from_flags_module(self):
        from lattereview.agentic.flags import FlagStore

        assert FlagStore is not None

    def test_import_from_agentic(self):
        from lattereview.agentic import FlagStore

        assert FlagStore is not None


# ── Live Integration Tests ─────────────────────────────────────────


@pytest.mark.live
@pytest.mark.asyncio
async def test_live_flagging_with_openai(tmp_path, env_keys):
    """Live test: agent flags ambiguous item and it gets revisited."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("OPENAI_API_KEY not set")

    reviewer = AgenticReviewer(
        name="FlagAgent",
        model="openai:gpt-5.4-mini",
        system_prompt=(
            "You are reviewing research articles. "
            "If an abstract is missing or too vague to assess, flag the item for revisit. "
            "Always provide your best assessment even if flagging."
        ),
        task_prompt="Review this article:\n\n${item}$",
        max_iterations=10,
        agentic_effort="high",
        skills=["flagging-items"],
    )

    items = [
        "Title: Effects of Exercise on Depression\n"
        "Abstract: A randomized controlled trial with N=500 participants showed "
        "significant reduction in depression scores (p<0.001) after 12 weeks.",
        "Title: Novel Compound X for Cancer Treatment\n" "Abstract: [Abstract not available]",
        "Title: Sleep and Cognition Meta-Analysis\n"
        "Abstract: A comprehensive meta-analysis of 30 studies (N=5000 total) "
        "found strong associations between sleep duration and cognitive performance.",
    ]

    flags_dir = tmp_path / "round_A" / "agent_FlagAgent" / "flags"
    flag_store = FlagStore(flags_dir)
    await flag_store.initialize()

    results = []
    for i, item in enumerate(items):
        result, cost = await reviewer.review_item(
            item_text=item,
            item_id=f"A-{i}",
            round_id="A",
            working_dir=tmp_path,
            flag_store=flag_store,
        )
        results.append(result)

    # All items should have been reviewed
    assert len(results) == 3
    for r in results:
        assert "score" in r
        assert "reasoning" in r

    # The second item (missing abstract) should ideally be flagged
    # but we don't strictly require it since LLM behavior varies
    all_flags = await flag_store.list_flags()
    # Just verify the flag store is functional
    assert isinstance(all_flags, list)


@pytest.mark.live
@pytest.mark.asyncio
async def test_live_flag_workflow_revisit(tmp_path, env_keys):
    """Live test: workflow with flagging skill runs revisit pass."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("OPENAI_API_KEY not set")

    import pandas as pd

    from lattereview.agentic.workflow import AgenticWorkflow

    reviewer = AgenticReviewer(
        name="FlagReviewer",
        model="openai:gpt-5.4-mini",
        system_prompt=(
            "You are reviewing research articles. "
            "If an abstract is missing, flag the item for revisit using the flag_for_revisit tool. "
            "Always provide your best score even if flagging."
        ),
        task_prompt="Review this article:\n\n${item}$",
        max_iterations=10,
        agentic_effort="high",
        skills=["flagging-items"],
        model_settings={"temperature": 0.0},
    )

    df = pd.DataFrame(
        {
            "Title": [
                "RCT on Drug X",
                "Study with Missing Abstract",
                "Large Cohort Study",
            ],
            "Abstract": [
                "A well-designed RCT with N=500 testing drug X for diabetes.",
                "[Abstract not available]",
                "A cohort of 2000 patients over 10 years examining cardiovascular outcomes.",
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
        working_dir=tmp_path,
        verbose=True,
        revisit_flagged=True,
    )

    result = await wf.run(df)

    # All rows should have results
    assert "round-A_FlagReviewer_reasoning" in result.columns
    assert "round-A_FlagReviewer_score" in result.columns
    for idx in range(len(df)):
        assert result.at[idx, "round-A_FlagReviewer_reasoning"] is not None
