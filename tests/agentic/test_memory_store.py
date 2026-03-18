"""Tests for memory system — MemoryIndex, MemoryStore, and managing-memory skill."""

import asyncio
import json
import os

import pytest

from lattereview.agentic.memory.index import MemoryIndex
from lattereview.agentic.memory.store import MemoryStore

# ── MemoryIndex Tests ──────────────────────────────────────────────


class TestMemoryIndex:
    @pytest.mark.asyncio
    async def test_empty_index(self, tmp_path):
        idx = MemoryIndex(tmp_path / "memory" / "_index.json")
        await idx.load()
        assert len(idx) == 0
        assert idx.entries == []

    @pytest.mark.asyncio
    async def test_add_entry(self, tmp_path):
        idx = MemoryIndex(tmp_path / "_index.json")
        await idx.load()
        await idx.add("mem_001", "Test Title", "Test brief")
        assert len(idx) == 1
        assert idx.entries[0]["id"] == "mem_001"
        assert idx.entries[0]["title"] == "Test Title"
        assert idx.entries[0]["brief"] == "Test brief"

    @pytest.mark.asyncio
    async def test_add_persists_to_disk(self, tmp_path):
        idx = MemoryIndex(tmp_path / "_index.json")
        await idx.load()
        await idx.add("mem_001", "Title", "Brief")

        # Load fresh instance
        idx2 = MemoryIndex(tmp_path / "_index.json")
        await idx2.load()
        assert len(idx2) == 1
        assert idx2.entries[0]["id"] == "mem_001"

    @pytest.mark.asyncio
    async def test_add_deduplicates(self, tmp_path):
        idx = MemoryIndex(tmp_path / "_index.json")
        await idx.load()
        await idx.add("mem_001", "Title 1", "Brief 1")
        await idx.add("mem_001", "Title 2", "Brief 2")
        assert len(idx) == 1
        assert idx.entries[0]["title"] == "Title 2"

    @pytest.mark.asyncio
    async def test_remove_entry(self, tmp_path):
        idx = MemoryIndex(tmp_path / "_index.json")
        await idx.load()
        await idx.add("mem_001", "Title", "Brief")
        removed = await idx.remove("mem_001")
        assert removed is True
        assert len(idx) == 0

    @pytest.mark.asyncio
    async def test_remove_nonexistent(self, tmp_path):
        idx = MemoryIndex(tmp_path / "_index.json")
        await idx.load()
        removed = await idx.remove("mem_999")
        assert removed is False

    @pytest.mark.asyncio
    async def test_get_summaries(self, tmp_path):
        idx = MemoryIndex(tmp_path / "_index.json")
        await idx.load()
        await idx.add("mem_001", "Title 1", "Brief 1")
        await idx.add("mem_002", "Title 2", "Brief 2")
        summaries = await idx.get_summaries()
        assert len(summaries) == 2
        assert summaries[0] == {"id": "mem_001", "brief": "Brief 1"}
        assert summaries[1] == {"id": "mem_002", "brief": "Brief 2"}

    @pytest.mark.asyncio
    async def test_get_next_id_empty(self, tmp_path):
        idx = MemoryIndex(tmp_path / "_index.json")
        await idx.load()
        assert idx.get_next_id() == "mem_001"

    @pytest.mark.asyncio
    async def test_get_next_id_sequential(self, tmp_path):
        idx = MemoryIndex(tmp_path / "_index.json")
        await idx.load()
        await idx.add("mem_001", "T1", "B1")
        assert idx.get_next_id() == "mem_002"
        await idx.add("mem_002", "T2", "B2")
        assert idx.get_next_id() == "mem_003"

    @pytest.mark.asyncio
    async def test_get_next_id_with_gap(self, tmp_path):
        idx = MemoryIndex(tmp_path / "_index.json")
        await idx.load()
        await idx.add("mem_001", "T1", "B1")
        await idx.add("mem_005", "T5", "B5")
        # Next ID is after highest, not gap-filling
        assert idx.get_next_id() == "mem_006"

    @pytest.mark.asyncio
    async def test_atomic_save(self, tmp_path):
        """Verify no .tmp files left after save."""
        idx = MemoryIndex(tmp_path / "_index.json")
        await idx.load()
        await idx.add("mem_001", "Title", "Brief")
        assert not (tmp_path / "_index.tmp").exists()
        assert (tmp_path / "_index.json").exists()

    @pytest.mark.asyncio
    async def test_creates_parent_dirs(self, tmp_path):
        deep_path = tmp_path / "a" / "b" / "c" / "_index.json"
        idx = MemoryIndex(deep_path)
        await idx.load()
        await idx.add("mem_001", "Title", "Brief")
        assert deep_path.exists()

    @pytest.mark.asyncio
    async def test_multiple_entries(self, tmp_path):
        idx = MemoryIndex(tmp_path / "_index.json")
        await idx.load()
        for i in range(10):
            await idx.add(f"mem_{i+1:03d}", f"Title {i+1}", f"Brief {i+1}")
        assert len(idx) == 10


# ── MemoryStore Tests ──────────────────────────────────────────────


class TestMemoryStore:
    @pytest.mark.asyncio
    async def test_initialize_creates_dir(self, tmp_path):
        store = MemoryStore(tmp_path / "memory")
        await store.initialize()
        assert store.memory_dir.exists()

    @pytest.mark.asyncio
    async def test_save_and_load(self, tmp_path):
        store = MemoryStore(tmp_path / "memory")
        await store.initialize()

        mem_id = await store.save(
            title="Test Memory",
            brief="A test memory for validation",
            content="This is the full content of the memory.",
        )

        assert mem_id == "mem_001"
        content = await store.load("mem_001")
        assert content == "This is the full content of the memory."

    @pytest.mark.asyncio
    async def test_save_multiple(self, tmp_path):
        store = MemoryStore(tmp_path / "memory")
        await store.initialize()

        id1 = await store.save("T1", "B1", "Content 1")
        id2 = await store.save("T2", "B2", "Content 2")
        id3 = await store.save("T3", "B3", "Content 3")

        assert id1 == "mem_001"
        assert id2 == "mem_002"
        assert id3 == "mem_003"

    @pytest.mark.asyncio
    async def test_load_nonexistent(self, tmp_path):
        store = MemoryStore(tmp_path / "memory")
        await store.initialize()
        content = await store.load("mem_999")
        assert content is None

    @pytest.mark.asyncio
    async def test_load_multiple(self, tmp_path):
        store = MemoryStore(tmp_path / "memory")
        await store.initialize()

        await store.save("T1", "B1", "Content 1")
        await store.save("T2", "B2", "Content 2")

        results = await store.load_multiple(["mem_001", "mem_002", "mem_999"])
        assert results["mem_001"] == "Content 1"
        assert results["mem_002"] == "Content 2"
        assert results["mem_999"] is None

    @pytest.mark.asyncio
    async def test_delete(self, tmp_path):
        store = MemoryStore(tmp_path / "memory")
        await store.initialize()

        await store.save("T1", "B1", "Content 1")
        assert await store.count() == 1

        deleted = await store.delete("mem_001")
        assert deleted is True
        assert await store.count() == 0
        assert await store.load("mem_001") is None

    @pytest.mark.asyncio
    async def test_delete_nonexistent(self, tmp_path):
        store = MemoryStore(tmp_path / "memory")
        await store.initialize()
        deleted = await store.delete("mem_999")
        assert deleted is False

    @pytest.mark.asyncio
    async def test_delete_removes_file(self, tmp_path):
        store = MemoryStore(tmp_path / "memory")
        await store.initialize()
        await store.save("T1", "B1", "Content 1")

        mem_path = tmp_path / "memory" / "mem_001.md"
        assert mem_path.exists()
        await store.delete("mem_001")
        assert not mem_path.exists()

    @pytest.mark.asyncio
    async def test_list_memories(self, tmp_path):
        store = MemoryStore(tmp_path / "memory")
        await store.initialize()

        await store.save("Title 1", "Brief 1", "Content 1")
        await store.save("Title 2", "Brief 2", "Content 2")

        entries = await store.list_memories()
        assert len(entries) == 2
        assert entries[0]["id"] == "mem_001"
        assert entries[0]["title"] == "Title 1"
        assert entries[0]["brief"] == "Brief 1"

    @pytest.mark.asyncio
    async def test_get_summaries(self, tmp_path):
        store = MemoryStore(tmp_path / "memory")
        await store.initialize()

        await store.save("Title 1", "Brief 1", "Content 1")
        summaries = await store.get_summaries()
        assert len(summaries) == 1
        assert summaries[0] == {"id": "mem_001", "brief": "Brief 1"}

    @pytest.mark.asyncio
    async def test_max_memories_enforced(self, tmp_path):
        store = MemoryStore(tmp_path / "memory", max_memories=3)
        await store.initialize()

        await store.save("T1", "B1", "C1")
        await store.save("T2", "B2", "C2")
        await store.save("T3", "B3", "C3")

        result = await store.save("T4", "B4", "C4")
        assert result.startswith("ERROR:")
        assert await store.count() == 3

    @pytest.mark.asyncio
    async def test_max_memories_after_delete(self, tmp_path):
        store = MemoryStore(tmp_path / "memory", max_memories=2)
        await store.initialize()

        await store.save("T1", "B1", "C1")
        await store.save("T2", "B2", "C2")
        result = await store.save("T3", "B3", "C3")
        assert result.startswith("ERROR:")

        await store.delete("mem_001")
        result = await store.save("T3", "B3", "C3")
        assert result == "mem_003"

    @pytest.mark.asyncio
    async def test_count(self, tmp_path):
        store = MemoryStore(tmp_path / "memory")
        await store.initialize()
        assert await store.count() == 0
        await store.save("T1", "B1", "C1")
        assert await store.count() == 1

    @pytest.mark.asyncio
    async def test_persistence_across_instances(self, tmp_path):
        mem_dir = tmp_path / "memory"

        store1 = MemoryStore(mem_dir)
        await store1.initialize()
        await store1.save("Title", "Brief", "Content")

        store2 = MemoryStore(mem_dir)
        await store2.initialize()
        assert await store2.count() == 1
        content = await store2.load("mem_001")
        assert content == "Content"

    @pytest.mark.asyncio
    async def test_memory_file_format(self, tmp_path):
        store = MemoryStore(tmp_path / "memory")
        await store.initialize()
        await store.save("T", "B", "# Heading\n\nParagraph with **bold**.")

        mem_path = tmp_path / "memory" / "mem_001.md"
        assert mem_path.exists()
        assert mem_path.read_text() == "# Heading\n\nParagraph with **bold**."

    @pytest.mark.asyncio
    async def test_index_file_format(self, tmp_path):
        store = MemoryStore(tmp_path / "memory")
        await store.initialize()
        await store.save("My Title", "My Brief", "Content")

        index_path = tmp_path / "memory" / "_index.json"
        data = json.loads(index_path.read_text())
        assert "memories" in data
        assert len(data["memories"]) == 1
        assert data["memories"][0] == {
            "id": "mem_001",
            "title": "My Title",
            "brief": "My Brief",
        }

    @pytest.mark.asyncio
    async def test_concurrent_saves(self, tmp_path):
        """Verify concurrent saves don't corrupt the index."""
        store = MemoryStore(tmp_path / "memory", max_memories=50)
        await store.initialize()

        async def save_one(i):
            await store.save(f"Title {i}", f"Brief {i}", f"Content {i}")

        await asyncio.gather(*[save_one(i) for i in range(20)])

        assert await store.count() == 20
        entries = await store.list_memories()
        ids = {e["id"] for e in entries}
        assert len(ids) == 20

    @pytest.mark.asyncio
    async def test_default_max_memories(self, tmp_path):
        store = MemoryStore(tmp_path / "memory")
        assert store.max_memories == 50


# ── Managing-Memory Skill Tests ───────────────────────────────────


class TestManagingMemorySkill:
    def test_skill_md_exists(self):
        from pathlib import Path

        skill_dir = Path(__file__).parents[2] / "lattereview" / "agentic" / "skills" / "builtin" / "managing-memory"
        assert (skill_dir / "SKILL.md").exists()
        assert (skill_dir / "tools.py").exists()

    def test_skill_md_frontmatter(self):
        from lattereview.agentic.skills.loader import parse_skill_md
        from pathlib import Path

        skill_dir = Path(__file__).parents[2] / "lattereview" / "agentic" / "skills" / "builtin" / "managing-memory"
        manifest = parse_skill_md(skill_dir)
        assert manifest is not None
        assert manifest.name == "managing-memory"
        assert "memories" in manifest.description.lower() or "memory" in manifest.description.lower()
        assert len(manifest.description) <= 1024

    def test_toolset_loadable(self):
        from lattereview.agentic.skills.loader import load_toolset, parse_skill_md
        from pathlib import Path

        skill_dir = Path(__file__).parents[2] / "lattereview" / "agentic" / "skills" / "builtin" / "managing-memory"
        manifest = parse_skill_md(skill_dir)
        ts = load_toolset(manifest)
        assert ts is not None

    def test_skill_discoverable(self):
        from lattereview.agentic.skills import SkillRegistry

        registry = SkillRegistry()
        registry.discover()
        names = registry.available_skills
        assert "managing-memory" in names

    def test_skill_enableable(self):
        from lattereview.agentic.skills import SkillRegistry

        registry = SkillRegistry()
        registry.discover()
        registry.enable(["managing-memory"])
        toolsets = registry.get_enabled_toolsets()
        assert len(toolsets) == 1
        descs = registry.get_enabled_descriptions()
        assert any(d["name"] == "managing-memory" for d in descs)


# ── Reviewer Integration Tests ─────────────────────────────────────


class TestReviewerMemoryIntegration:
    @pytest.mark.asyncio
    async def test_review_item_creates_memory_store(self, tmp_path):
        """Memory store should be auto-created when working_dir is set."""
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

        # Memory directory should have been created
        mem_dir = tmp_path / "round_A" / "agent_Reviewer" / "memory"
        assert mem_dir.exists()

    @pytest.mark.asyncio
    async def test_review_item_non_agentic_no_memory(self, tmp_path):
        """Non-agentic mode should not create memory store."""
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

        # No memory directory in non-agentic mode
        mem_dir = tmp_path / "round_A" / "agent_Reviewer" / "memory"
        assert not mem_dir.exists()

    @pytest.mark.asyncio
    async def test_review_item_with_external_memory_store(self, tmp_path):
        """Externally provided memory_store should be used."""
        from pydantic_ai.models.test import TestModel

        reviewer = AgenticReviewer(
            model=TestModel(),
            max_iterations=5,
        )

        mem_dir = tmp_path / "custom_memory"
        store = MemoryStore(mem_dir)
        await store.initialize()
        await store.save("Prior Insight", "Something useful", "Full content here")

        result, cost = await reviewer.review_item(
            item_text="Test item",
            item_id="1",
            round_id="A",
            memory_store=store,
        )

        assert result is not None

    @pytest.mark.asyncio
    async def test_memory_summaries_in_prompt(self, tmp_path):
        """Memory summaries should be included in the system prompt."""
        from lattereview.agentic.reviewer import AgenticReviewer

        reviewer = AgenticReviewer(max_iterations=5)

        prompt = reviewer._build_system_prompt(
            memory_summaries=[
                {"id": "mem_001", "brief": "Test pattern found"},
                {"id": "mem_002", "brief": "Another insight"},
            ]
        )

        assert "mem_001" in prompt
        assert "Test pattern found" in prompt
        assert "mem_002" in prompt
        assert "Your Memories" in prompt

    @pytest.mark.asyncio
    async def test_review_items_shares_memory_store(self, tmp_path):
        """review_items should share a single memory store across items."""
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
        # Only one memory directory should exist
        mem_dir = tmp_path / "round_A" / "agent_Reviewer" / "memory"
        assert mem_dir.exists()


# ── Workflow Memory Integration Tests ──────────────────────────────


class TestWorkflowMemoryIntegration:
    @pytest.mark.asyncio
    async def test_workflow_creates_memory_dirs(self, tmp_path):
        """Workflow should create memory directories when working_dir is set."""
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

        mem_dir = tmp_path / "round_A" / "agent_R1" / "memory"
        assert mem_dir.exists()


# ── Import Tests ───────────────────────────────────────────────────


class TestMemoryImports:
    def test_import_from_memory_module(self):
        from lattereview.agentic.memory import MemoryStore, MemoryIndex

        assert MemoryStore is not None
        assert MemoryIndex is not None

    def test_import_from_agentic(self):
        from lattereview.agentic import MemoryStore, MemoryIndex

        assert MemoryStore is not None
        assert MemoryIndex is not None


# ── Live Integration Tests ─────────────────────────────────────────


@pytest.mark.live
@pytest.mark.asyncio
async def test_live_memory_with_openai(tmp_path, env_keys):
    """Live test: agent uses memory tools during multi-item review."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("OPENAI_API_KEY not set")

    reviewer = AgenticReviewer(
        name="MemoryAgent",
        model="openai:gpt-5.4-mini",
        system_prompt=(
            "You are reviewing research articles. "
            "Save useful patterns you notice as memories for future reference. "
            "Check your memories before each review."
        ),
        task_prompt="Review this article excerpt:\n\n${item}$",
        max_iterations=10,
        agentic_effort="high",
        skills=["managing-memory"],
    )

    items = [
        "Title: Effects of Exercise on Depression\nAbstract: A randomized controlled trial with N=500 participants showed significant reduction in depression scores (p<0.001) after 12 weeks of moderate exercise.",
        "Title: Meditation and Anxiety\nAbstract: A small pilot study (N=15) found trends toward reduced anxiety (p=0.08) in meditators vs controls over 4 weeks.",
        "Title: Sleep Quality and Cognitive Performance\nAbstract: A large cohort study (N=2000) demonstrated strong associations between sleep duration and cognitive test scores (r=0.45, p<0.0001).",
    ]

    results = []
    mem_dir = tmp_path / "round_A" / "agent_MemoryAgent" / "memory"
    store = MemoryStore(mem_dir)
    await store.initialize()

    for i, item in enumerate(items):
        result, cost = await reviewer.review_item(
            item_text=item,
            item_id=str(i),
            round_id="A",
            working_dir=tmp_path,
            memory_store=store,
        )
        results.append(result)

    # All items should have been reviewed
    assert len(results) == 3
    for r in results:
        assert "score" in r
        assert "reasoning" in r


@pytest.mark.live
@pytest.mark.asyncio
async def test_live_memory_persistence(tmp_path, env_keys):
    """Live test: memories persist to disk and can be loaded by new store."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("OPENAI_API_KEY not set")

    # Create a store and manually save some memories
    mem_dir = tmp_path / "memory"
    store = MemoryStore(mem_dir)
    await store.initialize()
    await store.save(
        "Sample size matters",
        "Studies with N<50 are underpowered",
        "Small sample sizes (N<50) should be flagged as potentially underpowered. "
        "Look for confidence intervals and effect sizes as compensating factors.",
    )
    await store.save(
        "RCT gold standard",
        "Randomized controlled trials provide strongest evidence",
        "RCTs with proper blinding and adequate sample sizes are the gold standard. "
        "Observational studies should note this limitation.",
    )

    # Load from a new instance
    store2 = MemoryStore(mem_dir)
    await store2.initialize()
    assert await store2.count() == 2

    summaries = await store2.get_summaries()
    assert len(summaries) == 2

    content = await store2.load("mem_001")
    assert "underpowered" in content


# Need the import at module level for the test classes
from lattereview.agentic.reviewer import AgenticReviewer
