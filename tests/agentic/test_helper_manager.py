"""Tests for helper agent system — HelperAgentManager + discussing-with-helpers skill."""

import pytest

from pydantic import BaseModel

from lattereview.agentic.reviewer import AgenticReviewer
from lattereview.agentic.helpers.manager import HelperAgentManager, _build_helper_system_prompt
from lattereview.agentic.output_models import ScoringOutput
from lattereview.agentic.deps import ReviewDeps

# ---------------------------------------------------------------------------
# HelperAgentManager — creation and properties
# ---------------------------------------------------------------------------


class TestHelperAgentManagerInit:
    def test_creation_with_helpers(self):
        from pydantic_ai.models.test import TestModel

        h1 = AgenticReviewer(name="Expert", model=TestModel(), max_iterations=5)
        h2 = AgenticReviewer(name="Specialist", model=TestModel(), max_iterations=5)
        manager = HelperAgentManager(helpers=[h1, h2], helper_max_iterations=3)

        assert manager.helper_count == 2
        assert manager.helper_max_iterations == 3
        assert manager.get_helper_names() == ["Expert", "Specialist"]

    def test_empty_helpers(self):
        manager = HelperAgentManager(helpers=[])
        assert manager.helper_count == 0
        assert manager.get_helper_names() == []

    def test_default_max_iterations(self):
        manager = HelperAgentManager(helpers=[])
        assert manager.helper_max_iterations == 5

    def test_helpers_property(self):
        from pydantic_ai.models.test import TestModel

        h = AgenticReviewer(name="H", model=TestModel(), max_iterations=1)
        manager = HelperAgentManager(helpers=[h])
        assert len(manager.helpers) == 1
        assert manager.helpers[0].name == "H"


# ---------------------------------------------------------------------------
# HelperAgentManager — discussion
# ---------------------------------------------------------------------------


class TestHelperDiscussion:
    @pytest.mark.asyncio
    async def test_discuss_with_test_model(self):
        """Helper responds using TestModel (returns plain string)."""
        from pydantic_ai.models.test import TestModel

        helper = AgenticReviewer(
            name="TestHelper",
            model=TestModel(),
            max_iterations=1,
            backstory="You are a biomedical expert.",
        )
        manager = HelperAgentManager(helpers=[helper], helper_max_iterations=3)

        response, helper_name, cost = await manager.discuss(
            question="Is this study design appropriate?",
            item_text="A randomized controlled trial of 100 patients.",
            item_id="test_001",
        )

        assert isinstance(response, str)
        assert helper_name == "TestHelper"
        assert isinstance(cost, float)

    @pytest.mark.asyncio
    async def test_discuss_with_specific_helper(self):
        """Targeting a specific helper by index."""
        from pydantic_ai.models.test import TestModel

        h1 = AgenticReviewer(name="Primary", model=TestModel(), max_iterations=1)
        h2 = AgenticReviewer(name="Secondary", model=TestModel(), max_iterations=1)
        manager = HelperAgentManager(helpers=[h1, h2])

        _, name1, _ = await manager.discuss(question="Q?", item_text="Item", helper_index=0)
        assert name1 == "Primary"

        _, name2, _ = await manager.discuss(question="Q?", item_text="Item", helper_index=1)
        assert name2 == "Secondary"

    @pytest.mark.asyncio
    async def test_discuss_invalid_index_raises(self):
        """Out-of-range helper index raises IndexError."""
        from pydantic_ai.models.test import TestModel

        h = AgenticReviewer(name="H", model=TestModel(), max_iterations=1)
        manager = HelperAgentManager(helpers=[h])

        with pytest.raises(IndexError, match="out of range"):
            await manager.discuss(question="Q?", item_text="Item", helper_index=5)

    @pytest.mark.asyncio
    async def test_discuss_negative_index_raises(self):
        """Negative helper index raises IndexError."""
        from pydantic_ai.models.test import TestModel

        h = AgenticReviewer(name="H", model=TestModel(), max_iterations=1)
        manager = HelperAgentManager(helpers=[h])

        with pytest.raises(IndexError, match="out of range"):
            await manager.discuss(question="Q?", item_text="Item", helper_index=-1)

    @pytest.mark.asyncio
    async def test_discuss_with_working_dir(self, tmp_path):
        """Helper creates memory dir when working_dir is set."""
        from pydantic_ai.models.test import TestModel

        helper = AgenticReviewer(name="MemHelper", model=TestModel(), max_iterations=1)
        manager = HelperAgentManager(helpers=[helper])

        response, name, cost = await manager.discuss(
            question="What do you think?",
            item_text="Some item text",
            item_id="item_1",
            round_id="A",
            working_dir=tmp_path,
        )

        assert isinstance(response, str)
        # Memory dir should have been created
        memory_dir = tmp_path / "round_A" / "helper_MemHelper" / "memory"
        assert memory_dir.exists()


# ---------------------------------------------------------------------------
# Helper system prompt builder
# ---------------------------------------------------------------------------


class TestBuildHelperSystemPrompt:
    def test_basic_prompt(self):
        prompt = _build_helper_system_prompt(
            helper_name="Expert",
            helper_backstory="You specialize in epidemiology.",
            helper_system_prompt="",
        )
        assert "Expert" in prompt
        assert "epidemiology" in prompt
        assert "being consulted" in prompt

    def test_prompt_with_instructions(self):
        prompt = _build_helper_system_prompt(
            helper_name="Expert",
            helper_backstory="",
            helper_system_prompt="Focus on methodology assessment.",
        )
        assert "methodology assessment" in prompt

    def test_prompt_with_memories(self):
        prompt = _build_helper_system_prompt(
            helper_name="Expert",
            helper_backstory="",
            helper_system_prompt="",
            memory_summaries=[
                {"id": "mem_001", "brief": "Small sample sizes are common"},
            ],
        )
        assert "mem_001" in prompt
        assert "Small sample sizes" in prompt

    def test_prompt_with_skill_descriptions(self):
        prompt = _build_helper_system_prompt(
            helper_name="Expert",
            helper_backstory="",
            helper_system_prompt="",
            skill_descriptions=[
                {"name": "searching-pubmed", "description": "Search PubMed"},
            ],
        )
        assert "searching-pubmed" in prompt
        assert "Search PubMed" in prompt

    def test_prompt_no_extras(self):
        prompt = _build_helper_system_prompt(
            helper_name="Bot",
            helper_backstory="",
            helper_system_prompt="",
        )
        assert "Bot" in prompt
        assert "Your Memories" not in prompt
        assert "Available Skills" not in prompt


# ---------------------------------------------------------------------------
# Recursion safety — helpers must NOT get discussing-with-helpers skill
# ---------------------------------------------------------------------------


class TestRecursionSafety:
    @pytest.mark.asyncio
    async def test_helper_skills_exclude_discussion(self):
        """When a helper has skills including discussing-with-helpers, it must
        be excluded from the helper's skill list during consultation."""
        from pydantic_ai.models.test import TestModel

        helper = AgenticReviewer(
            name="SkillHelper",
            model=TestModel(),
            max_iterations=5,
            skills=["searching-content", "discussing-with-helpers"],
        )
        manager = HelperAgentManager(helpers=[helper])

        # This should NOT raise — the discussing-with-helpers skill should be
        # silently excluded during helper consultation
        response, name, cost = await manager.discuss(
            question="Is this sample size adequate?",
            item_text="A study with N=10.",
        )
        assert isinstance(response, str)

    def test_non_agentic_helper_gets_no_skills(self):
        """Non-agentic helpers (max_iterations=1) never get skills."""
        from pydantic_ai.models.test import TestModel

        helper = AgenticReviewer(
            name="Simple",
            model=TestModel(),
            max_iterations=1,
            skills=["searching-content"],
        )
        assert not helper.is_agentic


# ---------------------------------------------------------------------------
# Integration — AgenticReviewer creates HelperAgentManager
# ---------------------------------------------------------------------------


class TestReviewerHelperIntegration:
    def test_reviewer_with_helpers_config(self):
        """AgenticReviewer accepts helpers list."""
        from pydantic_ai.models.test import TestModel

        helper = AgenticReviewer(name="H", model=TestModel(), max_iterations=1)
        reviewer = AgenticReviewer(
            model=TestModel(),
            max_iterations=10,
            helpers=[helper],
            helper_max_iterations=3,
        )
        assert len(reviewer.helpers) == 1
        assert reviewer.helper_max_iterations == 3

    def test_reviewer_no_helpers_default(self):
        """Reviewer defaults to empty helpers list."""
        reviewer = AgenticReviewer()
        assert reviewer.helpers == []
        assert reviewer.helper_max_iterations == 5

    @pytest.mark.asyncio
    async def test_review_item_creates_helper_manager(self, tmp_path):
        """review_item creates HelperAgentManager when helpers are configured."""
        from pydantic_ai.models.test import TestModel

        helper = AgenticReviewer(
            name="TestHelper",
            model=TestModel(),
            max_iterations=1,
        )
        reviewer = AgenticReviewer(
            model=TestModel(),
            max_iterations=5,
            helpers=[helper],
        )

        # review_item should work without error even with helpers configured
        response, cost = await reviewer.review_item(
            item_text="A study on machine learning.",
            item_id="test_001",
            working_dir=tmp_path,
        )
        assert "reasoning" in response

    @pytest.mark.asyncio
    async def test_non_agentic_reviewer_ignores_helpers(self):
        """Non-agentic reviewer (max_iterations=1) ignores helpers."""
        from pydantic_ai.models.test import TestModel

        helper = AgenticReviewer(name="H", model=TestModel(), max_iterations=1)
        reviewer = AgenticReviewer(
            model=TestModel(),
            max_iterations=1,
            helpers=[helper],
        )

        response, cost = await reviewer.review_item(item_text="Test")
        assert "reasoning" in response


# ---------------------------------------------------------------------------
# Skill discovery and registration
# ---------------------------------------------------------------------------


class TestDiscussingWithHelpersSkill:
    def test_skill_md_exists(self):
        from pathlib import Path

        skill_dir = (
            Path(__file__).resolve().parents[2]
            / "lattereview"
            / "agentic"
            / "skills"
            / "builtin"
            / "discussing-with-helpers"
        )
        assert (skill_dir / "SKILL.md").exists()
        assert (skill_dir / "tools.py").exists()

    def test_skill_frontmatter(self):
        from lattereview.agentic.skills.loader import parse_skill_md
        from pathlib import Path

        skill_dir = (
            Path(__file__).resolve().parents[2]
            / "lattereview"
            / "agentic"
            / "skills"
            / "builtin"
            / "discussing-with-helpers"
        )
        manifest = parse_skill_md(skill_dir)
        assert manifest.name == "discussing-with-helpers"
        assert "helper" in manifest.description.lower()

    def test_skill_discoverable(self):
        from lattereview.agentic.skills import SkillRegistry

        registry = SkillRegistry()
        names = registry.discover()
        assert "discussing-with-helpers" in names

    def test_skill_enableable(self):
        from lattereview.agentic.skills import SkillRegistry

        registry = SkillRegistry()
        registry.discover()
        enabled = registry.enable(["discussing-with-helpers"])
        assert "discussing-with-helpers" in enabled

    def test_skill_toolset_has_discuss_tool(self):
        from lattereview.agentic.skills import SkillRegistry

        registry = SkillRegistry()
        registry.discover()
        registry.enable(["discussing-with-helpers"])
        toolsets = registry.get_enabled_toolsets()
        assert len(toolsets) == 1


# ---------------------------------------------------------------------------
# Tool function — discuss_with_helper
# ---------------------------------------------------------------------------


class TestDiscussWithHelperTool:
    @pytest.mark.asyncio
    async def test_reviewer_with_discussion_skill_and_helpers(self, tmp_path):
        """Reviewer with both discussion skill and helpers can review items."""
        from pydantic_ai.models.test import TestModel

        helper = AgenticReviewer(
            name="Expert",
            model=TestModel(),
            max_iterations=1,
        )
        reviewer = AgenticReviewer(
            model=TestModel(),
            max_iterations=5,
            helpers=[helper],
            skills=["discussing-with-helpers"],
        )

        # Should work — the discussion skill is wired to the helper manager
        response, cost = await reviewer.review_item(
            item_text="A clinical trial on a new drug.",
            working_dir=tmp_path,
        )
        assert "reasoning" in response


# ---------------------------------------------------------------------------
# ReviewDeps — helper_manager field
# ---------------------------------------------------------------------------


class TestReviewDepsHelperManager:
    def test_deps_default_none(self):
        deps = ReviewDeps(
            item_id="1",
            item_text="text",
            agent_name="R",
            round_id="A",
            max_iterations=5,
        )
        assert deps.helper_manager is None

    def test_deps_with_helper_manager(self):
        from pydantic_ai.models.test import TestModel

        helper = AgenticReviewer(name="H", model=TestModel(), max_iterations=1)
        manager = HelperAgentManager(helpers=[helper])

        deps = ReviewDeps(
            item_id="1",
            item_text="text",
            agent_name="R",
            round_id="A",
            max_iterations=5,
            helper_manager=manager,
        )
        assert deps.helper_manager is not None
        assert deps.helper_manager.helper_count == 1


# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------


class TestHelperImports:
    def test_import_from_helpers(self):
        from lattereview.agentic.helpers import HelperAgentManager

        assert HelperAgentManager is not None

    def test_import_from_agentic(self):
        from lattereview.agentic import HelperAgentManager

        assert HelperAgentManager is not None


# ---------------------------------------------------------------------------
# Live integration tests — require API keys in .env
# ---------------------------------------------------------------------------


class TestLiveHelperDiscussion:
    """Live integration tests — primary agent consults helper with different model."""

    @pytest.mark.live
    @pytest.mark.asyncio
    async def test_live_helper_discussion_openai(self, env_keys):
        """Primary (OpenAI) consults helper (OpenAI) for expert opinion."""
        import os

        if not os.environ.get("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set")

        helper = AgenticReviewer(
            name="MethodsExpert",
            model="openai:gpt-5.4-mini",
            max_iterations=1,
            backstory="You are an expert in research methodology and study design.",
            model_settings={"temperature": 0.0},
        )
        manager = HelperAgentManager(helpers=[helper], helper_max_iterations=3)

        response, name, cost = await manager.discuss(
            question="Is a sample size of 15 patients adequate for a randomized controlled trial on hypertension treatment?",
            item_text="A randomized controlled trial with 15 patients examining a novel antihypertensive agent over 6 months.",
            item_id="live_001",
        )

        assert isinstance(response, str)
        assert len(response) > 10  # Meaningful response
        assert name == "MethodsExpert"

    @pytest.mark.live
    @pytest.mark.asyncio
    async def test_live_reviewer_with_helper_cross_model(self, env_keys, tmp_path):
        """Primary reviewer (OpenAI) with helper (Anthropic) — cross-model discussion."""
        import os

        if not os.environ.get("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set")
        if not os.environ.get("ANTHROPIC_API_KEY"):
            pytest.skip("ANTHROPIC_API_KEY not set")

        helper = AgenticReviewer(
            name="AnthropicExpert",
            model="anthropic:claude-haiku-4-5-20251001",
            max_iterations=1,
            backstory="You are a biostatistics expert.",
            model_settings={"temperature": 0.0},
        )
        reviewer = AgenticReviewer(
            name="PrimaryReviewer",
            model="openai:gpt-5.4-mini",
            max_iterations=5,
            agentic_effort="high",
            helpers=[helper],
            helper_max_iterations=3,
            skills=["discussing-with-helpers"],
            system_prompt=(
                "You are reviewing clinical trial abstracts. " "Consult your helper agent for methodology questions."
            ),
            output_type=ScoringOutput,
            model_settings={"temperature": 0.0},
        )

        response, cost = await reviewer.review_item(
            item_text=(
                "A single-arm pilot study with 8 participants evaluated a new "
                "cognitive behavioral therapy protocol for treatment-resistant "
                "depression over 12 weeks. Results showed a 40% reduction in "
                "PHQ-9 scores (p=0.03)."
            ),
            item_id="live_002",
            working_dir=tmp_path,
        )

        assert isinstance(response["reasoning"], str)
        assert isinstance(response["score"], int)
