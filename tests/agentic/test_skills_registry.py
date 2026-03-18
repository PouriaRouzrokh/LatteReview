"""Tests for the skills system — discovery, registry, loading, and integration."""

import os
import textwrap
import tempfile

import pytest
from pydantic_ai.models.test import TestModel

from lattereview.agentic.skills.base import SkillManifest, validate_skill_name
from lattereview.agentic.skills.loader import parse_skill_md, discover_skills, load_toolset, BUILTIN_SKILLS_DIR
from lattereview.agentic.skills.registry import SkillRegistry
from lattereview.agentic.reviewer import AgenticReviewer
from lattereview.agentic.output_models import ScoringOutput

# ---------------------------------------------------------------------------
# SkillManifest & Validation
# ---------------------------------------------------------------------------


class TestSkillNameValidation:
    def test_valid_names(self):
        for name in ["searching-content", "managing-memory", "my-skill-1", "a", "x" * 64]:
            validate_skill_name(name)

    def test_invalid_uppercase(self):
        with pytest.raises(ValueError, match="Invalid skill name"):
            validate_skill_name("MySkill")

    def test_invalid_underscore(self):
        with pytest.raises(ValueError, match="Invalid skill name"):
            validate_skill_name("my_skill")

    def test_invalid_too_long(self):
        with pytest.raises(ValueError, match="Invalid skill name"):
            validate_skill_name("a" * 65)

    def test_invalid_reserved_word_anthropic(self):
        with pytest.raises(ValueError, match="reserved word"):
            validate_skill_name("anthropic-search")

    def test_invalid_reserved_word_claude(self):
        with pytest.raises(ValueError, match="reserved word"):
            validate_skill_name("claude-helper")

    def test_invalid_starts_with_hyphen(self):
        with pytest.raises(ValueError, match="Invalid skill name"):
            validate_skill_name("-bad-name")

    def test_empty_string(self):
        with pytest.raises(ValueError, match="Invalid skill name"):
            validate_skill_name("")


class TestSkillManifest:
    def test_valid_manifest(self):
        from pathlib import Path

        m = SkillManifest(
            name="searching-content",
            description="Searches within text for patterns.",
            path=Path("/tmp/test"),
        )
        assert m.name == "searching-content"
        assert not m.has_toolset

    def test_description_too_long(self):
        from pathlib import Path

        with pytest.raises(ValueError, match="exceeds"):
            SkillManifest(
                name="test-skill",
                description="x" * 1025,
                path=Path("/tmp/test"),
            )

    def test_to_description_dict(self):
        from pathlib import Path

        m = SkillManifest(
            name="my-skill",
            description="Does stuff.",
            path=Path("/tmp"),
        )
        d = m.to_description_dict()
        assert d == {"name": "my-skill", "description": "Does stuff."}


# ---------------------------------------------------------------------------
# SKILL.md Parsing
# ---------------------------------------------------------------------------


class TestParseSkillMd:
    def test_parse_valid_skill_md(self, tmp_path):
        skill_dir = tmp_path / "my-skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(textwrap.dedent("""\
            ---
            name: my-skill
            description: Does something useful. Use when the agent needs to do something.
            ---

            # My Skill

            Some detailed instructions here.
            """))
        manifest = parse_skill_md(skill_dir)
        assert manifest is not None
        assert manifest.name == "my-skill"
        assert "Does something useful" in manifest.description
        assert "# My Skill" in manifest.body

    def test_parse_missing_skill_md(self, tmp_path):
        skill_dir = tmp_path / "empty"
        skill_dir.mkdir()
        assert parse_skill_md(skill_dir) is None

    def test_parse_no_frontmatter(self, tmp_path):
        skill_dir = tmp_path / "bad"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("Just some text without frontmatter.")
        assert parse_skill_md(skill_dir) is None

    def test_parse_missing_name(self, tmp_path):
        skill_dir = tmp_path / "no-name"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(textwrap.dedent("""\
            ---
            description: Missing name field.
            ---
            Body.
            """))
        assert parse_skill_md(skill_dir) is None

    def test_parse_missing_description(self, tmp_path):
        skill_dir = tmp_path / "no-desc"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(textwrap.dedent("""\
            ---
            name: no-desc
            ---
            Body.
            """))
        assert parse_skill_md(skill_dir) is None


# ---------------------------------------------------------------------------
# Skill Discovery
# ---------------------------------------------------------------------------


class TestDiscoverSkills:
    def test_discover_builtin_searching_content(self):
        """The searching-content skill should be discoverable from builtins."""
        manifests = discover_skills()
        names = [m.name for m in manifests]
        assert "searching-content" in names

    def test_discover_custom_path(self, tmp_path):
        """Custom skill paths are scanned alongside builtins."""
        skill_dir = tmp_path / "custom-skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(textwrap.dedent("""\
            ---
            name: custom-skill
            description: A custom user skill.
            ---
            Custom body.
            """))
        manifests = discover_skills(tmp_path)
        names = [m.name for m in manifests]
        assert "custom-skill" in names
        assert "searching-content" in names  # builtins still present

    def test_discover_skips_hidden_dirs(self, tmp_path):
        hidden = tmp_path / ".hidden"
        hidden.mkdir()
        (hidden / "SKILL.md").write_text(textwrap.dedent("""\
            ---
            name: hidden-skill
            description: Should be skipped.
            ---
            """))
        manifests = discover_skills(tmp_path)
        names = [m.name for m in manifests]
        assert "hidden-skill" not in names

    def test_discover_no_duplicate_names(self, tmp_path):
        """If custom skill has same name as builtin, custom is skipped (builtin wins)."""
        skill_dir = tmp_path / "searching-content"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(textwrap.dedent("""\
            ---
            name: searching-content
            description: Duplicate of builtin.
            ---
            """))
        manifests = discover_skills(tmp_path)
        content_skills = [m for m in manifests if m.name == "searching-content"]
        assert len(content_skills) == 1  # No duplicates


# ---------------------------------------------------------------------------
# Toolset Loading
# ---------------------------------------------------------------------------


class TestLoadToolset:
    def test_load_searching_content_toolset(self):
        """Load the searching-content skill's toolset."""
        manifests = discover_skills()
        content = next(m for m in manifests if m.name == "searching-content")
        toolset = load_toolset(content)
        assert toolset is not None

    def test_load_missing_tools_py(self, tmp_path):
        """Skills without tools.py should raise FileNotFoundError."""
        skill_dir = tmp_path / "no-tools"
        skill_dir.mkdir()
        manifest = SkillManifest(
            name="no-tools",
            description="No tools file.",
            path=skill_dir,
        )
        with pytest.raises(FileNotFoundError, match="no tools.py"):
            load_toolset(manifest)

    def test_load_tools_py_without_toolset_var(self, tmp_path):
        """tools.py that doesn't export `toolset` should raise AttributeError."""
        skill_dir = tmp_path / "bad-tools"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(textwrap.dedent("""\
            ---
            name: bad-tools
            description: Has tools.py but no toolset var.
            ---
            """))
        (skill_dir / "tools.py").write_text("x = 42\n")
        manifest = parse_skill_md(skill_dir)
        with pytest.raises(AttributeError, match="must export"):
            load_toolset(manifest)


# ---------------------------------------------------------------------------
# SkillRegistry
# ---------------------------------------------------------------------------


class TestSkillRegistry:
    def test_discover(self):
        registry = SkillRegistry()
        names = registry.discover()
        assert "searching-content" in names
        assert "searching-content" in registry.available_skills

    def test_enable(self):
        registry = SkillRegistry()
        registry.discover()
        enabled = registry.enable(["searching-content"])
        assert "searching-content" in enabled
        assert "searching-content" in registry.enabled_skills

    def test_enable_unknown_skill_raises(self):
        registry = SkillRegistry()
        registry.discover()
        with pytest.raises(ValueError, match="not found"):
            registry.enable(["nonexistent-skill"])

    def test_enable_auto_discovers(self):
        """Enabling without prior discover() should auto-discover."""
        registry = SkillRegistry()
        registry.enable(["searching-content"])
        assert "searching-content" in registry.enabled_skills

    def test_disable(self):
        registry = SkillRegistry()
        registry.discover()
        registry.enable(["searching-content"])
        registry.disable("searching-content")
        assert "searching-content" not in registry.enabled_skills

    def test_disable_all(self):
        registry = SkillRegistry()
        registry.discover()
        registry.enable(["searching-content"])
        registry.disable_all()
        assert registry.enabled_skills == []

    def test_get_enabled_toolsets(self):
        registry = SkillRegistry()
        registry.discover()
        registry.enable(["searching-content"])
        toolsets = registry.get_enabled_toolsets()
        assert len(toolsets) == 1

    def test_get_enabled_descriptions(self):
        registry = SkillRegistry()
        registry.discover()
        registry.enable(["searching-content"])
        descs = registry.get_enabled_descriptions()
        assert len(descs) == 1
        assert descs[0]["name"] == "searching-content"
        assert "description" in descs[0]

    def test_get_skill_details(self):
        registry = SkillRegistry()
        registry.discover()
        details = registry.get_skill_details("searching-content")
        assert "regex_search" in details
        assert "keyword_search" in details

    def test_get_skill_details_unknown_raises(self):
        registry = SkillRegistry()
        registry.discover()
        with pytest.raises(ValueError, match="not found"):
            registry.get_skill_details("nonexistent")

    def test_get_manifest(self):
        registry = SkillRegistry()
        registry.discover()
        manifest = registry.get_manifest("searching-content")
        assert manifest is not None
        assert manifest.name == "searching-content"

    def test_get_manifest_unknown(self):
        registry = SkillRegistry()
        registry.discover()
        assert registry.get_manifest("nonexistent") is None

    def test_build_meta_toolset(self):
        registry = SkillRegistry()
        registry.discover()
        registry.enable(["searching-content"])
        meta_toolset = registry.build_meta_toolset()
        assert meta_toolset is not None

    def test_custom_skill_path(self, tmp_path):
        """Registry can discover skills from custom paths."""
        skill_dir = tmp_path / "my-custom"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(textwrap.dedent("""\
            ---
            name: my-custom
            description: Custom test skill.
            ---
            Custom body.
            """))
        (skill_dir / "tools.py").write_text(textwrap.dedent("""\
            from pydantic_ai.toolsets import FunctionToolset
            from pydantic_ai import RunContext

            toolset = FunctionToolset()

            @toolset.tool
            async def custom_tool(ctx: RunContext, query: str) -> str:
                \"\"\"A custom tool.\"\"\"
                return f"custom: {query}"
            """))
        registry = SkillRegistry()
        registry.discover(tmp_path)
        registry.enable(["my-custom"])
        assert "my-custom" in registry.enabled_skills
        assert len(registry.get_enabled_toolsets()) == 1


# ---------------------------------------------------------------------------
# AgenticReviewer Skills Integration
# ---------------------------------------------------------------------------


class TestReviewerSkillsIntegration:
    def test_reviewer_with_skills_non_agentic_ignores(self):
        """Non-agentic reviewer ignores skills config."""
        r = AgenticReviewer(
            model=TestModel(),
            max_iterations=1,
            skills=["searching-content"],
        )
        toolsets, descs = r._setup_skills()
        assert toolsets == []
        assert descs == []

    def test_reviewer_with_skills_agentic_sets_up(self):
        """Agentic reviewer with skills configured sets up toolsets."""
        r = AgenticReviewer(
            model=TestModel(),
            max_iterations=5,
            skills=["searching-content"],
        )
        toolsets, descs = r._setup_skills()
        # Should have: searching-content toolset + meta toolset
        assert len(toolsets) == 2
        assert len(descs) == 1
        assert descs[0]["name"] == "searching-content"

    def test_reviewer_no_skills_empty(self):
        """Agentic reviewer without skills returns empty."""
        r = AgenticReviewer(
            model=TestModel(),
            max_iterations=5,
        )
        toolsets, descs = r._setup_skills()
        assert toolsets == []
        assert descs == []

    @pytest.mark.asyncio
    async def test_review_item_with_skills(self):
        """review_item with skills configured should work end-to-end with TestModel."""
        r = AgenticReviewer(
            model=TestModel(),
            max_iterations=5,
            skills=["searching-content"],
            system_prompt="Review with content search.",
        )
        response, cost = await r.review_item(
            item_text="A study about machine learning in healthcare.",
            item_id="skill_test",
        )
        assert "reasoning" in response
        assert "score" in response

    @pytest.mark.asyncio
    async def test_review_item_skill_descriptions_in_prompt(self):
        """Skills should inject descriptions into the system prompt."""
        r = AgenticReviewer(
            model=TestModel(),
            max_iterations=5,
            skills=["searching-content"],
        )
        prompt = r._build_system_prompt(
            skill_descriptions=[{"name": "searching-content", "description": "Searches text."}],
        )
        assert "searching-content" in prompt
        assert "Available Skills" in prompt


# ---------------------------------------------------------------------------
# Content Search Skill Tools (direct invocation via toolset internals)
# ---------------------------------------------------------------------------


class TestContentSearchTools:
    def _get_content_toolset(self):
        """Load the searching-content toolset."""
        registry = SkillRegistry()
        registry.discover()
        registry.enable(["searching-content"])
        toolsets = registry.get_enabled_toolsets()
        return toolsets[0]

    def test_toolset_has_expected_tools(self):
        """searching-content toolset should have regex_search and keyword_search."""
        toolset = self._get_content_toolset()
        tool_names = set(toolset.tools.keys())
        assert "regex_search" in tool_names
        assert "keyword_search" in tool_names


# ---------------------------------------------------------------------------
# Live Integration Tests
# ---------------------------------------------------------------------------


class TestLiveSkills:
    @pytest.mark.live
    @pytest.mark.asyncio
    async def test_live_review_with_content_search(self, env_keys):
        """Live test: agent uses searching-content skill during review."""
        if not os.environ.get("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set")

        r = AgenticReviewer(
            name="SkillTester",
            model="openai:gpt-5.4-mini",
            max_iterations=5,
            agentic_effort="high",
            skills=["searching-content"],
            system_prompt=(
                "You are a paper reviewer. Use the content search tools to find "
                "specific patterns in the text before scoring."
            ),
            task_prompt="Search for statistical methods mentioned and score quality 1-10:\n\n${item}$",
            output_type=ScoringOutput,
            model_settings={"temperature": 0.0},
        )

        response, cost = await r.review_item(
            item_text=(
                "Methods: We conducted a randomized double-blind placebo-controlled trial "
                "with 500 participants. The primary outcome was assessed using a mixed-effects "
                "model. Statistical significance was set at p < 0.05. Results showed a "
                "significant reduction in blood pressure (p = 0.003, 95% CI: -8.5 to -2.1 mmHg)."
            ),
            item_id="live_skill_001",
        )

        assert isinstance(response["reasoning"], str)
        assert isinstance(response["score"], int)
        assert 0 <= response["certainty"] <= 100
