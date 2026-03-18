"""Tests for prompts.py — system prompt builders."""

import pytest

from lattereview.agentic.output_models import ScoringOutput, EvaluationOutput
from lattereview.agentic.prompts import build_system_prompt, build_task_prompt


class TestBuildSystemPrompt:
    def test_non_agentic_prompt_minimal(self):
        prompt = build_system_prompt(
            name="TestReviewer",
            backstory="",
            system_prompt="",
            output_type=ScoringOutput,
            max_iterations=1,
        )
        assert "TestReviewer" in prompt
        assert "reasoning" in prompt
        assert "score" in prompt
        # Should NOT contain agentic sections
        assert "Tool Usage Guidance" not in prompt
        assert "Available Skills" not in prompt
        assert "Your Memories" not in prompt

    def test_non_agentic_with_backstory_and_instructions(self):
        prompt = build_system_prompt(
            name="Expert",
            backstory="You are a biomedical expert.",
            system_prompt="Focus on methodology quality.",
            output_type=ScoringOutput,
            max_iterations=1,
        )
        assert "Expert" in prompt
        assert "biomedical expert" in prompt
        assert "methodology quality" in prompt

    def test_agentic_prompt_has_tool_guidance(self):
        prompt = build_system_prompt(
            name="Agent",
            backstory="",
            system_prompt="Review papers.",
            output_type=ScoringOutput,
            max_iterations=20,
            agentic_effort="medium",
        )
        assert "Tool Usage Guidance" in prompt
        assert "20 iterations" in prompt

    def test_effort_levels(self):
        for effort in ("low", "medium", "high"):
            prompt = build_system_prompt(
                name="R",
                backstory="",
                system_prompt="",
                output_type=ScoringOutput,
                max_iterations=10,
                agentic_effort=effort,
            )
            assert "Tool Usage Guidance" in prompt

    def test_low_effort_contains_sparingly(self):
        prompt = build_system_prompt(
            name="R",
            backstory="",
            system_prompt="",
            output_type=ScoringOutput,
            max_iterations=10,
            agentic_effort="low",
        )
        assert "sparingly" in prompt

    def test_high_effort_contains_proactively(self):
        prompt = build_system_prompt(
            name="R",
            backstory="",
            system_prompt="",
            output_type=ScoringOutput,
            max_iterations=10,
            agentic_effort="high",
        )
        assert "proactively" in prompt

    def test_limited_budget_warning(self):
        prompt = build_system_prompt(
            name="R",
            backstory="",
            system_prompt="",
            output_type=ScoringOutput,
            max_iterations=3,
            agentic_effort="medium",
        )
        assert "limited budget" in prompt

    def test_skill_descriptions_in_prompt(self):
        skills = [
            {"name": "searching-pubmed", "description": "Searches PubMed for articles."},
            {"name": "managing-memory", "description": "Manages agent memory."},
        ]
        prompt = build_system_prompt(
            name="R",
            backstory="",
            system_prompt="",
            output_type=ScoringOutput,
            max_iterations=10,
            enabled_skill_descriptions=skills,
        )
        assert "Available Skills" in prompt
        assert "searching-pubmed" in prompt
        assert "managing-memory" in prompt

    def test_memory_summaries_in_prompt(self):
        memories = [
            {"id": "mem_001", "brief": "RCTs generally scored higher"},
            {"id": "mem_002", "brief": "Check for sample size reporting"},
        ]
        prompt = build_system_prompt(
            name="R",
            backstory="",
            system_prompt="",
            output_type=ScoringOutput,
            max_iterations=10,
            memory_summaries=memories,
        )
        assert "Your Memories (2 total)" in prompt
        assert "mem_001" in prompt
        assert "RCTs generally scored higher" in prompt
        assert "load_memory" in prompt

    def test_non_agentic_ignores_skills_and_memories(self):
        """Skills and memories should NOT appear in non-agentic mode."""
        prompt = build_system_prompt(
            name="R",
            backstory="",
            system_prompt="",
            output_type=ScoringOutput,
            max_iterations=1,
            enabled_skill_descriptions=[{"name": "x", "description": "y"}],
            memory_summaries=[{"id": "m1", "brief": "something"}],
        )
        assert "Available Skills" not in prompt
        assert "Your Memories" not in prompt

    def test_output_schema_describes_fields(self):
        prompt = build_system_prompt(
            name="R",
            backstory="",
            system_prompt="",
            output_type=EvaluationOutput,
            max_iterations=1,
        )
        assert "decision" in prompt
        assert "reasoning" in prompt
        assert "certainty" in prompt


class TestBuildTaskPrompt:
    def test_basic_substitution(self):
        result = build_task_prompt("Review this: ${item}$", "A study about COVID-19.")
        assert result == "Review this: A study about COVID-19."

    def test_no_placeholder(self):
        result = build_task_prompt("No placeholder here", "item text")
        assert result == "No placeholder here"

    def test_multiple_occurrences(self):
        result = build_task_prompt("First: ${item}$\nSecond: ${item}$", "data")
        assert result == "First: data\nSecond: data"

    def test_empty_item(self):
        result = build_task_prompt("Item: ${item}$", "")
        assert result == "Item: "
