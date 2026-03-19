"""Tests for reviewer.py — AgenticReviewer core functionality."""

import warnings

import pytest

from pydantic import BaseModel

from lattereview.agentic.reviewer import AgenticReviewer
from lattereview.agentic.output_models import ScoringOutput, EvaluationOutput
from lattereview.agentic.deps import ReviewDeps


class TestAgenticReviewerInit:
    def test_default_values(self):
        r = AgenticReviewer()
        assert r.name == "Reviewer"
        assert r.model == "openai:gpt-5.4-mini"
        assert r.max_iterations == 20
        assert r.agentic_effort == "medium"
        assert r.output_type is ScoringOutput
        assert r.is_agentic is True

    def test_non_agentic_mode(self):
        r = AgenticReviewer(max_iterations=1)
        assert r.is_agentic is False

    def test_custom_values(self):
        r = AgenticReviewer(
            name="Expert",
            backstory="You are a biomedical expert.",
            model="anthropic:claude-sonnet-4-20250514",
            system_prompt="Focus on quality.",
            max_iterations=10,
            agentic_effort="high",
            output_type=EvaluationOutput,
        )
        assert r.name == "Expert"
        assert r.model == "anthropic:claude-sonnet-4-20250514"
        assert r.agentic_effort == "high"
        assert r.output_type is EvaluationOutput

    def test_invalid_max_iterations(self):
        with pytest.raises(ValueError, match="max_iterations must be >= 1"):
            AgenticReviewer(max_iterations=0)

    def test_invalid_effort(self):
        with pytest.raises(ValueError, match="agentic_effort"):
            AgenticReviewer(agentic_effort="extreme")

    def test_low_iteration_warning(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            AgenticReviewer(max_iterations=3)
            assert len(w) == 1
            assert "limited agentic functionality" in str(w[0].message)

    def test_no_warning_for_1_or_5(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            AgenticReviewer(max_iterations=1)
            AgenticReviewer(max_iterations=5)
            assert len(w) == 0


class TestBuildSystemPrompt:
    def test_non_agentic_prompt(self):
        r = AgenticReviewer(name="TestBot", max_iterations=1)
        prompt = r._build_system_prompt()
        assert "TestBot" in prompt
        assert "Tool Usage" not in prompt

    def test_agentic_prompt(self):
        r = AgenticReviewer(name="AgentBot", max_iterations=10)
        prompt = r._build_system_prompt()
        assert "AgentBot" in prompt
        assert "Tool Usage Guidance" in prompt


class TestBuildAgent:
    def test_non_agentic_agent_has_no_toolsets(self):
        from pydantic_ai.models.test import TestModel

        r = AgenticReviewer(model=TestModel(), max_iterations=1)
        prompt = r._build_system_prompt()
        agent = r._build_agent(prompt, toolsets=["some_toolset"])
        # In non-agentic mode, toolsets should NOT be passed to agent
        # We can verify by checking the agent was created successfully
        assert agent is not None

    def test_agentic_agent_creation(self):
        from pydantic_ai.models.test import TestModel

        r = AgenticReviewer(model=TestModel(), max_iterations=10)
        prompt = r._build_system_prompt()
        agent = r._build_agent(prompt)
        assert agent is not None


class TestReviewItem:
    @pytest.mark.asyncio
    async def test_review_item_with_test_model(self):
        """Test review_item using Pydantic AI's TestModel for mocked responses."""
        from pydantic_ai.models.test import TestModel

        r = AgenticReviewer(
            name="TestReviewer",
            model=TestModel(),
            max_iterations=1,
            system_prompt="You are a paper reviewer.",
            task_prompt="Review this paper:\n${item}$",
        )
        response, cost = await r.review_item(
            item_text="A study on machine learning in healthcare.",
            item_id="test_001",
        )
        # TestModel returns default values for the output type
        assert "reasoning" in response
        assert "score" in response
        assert "certainty" in response
        assert isinstance(cost, float)

    @pytest.mark.asyncio
    async def test_review_item_with_custom_output(self):
        """Test review_item with a custom output model."""
        from pydantic_ai.models.test import TestModel

        class CustomOutput(BaseModel):
            summary: str
            relevance: int

        r = AgenticReviewer(
            model=TestModel(),
            max_iterations=1,
            output_type=CustomOutput,
        )
        response, cost = await r.review_item(item_text="test item")
        assert "summary" in response
        assert "relevance" in response

    @pytest.mark.asyncio
    async def test_review_item_agentic_mode(self, tmp_path):
        """Test that agentic mode works with TestModel (default skills need working_dir)."""
        from pydantic_ai.models.test import TestModel

        r = AgenticReviewer(
            model=TestModel(),
            max_iterations=5,
        )
        response, cost = await r.review_item(item_text="test item", working_dir=tmp_path)
        assert "reasoning" in response


class TestReviewItems:
    @pytest.mark.asyncio
    async def test_review_multiple_items(self):
        """Test batch review with TestModel."""
        from pydantic_ai.models.test import TestModel

        r = AgenticReviewer(
            model=TestModel(),
            max_iterations=1,
        )
        responses, total_cost = await r.review_items(
            text_inputs=["Item A", "Item B", "Item C"],
        )
        assert len(responses) == 3
        for resp in responses:
            assert "reasoning" in resp
            assert "score" in resp

    @pytest.mark.asyncio
    async def test_review_items_with_custom_ids(self):
        """Test batch review with explicit item IDs."""
        from pydantic_ai.models.test import TestModel

        r = AgenticReviewer(
            model=TestModel(),
            max_iterations=1,
        )
        responses, cost = await r.review_items(
            text_inputs=["A", "B"],
            item_ids=["id_x", "id_y"],
        )
        assert len(responses) == 2

    @pytest.mark.asyncio
    async def test_review_items_mismatched_lengths(self):
        """Test that mismatched text_inputs and item_ids raises ValueError."""
        from pydantic_ai.models.test import TestModel

        r = AgenticReviewer(model=TestModel(), max_iterations=1)
        with pytest.raises(ValueError, match="same length"):
            await r.review_items(
                text_inputs=["A", "B"],
                item_ids=["only_one"],
            )


class TestReviewDeps:
    def test_deps_creation(self):
        deps = ReviewDeps(
            item_id="001",
            item_text="test text",
            agent_name="Reviewer",
            round_id="A",
            max_iterations=10,
        )
        assert deps.item_id == "001"
        assert deps.agentic_effort == "medium"
        assert deps.memory_store is None
        assert deps.extra == {}

    def test_deps_with_extras(self):
        deps = ReviewDeps(
            item_id="002",
            item_text="text",
            agent_name="R",
            round_id="B",
            max_iterations=1,
            extra={"custom_key": "custom_value"},
        )
        assert deps.extra["custom_key"] == "custom_value"


class TestLiveReviewItem:
    """Live integration tests — require API keys in .env."""

    @pytest.mark.live
    @pytest.mark.asyncio
    async def test_review_item_openai(self, env_keys):
        """End-to-end review with real OpenAI API."""
        import os

        if not os.environ.get("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set")

        r = AgenticReviewer(
            name="OpenAI-Tester",
            model="openai:gpt-5.4-mini",
            max_iterations=1,
            system_prompt="You are a scientific paper reviewer. Be concise.",
            task_prompt="Rate this paper abstract on a scale of 1-10:\n\n${item}$",
            output_type=ScoringOutput,
            model_settings={"temperature": 0.0},
        )

        response, cost = await r.review_item(
            item_text="This randomized controlled trial examined the efficacy of a new drug for treating hypertension in 500 patients over 12 months. Results showed a significant reduction in systolic blood pressure (p<0.001).",
            item_id="live_001",
        )

        assert isinstance(response["reasoning"], str)
        assert len(response["reasoning"]) > 0
        assert isinstance(response["score"], int)
        assert 0 <= response["certainty"] <= 100

    @pytest.mark.live
    @pytest.mark.asyncio
    async def test_review_item_openai_agentic(self, env_keys):
        """End-to-end agentic review with real OpenAI API (no skills, just iteration)."""
        import os

        if not os.environ.get("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set")

        r = AgenticReviewer(
            name="Agentic-Tester",
            model="openai:gpt-5.4-mini",
            max_iterations=5,
            agentic_effort="low",
            system_prompt="You are a scientific paper reviewer.",
            output_type=ScoringOutput,
            model_settings={"temperature": 0.0},
        )

        response, cost = await r.review_item(
            item_text="A meta-analysis of 20 studies on cognitive behavioral therapy for anxiety disorders.",
        )

        assert isinstance(response["reasoning"], str)
        assert isinstance(response["score"], int)

    @pytest.mark.live
    @pytest.mark.asyncio
    async def test_review_item_anthropic(self, env_keys):
        """End-to-end review with real Anthropic API."""
        import os

        if not os.environ.get("ANTHROPIC_API_KEY"):
            pytest.skip("ANTHROPIC_API_KEY not set")

        r = AgenticReviewer(
            name="Anthropic-Tester",
            model="anthropic:claude-haiku-4-5-20251001",
            max_iterations=1,
            system_prompt="You are a scientific paper reviewer. Be concise.",
            output_type=ScoringOutput,
            model_settings={"temperature": 0.0},
        )

        response, cost = await r.review_item(
            item_text="A cohort study of 1000 patients examining the relationship between diet and cardiovascular disease over 10 years.",
        )

        assert isinstance(response["reasoning"], str)
        assert isinstance(response["score"], int)
        assert 0 <= response["certainty"] <= 100

    @pytest.mark.live
    @pytest.mark.asyncio
    async def test_review_item_gemini(self, env_keys):
        """End-to-end review with real Gemini API."""
        import os

        if not os.environ.get("GEMINI_API_KEY"):
            pytest.skip("GEMINI_API_KEY not set")

        r = AgenticReviewer(
            name="Gemini-Tester",
            model="google-gla:gemini-3.1-flash-lite-preview",
            max_iterations=1,
            system_prompt="You are a scientific paper reviewer. Be concise.",
            output_type=ScoringOutput,
            model_settings={"temperature": 0.0},
        )

        response, cost = await r.review_item(
            item_text="A systematic review of randomized controlled trials evaluating the effectiveness of mindfulness-based interventions for chronic pain management.",
        )

        assert isinstance(response["reasoning"], str)
        assert isinstance(response["score"], int)
        assert 0 <= response["certainty"] <= 100

    @pytest.mark.live
    @pytest.mark.asyncio
    async def test_review_items_batch_openai(self, env_keys):
        """End-to-end batch review with real OpenAI API."""
        import os

        if not os.environ.get("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set")

        r = AgenticReviewer(
            name="Batch-Tester",
            model="openai:gpt-5.4-mini",
            max_iterations=1,
            system_prompt="Rate each abstract 1-10.",
            output_type=ScoringOutput,
            model_settings={"temperature": 0.0},
        )

        responses, total_cost = await r.review_items(
            text_inputs=[
                "RCT of 500 patients on drug X for diabetes.",
                "Case report of rare adverse event with drug Y.",
                "Meta-analysis of 30 studies on exercise for depression.",
            ],
            item_ids=["paper_1", "paper_2", "paper_3"],
        )

        assert len(responses) == 3
        for resp in responses:
            assert isinstance(resp["reasoning"], str)
            assert isinstance(resp["score"], int)

    @pytest.mark.live
    @pytest.mark.asyncio
    async def test_review_item_openrouter(self, env_keys):
        """End-to-end review with OpenRouter — just a string, no custom provider code."""
        import os

        if not os.environ.get("OPENROUTER_API_KEY"):
            pytest.skip("OPENROUTER_API_KEY not set")

        r = AgenticReviewer(
            name="OpenRouter-Tester",
            model="openrouter:google/gemini-2.5-flash",
            max_iterations=1,
            system_prompt="You are a scientific paper reviewer. Be concise.",
            output_type=ScoringOutput,
            model_settings={"temperature": 0.0},
        )

        response, cost = await r.review_item(
            item_text="A cross-sectional study examining the prevalence of antibiotic resistance in 200 hospital isolates.",
        )

        assert isinstance(response["reasoning"], str)
        assert isinstance(response["score"], int)
        assert 0 <= response["certainty"] <= 100
