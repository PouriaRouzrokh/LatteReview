"""Tests for RFD-8: Search Skills Bundle.

Unit tests mock all external API calls. Live tests make real queries.

Because skill directories use hyphens (e.g., searching-duckduckgo/), we
cannot use normal Python imports. Instead we load tools modules via
importlib, matching how the SkillRegistry loader works.
"""

import importlib.util
import os
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic_ai import ModelRetry
from pydantic_ai.models.test import TestModel

from lattereview.agentic.deps import ReviewDeps
from lattereview.agentic.output_models import ScoringOutput
from lattereview.agentic.reviewer import AgenticReviewer
from lattereview.agentic.skills.loader import BUILTIN_SKILLS_DIR
from lattereview.agentic.skills.registry import SkillRegistry

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SKILL_CACHE: dict[str, ModuleType] = {}


def _load_tools_module(skill_name: str) -> ModuleType:
    """Load a skill's tools.py via importlib (handles hyphen directories)."""
    if skill_name in _SKILL_CACHE:
        return _SKILL_CACHE[skill_name]
    tools_py = BUILTIN_SKILLS_DIR / skill_name / "tools.py"
    module_name = f"_test_skill_{skill_name.replace('-', '_')}"
    spec = importlib.util.spec_from_file_location(module_name, tools_py)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    _SKILL_CACHE[skill_name] = mod
    return mod


def _make_deps(**overrides):
    """Create a minimal ReviewDeps for tool tests."""
    defaults = dict(
        item_id="test_001",
        item_text="A study about machine learning in healthcare.",
        agent_name="tester",
        round_id="A",
        max_iterations=5,
        agentic_effort="medium",
    )
    defaults.update(overrides)
    return ReviewDeps(**defaults)


def _make_ctx(deps=None):
    """Create a mock RunContext wrapping ReviewDeps."""
    ctx = MagicMock()
    ctx.deps = deps or _make_deps()
    return ctx


# ---------------------------------------------------------------------------
# Skill Discovery — all 5 search skills should be discoverable
# ---------------------------------------------------------------------------


class TestSearchSkillDiscovery:
    """Verify all search skills are discovered by the registry."""

    SEARCH_SKILLS = [
        "searching-google",
        "searching-duckduckgo",
        "searching-pubmed",
        "searching-semantic-scholar",
        "searching-arxiv",
    ]

    def test_all_search_skills_discoverable(self):
        registry = SkillRegistry()
        names = registry.discover()
        for skill in self.SEARCH_SKILLS:
            assert skill in names, f"{skill} not discovered"

    def test_all_search_skills_have_descriptions(self):
        registry = SkillRegistry()
        registry.discover()
        for skill in self.SEARCH_SKILLS:
            manifest = registry.get_manifest(skill)
            assert manifest is not None
            assert len(manifest.description) > 20

    def test_all_search_skills_have_body(self):
        registry = SkillRegistry()
        registry.discover()
        for skill in self.SEARCH_SKILLS:
            details = registry.get_skill_details(skill)
            assert "Available Tools" in details

    def test_enable_all_search_skills(self):
        registry = SkillRegistry()
        registry.discover()
        enabled = registry.enable(self.SEARCH_SKILLS)
        assert set(enabled) == set(self.SEARCH_SKILLS)

    def test_get_toolsets_for_all(self):
        registry = SkillRegistry()
        registry.discover()
        registry.enable(self.SEARCH_SKILLS)
        toolsets = registry.get_enabled_toolsets()
        assert len(toolsets) == 5


# ---------------------------------------------------------------------------
# Toolset Verification — check each skill exports expected tools
# ---------------------------------------------------------------------------


class TestSearchSkillToolsets:
    def _get_toolset(self, skill_name):
        registry = SkillRegistry()
        registry.discover()
        registry.enable([skill_name])
        return registry.get_enabled_toolsets()[0]

    def test_google_tools(self):
        ts = self._get_toolset("searching-google")
        assert "google_search" in ts.tools

    def test_duckduckgo_tools(self):
        ts = self._get_toolset("searching-duckduckgo")
        assert "duckduckgo_search" in ts.tools

    def test_pubmed_tools(self):
        ts = self._get_toolset("searching-pubmed")
        names = set(ts.tools.keys())
        assert "search_pubmed" in names
        assert "get_abstract" in names
        assert "get_full_text" in names

    def test_semantic_scholar_tools(self):
        ts = self._get_toolset("searching-semantic-scholar")
        names = set(ts.tools.keys())
        assert "search_papers" in names
        assert "get_paper_details" in names

    def test_arxiv_tools(self):
        ts = self._get_toolset("searching-arxiv")
        names = set(ts.tools.keys())
        assert "search_arxiv" in names
        assert "get_paper" in names


# ---------------------------------------------------------------------------
# DuckDuckGo — mock tests
# ---------------------------------------------------------------------------


class TestDuckDuckGoSearch:
    def _get_mod(self):
        return _load_tools_module("searching-duckduckgo")

    @pytest.mark.asyncio
    async def test_search_returns_results(self):
        mod = self._get_mod()
        mock_results = [
            {"title": "ML in Healthcare", "body": "A comprehensive review...", "href": "https://example.com/1"},
            {"title": "Deep Learning", "body": "Neural network approaches...", "href": "https://example.com/2"},
        ]

        mock_ddgs_instance = MagicMock()
        mock_ddgs_instance.__enter__ = MagicMock(return_value=mock_ddgs_instance)
        mock_ddgs_instance.__exit__ = MagicMock(return_value=False)
        mock_ddgs_instance.text = MagicMock(return_value=mock_results)

        mock_ddgs_cls = MagicMock(return_value=mock_ddgs_instance)
        mock_ddgs_module = ModuleType("duckduckgo_search")
        mock_ddgs_module.DDGS = mock_ddgs_cls

        with patch.dict(sys.modules, {"duckduckgo_search": mock_ddgs_module}):
            ctx = _make_ctx()
            result = await mod.duckduckgo_search(ctx, "machine learning healthcare")

        assert "ML in Healthcare" in result
        assert "Deep Learning" in result
        assert "example.com" in result

    @pytest.mark.asyncio
    async def test_search_no_results(self):
        mod = self._get_mod()
        mock_ddgs_instance = MagicMock()
        mock_ddgs_instance.__enter__ = MagicMock(return_value=mock_ddgs_instance)
        mock_ddgs_instance.__exit__ = MagicMock(return_value=False)
        mock_ddgs_instance.text = MagicMock(return_value=[])

        mock_ddgs_cls = MagicMock(return_value=mock_ddgs_instance)
        mock_ddgs_module = ModuleType("duckduckgo_search")
        mock_ddgs_module.DDGS = mock_ddgs_cls

        with patch.dict(sys.modules, {"duckduckgo_search": mock_ddgs_module}):
            ctx = _make_ctx()
            result = await mod.duckduckgo_search(ctx, "xyznonexistent")

        assert "No results found" in result

    @pytest.mark.asyncio
    async def test_max_results_capped(self):
        mod = self._get_mod()
        mock_ddgs_instance = MagicMock()
        mock_ddgs_instance.__enter__ = MagicMock(return_value=mock_ddgs_instance)
        mock_ddgs_instance.__exit__ = MagicMock(return_value=False)
        mock_ddgs_instance.text = MagicMock(
            return_value=[{"title": f"R{i}", "body": "...", "href": "url"} for i in range(10)]
        )

        mock_ddgs_cls = MagicMock(return_value=mock_ddgs_instance)
        mock_ddgs_module = ModuleType("duckduckgo_search")
        mock_ddgs_module.DDGS = mock_ddgs_cls

        with patch.dict(sys.modules, {"duckduckgo_search": mock_ddgs_module}):
            ctx = _make_ctx()
            await mod.duckduckgo_search(ctx, "test", max_results=50)

        # max_results should be capped to 10
        mock_ddgs_instance.text.assert_called_once_with("test", max_results=10)


# ---------------------------------------------------------------------------
# Google Search — mock tests
# ---------------------------------------------------------------------------


class TestGoogleSearch:
    def _get_mod(self):
        return _load_tools_module("searching-google")

    @pytest.mark.asyncio
    async def test_missing_api_key_raises_model_retry(self):
        mod = self._get_mod()

        # Mock google.genai so it imports successfully
        mock_genai = MagicMock()
        mock_google = ModuleType("google")
        mock_google.genai = mock_genai
        sys.modules.setdefault("google", mock_google)

        with patch.dict(sys.modules, {"google": mock_google, "google.genai": mock_genai}):
            env = {k: v for k, v in os.environ.items() if k != "GEMINI_API_KEY"}
            with patch.dict(os.environ, env, clear=True):
                ctx = _make_ctx()
                with pytest.raises(ModelRetry, match="GEMINI_API_KEY"):
                    await mod.google_search(ctx, "test query")

    @pytest.mark.asyncio
    async def test_search_with_grounding_results(self):
        mod = self._get_mod()

        mock_web = MagicMock()
        mock_web.title = "Test Article"
        mock_web.uri = "https://example.com/article"

        mock_chunk = MagicMock()
        mock_chunk.web = mock_web

        mock_metadata = MagicMock()
        mock_metadata.grounding_chunks = [mock_chunk]

        mock_candidate = MagicMock()
        mock_candidate.grounding_metadata = mock_metadata

        mock_response = MagicMock()
        mock_response.candidates = [mock_candidate]
        mock_response.text = "Some text"

        mock_client = MagicMock()
        mock_client.models.generate_content = MagicMock(return_value=mock_response)

        mock_genai = MagicMock()
        mock_genai.Client = MagicMock(return_value=mock_client)
        mock_google = ModuleType("google")
        mock_google.genai = mock_genai

        with patch.dict(sys.modules, {"google": mock_google, "google.genai": mock_genai}):
            with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}):
                ctx = _make_ctx()
                result = await mod.google_search(ctx, "test query")

        assert "Test Article" in result
        assert "example.com" in result

    @pytest.mark.asyncio
    async def test_search_fallback_to_text(self):
        mod = self._get_mod()

        mock_candidate = MagicMock()
        mock_candidate.grounding_metadata = None

        mock_response = MagicMock()
        mock_response.candidates = [mock_candidate]
        mock_response.text = "Here is what I found about the topic."

        mock_client = MagicMock()
        mock_client.models.generate_content = MagicMock(return_value=mock_response)

        mock_genai = MagicMock()
        mock_genai.Client = MagicMock(return_value=mock_client)
        mock_google = ModuleType("google")
        mock_google.genai = mock_genai

        with patch.dict(sys.modules, {"google": mock_google, "google.genai": mock_genai}):
            with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}):
                ctx = _make_ctx()
                result = await mod.google_search(ctx, "test query")

        assert "Here is what I found" in result


# ---------------------------------------------------------------------------
# PubMed — mock tests
# ---------------------------------------------------------------------------


class TestPubMedSearch:
    def _get_mod(self):
        return _load_tools_module("searching-pubmed")

    def _make_mock_article(self, pmid="12345678", title="Test Article", abstract="Test abstract text."):
        article = MagicMock()
        article.pubmed_id = pmid
        article.title = title
        article.abstract = abstract
        article.journal = "Test Journal"
        pub_date = MagicMock()
        pub_date.year = 2024
        article.publication_date = pub_date
        article.authors = [
            {"firstname": "John", "lastname": "Doe"},
            {"firstname": "Jane", "lastname": "Smith"},
        ]
        article.xml = None
        return article

    def _mock_pymed(self, articles):
        """Create a mock pymed module returning the given articles."""
        mock_pubmed_instance = MagicMock()
        mock_pubmed_instance.query = MagicMock(return_value=articles)
        mock_pubmed_cls = MagicMock(return_value=mock_pubmed_instance)
        mock_pymed = ModuleType("pymed")
        mock_pymed.PubMed = mock_pubmed_cls
        return mock_pymed

    @pytest.mark.asyncio
    async def test_search_returns_results(self):
        mod = self._get_mod()
        mock_article = self._make_mock_article()
        mock = self._mock_pymed([mock_article])

        with patch.dict(sys.modules, {"pymed": mock}):
            ctx = _make_ctx()
            result = await mod.search_pubmed(ctx, "machine learning")

        assert "Test Article" in result
        assert "12345678" in result
        assert "John Doe" in result

    @pytest.mark.asyncio
    async def test_search_no_results(self):
        mod = self._get_mod()
        mock = self._mock_pymed([])

        with patch.dict(sys.modules, {"pymed": mock}):
            ctx = _make_ctx()
            result = await mod.search_pubmed(ctx, "xyznonexistent")

        assert "No PubMed results" in result

    @pytest.mark.asyncio
    async def test_get_abstract(self):
        mod = self._get_mod()
        mock_article = self._make_mock_article(abstract="This is the full abstract of the study.")
        mock = self._mock_pymed([mock_article])

        with patch.dict(sys.modules, {"pymed": mock}):
            ctx = _make_ctx()
            result = await mod.get_abstract(ctx, "12345678")

        assert "full abstract" in result
        assert "Test Article" in result

    @pytest.mark.asyncio
    async def test_get_abstract_not_found(self):
        mod = self._get_mod()
        mock = self._mock_pymed([])

        with patch.dict(sys.modules, {"pymed": mock}):
            ctx = _make_ctx()
            result = await mod.get_abstract(ctx, "99999999")

        assert "No article found" in result

    @pytest.mark.asyncio
    async def test_get_full_text_fallback_to_abstract(self):
        mod = self._get_mod()
        mock_article = self._make_mock_article(abstract="Abstract fallback text.")
        mock = self._mock_pymed([mock_article])

        with patch.dict(sys.modules, {"pymed": mock}):
            ctx = _make_ctx()
            result = await mod.get_full_text(ctx, "12345678")

        assert "Full text not available" in result
        assert "Abstract fallback text" in result

    @pytest.mark.asyncio
    async def test_multiline_pmid_handled(self):
        mod = self._get_mod()
        mock_article = self._make_mock_article(pmid="12345678\n87654321")
        mock = self._mock_pymed([mock_article])

        with patch.dict(sys.modules, {"pymed": mock}):
            ctx = _make_ctx()
            result = await mod.search_pubmed(ctx, "test")

        assert "12345678" in result
        assert "87654321" not in result


# ---------------------------------------------------------------------------
# Semantic Scholar — mock tests
# ---------------------------------------------------------------------------


def _mock_httpx_client(response_data):
    """Create a mock httpx.AsyncClient that returns response_data on get()."""
    mock_resp = MagicMock()
    mock_resp.json = MagicMock(return_value=response_data)
    mock_resp.raise_for_status = MagicMock()

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.get = AsyncMock(return_value=mock_resp)

    mock_httpx = ModuleType("httpx")
    mock_httpx.AsyncClient = MagicMock(return_value=mock_client)
    return mock_httpx, mock_client


class TestSemanticScholarSearch:
    def _get_mod(self):
        return _load_tools_module("searching-semantic-scholar")

    @pytest.mark.asyncio
    async def test_search_returns_results(self):
        mod = self._get_mod()
        data = {
            "total": 100,
            "data": [
                {
                    "paperId": "abc123",
                    "title": "Attention Is All You Need",
                    "authors": [{"name": "Vaswani"}, {"name": "Shazeer"}],
                    "year": 2017,
                    "citationCount": 50000,
                    "externalIds": {"DOI": "10.1234/test"},
                    "abstract": "We propose a new architecture...",
                },
            ],
        }
        mock_httpx, _ = _mock_httpx_client(data)

        with patch.dict(sys.modules, {"httpx": mock_httpx}):
            ctx = _make_ctx()
            result = await mod.search_papers(ctx, "transformer")

        assert "Attention Is All You Need" in result
        assert "Vaswani" in result
        assert "50000" in result

    @pytest.mark.asyncio
    async def test_search_no_results(self):
        mod = self._get_mod()
        mock_httpx, _ = _mock_httpx_client({"total": 0, "data": []})

        with patch.dict(sys.modules, {"httpx": mock_httpx}):
            ctx = _make_ctx()
            result = await mod.search_papers(ctx, "xyznonexistent")

        assert "No Semantic Scholar results" in result

    @pytest.mark.asyncio
    async def test_get_paper_details(self):
        mod = self._get_mod()
        data = {
            "paperId": "abc123",
            "title": "Test Paper",
            "authors": [{"name": "Author One"}, {"name": "Author Two"}],
            "year": 2023,
            "venue": "NeurIPS",
            "citationCount": 100,
            "url": "https://semanticscholar.org/paper/abc123",
            "abstract": "This paper presents a novel approach.",
            "externalIds": {},
            "references": [{"title": "Ref Paper 1"}, {"title": "Ref Paper 2"}],
            "citations": [{"title": "Citing Paper 1"}],
        }
        mock_httpx, _ = _mock_httpx_client(data)

        with patch.dict(sys.modules, {"httpx": mock_httpx}):
            ctx = _make_ctx()
            result = await mod.get_paper_details(ctx, "abc123")

        assert "Test Paper" in result
        assert "NeurIPS" in result
        assert "novel approach" in result
        assert "Ref Paper 1" in result
        assert "Citing Paper 1" in result

    @pytest.mark.asyncio
    async def test_optional_api_key_header(self):
        mod = self._get_mod()
        mock_httpx, mock_client = _mock_httpx_client({"total": 0, "data": []})

        with patch.dict(os.environ, {"S2_API_KEY": "test-key-123"}):
            with patch.dict(sys.modules, {"httpx": mock_httpx}):
                ctx = _make_ctx()
                await mod.search_papers(ctx, "test")

        # Verify API key was passed in headers
        call_kwargs = mock_client.get.call_args
        headers = call_kwargs.kwargs.get("headers", {})
        assert headers.get("x-api-key") == "test-key-123"


# ---------------------------------------------------------------------------
# arXiv — mock tests
# ---------------------------------------------------------------------------


def _make_mock_arxiv_result(arxiv_id="2301.12345", title="Test Paper"):
    result = MagicMock()
    result.entry_id = f"http://arxiv.org/abs/{arxiv_id}"
    result.title = title
    result.summary = "This paper presents novel methods."

    author1 = MagicMock()
    author1.name = "Author One"
    author2 = MagicMock()
    author2.name = "Author Two"
    result.authors = [author1, author2]

    pub_date = MagicMock()
    pub_date.year = 2023
    pub_date.strftime = MagicMock(return_value="2023-01-15")
    result.published = pub_date
    result.updated = pub_date

    result.categories = ["cs.CL", "cs.AI"]
    result.pdf_url = f"https://arxiv.org/pdf/{arxiv_id}"
    return result


def _mock_arxiv_module(results):
    """Create a mock arxiv module that returns given results."""
    mock_client_instance = MagicMock()
    mock_client_instance.results = MagicMock(return_value=results)

    mock_arxiv = ModuleType("arxiv")
    mock_arxiv.Client = MagicMock(return_value=mock_client_instance)
    mock_arxiv.Search = MagicMock()
    mock_arxiv.SortCriterion = MagicMock()
    mock_arxiv.SortCriterion.Relevance = "relevance"
    return mock_arxiv


class TestArxivSearch:
    def _get_mod(self):
        return _load_tools_module("searching-arxiv")

    @pytest.mark.asyncio
    async def test_search_returns_results(self):
        mod = self._get_mod()
        mock_result = _make_mock_arxiv_result()
        mock_arxiv = _mock_arxiv_module([mock_result])

        with patch.dict(sys.modules, {"arxiv": mock_arxiv}):
            ctx = _make_ctx()
            result = await mod.search_arxiv(ctx, "transformer attention")

        assert "Test Paper" in result
        assert "2301.12345" in result
        assert "cs.CL" in result

    @pytest.mark.asyncio
    async def test_search_no_results(self):
        mod = self._get_mod()
        mock_arxiv = _mock_arxiv_module([])

        with patch.dict(sys.modules, {"arxiv": mock_arxiv}):
            ctx = _make_ctx()
            result = await mod.search_arxiv(ctx, "xyznonexistent")

        assert "No arXiv results" in result

    @pytest.mark.asyncio
    async def test_get_paper(self):
        mod = self._get_mod()
        mock_result = _make_mock_arxiv_result(arxiv_id="2301.12345", title="Detailed Paper")
        mock_arxiv = _mock_arxiv_module([mock_result])

        with patch.dict(sys.modules, {"arxiv": mock_arxiv}):
            ctx = _make_ctx()
            result = await mod.get_paper(ctx, "2301.12345")

        assert "Detailed Paper" in result
        assert "novel methods" in result
        assert "Author One" in result

    @pytest.mark.asyncio
    async def test_get_paper_not_found(self):
        mod = self._get_mod()
        mock_arxiv = _mock_arxiv_module([])

        with patch.dict(sys.modules, {"arxiv": mock_arxiv}):
            ctx = _make_ctx()
            result = await mod.get_paper(ctx, "9999.99999")

        assert "No paper found" in result


# ---------------------------------------------------------------------------
# Missing dependency handling — ModelRetry
# ---------------------------------------------------------------------------


class TestMissingDependencies:
    @pytest.mark.asyncio
    async def test_duckduckgo_missing_import(self):
        """When duckduckgo-search is not installed, tool raises ModelRetry."""
        mod = _load_tools_module("searching-duckduckgo")

        # Inject None for duckduckgo_search to simulate ImportError
        with patch.dict(sys.modules, {"duckduckgo_search": None}):
            ctx = _make_ctx()
            with pytest.raises(ModelRetry, match="duckduckgo-search"):
                await mod.duckduckgo_search(ctx, "test")

    @pytest.mark.asyncio
    async def test_pubmed_missing_import(self):
        """When pymed is not installed, tool raises ModelRetry."""
        mod = _load_tools_module("searching-pubmed")

        with patch.dict(sys.modules, {"pymed": None}):
            ctx = _make_ctx()
            with pytest.raises(ModelRetry, match="pymed"):
                await mod.search_pubmed(ctx, "test")

    @pytest.mark.asyncio
    async def test_arxiv_missing_import(self):
        """When arxiv is not installed, tool raises ModelRetry."""
        mod = _load_tools_module("searching-arxiv")

        with patch.dict(sys.modules, {"arxiv": None}):
            ctx = _make_ctx()
            with pytest.raises(ModelRetry, match="arxiv"):
                await mod.search_arxiv(ctx, "test")

    @pytest.mark.asyncio
    async def test_semantic_scholar_missing_httpx(self):
        """When httpx is not installed, tool raises ModelRetry."""
        mod = _load_tools_module("searching-semantic-scholar")

        with patch.dict(sys.modules, {"httpx": None}):
            ctx = _make_ctx()
            with pytest.raises(ModelRetry, match="httpx"):
                await mod.search_papers(ctx, "test")

    @pytest.mark.asyncio
    async def test_google_missing_genai(self):
        """When google-genai is not installed, tool raises ModelRetry."""
        mod = _load_tools_module("searching-google")

        with patch.dict(sys.modules, {"google": None, "google.genai": None}):
            ctx = _make_ctx()
            with pytest.raises(ModelRetry, match="google-genai"):
                await mod.google_search(ctx, "test")


# ---------------------------------------------------------------------------
# AgenticReviewer Integration — search skills
# ---------------------------------------------------------------------------


class TestReviewerSearchSkillsIntegration:
    def test_reviewer_with_search_skills_setup(self):
        """Agentic reviewer with search skills configures toolsets."""
        r = AgenticReviewer(
            model=TestModel(),
            max_iterations=5,
            skills=["searching-duckduckgo", "searching-pubmed"],
        )
        toolsets, descs = r._setup_skills()
        # 2 user skills + 2 default skills (managing-memory, flagging-items)
        assert len(toolsets) == 4
        desc_names = {d["name"] for d in descs}
        assert "searching-duckduckgo" in desc_names
        assert "searching-pubmed" in desc_names
        assert "managing-memory" in desc_names
        assert "flagging-items" in desc_names

    def test_reviewer_with_all_search_skills(self):
        """All 5 search skills can be enabled together."""
        r = AgenticReviewer(
            model=TestModel(),
            max_iterations=5,
            skills=[
                "searching-google",
                "searching-duckduckgo",
                "searching-pubmed",
                "searching-semantic-scholar",
                "searching-arxiv",
            ],
        )
        toolsets, descs = r._setup_skills()
        # 5 user skills + 2 default skills
        assert len(toolsets) == 7
        assert len(descs) == 7

    @pytest.mark.asyncio
    async def test_review_item_with_search_skill(self, tmp_path):
        """review_item works with search skills enabled (TestModel, uses searching-content)."""
        r = AgenticReviewer(
            model=TestModel(),
            max_iterations=5,
            skills=["searching-content"],
            system_prompt="Review this item.",
        )
        response, cost = await r.review_item(
            item_text="A study about CRISPR gene editing.",
            item_id="search_test",
            working_dir=tmp_path,
        )
        assert "reasoning" in response
        assert "score" in response

    def test_search_skill_descriptions_not_in_prompt(self):
        """Skill descriptions are no longer injected into the system prompt."""
        r = AgenticReviewer(
            model=TestModel(),
            max_iterations=5,
            skills=["searching-pubmed", "searching-arxiv"],
        )
        prompt = r._build_system_prompt()
        assert "Available Skills" not in prompt


# ---------------------------------------------------------------------------
# Live Integration Tests — real API calls
# ---------------------------------------------------------------------------


class TestLiveSearchSkills:
    @pytest.mark.live
    @pytest.mark.asyncio
    async def test_live_duckduckgo_search(self, env_keys):
        """Live: DuckDuckGo search returns real results."""
        try:
            from duckduckgo_search import DDGS
        except ImportError:
            pytest.skip("duckduckgo-search not installed")

        mod = _load_tools_module("searching-duckduckgo")
        ctx = _make_ctx()
        result = await mod.duckduckgo_search(ctx, "systematic review machine learning", max_results=3)
        assert "DuckDuckGo results" in result
        assert "1." in result

    @pytest.mark.live
    @pytest.mark.asyncio
    async def test_live_pubmed_search(self, env_keys):
        """Live: PubMed search returns real results."""
        try:
            from pymed import PubMed
        except ImportError:
            pytest.skip("pymed not installed")

        mod = _load_tools_module("searching-pubmed")
        ctx = _make_ctx()
        result = await mod.search_pubmed(ctx, "CRISPR gene therapy 2024", max_results=3)
        assert "PubMed results" in result
        assert "PMID" in result

    @pytest.mark.live
    @pytest.mark.asyncio
    async def test_live_pubmed_get_abstract(self, env_keys):
        """Live: PubMed get_abstract retrieves a real abstract."""
        try:
            from pymed import PubMed
        except ImportError:
            pytest.skip("pymed not installed")

        mod = _load_tools_module("searching-pubmed")
        ctx = _make_ctx()
        result = await mod.get_abstract(ctx, "23287718")
        assert "PMID" in result or "23287718" in result

    @pytest.mark.live
    @pytest.mark.asyncio
    async def test_live_semantic_scholar_search(self, env_keys):
        """Live: Semantic Scholar search returns real results."""
        try:
            import httpx
        except ImportError:
            pytest.skip("httpx not installed")

        mod = _load_tools_module("searching-semantic-scholar")
        ctx = _make_ctx()
        result = await mod.search_papers(ctx, "attention is all you need", max_results=3)
        assert "Semantic Scholar results" in result
        assert "1." in result

    @pytest.mark.live
    @pytest.mark.asyncio
    async def test_live_arxiv_search(self, env_keys):
        """Live: arXiv search returns real results."""
        try:
            import arxiv
        except ImportError:
            pytest.skip("arxiv not installed")

        mod = _load_tools_module("searching-arxiv")
        ctx = _make_ctx()
        result = await mod.search_arxiv(ctx, "large language models", max_results=3)
        assert "arXiv results" in result
        assert "arXiv:" in result

    @pytest.mark.live
    @pytest.mark.asyncio
    async def test_live_arxiv_get_paper(self, env_keys):
        """Live: arXiv get_paper retrieves a known paper."""
        try:
            import arxiv
        except ImportError:
            pytest.skip("arxiv not installed")

        mod = _load_tools_module("searching-arxiv")
        ctx = _make_ctx()
        result = await mod.get_paper(ctx, "1706.03762")
        assert "Abstract" in result

    @pytest.mark.live
    @pytest.mark.asyncio
    async def test_live_google_search(self, env_keys):
        """Live: Google search via Gemini API returns results."""
        if not os.environ.get("GEMINI_API_KEY"):
            pytest.skip("GEMINI_API_KEY not set")

        try:
            from google import genai
        except ImportError:
            pytest.skip("google-genai not installed")

        mod = _load_tools_module("searching-google")
        ctx = _make_ctx()
        result = await mod.google_search(ctx, "systematic review methodology", max_results=3)
        assert "Google search" in result

    @pytest.mark.live
    @pytest.mark.asyncio
    async def test_live_reviewer_with_duckduckgo(self, env_keys):
        """Live: Full reviewer pipeline with DuckDuckGo search skill."""
        if not os.environ.get("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set")

        try:
            from duckduckgo_search import DDGS
        except ImportError:
            pytest.skip("duckduckgo-search not installed")

        r = AgenticReviewer(
            name="SearchTester",
            model="openai:gpt-5.4-mini",
            max_iterations=5,
            agentic_effort="high",
            skills=["searching-duckduckgo"],
            system_prompt="You are a paper reviewer. Use web search to verify claims in the text before scoring.",
            task_prompt="Verify the claims and score quality 1-10:\n\n${item}$",
            output_type=ScoringOutput,
            model_settings={"temperature": 0.0},
        )

        response, cost = await r.review_item(
            item_text=(
                "A recent study by Smith et al. (2024) found that CRISPR-Cas9 "
                "can correct sickle cell disease mutations with 95% efficiency."
            ),
            item_id="live_search_001",
        )

        assert isinstance(response["reasoning"], str)
        assert isinstance(response["score"], int)
