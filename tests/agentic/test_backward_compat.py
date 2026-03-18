"""Tests for RFD-10: Backward Compatibility + Deprecation.

Verifies that v1 imports still work, deprecation warnings are emitted,
and v1/v2 can coexist in the same script.
"""

import importlib
import sys
import warnings

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fresh_import(module_name: str):
    """Force a fresh import of a module by removing it and its submodules from sys.modules."""
    to_remove = [k for k in sys.modules if k == module_name or k.startswith(module_name + ".")]
    for k in to_remove:
        del sys.modules[k]
    return importlib.import_module(module_name)


# ---------------------------------------------------------------------------
# lattereview.agents — deprecation warning + imports still work
# ---------------------------------------------------------------------------


class TestAgentsDeprecation:
    """Verify lattereview.agents emits DeprecationWarning but remains functional."""

    def test_agents_import_emits_deprecation_warning(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _fresh_import("lattereview.agents")

        dep_warnings = [w for w in caught if issubclass(w.category, DeprecationWarning)]
        assert len(dep_warnings) >= 1
        msg = str(dep_warnings[0].message)
        assert "lattereview.agents is deprecated" in msg
        assert "lattereview.agentic" in msg

    def test_agents_warning_mentions_migration_paths(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _fresh_import("lattereview.agents")

        dep_warnings = [w for w in caught if issubclass(w.category, DeprecationWarning)]
        msg = str(dep_warnings[0].message)
        assert "ScoringReviewer" in msg
        assert "TitleAbstractReviewer" in msg
        assert "AbstractionReviewer" in msg
        assert "AgenticReviewer" in msg

    def test_basic_reviewer_importable(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            mod = _fresh_import("lattereview.agents")
        assert hasattr(mod, "BasicReviewer")

    def test_scoring_reviewer_importable(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            mod = _fresh_import("lattereview.agents")
        assert hasattr(mod, "ScoringReviewer")

    def test_title_abstract_reviewer_importable(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            mod = _fresh_import("lattereview.agents")
        assert hasattr(mod, "TitleAbstractReviewer")

    def test_abstraction_reviewer_importable(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            mod = _fresh_import("lattereview.agents")
        assert hasattr(mod, "AbstractionReviewer")


# ---------------------------------------------------------------------------
# lattereview.providers — deprecation warning + imports still work
# ---------------------------------------------------------------------------


class TestProvidersDeprecation:
    """Verify lattereview.providers emits DeprecationWarning but remains functional."""

    def test_providers_import_emits_deprecation_warning(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _fresh_import("lattereview.providers")

        dep_warnings = [w for w in caught if issubclass(w.category, DeprecationWarning)]
        assert len(dep_warnings) >= 1
        msg = str(dep_warnings[0].message)
        assert "lattereview.providers is deprecated" in msg
        assert "Pydantic AI model strings" in msg

    def test_providers_warning_mentions_model_string_examples(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _fresh_import("lattereview.providers")

        dep_warnings = [w for w in caught if issubclass(w.category, DeprecationWarning)]
        msg = str(dep_warnings[0].message)
        assert "openai:gpt-4o" in msg

    def test_openai_provider_importable(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            mod = _fresh_import("lattereview.providers")
        assert hasattr(mod, "OpenAIProvider")

    def test_google_provider_importable(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            mod = _fresh_import("lattereview.providers")
        assert hasattr(mod, "GoogleProvider")

    def test_litellm_provider_importable(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            mod = _fresh_import("lattereview.providers")
        assert hasattr(mod, "LiteLLMProvider")

    def test_ollama_provider_importable(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            mod = _fresh_import("lattereview.providers")
        assert hasattr(mod, "OllamaProvider")


# ---------------------------------------------------------------------------
# lattereview.workflows — deprecation warning + imports still work
# ---------------------------------------------------------------------------


class TestWorkflowsDeprecation:
    """Verify lattereview.workflows emits DeprecationWarning but remains functional."""

    def test_workflows_import_emits_deprecation_warning(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _fresh_import("lattereview.workflows")

        dep_warnings = [w for w in caught if issubclass(w.category, DeprecationWarning)]
        assert len(dep_warnings) >= 1
        msg = str(dep_warnings[0].message)
        assert "lattereview.workflows is deprecated" in msg
        assert "AgenticWorkflow" in msg

    def test_review_workflow_importable(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            mod = _fresh_import("lattereview.workflows")
        assert hasattr(mod, "ReviewWorkflow")


# ---------------------------------------------------------------------------
# Coexistence — v1 and v2 can be used in the same script
# ---------------------------------------------------------------------------


class TestV1V2Coexistence:
    """Verify v1 and v2 modules can coexist without interference."""

    def test_v1_agents_and_v2_agentic_coexist(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from lattereview.agents import ScoringReviewer as V1ScoringReviewer
            from lattereview.agentic import ScoringReviewer as V2ScoringReviewer

        # Both are importable and are distinct classes
        assert V1ScoringReviewer is not V2ScoringReviewer

    def test_v1_agents_and_v2_agentic_all_types_coexist(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from lattereview.agents import (
                BasicReviewer,
                ScoringReviewer,
                TitleAbstractReviewer,
                AbstractionReviewer,
            )
            from lattereview.agentic import (
                AgenticReviewer,
                ScoringReviewer as V2Scoring,
                TitleAbstractReviewer as V2TitleAbstract,
                AbstractionReviewer as V2Abstraction,
            )

        # All v1 classes exist
        assert BasicReviewer is not None
        assert ScoringReviewer is not None
        assert TitleAbstractReviewer is not None
        assert AbstractionReviewer is not None

        # All v2 classes exist
        assert AgenticReviewer is not None
        assert V2Scoring is not None
        assert V2TitleAbstract is not None
        assert V2Abstraction is not None

    def test_v1_workflows_and_v2_workflow_coexist(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from lattereview.workflows import ReviewWorkflow
            from lattereview.agentic import AgenticWorkflow

        assert ReviewWorkflow is not AgenticWorkflow

    def test_v1_providers_and_v2_model_strings_coexist(self):
        """v1 providers and v2 (which uses model strings, not provider classes) coexist."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from lattereview.providers import OpenAIProvider
            from lattereview.agentic import AgenticReviewer

        # v1 provider is a class, v2 reviewer accepts model strings
        assert OpenAIProvider is not None
        assert AgenticReviewer is not None


# ---------------------------------------------------------------------------
# Warning type correctness
# ---------------------------------------------------------------------------


class TestWarningTypes:
    """Verify warnings use the correct category for filtering."""

    @pytest.mark.parametrize(
        "module_name",
        ["lattereview.agents", "lattereview.providers", "lattereview.workflows"],
    )
    def test_warning_is_deprecation_warning_type(self, module_name):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _fresh_import(module_name)

        dep_warnings = [w for w in caught if issubclass(w.category, DeprecationWarning)]
        assert len(dep_warnings) >= 1, f"No DeprecationWarning from {module_name}"

    @pytest.mark.parametrize(
        "module_name",
        ["lattereview.agents", "lattereview.providers", "lattereview.workflows"],
    )
    def test_warning_mentions_v3_removal(self, module_name):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _fresh_import(module_name)

        dep_warnings = [w for w in caught if issubclass(w.category, DeprecationWarning)]
        msg = str(dep_warnings[0].message)
        assert "v3.0" in msg, f"Warning from {module_name} should mention v3.0 removal"
