"""Tests for output_models.py — Pydantic output model definitions and dynamic model builder."""

import pytest
from pydantic import BaseModel, Field, ValidationError

from lattereview.agentic.output_models import (
    ScoringOutput,
    EvaluationOutput,
    AbstractionOutput,
    build_dynamic_output_model,
)


class TestScoringOutput:
    def test_valid_scoring_output(self):
        output = ScoringOutput(reasoning="Good paper", score=8, certainty=85)
        assert output.reasoning == "Good paper"
        assert output.score == 8
        assert output.certainty == 85

    def test_certainty_bounds(self):
        # Valid boundaries
        ScoringOutput(reasoning="x", score=1, certainty=0)
        ScoringOutput(reasoning="x", score=1, certainty=100)

        # Out of bounds
        with pytest.raises(ValidationError):
            ScoringOutput(reasoning="x", score=1, certainty=-1)
        with pytest.raises(ValidationError):
            ScoringOutput(reasoning="x", score=1, certainty=101)

    def test_model_dump(self):
        output = ScoringOutput(reasoning="test", score=5, certainty=50)
        d = output.model_dump()
        assert d == {"reasoning": "test", "score": 5, "certainty": 50}

    def test_model_json_schema(self):
        schema = ScoringOutput.model_json_schema()
        assert "reasoning" in schema["properties"]
        assert "score" in schema["properties"]
        assert "certainty" in schema["properties"]


class TestEvaluationOutput:
    def test_valid_evaluation(self):
        output = EvaluationOutput(reasoning="Relevant study", decision="include", certainty=90)
        assert output.decision == "include"

    def test_model_dump(self):
        output = EvaluationOutput(reasoning="x", decision="exclude", certainty=75)
        d = output.model_dump()
        assert d["decision"] == "exclude"


class TestAbstractionOutput:
    def test_valid_abstraction(self):
        output = AbstractionOutput(
            reasoning="Extracted key findings",
            extracted_data={"sample_size": 100, "method": "RCT"},
        )
        assert output.extracted_data["sample_size"] == 100


class TestBuildDynamicOutputModel:
    def test_simple_fields(self):
        Model = build_dynamic_output_model("TestModel", {"score": int, "reasoning": str})
        instance = Model(score=5, reasoning="test")
        assert instance.score == 5
        assert instance.reasoning == "test"

    def test_with_field_constraints(self):
        Model = build_dynamic_output_model(
            "ConstrainedModel",
            {
                "score": (int, Field(ge=0, le=10)),
                "reasoning": (str, Field(description="Why this score")),
            },
        )
        instance = Model(score=7, reasoning="good")
        assert instance.score == 7

        with pytest.raises(ValidationError):
            Model(score=11, reasoning="bad")

    def test_dynamic_model_is_basemodel(self):
        Model = build_dynamic_output_model("MyModel", {"x": int})
        assert issubclass(Model, BaseModel)

    def test_model_dump(self):
        Model = build_dynamic_output_model("DumpModel", {"a": str, "b": int})
        d = Model(a="hello", b=42).model_dump()
        assert d == {"a": "hello", "b": 42}

    def test_model_json_schema(self):
        Model = build_dynamic_output_model(
            "SchemaModel",
            {"name": (str, Field(description="The name")), "value": int},
        )
        schema = Model.model_json_schema()
        assert "name" in schema["properties"]
        assert schema["properties"]["name"]["description"] == "The name"
