"""Unit tests for SystemOneProvider and the question/answer models (no network)."""

import asyncio

import httpx
import pytest

from lattereview.providers import SystemOneProvider, Noul, Choice, Score
from lattereview.providers.base_provider import ClientCreationError, ProviderError
from lattereview.providers.system_one_provider import SystemOneResponseError

from systemone_fixtures import RESPONSES, REQUEST_QUESTIONS


def ok(data):
    return httpx.Response(200, json=data)


# --- Questions ---------------------------------------------------------------------------------------------------


def test_question_payloads():
    assert Noul("Is it an RCT?").to_payload() == {"type": "noul", "instructions": "Is it an RCT?"}
    assert Noul("Is it an RCT?", true="yes", false="no").to_payload()["criteria"] == {"true": "yes", "false": "no"}
    assert Noul("Is it an RCT?", true="yes").to_payload()["criteria"] == {"true": "yes"}
    assert Choice("Modality?", ["CT", "MRI"]).to_payload() == {
        "type": "choice",
        "instructions": "Modality?",
        "criteria": {"CT": "CT", "MRI": "MRI"},
    }
    assert Choice(instructions="Modality?", options={"CT": "computed tomography", "MRI": "MR"}).options["CT"] == (
        "computed tomography"
    )
    assert Score("Quality?", ["low", "high"]).to_payload() == {
        "type": "score",
        "instructions": "Quality?",
        "criteria": ["low", "high"],
    }
    structured = Score({"task": "rate", "focus": "methods"}, [{"level": "low"}, {"level": "high"}])
    assert structured.to_payload()["instructions"] == {"task": "rate", "focus": "methods"}


@pytest.mark.parametrize(
    "build",
    [
        lambda: Noul(""),
        lambda: Noul("   "),
        lambda: Choice("Modality?", ["CT"]),
        lambda: Choice("Modality?", ["CT", "CT"]),
        lambda: Score("Quality?", ["only"]),
        lambda: Score("Quality?", [str(i) for i in range(11)]),
        lambda: Score("Quality?", ["low", "low"]),
        lambda: Noul("a", "b"),
        lambda: Choice("Modality?", ["CT", "MRI"], options=["CT", "MRI"]),
    ],
)
def test_question_validation(build):
    with pytest.raises((ValueError, TypeError)):
        build()


# --- Configuration -----------------------------------------------------------------------------------------------


def test_backend_presets(monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "ts-key")
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")
    typesafe = SystemOneProvider()
    assert typesafe.endpoint == "https://api.typesafe.ai/v1/systemone"
    assert (typesafe.model, typesafe.api_key, typesafe.input_price_per_million) == ("jev-latest", "ts-key", 0.042)
    openrouter = SystemOneProvider(backend="openrouter", model="typesafe/jev-1.13")
    assert openrouter.endpoint == "https://openrouter.ai/api/v1/systemone"
    assert (openrouter.model, openrouter.api_key) == ("typesafe/jev-1.13", "or-key")
    assert SystemOneProvider(api_key="explicit").api_key == "explicit"


def test_perplexity_and_openai_presets(monkeypatch):
    monkeypatch.setenv("PERPLEXITY_API_KEY", "pplx-key")
    monkeypatch.setenv("OPENAI_API_KEY", "oa-key")
    perplexity = SystemOneProvider(backend="perplexity")
    assert perplexity.endpoint == "https://api.perplexity.ai/v1/decisions" and perplexity.protocol == "systemone"
    assert (perplexity.model, perplexity.api_key, perplexity.input_price_per_million) == (
        "pplx-decider-v1.1-27b",
        "pplx-key",
        0.02,
    )
    assert perplexity.requests_per_minute == 500
    openai = SystemOneProvider(backend="openai")
    assert openai.endpoint == "https://api.openai.com/v1/decisions" and openai.protocol == "openai"
    assert (openai.model, openai.api_key, openai.input_price_per_million) == ("gpt-6-luna", "oa-key", 0.10)
    assert openai.requests_per_minute == 0


def test_perplexity_key_falls_back_to_litellm_name(monkeypatch):
    monkeypatch.delenv("PERPLEXITY_API_KEY", raising=False)
    monkeypatch.setenv("PERPLEXITYAI_API_KEY", "litellm-style-key")
    assert SystemOneProvider(backend="perplexity").api_key == "litellm-style-key"
    monkeypatch.delenv("PERPLEXITYAI_API_KEY")
    with pytest.raises(ClientCreationError, match="PERPLEXITY_API_KEY"):
        SystemOneProvider(backend="perplexity")


def test_protocol_applies_only_to_custom_servers(monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "ts-key")
    with pytest.raises(ClientCreationError, match="only to a custom base_url"):
        SystemOneProvider(backend="typesafe", protocol="openai")
    assert SystemOneProvider(backend="typesafe", protocol="systemone").protocol == "systemone"


@pytest.mark.parametrize(
    "base_url, protocol, endpoint",
    [
        ("https://gateway.example", "openai", "https://gateway.example/v1/decisions"),
        ("https://eu.gateway.example/v1/decisions/", "openai", "https://eu.gateway.example/v1/decisions"),
        ("http://localhost:8000/v1/decisions", None, "http://localhost:8000/v1/decisions"),  # a self-hosted decider
    ],
)
def test_custom_decisions_urls(base_url, protocol, endpoint):
    provider = SystemOneProvider(base_url=base_url, protocol=protocol)
    assert provider.endpoint == endpoint and provider.protocol == (protocol or "systemone")


@pytest.mark.parametrize(
    "base_url",
    [
        "http://localhost:3000",
        "http://localhost:3000/",
        "http://localhost:3000/v1/systemone",
        "http://localhost:3000/v1/systemone/",
    ],
)
def test_custom_base_url(monkeypatch, base_url):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    provider = SystemOneProvider(base_url=base_url)
    assert provider.endpoint == "http://localhost:3000/v1/systemone"
    assert provider.model is None and provider.api_key is None
    assert provider.input_price_per_million == 0.0


def test_missing_key_is_a_clear_error(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(ClientCreationError, match="OPENROUTER_API_KEY"):
        SystemOneProvider(backend="openrouter")


def test_api_key_hidden_from_repr(monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "secret-value")
    assert "secret-value" not in repr(SystemOneProvider())


# --- Requests and normalization ----------------------------------------------------------------------------------


async def test_typesafe_response_normalized(make_provider):
    provider, recorder = make_provider(ok(RESPONSES["typesafe"]))
    result = await provider.decide("Title: a study", REQUEST_QUESTIONS)

    request = recorder.requests[0]
    assert str(request.url) == "https://api.typesafe.ai/v1/systemone"
    assert request.headers["authorization"] == "Bearer test-typesafe-key"
    body = recorder.bodies[0]
    assert body["model"] == "jev-latest" and body["state"] == "Title: a study"
    assert body["questions"]["mod"]["criteria"]["XR"] == "radiography"

    dl, mod, q, plain = (result.answers[k] for k in ("dl", "mod", "q", "plain"))
    assert (dl.type, dl.value, dl.confidence, dl.probabilities) == ("noul", 0.99, None, None)
    assert (mod.value, mod.label, mod.confidence) == ("XR", "XR", 1.0)
    assert list(mod.probabilities) == ["CT", "MRI", "XR", "US"]  # the question's order, not the response's
    assert (q.value, q.level, q.label) == (2.0, 2, "external")
    assert q.probabilities == {"none": 0.0, "internal only": 0.0, "external": 1.0}
    assert plain.value == 0.01
    assert result.model == "jev-1.13.0" and result.input_tokens == 478
    assert result.cost == pytest.approx(478 * 0.042 / 1e6)
    assert result.raw == RESPONSES["typesafe"]


async def test_openrouter_response_normalized(make_provider):
    provider, recorder = make_provider(ok(RESPONSES["openrouter"]), backend="openrouter")
    result = await provider.decide("Title: a study", REQUEST_QUESTIONS)
    assert str(recorder.requests[0].url) == "https://openrouter.ai/api/v1/systemone"
    assert recorder.bodies[0]["model"] == "~typesafe/jev-latest"
    q = result.answers["q"]
    assert isinstance(q.value, float) and q.value == 2.0  # whole-number score normalized to float
    assert q.label == "external" and q.confidence == 1.0
    assert result.cost == RESPONSES["openrouter"]["usage"]["cost"]  # reported cost wins over the token price


async def test_perplexity_response_normalized(make_provider):
    """Perplexity speaks the System One protocol at /v1/decisions; without a reported cost, its list price is used."""
    response = {key: value for key, value in RESPONSES["perplexity_openrouter"].items() if key != "usage"}
    response["usage"] = {"input_tokens": 591, "output_tokens": 4}  # Perplexity's own usage has no cost
    provider, recorder = make_provider(ok(response), backend="perplexity")
    result = await provider.decide("Title: a study", REQUEST_QUESTIONS)
    assert str(recorder.requests[0].url) == "https://api.perplexity.ai/v1/decisions"
    assert recorder.requests[0].headers["authorization"] == "Bearer test-perplexity-key"
    body = recorder.bodies[0]
    assert set(body) == {"model", "state", "questions"}  # Perplexity rejects unknown top-level fields
    assert body["model"] == "pplx-decider-v1.1-27b" and body["questions"]["q"]["criteria"][0] == "none"
    mod, q = result.answers["mod"], result.answers["q"]
    assert mod.value == "XR" and list(mod.probabilities) == ["CT", "MRI", "XR", "US"]
    assert (q.label, q.level) == ("external", 2) and 0.99 < q.confidence < 1
    assert result.cost == pytest.approx(591 * 0.02 / 1e6)


async def test_perplexity_decider_on_openrouter(make_provider):
    provider, recorder = make_provider(
        ok(RESPONSES["perplexity_openrouter"]), backend="openrouter", model="perplexity/pplx-decider-v1.1-27b"
    )
    result = await provider.decide("Title: a study", REQUEST_QUESTIONS)
    assert recorder.bodies[0]["model"] == "perplexity/pplx-decider-v1.1-27b"
    assert result.answers["dl"].value > 0.99 and result.answers["plain"].value < 0.01
    assert result.cost == RESPONSES["perplexity_openrouter"]["usage"]["cost"]


async def test_openai_request_translated(make_provider):
    provider, recorder = make_provider(ok(RESPONSES["openai"]), backend="openai")
    await provider.decide("Title: a study", REQUEST_QUESTIONS)
    assert str(recorder.requests[0].url) == "https://api.openai.com/v1/decisions"
    assert recorder.requests[0].headers["authorization"] == "Bearer test-openai-key"
    body = recorder.bodies[0]
    assert set(body) == {"model", "input", "questions"} and body["model"] == "gpt-6-luna"
    assert body["input"] == "Title: a study"
    dl, mod, q, plain = body["questions"]
    assert dl == {
        "type": "predicate",
        "name": "dl",
        "instructions": "Does the study use deep learning?\nAnswer true if: uses deep learning\nAnswer false if: does not",
    }
    assert mod["type"] == "choice" and mod["choices"][2] == {"value": "XR", "description": "radiography"}
    assert q["type"] == "score" and q["levels"] == [
        {"label": "none"},
        {"label": "internal only"},
        {"label": "external"},
    ]
    assert plain == {"type": "predicate", "name": "plain", "instructions": "Is this a randomized controlled trial?"}


async def test_openai_request_serializes_json_and_skips_redundant_descriptions(make_provider):
    provider, recorder = make_provider(ok(RESPONSES["openai"]), backend="openai")
    questions = {
        "organ": Choice("Organ?", ["brain", "lung"]),
        "level": Score({"task": "rate"}, [{"level": "low"}, {"level": "high"}]),
    }
    with pytest.raises(SystemOneResponseError):  # the recorded answers belong to other questions
        await provider.decide({"item": "text", "additional_context": "ctx"}, questions)
    body = recorder.bodies[0]
    assert body["input"] == '{"item": "text", "additional_context": "ctx"}'
    organ, level = body["questions"]
    assert organ["choices"] == [{"value": "brain"}, {"value": "lung"}]
    assert level["instructions"] == '{"task": "rate"}' and level["levels"][0] == {"label": '{"level": "low"}'}


async def test_openai_response_normalized(make_provider):
    provider, _ = make_provider(ok(RESPONSES["openai"]), backend="openai")
    result = await provider.decide("Title: a study", REQUEST_QUESTIONS)
    dl, mod, q, plain = (result.answers[k] for k in ("dl", "mod", "q", "plain"))
    assert (dl.type, dl.value, dl.confidence, dl.refused) == ("noul", 1.0, None, False)
    assert (mod.value, mod.label, mod.confidence) == ("XR", "XR", 1.0)
    assert mod.probabilities == {"CT": 0.0, "MRI": 0.0, "XR": 1.0, "US": 0.0}
    assert (q.value, q.level, q.label, q.confidence) == (2.0, 2, "external", 1.0)
    assert list(q.probabilities) == ["none", "internal only", "external"]
    assert plain.value == 0.04
    assert result.model == "gpt-6-luna" and result.input_tokens == 601
    assert result.cost == pytest.approx(601 * 0.10 / 1e6)  # OpenAI reports no cost
    assert result.raw == RESPONSES["openai"]  # the untranslated response


async def test_openai_refusal_is_an_answer_not_an_error(make_provider):
    provider, _ = make_provider(ok(RESPONSES["openai_refusal"]), backend="openai")
    result = await provider.decide("text", {"complete": Noul("Is it complete?"), "plain": Noul("Is it English?")})
    complete, plain = result.answers["complete"], result.answers["plain"]
    assert (complete.type, complete.value, complete.refused) == ("noul", None, True)
    assert (plain.value, plain.refused) == (1.0, False)


async def test_openai_answers_without_names_match_by_position(make_provider):
    response = {
        "answers": [{"type": "predicate", "probability": 0.3}, {"type": "choice", "choice": "MRI"}],
        "usage": {"input_tokens": 10},
    }
    provider, _ = make_provider(ok(response), backend="openai")
    result = await provider.decide("text", {"a": Noul("A?"), "b": Choice("Modality?", ["CT", "MRI"])})
    assert (result.answers["a"].value, result.answers["b"].value) == (0.3, "MRI")
    assert result.answers["b"].probabilities is None


@pytest.mark.parametrize(
    "response",
    [
        {"answers": {"dl": {"type": "predicate", "probability": 0.9}}},  # not a list
        {"answers": [{"type": "predicate", "name": "dl"}]},  # predicate without a probability
        {"answers": [{"type": "choice", "name": "dl", "probabilities": [{"value": "CT"}]}]},  # malformed probabilities
        {"answers": [{"type": "predicate", "name": "other", "probability": 0.9}]},  # the question is not answered
        {"answers": [{"type": "mystery", "name": "dl"}]},  # unknown answer type
    ],
)
async def test_malformed_openai_responses(make_provider, response):
    provider, _ = make_provider(ok(response), backend="openai")
    with pytest.raises(SystemOneResponseError) as error:
        await provider.decide("state", {"dl": Noul("Deep learning?")})
    assert error.value.status_code == 200


async def test_openai_errors_are_readable_and_not_retried(make_provider):
    body = {"error": {"message": "The model `gpt-6-luna-nope` does not exist.", "type": "invalid_request_error"}}
    provider, recorder = make_provider(httpx.Response(404, json=body), backend="openai")
    with pytest.raises(SystemOneResponseError, match="HTTP 404: The model `gpt-6-luna-nope` does not exist"):
        await provider.decide("state", {"dl": Noul("Deep learning?")})
    assert len(recorder.requests) == 1


async def test_clone_style_response_without_legend_or_confidence(make_provider):
    """Self-hosted clones may omit legend, confidence, choice, or cost; answers must still normalize."""
    response = {
        "answers": {
            "dl": {"type": "noul", "noul": 0.7},
            "mod": {"type": "choice", "probabilities": {"CT": 0.1, "MRI": 0.2, "XR": 0.6, "US": 0.1}},
            "q": {"type": "score", "probabilities": {"0": 0.2, "1": 0.5, "2": 0.3}},
            "plain": {"type": "noul", "noul": 0.1},
        },
        "usage": {"input_tokens": 500},
    }
    provider, recorder = make_provider(ok(response), base_url="http://localhost:3000")
    result = await provider.decide("state", REQUEST_QUESTIONS)
    assert "model" not in recorder.bodies[0]
    assert "authorization" not in recorder.requests[0].headers
    mod, q = result.answers["mod"], result.answers["q"]
    assert (mod.value, mod.confidence) == ("XR", None)
    assert q.value == pytest.approx(1.1) and (q.level, q.label) == (1, "internal only")
    assert q.probabilities == {"none": 0.2, "internal only": 0.5, "external": 0.3}
    assert result.cost == 0.0  # custom URL, no price set


async def test_score_without_probabilities_uses_rounded_score(make_provider):
    response = {"answers": {"q": {"type": "score", "score": 1.4}}}
    provider, _ = make_provider(ok(response), base_url="http://localhost:3000")
    q = (await provider.decide("state", {"q": REQUEST_QUESTIONS["q"]})).answers["q"]
    assert (q.value, q.level, q.label, q.probabilities) == (1.4, 1, "internal only", None)


async def test_custom_price_for_paid_gateways(make_provider):
    response = {"answers": {"plain": {"type": "noul", "noul": 0.5}}, "usage": {"input_tokens": 1_000_000}}
    provider, _ = make_provider(ok(response), base_url="https://gateway.example", input_price_per_million=0.05)
    result = await provider.decide("state", {"plain": Noul("Is it?")})
    assert result.cost == pytest.approx(0.05)


async def test_raw_dict_questions_and_structured_state(make_provider):
    response = {"answers": {"a": {"type": "choice", "choice": "CT"}, "b": {"type": "score", "score": 0}}}
    provider, recorder = make_provider(ok(response))
    questions = {
        "a": {"type": "choice", "instructions": "Modality?", "criteria": {"CT": "ct", "MRI": "mr"}},
        "b": {"type": "score", "instructions": "Quality?", "criteria": ["low", "high"]},
    }
    result = await provider.decide({"item": "text", "additional_context": "ctx"}, questions)
    assert recorder.bodies[0]["state"] == {"item": "text", "additional_context": "ctx"}
    assert recorder.bodies[0]["questions"] == questions
    assert result.answers["a"].value == "CT" and result.answers["b"].label == "low"


@pytest.mark.parametrize(
    "response",
    [
        {"answers": {"dl": {"type": "noul", "noul": 0.9}}},  # a question is missing
        {"result": {}},  # no answers at all
        {"answers": {"dl": {"type": "noul"}, "mod": {}, "q": {}, "plain": {}}},  # unreadable answers
        {"answers": {"dl": {"type": "noul", "noul": 0.9}, "mod": {"type": "choice"}}},  # choice without an answer
        {"answers": {"dl": {"type": "noul", "noul": 0.9}, "mod": {"choice": "CT"}, "q": {"type": "score"}}},  # score
    ],
)
async def test_malformed_responses(make_provider, response):
    provider, _ = make_provider(ok(response))
    with pytest.raises(SystemOneResponseError) as error:
        await provider.decide("state", REQUEST_QUESTIONS)
    assert error.value.retryable


async def test_invalid_inputs(make_provider):
    provider, recorder = make_provider(ok({}))
    for state, questions in [("", REQUEST_QUESTIONS), ("state", {}), ("state", {"x": "not a question"})]:
        with pytest.raises(ProviderError):
            await provider.decide(state, questions)
    with pytest.raises(ProviderError, match="unknown type"):
        await provider.decide("state", {"x": {"type": "bogus", "instructions": "x"}})
    assert recorder.requests == []


# --- Retries and errors ------------------------------------------------------------------------------------------


@pytest.mark.parametrize("status", [429, 529, 500, 503])
async def test_transient_errors_are_retried(make_provider, status):
    provider, recorder = make_provider(httpx.Response(status, json={"detail": "busy"}), ok(RESPONSES["typesafe"]))
    result = await provider.decide("state", REQUEST_QUESTIONS)
    assert len(recorder.requests) == 2 and result.answers["mod"].value == "XR"


async def test_timeouts_are_retried(make_provider):
    provider, recorder = make_provider(httpx.ReadTimeout("slow"), ok(RESPONSES["typesafe"]))
    await provider.decide("state", REQUEST_QUESTIONS)
    assert len(recorder.requests) == 2


async def test_retry_after_is_honored(make_provider, monkeypatch):
    delays = []

    async def record_sleep(delay):
        delays.append(delay)

    monkeypatch.setattr("lattereview.providers.system_one_provider.asyncio.sleep", record_sleep)
    rate_limited = httpx.Response(429, headers={"Retry-After": "7"}, json={"detail": "slow down"})
    provider, _ = make_provider(rate_limited, ok(RESPONSES["typesafe"]), requests_per_minute=0)
    await provider.decide("state", REQUEST_QUESTIONS)
    assert delays == [7.0]


async def test_requests_are_paced(make_provider, monkeypatch):
    delays = []

    async def record_sleep(delay):
        delays.append(delay)

    monkeypatch.setattr("lattereview.providers.system_one_provider.asyncio.sleep", record_sleep)
    provider, recorder = make_provider(ok(RESPONSES["typesafe"]), requests_per_minute=600)
    await asyncio.gather(*(provider.decide("state", REQUEST_QUESTIONS) for _ in range(3)))
    assert len(recorder.requests) == 3
    assert sorted(delays) == pytest.approx([0.1, 0.2], abs=0.02)  # the first request goes at once


def test_pacing_defaults(monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "key")
    assert SystemOneProvider().requests_per_minute == 1000
    assert SystemOneProvider(requests_per_minute=0).requests_per_minute == 0
    assert SystemOneProvider(base_url="http://localhost:3000").requests_per_minute is None


async def test_retries_give_up_after_max_retries(make_provider):
    provider, recorder = make_provider(httpx.Response(529, json={"detail": "overloaded"}), max_retries=2)
    with pytest.raises(SystemOneResponseError, match="HTTP 529") as error:
        await provider.decide("state", REQUEST_QUESTIONS)
    assert len(recorder.requests) == 3 and error.value.retryable


@pytest.mark.parametrize(
    "status, body, message",
    [
        (401, RESPONSES["typesafe_401"], "Cannot authenticate"),
        (422, RESPONSES["fastapi_422"], "body.questions.a.choice.criteria: Input should be a valid dictionary"),
        (400, RESPONSES["openrouter_400"], "Invalid discriminator value"),
        (400, {"detail": {"error_type": "api_usage_error", "message": "Unknown model: jev-nope"}}, "Unknown model"),
    ],
)
async def test_client_errors_are_not_retried(make_provider, status, body, message):
    provider, recorder = make_provider(httpx.Response(status, json=body))
    with pytest.raises(SystemOneResponseError, match=message) as error:
        await provider.decide("state", REQUEST_QUESTIONS)
    assert len(recorder.requests) == 1
    assert error.value.status_code == status and not error.value.retryable


async def test_non_json_error_body(make_provider):
    provider, _ = make_provider(httpx.Response(404, text="Not Found"))
    with pytest.raises(SystemOneResponseError, match="HTTP 404: Not Found"):
        await provider.decide("state", REQUEST_QUESTIONS)


# --- Client lifecycle --------------------------------------------------------------------------------------------


def test_client_is_recreated_for_each_event_loop(make_provider):
    """Scripts that call asyncio.run() twice must not reuse a client bound to a closed loop."""
    provider, recorder = make_provider(ok(RESPONSES["typesafe"]))
    clients = []

    async def run():
        await provider.decide("state", REQUEST_QUESTIONS)
        clients.append(provider._client)

    asyncio.run(run())
    asyncio.run(run())
    assert len(recorder.requests) == 2 and clients[0] is not clients[1]


async def test_client_is_shared_and_closable(make_provider):
    provider, _ = make_provider(ok(RESPONSES["typesafe"]))
    await asyncio.gather(*(provider.decide("state", REQUEST_QUESTIONS) for _ in range(3)))
    client = provider._client
    await provider.decide("state", REQUEST_QUESTIONS)
    assert provider._client is client
    await provider.aclose()
    assert client.is_closed and provider._client is None
    await provider.decide("state", REQUEST_QUESTIONS)  # reopens transparently
    await provider.aclose()


async def test_cost_reported_as_a_breakdown(make_provider):
    response = {
        "answers": {"plain": {"type": "noul", "noul": 0.5}},
        "usage": {"input_tokens": 10, "cost": {"total_cost": 0.003}},
    }
    provider, _ = make_provider(ok(response), backend="perplexity")
    assert (await provider.decide("state", {"plain": Noul("Is it?")})).cost == 0.003
