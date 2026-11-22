import copy
import json
from dataclasses import replace

import pytest

from lm15 import (
    AsyncLMRouter,
    Config,
    DataPart,
    ImagePart,
    LMRouter,
    Message,
    OpenAIDecisionsLM,
    ProviderError,
    Request,
    RouterConfig,
    UnsupportedFeatureError,
    bounding_box,
    choice,
    judgments,
    score,
    yes_no,
)
from lm15.testing import FakeResponse, FakeTransport

REQUEST = Request(
    model="openai-decisions:gpt-6-luna",
    messages=(Message.user(DataPart(value={"ticket": "Charged twice"})),),
    config=Config(
        response_format=judgments(
            urgent=yes_no("Is it urgent?"),
            department=choice("Which department?", ["billing", "other"]),
            severity=score("How severe?", ["low", "medium", "high"]),
        )
    ),
)
BODY = {
    "model": "gpt-6-luna",
    "answers": [
        {"name": "urgent", "type": "predicate", "probability": 0.8},
        {
            "name": "department",
            "type": "choice",
            "choice": "billing",
            "confidence": 0.9,
            "probabilities": [{"value": "billing", "probability": 0.9}, {"value": "other", "probability": 0.1}],
        },
        {
            "name": "severity",
            "type": "score",
            "score": 1.6,
            "confidence": 0.7,
            "probabilities": [
                {"value": 0, "label": "0", "probability": 0.1},
                {"value": 1, "label": "1", "probability": 0.2},
                {"value": 2, "label": "2", "probability": 0.7},
            ],
        },
    ],
    "usage": {
        "input_tokens": 100,
        "output_tokens": 0,
        "total_tokens": 100,
        "input_tokens_details": {"cached_tokens": 20, "cache_write_tokens": 10},
    },
}


def transport(body=BODY, status=200):
    return FakeTransport(
        [FakeResponse(status=status, body=json.dumps(body).encode(), headers=[("x-request-id", "req-test")])]
    )


def test_router_wire_and_all_answer_types():
    wire = transport()
    router = LMRouter(RouterConfig(api_keys={"openai-decisions": "test-key"}, transport=wire))
    response = router.complete(REQUEST)
    sent = wire.requests[0]
    assert sent.url == "https://api.openai.com/v1/decisions"
    payload = json.loads(sent.body)
    assert payload["model"] == "gpt-6-luna"
    assert payload["questions"][1]["choices"] == [{"value": "billing"}, {"value": "other"}]
    assert payload["questions"][2]["levels"][1] == {"label": "1", "description": "medium"}
    assert response.data == {"urgent": True, "department": "billing", "severity": 2}
    assert response.expected("severity") == pytest.approx(1.6)
    assert response.id == "req-test"
    assert response.usage.cache_read_tokens == 20
    assert response.usage.cache_write_tokens == 10
    assert response.usage.output_tokens == 0
    assert response.provider_data["openai-decisions"] == BODY
    assert response.provider_data["decisions"]["answers"]["urgent"] == {"noul": 0.8}


def test_structured_overrides_and_image_input():
    request = replace(
        REQUEST,
        messages=(Message.user(["Inspect", ImagePart(media_type="image/png", url="https://example.com/image.png")]),),
        config=replace(
            REQUEST.config,
            extensions={
                "questions": {
                    "urgent": {"type": "noul", "instructions": {"rule": "Urgent?"}, "criteria": {"true": "today"}},
                    "department": {"criteria": {"billing": {"examples": ["refund"]}, "other": None}},
                }
            },
        ),
    )
    payload = json.loads(OpenAIDecisionsLM(api_key="test").build_request(request, False).body)
    assert payload["input"][0]["content"][1]["image_url"] == "https://example.com/image.png"
    assert json.loads(payload["questions"][0]["instructions"])["criteria"] == {"true": "today"}
    assert json.loads(payload["questions"][1]["choices"][0]["description"]) == {"examples": ["refund"]}


@pytest.mark.parametrize("field,value", [("max_tokens", 10), ("store", False)])
def test_rejects_generation_settings_before_transport(field, value):
    wire = transport()
    lm = OpenAIDecisionsLM(api_key="test", transport=wire)
    with pytest.raises(UnsupportedFeatureError):
        lm.complete(replace(REQUEST, config=replace(REQUEST.config, **{field: value})))
    assert not wire.requests


def test_rejects_boxes_and_streaming():
    lm = OpenAIDecisionsLM(api_key="test")
    with pytest.raises(UnsupportedFeatureError, match="bounding boxes"):
        lm.build_request(replace(REQUEST, config=Config(response_format=judgments(box=bounding_box("Locate")))), False)
    with pytest.raises(UnsupportedFeatureError, match="streaming"):
        lm.build_request(REQUEST, True)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda b: b["answers"].pop(),
        lambda b: b["answers"][0].update(name="wrong"),
        lambda b: b["answers"][0].update(probability=float("nan")),
        lambda b: b["answers"][0].update(probability=True),
        lambda b: b["answers"][1]["probabilities"][1].update(value="billing"),
        lambda b: b["answers"][1].update(confidence=2),
        lambda b: b["answers"][2]["probabilities"][0].update(value=False),
        lambda b: b["answers"][2]["probabilities"][0].update(label="wrong"),
        lambda b: b["answers"][2].update(score=10**400),
    ],
)
def test_rejects_malformed_answers(mutation):
    body = copy.deepcopy(BODY)
    mutation(body)
    with pytest.raises(ProviderError, match="Malformed OpenAI Decisions"):
        OpenAIDecisionsLM(api_key="test", transport=transport(body)).complete(REQUEST)


def test_refusal_preserves_other_answers_and_usage():
    body = copy.deepcopy(BODY)
    body["answers"][0] = {"name": "urgent", "type": "refusal"}
    response = OpenAIDecisionsLM(api_key="test", transport=transport(body)).complete(REQUEST)
    assert response.finish_reason == "content_filter"
    assert "urgent" not in response.data
    assert response.data["department"] == "billing"
    assert response.usage.input_tokens == 100
    assert response.provider_data["decisions"]["refusals"] == ["urgent"]


@pytest.mark.asyncio
async def test_async_router_uses_same_codec():
    from tests.test_router import FakeAsyncResponse, FakeAsyncTransport

    wire = FakeAsyncTransport([FakeAsyncResponse(200, json.dumps(BODY).encode())])
    router = AsyncLMRouter(RouterConfig(api_keys={"openai-decisions": "test"}, transport=wire))
    response = await router.complete(REQUEST)
    assert response.data["department"] == "billing"


@pytest.mark.parametrize("policy", ["note", "silent", "refuse"])
def test_sampling_hints_follow_adaptation_policy(policy):
    wire = transport()
    lm = OpenAIDecisionsLM(api_key="test", transport=wire, adaptations=policy)
    request = replace(REQUEST, config=replace(REQUEST.config, temperature=0.2))
    if policy == "refuse":
        with pytest.raises(UnsupportedFeatureError, match="config.temperature"):
            lm.complete(request)
        assert not wire.requests
    else:
        response = lm.complete(request)
        assert "temperature" not in json.loads(wire.requests[0].body)
        assert response.data["urgent"] is True
        if policy == "note":
            assert [(a.field, a.action) for a in response.adaptations] == [("config.temperature", "dropped")]
        else:
            assert response.adaptations == ()
