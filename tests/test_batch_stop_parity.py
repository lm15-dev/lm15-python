"""Source-only batch stop regressions; run explicitly with pytest when permitted."""
import asyncio
import json

import pytest

from lm15 import BatchRequest, Config, Message, Request, UnsupportedFeatureError
from lm15.providers import AnthropicLM, GeminiLM, OpenAILM
from lm15.providers.async_base import AsyncOpenAILM


class NoTransport:
    def stream(self, request):
        raise AssertionError("batch touched transport before refusal")


def no_credential():
    raise AssertionError("batch resolved credentials before refusal")


def batch(model="gpt-4.1", later=False):
    stopped = Request(model=model, messages=(Message.user("hi"),), config=Config(stop=("STOP",)))
    ordinary = Request(model=model, messages=(Message.user("first"),))
    return BatchRequest(requests=(ordinary, stopped) if later else (stopped,))


def refused(call):
    with pytest.raises(UnsupportedFeatureError) as caught:
        call()
    assert caught.value.feature == "config.stop"
    assert "batch cannot close" in str(caught.value)
    assert "complete()/stream()" in str(caught.value)


@pytest.mark.parametrize("policy", ["note", "silent", "refuse"])
@pytest.mark.parametrize("later", [False, True])
def test_responses_batch_refuses_before_credentials_or_upload(policy, later):
    lm = OpenAILM(api_key=no_credential, transport=NoTransport(), adaptations=policy)
    request = batch(later=later)
    refused(lambda: lm.batch_submit(request))
    refused(lambda: lm._batch_upload_request(request))
    refused(lambda: lm._batch_submit_request(request, {"id": "existing-upload"}))


@pytest.mark.parametrize("policy", ["note", "silent", "refuse"])
@pytest.mark.parametrize("later", [False, True])
def test_async_responses_batch_uses_same_preflight(policy, later):
    lm = AsyncOpenAILM(api_key=no_credential, transport=NoTransport(), adaptations=policy)
    refused(lambda: asyncio.run(lm.batch_submit(batch(later=later))))


@pytest.mark.parametrize("policy", ["note", "silent", "refuse"])
@pytest.mark.parametrize("provider,model", [(AnthropicLM, "claude-haiku-4-5"), (GeminiLM, "gemini-2.5-flash")])
def test_native_batch_stops_are_preserved(provider, model, policy):
    lm = provider(api_key="synthetic", transport=NoTransport(), adaptations=policy)
    wire = lm._batch_submit_request(batch(model), None)
    body = json.loads(wire.body)
    if provider is AnthropicLM:
        assert body["requests"][0]["params"]["stop_sequences"] == ["STOP"]
    else:
        nested = body["batch"]["inputConfig"]["requests"]["requests"][0]["request"]
        assert nested["generationConfig"]["stopSequences"] == ["STOP"]


def test_harmless_batch_label_mapping_remains_available():
    request = BatchRequest(requests=batch("claude-haiku-4-5").requests, label="local-label")
    lm = AnthropicLM(api_key="synthetic", transport=NoTransport())
    body = json.loads(lm._batch_submit_request(request, None).body)
    assert "label" not in body
    assert body["requests"][0]["params"]["stop_sequences"] == ["STOP"]
