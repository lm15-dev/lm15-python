"""Malformed measurements are provider faults, never invented numbers (MAP-14).

Regression coverage for the September parity repair; no live provider calls.
"""
from copy import deepcopy
import json

import pytest

from lm15 import Config, DataPart, Message, Request, TypeSafeLM, judgments, choice, score, yes_no
from lm15.errors import ProviderError, RETRYABLE_ERRORS
from lm15.providers import OpenAIChatLM
from lm15.providers.base import HttpResponse
from lm15.testing import FakeResponse, FakeTransport


FORMAT = judgments(flag=yes_no("True?"), pick=choice("Which?", ["a", "b"]), level=score("Level?", ["low", "high"]))
REQUEST = Request(model="jev-latest", messages=(Message.user("state"),), config=Config(response_format=FORMAT))
BODY = {
    "answers": {
        "flag": {"type": "noul", "noul": 0.7},
        "pick": {"type": "choice", "choice": "a", "probabilities": {"a": 0.7, "b": 0.3}},
        "level": {"type": "score", "score": 0.8, "confidence": 0.5, "probabilities": {"0": 0.2, "1": 0.8}},
    },
}


def response(body, *, raw=False):
    return HttpResponse(status=200, reason="OK", headers=[("X-TypeSafe-Request-ID", "req-parity"), ("Content-Type", "application/json")],
                        body=body if raw else json.dumps(body).encode(), provider="typesafe")


def parse(body):
    return TypeSafeLM(api_key="k", transport=FakeTransport([])).parse_response(REQUEST, response(body))


def assert_fault(body):
    with pytest.raises(ProviderError) as caught:
        parse(body)
    error = caught.value
    assert error.code == "provider" and not isinstance(error, RETRYABLE_ERRORS)
    assert error.provider == "typesafe" and error.status == 200 and error.request_id == "req-parity"


@pytest.mark.parametrize("body", [None, [], 1, "x", {}, {"answers": None}, {"answers": []}, {"answers": {}}])
def test_wrong_top_level_or_missing_answers_is_a_provider_fault(body):
    assert_fault(body)


@pytest.mark.parametrize("name", ["flag", "pick", "level"])
@pytest.mark.parametrize("bad", [None, [], "answer", 1, {}, {"type": "unknown"}, {"type": []}])
def test_wrong_answer_shape_is_a_provider_fault(name, bad):
    body = deepcopy(BODY)
    body["answers"][name] = bad
    assert_fault(body)


@pytest.mark.parametrize("name", ["flag", "pick", "level"])
def test_missing_declared_answer_is_a_provider_fault(name):
    body = deepcopy(BODY)
    del body["answers"][name]
    assert_fault(body)


def test_undeclared_answer_is_a_provider_fault():
    body = deepcopy(BODY)
    body["answers"]["extra"] = {"type": "noul", "noul": 1}
    assert_fault(body)


@pytest.mark.parametrize("bad", [None, True, False, "0.2", [], {}, -0.1, 1.1, float("nan"), float("inf"), float("-inf"), 10**400])
@pytest.mark.parametrize("name,key", [("flag", "noul"), ("pick", "a"), ("level", "0")])
def test_probabilities_are_numbers_not_coercions_and_are_bounded(name, key, bad):
    body = deepcopy(BODY)
    target = body["answers"][name] if name == "flag" else body["answers"][name]["probabilities"]
    target[key] = bad
    assert_fault(body)


@pytest.mark.parametrize("name", ["pick", "level"])
@pytest.mark.parametrize("bad", [None, [], {}, "distribution", {"unknown": 1}])
def test_distribution_must_be_a_complete_declared_map(name, bad):
    body = deepcopy(BODY)
    body["answers"][name]["probabilities"] = bad
    assert_fault(body)


@pytest.mark.parametrize("name", ["pick", "level"])
@pytest.mark.parametrize("extra", [False, True])
def test_distribution_neither_fills_missing_keys_nor_discards_extra_keys(name, extra):
    body = deepcopy(BODY)
    dist = body["answers"][name]["probabilities"]
    if extra:
        dist["undeclared"] = 0
    else:
        del dist[next(iter(dist))]
    assert_fault(body)


@pytest.mark.parametrize("bad", [None, [], {}, 0, True, "undeclared"])
def test_choice_must_be_a_declared_string(bad):
    body = deepcopy(BODY)
    body["answers"]["pick"]["choice"] = bad
    assert_fault(body)


@pytest.mark.parametrize("name,field", [("flag", "noul"), ("pick", "choice"), ("pick", "probabilities"), ("level", "probabilities")])
def test_required_measurement_fields_are_not_defaulted(name, field):
    body = deepcopy(BODY)
    del body["answers"][name][field]
    assert_fault(body)


@pytest.mark.parametrize("dist", [{"a": 0.333, "b": 0.333}, {"a": 0.9, "b": 0.9}, {"a": 0.0, "b": 0.0}])
def test_inv_052_does_not_validate_totals_or_normalize(dist):
    body = deepcopy(BODY)
    body["answers"]["pick"]["probabilities"] = dist
    result = parse(body)
    assert result.probabilities["pick"] == dist
    assert result.data["pick"] == "a"
    assert result.provider_data["typesafe"]["answers"] == body["answers"]
    part = DataPart(value={"pick": "a"}, probabilities={"pick": dist}, method="provider_classification")
    assert part.probabilities["pick"] == dist


@pytest.mark.parametrize("p", [0, 1, 0.5])
def test_noul_endpoints_and_threshold_are_not_missing(p):
    body = deepcopy(BODY)
    body["answers"]["flag"]["noul"] = p
    result = parse(body)
    assert result.data["flag"] is (p >= 0.5)
    assert result.probabilities["flag"] == {"true": p, "false": 1 - p}


@pytest.mark.parametrize("usage,expected", [(None, (None, None, None)), ({}, (None, None, None)),
    ({"input_tokens": 0}, (0, None, None)), ({"output_tokens": 0}, (None, 0, None)),
    ({"input_tokens": 0, "output_tokens": 0}, (0, 0, 0)),
    ({"input_tokens": 2.0, "output_tokens": 3}, (2, 3, 5))])
def test_usage_missing_null_and_explicit_zero_are_distinct(usage, expected):
    body = deepcopy(BODY)
    body["usage"] = usage
    u = parse(body).usage
    assert (u.input_tokens, u.output_tokens, u.total_tokens) == expected
    assert parse(BODY).usage.input_tokens is None


@pytest.mark.parametrize("usage", [[], "x", {"input_tokens": True}, {"output_tokens": "0"}, {"input_tokens": -1}, {"output_tokens": 1.2}])
def test_malformed_usage_is_a_provider_fault(usage):
    body = deepcopy(BODY)
    body["usage"] = usage
    assert_fault(body)


@pytest.mark.parametrize("header", ["x-typesafe-request-id", "apim-request-id", "x-amzn-requestid"])
def test_non_json_error_keeps_request_id_even_on_pure_parser_path(header):
    raw = HttpResponse(status=200, reason="OK", headers=[(header, "id"), ("content-type", "text/html")], body=b"<html>bad gateway</html>")
    with pytest.raises(ProviderError) as caught:
        raw.json()
    assert caught.value.request_id == "id" and not isinstance(caught.value, RETRYABLE_ERRORS)
    assert "text/html" in str(caught.value) and "<html>" in str(caught.value)


def test_complete_does_not_retry_a_missing_measurement():
    transport = FakeTransport([FakeResponse(status=200, body=b'{"answers": {}}')])
    with pytest.raises(ProviderError):
        TypeSafeLM(api_key="k", transport=transport).complete(REQUEST)
    assert len(transport.requests) == 1


def chat_response(body):
    return HttpResponse(status=200, reason="OK", headers=[("x-request-id", "score-id")], body=json.dumps(body).encode(), provider="openai-chat")


@pytest.mark.parametrize("tokens", [None, "x", [True], [-1], [1.5]])
def test_tokenize_measurements_are_not_coerced(tokens):
    lm = OpenAIChatLM(api_key="k", transport=FakeTransport([]))
    with pytest.raises(ProviderError) as caught:
        lm._judgment_tokens_from_body(chat_response({"tokens": tokens}))
    assert caught.value.request_id == "score-id" and not isinstance(caught.value, RETRYABLE_ERRORS)


@pytest.mark.parametrize("bad", [True, "-1", None, 0.1, float("nan"), float("inf")])
def test_token_scoring_rejects_invalid_log_probabilities(bad):
    lm = OpenAIChatLM(api_key="k", transport=FakeTransport([]))
    with pytest.raises(ProviderError):
        lm._judgment_scores_from_body(chat_response({"choices": [{"index": 0, "logprobs": {"top_logprobs": [{"token_id:1": bad}]}}]}), 1)


@pytest.mark.parametrize("indices", [[0, 0], [0, 2], [True, 0], ["0", 1]])
def test_scoring_indices_must_cover_each_prompt_once(indices):
    lm = OpenAIChatLM(api_key="k", transport=FakeTransport([]))
    with pytest.raises(ProviderError):
        lm._judgment_scores_from_body(chat_response({"choices": [{"index": i} for i in indices]}), 2)


def test_scoring_missing_usage_stays_unknown_and_missing_ids_stay_unmeasured():
    lm = OpenAIChatLM(api_key="k", transport=FakeTransport([]))
    scores, usage, model = lm._judgment_scores_from_body(chat_response({"choices": [{"index": 0}]}), 1)
    assert scores == [{}] and usage.input_tokens is None and usage.output_tokens is None and model is None
    _, usage, _ = lm._judgment_scores_from_body(chat_response({"choices": [{"index": 0}], "usage": {"prompt_tokens": 0}}), 1)
    assert usage.input_tokens == 0 and usage.output_tokens is None


@pytest.mark.parametrize("zero_mass", [False, True])
def test_token_scoring_zero_likelihood_is_not_a_fabricated_distribution(zero_mass):
    from lm15 import Usage
    from lm15.judgments import request_judgments

    req = Request(model="m", messages=(Message.user("state"),), config=Config(response_format=judgments(flag=yes_no("True?"))))
    found = request_judgments(req)
    j = found["flag"]
    paths = {"true": (1,), "false": (2,)}
    table = {(): {1: float("-inf"), 2: float("-inf") if zero_mass else -1.0}}
    lm = OpenAIChatLM(api_key="k", transport=FakeTransport([]))
    if zero_mass:
        with pytest.raises(ProviderError, match="zero likelihood"):
            lm._judgment_fold(req, found, [(j, paths, table)], Usage(), None, 1, 1)
    else:
        result = lm._judgment_fold(req, found, [(j, paths, table)], Usage(), None, 1, 1)
        assert result.probabilities["flag"] == {"true": 0, "false": 1}
        assert result.usage.input_tokens is None and result.usage.output_tokens is None


@pytest.mark.asyncio
@pytest.mark.parametrize("malformed", [False, True])
async def test_async_typesafe_uses_the_same_measurement_and_usage_rules(malformed):
    from lm15.providers.async_base import AsyncTypeSafeLM
    from lm15.transports import AsyncTransportResponse

    class Transport:
        calls = 0

        def stream(self, request):
            self.calls += 1

            async def chunks():
                yield json.dumps({"answers": {}} if malformed else BODY).encode()

            async def release(consumed):
                pass

            return AsyncTransportResponse(status=200, reason="OK", headers=[("x-typesafe-request-id", "async-id")],
                                          http_version="HTTP/1.1", chunks=chunks(), release=release)

    transport = Transport()
    lm = AsyncTypeSafeLM(api_key="k", transport=transport)
    if malformed:
        with pytest.raises(ProviderError) as caught:
            await lm.complete(REQUEST)
        assert caught.value.request_id == "async-id" and not isinstance(caught.value, RETRYABLE_ERRORS)
    else:
        result = await lm.complete(REQUEST)
        assert result.data == {"flag": True, "pick": "a", "level": 1}
        assert result.usage.input_tokens is None and result.usage.output_tokens is None
    assert transport.calls == 1
