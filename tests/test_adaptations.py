"""MAP-13 — adapt freely, never invisibly (changes/2026-09-14-adapt-visibly.md).

The mechanism: the Adaptation record, the three-position switch, plan(),
the record on Response and on the stream's start event, the client-side
stop, and the refusal's `feature` path.  Adapter-by-adapter verdicts live
next to each adapter's tests; this file is the machinery.
"""
from __future__ import annotations

import asyncio
import json

import pytest

import lm15
from lm15.judgments import judgments, yes_no
from lm15 import (
    Adaptation,
    AnthropicLM,
    Config,
    LMRouter,
    Message,
    OpenAILM,
    Request,
    Response,
    RouterConfig,
    StreamStartEvent,
    Usage,
)
from lm15.adaptation import ADAPTATION_ACTIONS, adapt, collecting, current_policy, nearest_effort
from lm15.errors import UnsupportedFeatureError
from lm15.providers import AsyncOpenAILM
from lm15.serde import response_from_dict, response_to_dict, stream_event_from_dict, stream_event_to_dict
from lm15.testing import FakeResponse, FakeTransport
from lm15.types import StreamDeltaEvent, StreamEndEvent, TextDelta, TextPart


# ─── the record ──────────────────────────────────────────────────────

def test_adaptation_is_validated_and_exported() -> None:
    a = Adaptation(field="config.seed", action="dropped", reason="no seed here", asked=7)
    assert (a.field, a.action, a.asked, a.applied) == ("config.seed", "dropped", 7, None)
    assert lm15.Adaptation is Adaptation and lm15.AdaptationPolicy
    assert set(ADAPTATION_ACTIONS) == {"dropped", "clamped", "substituted", "client_side", "satisfied", "defaulted"}
    with pytest.raises(ValueError, match="action"):
        Adaptation(field="x", action="ignored", reason="r")
    with pytest.raises(ValueError, match="reason"):
        Adaptation(field="x", action="dropped", reason="")


def test_response_and_start_event_carry_the_record_and_omit_it_when_empty() -> None:
    plain = Response(id=None, model="m", message=Message.assistant("hi"), finish_reason="stop", usage=Usage())
    assert plain.adaptations == () and "adaptations" not in response_to_dict(plain)
    noted = Response(id=None, model="m", message=Message.assistant("hi"), finish_reason="stop", usage=Usage(),
                     adaptations=(Adaptation(field="config.seed", action="dropped", reason="r", asked=7),))
    d = response_to_dict(noted)
    assert d["adaptations"] == [{"field": "config.seed", "action": "dropped", "reason": "r", "asked": 7}]
    assert response_from_dict(d) == noted
    assert "adaptations=['config.seed:dropped']" in repr(noted)
    start = StreamStartEvent(model="m", adaptations=noted.adaptations)
    assert stream_event_from_dict(stream_event_to_dict(start)) == start
    assert "adaptations" not in stream_event_to_dict(StreamStartEvent(model="m"))
    with pytest.raises(TypeError, match="Adaptation"):
        Response(id=None, model="m", message=Message.assistant("hi"), finish_reason="stop", usage=Usage(), adaptations=("x",))  # type: ignore[arg-type]


# ─── the switch ──────────────────────────────────────────────────────

def test_switch_note_silent_refuse() -> None:
    with collecting("note", provider="p") as scope:
        adapt("config.a", "dropped", "gone", asked=1)
        adapt("config.b", "defaulted", "filled", applied=2)
    assert [(a.field, a.action) for a in scope.records] == [("config.a", "dropped"), ("config.b", "defaulted")]
    with collecting("silent") as scope:
        adapt("config.a", "dropped", "gone")
    assert [a.field for a in scope.records] == ["config.a"]  # kept: behaviour reads it; hidden on the response
    with collecting("refuse", provider="p") as scope:
        # A required field the caller left open is filled, not refused.
        adapt("config.max_tokens", "defaulted", "the wire requires it", applied=10)
        with pytest.raises(UnsupportedFeatureError, match=r"p: config.a would be dropped: gone \(adaptations='refuse'\)") as err:
            adapt("config.a", "dropped", "gone")
        assert err.value.feature == "config.a" and err.value.provider == "p"
    assert [a.action for a in scope.records] == ["defaulted"]
    # No scope open: nothing is kept, nothing raises (a builder called directly).
    assert current_policy() == "note"
    adapt("config.a", "dropped", "gone")
    with pytest.raises(ValueError, match="adaptations must be one of"):
        with collecting("loud"):  # type: ignore[arg-type]
            pass


def test_policy_reaches_every_constructor_and_the_router() -> None:
    req = Request(model="claude-sonnet-4-5", messages=(Message.user("hi"),), config=Config(max_tokens=10, seed=7))
    assert [a.action for a in AnthropicLM(api_key="k").plan(req)] == ["dropped"]
    assert [a.action for a in AnthropicLM(api_key="k", adaptations="silent").plan(req)] == ["dropped"]  # a preview never hides
    with pytest.raises(UnsupportedFeatureError) as err:
        AnthropicLM(api_key="k", adaptations="refuse").plan(req)
    assert err.value.feature == "config.seed"
    with pytest.raises(ValueError, match="adaptations must be one of"):
        AnthropicLM(api_key="k", adaptations="maybe")  # type: ignore[arg-type]
    router = LMRouter(RouterConfig(env={}, api_keys={"anthropic": "k"}, adaptations="refuse"))
    assert router.lm("anthropic:m").adaptations == "refuse"
    with pytest.raises(UnsupportedFeatureError):
        router.plan(Request(model="anthropic:claude-sonnet-4-5", messages=(Message.user("hi"),), config=Config(max_tokens=10, seed=7)))
    with pytest.raises(ValueError):
        RouterConfig(adaptations="never")  # type: ignore[arg-type]
    # The OpenAI-chat door's client-keyword table points drop_params here.
    from lm15.router import _CLIENT_KEYWORDS
    assert "adaptations='silent'" in _CLIENT_KEYWORDS["drop_params"]


# ─── plan() and the record on complete / stream ──────────────────────

_ANTHROPIC_BODY = json.dumps({
    "id": "msg_1", "model": "claude-sonnet-4-5", "role": "assistant",
    "content": [{"type": "text", "text": "hello END world"}], "stop_reason": "end_turn",
    "usage": {"input_tokens": 1, "output_tokens": 1},
}).encode()


def test_plan_matches_complete_and_the_record_rides_the_response() -> None:
    req = Request(model="claude-sonnet-4-5", messages=(Message.user("hi"),), config=Config(seed=7, temperature=1.5))
    lm = AnthropicLM(api_key="k", transport=FakeTransport([FakeResponse(status=200, body=_ANTHROPIC_BODY)]))
    plan = lm.plan(req)
    assert [(a.field, a.action, a.applied) for a in plan] == [
        ("config.max_tokens", "defaulted", 16384), ("config.seed", "dropped", None), ("config.temperature", "clamped", 1.0),
    ]
    response = lm.complete(req)
    assert response.adaptations == plan and response.text == "hello END world"
    # plan() sent nothing: the fake served exactly one request, to complete().
    assert len(lm.transport.requests) == 1


def test_stream_stamps_the_record_on_the_start_event() -> None:
    sse = (
        b'event: message_start\ndata: {"type":"message_start","message":{"id":"msg_1","model":"claude-sonnet-4-5","role":"assistant","content":[],"usage":{"input_tokens":1,"output_tokens":0}}}\n\n'
        b'event: content_block_start\ndata: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}\n\n'
        b'event: content_block_delta\ndata: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"hi"}}\n\n'
        b'event: message_delta\ndata: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":1}}\n\n'
        b'event: message_stop\ndata: {"type":"message_stop"}\n\n'
    )
    req = Request(model="claude-sonnet-4-5", messages=(Message.user("hi"),), config=Config(max_tokens=10, seed=7))
    lm = AnthropicLM(api_key="k", transport=FakeTransport([FakeResponse(status=200, body=sse)]))
    events = list(lm.stream(req))
    assert events[0].type == "start" and [a.field for a in events[0].adaptations] == ["config.seed"]
    assert sum(e.type == "start" for e in events) == 1 and events[-1].type == "end"
    from lm15.result import materialize_response
    lm2 = AnthropicLM(api_key="k", transport=FakeTransport([FakeResponse(status=200, body=sse)]))
    assert [a.field for a in materialize_response(lm2.stream(req), req).adaptations] == ["config.seed"]


# ─── client-side stop on the Responses wire ──────────────────────────

_RESPONSES_BODY = json.dumps({
    "id": "resp_1", "model": "gpt-5", "status": "completed",
    "output": [{"type": "message", "content": [{"type": "output_text", "text": "alpha END beta"}]}],
    "usage": {"input_tokens": 3, "output_tokens": 9},
}).encode()


def test_complete_with_a_stop_streams_under_the_hood_and_cuts() -> None:
    # Decision 2026-09-14: a client-side stop on a non-streaming call is
    # honoured by streaming and closing at the cut. The provider controls
    # subsequent generation/billing; usage is not reported (never
    # estimated). The fake serves the streamed body the wire would send.
    req = Request(model="gpt-5", messages=(Message.user("hi"),), config=Config(stop=("END",)))
    body = _responses_sse("alpha ", "END beta")
    lm = OpenAILM(api_key="k", transport=FakeTransport([FakeResponse(status=200, body=body)]))
    response = lm.complete(req)
    assert response.text == "alpha " and response.finish_reason == "stop"
    assert response.usage == Usage()  # not reported: the final frame was never read
    assert response.adaptations[0].field == "config.stop" and response.adaptations[0].action == "client_side"
    sent = json.loads(lm.transport.requests[0].body)
    assert sent["stream"] is True and "stop" not in sent


def test_complete_with_a_stop_that_never_hits_keeps_the_usage() -> None:
    req = Request(model="gpt-5", messages=(Message.user("hi"),), config=Config(stop=("END",)))
    body = _responses_sse("alpha ", "beta")
    lm = OpenAILM(api_key="k", transport=FakeTransport([FakeResponse(status=200, body=body)]))
    response = lm.complete(req)
    assert response.text == "alpha beta" and response.usage.output_tokens == 9


def _responses_sse(*texts: str) -> bytes:
    frames = [
        'event: response.created\ndata: {"type":"response.created","response":{"id":"resp_1","model":"gpt-5"}}\n\n',
        'event: response.output_item.added\ndata: {"type":"response.output_item.added","output_index":0,"item":{"type":"message","id":"m1","role":"assistant","content":[]}}\n\n',
    ]
    for t in texts:
        frames.append('event: response.output_text.delta\ndata: ' + json.dumps({"type": "response.output_text.delta", "output_index": 0, "content_index": 0, "delta": t}) + "\n\n")
    frames.append('event: response.completed\ndata: {"type":"response.completed","response":{"id":"resp_1","model":"gpt-5","status":"completed","output":[],"usage":{"input_tokens":3,"output_tokens":9}}}\n\n')
    return "".join(frames).encode()


def test_stream_cuts_at_a_stop_sequence_split_across_deltas_and_closes_early() -> None:
    req = Request(model="gpt-5", messages=(Message.user("hi"),), config=Config(stop=("END",)))
    body = _responses_sse("alp", "ha E", "ND beta", " gamma")
    lm = OpenAILM(api_key="k", transport=FakeTransport([FakeResponse(status=200, body=body)]))
    events = list(lm.stream(req))
    text = "".join(e.delta.text for e in events if e.type == "delta" and isinstance(e.delta, TextDelta))
    assert text == "alpha "
    assert events[-1].type == "end" and events[-1].finish_reason == "stop" and events[-1].usage is None
    assert sum(e.type == "end" for e in events) == 1
    assert events[0].adaptations[0].action == "client_side"


def test_stream_without_a_hit_releases_the_withheld_tail() -> None:
    req = Request(model="gpt-5", messages=(Message.user("hi"),), config=Config(stop=("END",)))
    body = _responses_sse("alpha ", "beta")
    lm = OpenAILM(api_key="k", transport=FakeTransport([FakeResponse(status=200, body=body)]))
    events = list(lm.stream(req))
    text = "".join(e.delta.text for e in events if e.type == "delta" and isinstance(e.delta, TextDelta))
    assert text == "alpha beta" and events[-1].usage is not None and events[-1].usage.output_tokens == 9


def test_async_stream_cuts_at_stop_too() -> None:
    from tests.test_async_adapters import FakeAsyncTransport

    req = Request(model="gpt-5", messages=(Message.user("hi"),), config=Config(stop=("END",)))
    body = _responses_sse("alpha EN", "D beta")
    lm = AsyncOpenAILM(api_key="k", transport=FakeAsyncTransport(body))

    async def collect():
        return [e async for e in lm.stream(req)]

    events = asyncio.run(collect())
    text = "".join(e.delta.text for e in events if e.type == "delta" and isinstance(e.delta, TextDelta))
    assert text == "alpha " and events[-1].finish_reason == "stop"
    assert lm.plan(req)[0].action == "client_side"
    assert lm.adaptations == "note"


def test_apply_client_side_stop_on_a_response_value() -> None:
    from lm15.result import apply_client_side_stop

    r = Response(id=None, model="m", finish_reason="length", usage=Usage(),
                 message=Message.assistant((TextPart(text="one"), TextPart(text="two STOP three"), TextPart(text="four"))))
    cut = apply_client_side_stop(r, ("STOP",))
    assert [p.text for p in cut.message.parts] == ["one", "two "] and cut.finish_reason == "stop"
    assert apply_client_side_stop(r, ("nowhere",)) is r


# ─── shared clamp ────────────────────────────────────────────────────

def test_nearest_effort_ties_go_lower() -> None:
    assert nearest_effort("medium", ("low", "high")) == "low"
    assert nearest_effort("xhigh", ("low", "medium", "high")) == "high"
    assert nearest_effort("minimal", ("low", "high", "max")) == "low"
    assert nearest_effort("high", ("low", "high", "max")) == "high"
    with pytest.raises(ValueError):
        nearest_effort("low", ())


# ─── plan() is offline: no credential, no key required ───────────────

def test_plan_invokes_no_credential_and_needs_no_key() -> None:
    from lm15 import OpenAIChatLM

    calls: list[int] = []
    lm = OpenAIChatLM(api_key=lambda: calls.append(1) or "k")
    req = Request(model="gpt-4.1", messages=(Message.user("hi"),), config=Config(top_k=2))
    assert [a.action for a in lm.plan(req)] == ["dropped"]
    assert calls == []  # the build's bytes are discarded; the provider is not asked
    lm.build_request(req, stream=False)
    assert calls == [1]
    # A router with no key for the route still plans; lm() still refuses.
    router = LMRouter(RouterConfig(env={}))
    plan = router.plan(Request(model="anthropic:claude-sonnet-4-5", messages=(Message.user("hi"),), config=Config(seed=1)))
    assert [a.field for a in plan] == ["config.max_tokens", "config.seed"]
    with pytest.raises(lm15.MissingCredentialError):
        router.lm("anthropic:claude-sonnet-4-5")
    assert "anthropic" not in router._lms  # the planning LM is not cached


# ─── review of dspy#10409: the five gaps, each pinned ────────────────

def _text_event(text: str, index: int = 0) -> StreamDeltaEvent:
    return StreamDeltaEvent(TextDelta(text=text, part_index=index))


def test_stop_cutter_catches_a_sequence_split_into_pieces_shorter_than_itself() -> None:
    from lm15.result import _StopCutter

    # "ST" + "OP": the first chunk is shorter than the withheld tail; the
    # cutter once released "S" and missed the word.
    c = _StopCutter(("STOP",))
    assert c.feed(_text_event("ST")) == [] and c.feed(_text_event("OP")) == [] and c.cut
    c = _StopCutter(("STOP",))
    assert [c.feed(_text_event(piece)) for piece in ("S", "T", "O", "P")] == [[], [], [], []] and c.cut
    c = _StopCutter(("STOP",))
    assert [c.feed(_text_event(piece)) for piece in ("a", "b", "c", "d")] == [[], [], [], [_text_event("a")]] and not c.cut
    assert c.flush() == [_text_event("b"), _text_event("c"), _text_event("d")]


def test_stream_cuts_when_the_stop_word_arrives_letter_by_letter() -> None:
    req = Request(model="gpt-5", messages=(Message.user("hi"),), config=Config(stop=("STOP",)))
    body = _responses_sse("alpha ", "S", "T", "O", "P", " beta")
    lm = OpenAILM(api_key="k", transport=FakeTransport([FakeResponse(status=200, body=body)]))
    events = list(lm.stream(req))
    text = "".join(e.delta.text for e in events if e.type == "delta" and isinstance(e.delta, TextDelta))
    assert text == "alpha " and events[-1].finish_reason == "stop"


def test_silent_hides_the_record_and_changes_nothing_else() -> None:
    req = Request(model="gpt-5", messages=(Message.user("hi"),), config=Config(stop=("END",)))
    body = _responses_sse("alpha ", "END beta")
    noted = OpenAILM(api_key="k", transport=FakeTransport([FakeResponse(status=200, body=body)]))
    silent = OpenAILM(api_key="k", transport=FakeTransport([FakeResponse(status=200, body=body)]), adaptations="silent")
    a, b = noted.complete(req), silent.complete(req)
    assert a.text == b.text == "alpha "           # the cut happens under both
    assert a.adaptations and b.adaptations == ()  # only the record differs
    assert silent.plan(req)[0].action == "client_side"  # a preview never hides
    events = list(OpenAILM(api_key="k", transport=FakeTransport([FakeResponse(status=200, body=body)]), adaptations="silent").stream(req))
    assert events[0].adaptations == () and events[-1].finish_reason == "stop"
    # A narrowed tool list is narrowed under "silent" too.
    from lm15 import FunctionTool, ToolChoice
    tooled = Request(model="claude-sonnet-4-5", messages=(Message.user("hi"),), tools=(FunctionTool(name="a"), FunctionTool(name="b")),
                     config=Config(max_tokens=10, tool_choice=ToolChoice(mode="auto", allowed=("a",))))
    wire, _ = AnthropicLM(api_key="k", adaptations="silent")._build(tooled, stream=False)
    assert [t["name"] for t in json.loads(wire.body)["tools"]] == ["a"]


def test_plan_reads_no_stored_login_and_walks_no_chain(monkeypatch) -> None:
    import lm15.access as access
    from lm15.registry import PROVIDERS

    class Forbidden(dict):
        def get(self, key, default=None):
            raise AssertionError(f"plan() must not read a stored login ({key})")

        def __getitem__(self, key):
            raise AssertionError(f"plan() must not read a stored login ({key})")

    monkeypatch.setattr(access, "_CREDENTIAL_LOADERS", Forbidden())
    router = LMRouter(RouterConfig(env={}))
    for pid in PROVIDERS:
        model = {"deepseek-anthropic": "deepseek-v4-flash", "moonshotai-anthropic": "kimi-k3"}.get(pid, "m")
        # typesafe answers declared judgments only (MAP-14): a plain text
        # request is a refusal there, not a credential question.
        config = Config(response_format=judgments(ok=yes_no("Is it fine?"))) if pid == "typesafe" else Config()
        router.plan(Request(model=f"{pid}:{model}", messages=(Message.user("hi"),), config=config))
    assert router._lms == {}  # planning LMs are never cached


def test_router_is_thread_safe_on_first_calls() -> None:
    import threading

    router = LMRouter(RouterConfig(env={}, api_keys={"anthropic": "k", "openai": "k"}))
    transports: set[int] = set()
    lms: dict[str, set[int]] = {"anthropic": set(), "openai": set()}
    barrier = threading.Barrier(16)

    def go(model: str) -> None:
        barrier.wait()
        lm = router.lm(model)
        transports.add(id(lm.transport))
        lms[model.split(":")[0]].add(id(lm))

    threads = [threading.Thread(target=go, args=(m,)) for m in ["anthropic:x", "openai:y"] * 8]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(transports) == 1 and all(len(ids) == 1 for ids in lms.values())
    router.close()
    assert router._transport is None and router._lms == {}


def test_client_side_stop_reason_makes_no_billing_promise() -> None:
    lm = OpenAILM(api_key="k")
    (a,) = lm.plan(Request(model="gpt-5", messages=(Message.user("hi"),), config=Config(stop=("END",))))
    assert "not reported" in a.reason and "billed" not in a.reason.replace("and billing, is its own behaviour", "")


# ─── greptile on dspy#10409: four more, each pinned ──────────────────

def test_stop_sequence_spanning_two_text_parts_is_cut_on_complete() -> None:
    from lm15.result import apply_client_side_stop

    r = Response(id=None, model="m", finish_reason="length", usage=Usage(),
                 message=Message.assistant((TextPart(text="alpha S"), TextPart(text="TOP beta"), TextPart(text="gamma"))))
    cut = apply_client_side_stop(r, ("STOP",))
    assert [p.text for p in cut.message.parts] == ["alpha "] and cut.finish_reason == "stop"


def test_stop_sequence_spanning_two_text_parts_is_cut_on_stream() -> None:
    from lm15.result import _StopCutter

    c = _StopCutter(("STOP",))
    out = c.feed(_text_event("alpha S")) + c.feed(_text_event("TOP beta", 1))
    # Hold the original event until it is safe, rather than splitting it
    # just for early delivery and losing its attached fields.
    assert out == [_text_event("alpha ")] and c.cut
    # No hit: every original event is released under its own part index.
    c = _StopCutter(("STOP",))
    original = [_text_event(text, index) for index, text in enumerate("abcd")]
    out = [event for source in original for event in c.feed(source)] + c.flush()
    assert out == original and all(a is b for a, b in zip(out, original)) and not c.cut


def test_dropped_schema_is_not_sent_on_a_server_that_ignores_it() -> None:
    from tests._adapt import adapted

    lm = LMRouter(RouterConfig(env={"DEEPSEEK_API_KEY": "d"})).lm("deepseek-anthropic:deepseek-v4-flash")
    req = Request(model="deepseek-v4-flash", messages=(Message.user("x"),),
                  config=Config(max_tokens=10, response_format={"type": "json_schema", "name": "x", "schema": {"type": "object"}}))
    out = adapted(lm, req)
    assert out["config.response_format"].action == "dropped"
    assert "format" not in out["__body__"].get("output_config", {})  # dropped means not sent


def test_cache_breakpoint_walk_back_is_recorded_once() -> None:
    from lm15 import CacheConfig, OpenAIChatLM

    req = Request(model="gpt-5.6-sol", messages=(Message.user("prefix"), Message.assistant("ok"), Message.user("q")),
                  config=Config(cache=CacheConfig(prefix_until_index=1)))
    for lm in (OpenAILM(api_key="k"), OpenAIChatLM(api_key="k")):
        plan = lm.plan(req)
        assert [a.field for a in plan] == ["config.cache.prefix_until_index"]


def test_compat_guess_warning_never_prints_url_credentials() -> None:
    import warnings

    lm = OpenAILM(api_key="k", base_url="https://user:secret-token@openrouter.ai/api/v1?key=also-secret")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        lm.plan(Request(model="m", messages=(Message.user("hi"),)))  # the guess happens per request
    text = " ".join(str(w.message) for w in caught if issubclass(w.category, DeprecationWarning))
    assert "openrouter.ai" in text and "secret-token" not in text and "also-secret" not in text and "user" not in text
    # An out-of-range port must not turn the warning into a failure.
    lm = OpenAILM(api_key="k", base_url="https://openrouter.ai:99999/api/v1")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        lm.plan(Request(model="m", messages=(Message.user("hi"),)))
    assert any("openrouter.ai:99999" in str(w.message) for w in caught)
