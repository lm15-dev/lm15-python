"""The provider registry is the one place a provider is declared.

Every other table (ADAPTERS, ASYNC_ADAPTERS, CHAT_PRESET_ROUTES, the doctor's
known-provider list, the vet surface dump) is a view of it.  These tests pin
the rules the module docstring states, so a new provider that breaks one
fails here rather than drifting into a port.
"""

from __future__ import annotations

import dataclasses

import pytest

from lm15 import access
from lm15.compat import (
    ANTHROPIC_PRESET_BASE_URLS,
    OPENAI_CHAT_PRESET_BASE_URLS,
    OPENAI_RESPONSES_PRESET_BASE_URLS,
    AnthropicCompat,
    OpenAIChatCompat,
    OpenAIResponsesCompat,
)
from lm15.doctor import explain_auth
from lm15.features import AccessPolicy, EndpointSupport
from lm15.providers import GeminiLM, AnthropicLM, OpenAIChatLM, OpenAILM
from lm15.registry import PROVIDERS, ProviderDefinition, canonical_provider, lookup
from lm15.router import ADAPTERS, ASYNC_ADAPTERS, CHAT_PRESET_ROUTES, LMRouter, RouterConfig
from lm15.vet import op_surface_dump


class TestShape:
    def test_ids_are_canonical_and_match_access(self) -> None:
        for pid, d in PROVIDERS.items():
            assert pid == d.id == canonical_provider(pid)
            assert canonical_provider(d.access.provider) == pid

    def test_one_dialect_per_provider_string(self) -> None:
        # A provider string names ONE wire behavior; the adapter class is
        # the dialect, so two entries never share an id (a mapping cannot)
        # and every entry names a dialect the class actually speaks.
        dialect_of = {"openai-responses": "OpenAI", "openai-chat": "OpenAIChat", "anthropic": "Anthropic", "gemini": "Gemini", "typesafe": "TypeSafe"}
        bound_class = {"openai-responses": OpenAILM, "openai-chat": OpenAIChatLM, "anthropic": AnthropicLM, "gemini": GeminiLM}
        for d in PROVIDERS.values():
            assert d.adapter.__name__.endswith("LM")
            if d.bound:
                assert d.adapter is bound_class[d.dialect], d.id
            else:
                # Adapter-owned classes speak their dialect directly or by
                # subclassing it (XaiLM is an OpenAIChatLM).
                assert any(base.__name__ == f"{dialect_of[d.dialect]}LM" for base in d.adapter.__mro__), d.id

    def test_bound_entries_are_pure_data(self) -> None:
        for d in PROVIDERS.values():
            if not d.bound:
                assert d.compat is None
                assert d.placeholder_key is None
                continue
            if d.hosted:
                # A cloud door (AUTH-10): the URL is a template over settings,
                # so the compat-table URL rule does not apply; a preset is
                # optional and must exist for the dialect when named.
                assert d.access.host is not None and d.placeholder_key is None
                continue
            assert d.compat is not None
            if d.dialect == "openai-chat":
                OpenAIChatCompat.preset(d.compat)  # exists
                assert d.access.base_url == OPENAI_CHAT_PRESET_BASE_URLS[d.compat]
            elif d.dialect == "openai-responses":
                OpenAIResponsesCompat.preset(d.compat)
                assert d.access.base_url == OPENAI_RESPONSES_PRESET_BASE_URLS[d.compat]
            else:
                AnthropicCompat.preset(d.compat)
                assert d.access.base_url == ANTHROPIC_PRESET_BASE_URLS[d.compat]
            assert d.access.credential_policy == "key"

    def test_keyless_means_no_env_keys(self) -> None:
        for d in PROVIDERS.values():
            if d.placeholder_key is not None:
                assert d.env_keys == (), d.id
            else:
                assert d.credential_policy != "key" or d.env_keys, d.id

    def test_console_url_for_every_key_provider(self) -> None:
        # A user who lacks a key must be told where to get one.
        for d in PROVIDERS.values():
            if d.credential_policy != "oauth" and d.placeholder_key is None:
                assert d.console_url and d.console_url.startswith("https://"), d.id

    def test_definition_rejects_drift(self) -> None:
        bad_url = AccessPolicy(provider="groq", env_keys=("GROQ_API_KEY",), base_url="https://example.invalid")
        with pytest.raises(ValueError, match="compat table"):
            ProviderDefinition(id="groq", dialect="openai-chat", adapter=OpenAIChatLM,
                               async_adapter=OpenAIChatLM, access=bad_url, compat="groq")
        with pytest.raises(ValueError, match="hyphenated"):
            ProviderDefinition(id="openai_chat", dialect="openai-chat", adapter=OpenAIChatLM,
                               async_adapter=OpenAIChatLM, access=OpenAIChatLM.manifest)
        with pytest.raises(ValueError, match="names provider"):
            ProviderDefinition(id="groq", dialect="openai-chat", adapter=OpenAIChatLM,
                               async_adapter=OpenAIChatLM, access=access.OPENROUTER, compat="groq")
        with pytest.raises(ValueError, match="no env_keys"):
            ProviderDefinition(id="groq", dialect="openai-chat", adapter=OpenAIChatLM,
                               async_adapter=OpenAIChatLM, access=access.GROQ, compat="groq",
                               placeholder_key="x")

    def test_frozen(self) -> None:
        with pytest.raises(TypeError):
            PROVIDERS["deepseek"] = PROVIDERS["groq"]  # type: ignore[index]
        with pytest.raises(dataclasses.FrozenInstanceError):
            PROVIDERS["groq"].note = "x"  # type: ignore[misc]


class TestViews:
    def test_router_tables_are_views(self) -> None:
        owned = {d.id for d in PROVIDERS.values() if not d.bound}
        chat_bound = {d.id for d in PROVIDERS.values() if d.bound and d.dialect == "openai-chat"}
        assert set(ADAPTERS) == set(ASYNC_ADAPTERS) == owned
        assert set(CHAT_PRESET_ROUTES) == chat_bound  # the table's name says chat; other dialects route from the registry
        for pid, route in CHAT_PRESET_ROUTES.items():
            d = PROVIDERS[pid]
            assert route.env_keys == d.env_keys
            assert route.default_key == d.placeholder_key
        with pytest.raises(TypeError):
            ADAPTERS["x"] = OpenAIChatLM  # type: ignore[index]

    def test_surface_dump_reflects_every_provider(self) -> None:
        dumped = op_surface_dump({})["providers"]
        assert set(dumped) == set(PROVIDERS)
        for pid, d in PROVIDERS.items():
            assert dumped[pid]["env_keys"] == list(d.env_keys)
            assert dumped[pid]["auth_modes"] == list(d.access.auth_modes)
            assert dumped[pid]["supports"]["complete"] is d.supports.complete

    def test_doctor_knows_every_provider(self) -> None:
        for pid, d in PROVIDERS.items():
            if d.credential_policy == "oauth":
                continue  # reads a local file; covered by the auth fixtures
            report = explain_auth(pid, env={})
            kinds = [s.kind for s in report.steps]
            assert kinds[0] == "api_keys"
            for key in d.env_keys:
                assert f"env:{key}" in kinds
            if d.placeholder_key is not None:
                assert kinds[-1] == "placeholder" and report.configured
            elif d.credential_policy == "key":
                assert not report.configured

    def test_lookup_accepts_both_spellings(self) -> None:
        assert lookup("openai_chat") is PROVIDERS["openai-chat"]
        assert lookup("claude_code") is PROVIDERS["claude-code"]
        assert lookup("nope") is None


class TestBinding:
    def test_bound_lm_names_the_provider(self) -> None:
        # Before the registry, a routed groq LM called itself "openai_chat"
        # in errors and ModelInfo.provider.  The bound access policy makes
        # it name the provider, and carry the provider's env keys.
        router = LMRouter(RouterConfig(env={"GROQ_API_KEY": "g", "DEEPSEEK_API_KEY": "d"}))
        for pid in ("groq", "deepseek"):
            lm = router.lm(f"{pid}:model")
            assert isinstance(lm, OpenAIChatLM)
            assert lm.provider == pid
            assert lm.access is PROVIDERS[pid].access
            assert lm.base_url == PROVIDERS[pid].base_url
            assert lm.supports == EndpointSupport(complete=True, stream=True, models=True)
        # The second bound dialect: same key, a different wire, its own class.
        lm = router.lm("deepseek-anthropic:deepseek-v4-flash")
        assert isinstance(lm, AnthropicLM) and lm.provider == "deepseek-anthropic"
        assert lm.base_url == "https://api.deepseek.com/anthropic/v1"
        assert lm.supports.models is False
        assert router.resolve("deepseek-anthropic:deepseek-v4-flash").adapter == "AnthropicLM"
        from lm15.router import AsyncLMRouter
        from lm15.providers import AsyncAnthropicLM

        alm = AsyncLMRouter(RouterConfig(env={"DEEPSEEK_API_KEY": "d"})).lm("deepseek-anthropic:m")
        assert isinstance(alm, AsyncAnthropicLM) and alm.base_url == lm.base_url

    def test_deepseek_anthropic_declaration_and_refusals(self) -> None:
        import json

        from lm15 import Config, FunctionTool, Message, Reasoning, Request, ToolChoice
        from lm15.errors import UnsupportedFeatureError, UnsupportedModelError

        d = PROVIDERS["deepseek-anthropic"]
        assert d.dialect == "anthropic" and d.bound and d.compat == "deepseek"
        assert d.env_keys == ("DEEPSEEK_API_KEY",) and d.access.auth_header == "x-api-key"
        lm = LMRouter(RouterConfig(env={"DEEPSEEK_API_KEY": "d"})).lm("deepseek-anthropic:deepseek-v4-flash")
        say = (Message.user("x"),)
        tool = FunctionTool(name="w", description="d", parameters={"type": "object", "properties": {}})

        def body(cfg, model="deepseek-v4-flash"):
            return json.loads(lm.build_request(Request(model=model, messages=say, tools=(tool,), config=cfg), stream=False).body)

        # Thinking: DeepSeek's shape, and an explicit off reaches the wire
        # (absence means ON there; live 2026-09-03).
        assert "thinking" not in body(Config(max_tokens=100))
        assert body(Config(max_tokens=100, reasoning=Reasoning(effort="off")))["thinking"] == {"type": "disabled"}
        b = body(Config(max_tokens=100, reasoning=Reasoning(effort="low")))
        assert b["thinking"] == {"type": "enabled"} and b["output_config"] == {"effort": "low"} and b["max_tokens"] == 100
        # Silent cells (guide--anthropic-api.md; live 2026-09-03): MAP-13 drops
        # the setting and records it — the record is the visibility the old
        # refusal existed for; "refuse" keeps the refusal.
        from tests._adapt import adapted, refuses
        for cfg, field in (
            (Config(reasoning=Reasoning(effort="low", thinking_budget=4096)), "config.reasoning.thinking_budget"),  # budget_tokens ignored
            (Config(response_format={"type": "json_schema", "name": "x", "schema": {"type": "object"}}), "config.response_format"),  # schema ignored
            (Config(tool_choice=ToolChoice(mode="auto", parallel=False)), "config.tool_choice.parallel"),      # disable_parallel_tool_use ignored
        ):
            req = Request(model="deepseek-v4-flash", messages=say, tools=(tool,), config=cfg)
            assert adapted(lm, req)[field].action == "dropped"
            refuses(lm, req, field)
        assert "disable_parallel_tool_use" not in body(Config(tool_choice=ToolChoice(mode="auto", parallel=False)))["tool_choice"]
        # claude-* is silently served by a DeepSeek model: refuse before the wire.
        with pytest.raises(UnsupportedModelError, match="silently substituted"):
            body(Config(max_tokens=10), model="claude-opus-4-1")
        # cache marks: not placed, not an error (implicit caching applies).
        from lm15 import CacheConfig

        b = body(Config(max_tokens=10, cache=CacheConfig(mode="auto")))
        assert "cache_control" not in json.dumps(b)
        # Plain Anthropic is untouched: manual class still sends a budget.
        plain = json.loads(AnthropicLM(api_key="k").build_request(
            Request(model="claude-haiku-4-5", messages=say, config=Config(max_tokens=100, reasoning=Reasoning(effort="low"))), stream=False).body)
        assert plain["thinking"]["type"] == "enabled" and "budget_tokens" in plain["thinking"]

    def test_deepseek_declaration(self) -> None:
        d = PROVIDERS["deepseek"]
        assert d.dialect == "openai-chat" and d.bound
        assert d.env_keys == ("DEEPSEEK_API_KEY",)
        assert d.base_url == "https://api.deepseek.com"
        compat = OpenAIChatCompat.preset("deepseek")
        assert compat.thinking_format == "deepseek"
        assert compat.thinking_replay == "native"
        assert compat.assistant_reasoning_content == "include_empty"
        assert compat.max_tokens_field == "max_tokens"
        assert compat.user_field == "user_id"

    def test_user_field_rides_the_compat_name(self) -> None:
        # DeepSeek documents `user_id` and accepts `user` silently (live
        # 2026-09-03); OpenAI's dialect spells it `user`.
        import json

        from lm15 import Config, Message, Request

        req = Request(model="m", messages=(Message.user("x"),), config=Config(user_id="u1"))
        for preset, field_name in (("openai", "user"), ("groq", "user"), ("deepseek", "user_id")):
            body = json.loads(OpenAIChatLM(api_key="k", compat=preset).build_request(req, stream=False).body)
            assert body[field_name] == "u1", preset
            assert ({"user", "user_id"} - {field_name}).isdisjoint(body), preset
        with pytest.raises(ValueError):
            OpenAIChatCompat(user_field="uid")  # type: ignore[arg-type]

    def test_zai_declaration(self) -> None:
        d = PROVIDERS["zai"]
        assert d.dialect == "openai-chat" and d.bound
        assert d.env_keys == ("ZAI_API_KEY",)
        assert d.base_url == "https://api.z.ai/api/paas/v4"
        compat = OpenAIChatCompat.preset("zai")
        # The wire shape docs.z.ai documents (ChatThinking) — the deepseek
        # shape — not Qwen's enable_thinking the preset sent before 2026-09-03.
        assert compat.thinking_format == "deepseek"
        assert compat.thinking_replay == "native"
        assert compat.assistant_reasoning_content is None  # Z.AI answered 200 without it (live 2026-09-03)
        assert compat.user_field == "user_id"
        assert compat.forced_tool_choice == "reject" and compat.json_schema == "reject"

    def test_zai_refuses_silent_cells_before_the_wire(self) -> None:
        # Live 2026-09-03: tool_choice=required answered text, =none called
        # the tool, json_schema returned fenced free-form JSON — all HTTP 200.
        from lm15 import Config, FunctionTool, Message, Request, ToolChoice
        from lm15.errors import UnsupportedFeatureError

        from tests._adapt import adapted
        tool = FunctionTool(name="w", description="d", parameters={"type": "object", "properties": {}})
        lm = LMRouter(RouterConfig(env={"ZAI_API_KEY": "k"})).lm("zai:glm-5.3-flash")
        def req(cfg):
            return Request(model="glm-5.3-flash", messages=(Message.user("x"),), tools=(tool,), config=cfg)
        # MAP-13: "required" cannot be reproduced client-side and the program
        # depends on the call — still a refusal, naming its field.
        with pytest.raises(UnsupportedFeatureError, match="zai: .*silently ignored") as err:
            lm.build_request(req(Config(tool_choice=ToolChoice(mode="required"))), stream=False)
        assert err.value.feature == "config.tool_choice.mode"
        # "none" and an allowlist have a client-side form: no tools / only those tools.
        out = adapted(lm, req(Config(tool_choice=ToolChoice(mode="none"))))
        assert out["config.tool_choice.mode"].action == "client_side" and "tools" not in out["__body__"]
        out = adapted(lm, req(Config(tool_choice=ToolChoice(mode="auto", allowed=("w",)))))
        assert out["config.tool_choice.allowed"].action == "client_side" and out["__body__"]["tool_choice"] == "auto"
        # json_schema is accepted and ignored by the server: dropped and recorded.
        out = adapted(lm, req(Config(response_format={"type": "json_schema", "name": "x", "schema": {"type": "object"}})))
        assert out["config.response_format"].action == "dropped" and "response_format" not in out["__body__"]
        # The honoured forms still go out.
        for cfg in (Config(tool_choice=ToolChoice(mode="auto")), Config(response_format={"type": "json_object"})):
            lm.build_request(Request(model="glm-5.3-flash", messages=(Message.user("x"),), tools=(tool,), config=cfg), stream=False)
        # Other presets are untouched: the knob is per server, not dialect-wide.
        groq = LMRouter(RouterConfig(env={"GROQ_API_KEY": "k"})).lm("groq:m")
        groq.build_request(Request(model="m", messages=(Message.user("x"),), tools=(tool,), config=Config(tool_choice=ToolChoice(mode="required"))), stream=False)


class TestMoonshotai:
    """Moonshot AI / Kimi API Platform: the Chat Completions wire, live-verified
    2026-09-03 (lm15-contract/changes/2026-09-03-moonshotai-live.md)."""

    def test_declaration(self) -> None:
        d = PROVIDERS["moonshotai"]
        assert d.dialect == "openai-chat" and d.bound and d.adapter is OpenAIChatLM
        # lm15's own name first, then the name Moonshot's docs use; both vendor-named.
        assert d.env_keys == ("MOONSHOTAI_API_KEY", "MOONSHOT_API_KEY")
        assert d.base_url == "https://api.moonshot.ai/v1"
        assert d.access.supports == EndpointSupport(complete=True, stream=True, models=True)
        compat = OpenAIChatCompat.preset("moonshotai")
        assert compat.thinking_format == "kimi"
        assert compat.thinking_replay == "native"
        assert compat.assistant_reasoning_content is None  # K3 answered 200 to a tool loop without it (live 2026-09-03)
        assert compat.max_tokens_field == "max_completion_tokens"  # `max_tokens` is deprecated in the OpenAPI
        assert compat.user_field == "safety_identifier"
        assert compat.cache_control == "openai_implicit"  # prompt_cache_key only; caching is automatic
        assert compat.reasoning_efforts == ("low", "high", "max")
        assert compat.forced_tool_choice is None and compat.json_schema is None  # both honoured on K3 (live)

    def test_either_env_name_configures_it(self) -> None:
        for env in ({"MOONSHOTAI_API_KEY": "a"}, {"MOONSHOT_API_KEY": "b"}):
            lm = LMRouter(RouterConfig(env=env)).lm("moonshotai:kimi-k3")
            assert lm.base_url == "https://api.moonshot.ai/v1"
        report = explain_auth("moonshotai", env={"MOONSHOTAI_API_KEY": "a", "MOONSHOT_API_KEY": "b"})
        states = {s.kind: s.state for s in report.steps}
        assert states["env:MOONSHOTAI_API_KEY"] == "selected" and states["env:MOONSHOT_API_KEY"] == "shadowed"

    def test_kimi_reasoning_shape_is_split_by_intent(self) -> None:
        # An effort word is kimi-k3's documented field alone; off is the K2.x
        # family's `thinking` object alone (api--models-overview.md).  Live
        # 2026-09-03: K3 honoured the off object too, K2.6 ignored the word.
        import json

        from lm15 import Config, Message, Reasoning, Request

        lm = LMRouter(RouterConfig(env={"MOONSHOTAI_API_KEY": "k"})).lm("moonshotai:kimi-k3")

        def body(reasoning: Reasoning | None) -> dict:
            return json.loads(lm.build_request(Request(model="kimi-k3", messages=(Message.user("x"),), config=Config(max_tokens=50, reasoning=reasoning)), stream=False).body)

        low = body(Reasoning(effort="low"))
        assert low["reasoning_effort"] == "low" and "thinking" not in low
        off = body(Reasoning(effort="off"))
        assert off["thinking"] == {"type": "disabled"} and "reasoning_effort" not in off
        absent = body(None)
        assert "thinking" not in absent and "reasoning_effort" not in absent
        assert absent["max_completion_tokens"] == 50 and "max_tokens" not in absent

    def test_effort_words_without_a_native_level_are_clamped(self) -> None:
        # Live 2026-09-03: kimi-k3 answered HTTP 200 to `medium` AND to `bogus`
        # (17 reasoning tokens each) — the server validates nothing, so a word
        # outside low|high|max would downgrade silently.  MAP-13: the word is
        # clamped to the nearest declared level (a tie goes lower) and
        # recorded; "refuse" keeps the old MAP-7 rule 2 refusal.
        from lm15 import Config, Message, Reasoning, Request
        from tests._adapt import adapted, refuses

        lm = LMRouter(RouterConfig(env={"MOONSHOTAI_API_KEY": "k"})).lm("moonshotai:kimi-k3")
        for word, nearest in (("minimal", "low"), ("medium", "low"), ("xhigh", "high")):
            req = Request(model="kimi-k3", messages=(Message.user("x"),), config=Config(reasoning=Reasoning(effort=word)))
            out = adapted(lm, req)
            a = out["config.reasoning.effort"]
            assert (a.action, a.asked, a.applied) == ("clamped", word, nearest)
            assert out["__body__"]["reasoning_effort"] == nearest
            refuses(lm, req, "config.reasoning.effort")
        # The knob is per server: the plain dialect still passes any word through.
        groq = LMRouter(RouterConfig(env={"GROQ_API_KEY": "k"})).lm("groq:m")
        groq.build_request(Request(model="m", messages=(Message.user("x"),), config=Config(reasoning=Reasoning(effort="medium"))), stream=False)
        with pytest.raises(ValueError):
            OpenAIChatCompat(reasoning_efforts=("low", "off"))
        with pytest.raises(ValueError):
            OpenAIChatCompat(reasoning_efforts=("bogus",))

    def test_quota_error_type_is_billing_not_rate_limit(self) -> None:
        # platform.kimi.ai errors.md: `exceeded_current_quota_error` rides HTTP
        # 429 for an insufficient balance; retrying a drained balance is wrong.
        import json

        from lm15.errors import BillingError, RateLimitError

        lm = LMRouter(RouterConfig(env={"MOONSHOTAI_API_KEY": "k"})).lm("moonshotai:kimi-k3")
        err = lm.normalize_error(429, json.dumps({"error": {"type": "exceeded_current_quota_error", "message": "Account balance is insufficient"}}))
        assert isinstance(err, BillingError) and err.provider_code == "exceeded_current_quota_error"
        err = lm.normalize_error(429, json.dumps({"error": {"type": "rate_limit_reached_error", "message": "Organization-level RPM limit reached"}}))
        assert isinstance(err, RateLimitError)

    def test_responses_wire_declaration(self) -> None:
        d = PROVIDERS["moonshotai-responses"]
        assert d.dialect == "openai-responses" and d.bound and d.adapter is OpenAILM
        assert d.env_keys == ("MOONSHOTAI_API_KEY", "MOONSHOT_API_KEY")
        assert d.base_url == "https://api.moonshot.ai/v1"
        assert d.access.supports == EndpointSupport(complete=True, stream=True, responses_api=True, models=True)
        compat = OpenAIResponsesCompat.preset("moonshotai")
        assert compat.reasoning_format == "responses_reasoning" and compat.developer_role == "developer"
        assert compat.cache_control == "openai_implicit" and compat.builtin_tools == "verbatim"  # the canonical `web_search` is the wire type

    def test_responses_wire_replays_reasoning_as_summary_text(self) -> None:
        # Stateless server (store always false): the reasoning item goes back
        # with its summary text, which is what the server returned (live 2026-09-03).
        import json

        from lm15 import BuiltinTool, Config, Message, Reasoning, Request
        from lm15.types import ContinuationState, ThinkingPart, ToolCallPart

        lm = LMRouter(RouterConfig(env={"MOONSHOTAI_API_KEY": "k"})).lm("moonshotai-responses:kimi-k3")
        thinking = ThinkingPart(text="Need the tool.", continuation=(ContinuationState(provider="openai", kind="reasoning_item", data={"id": "rs_1"}),))
        msgs = (Message.user("weather?"), Message.assistant((thinking, ToolCallPart(id="get_weather_0", name="get_weather", input={"city": "Paris"}))), Message.tool("get_weather_0", "Sunny"))
        body = json.loads(lm.build_request(Request(model="kimi-k3", messages=msgs, tools=(BuiltinTool(name="web_search"),), config=Config(max_tokens=50, reasoning=Reasoning(effort="low"))), stream=False).body)
        reasoning_items = [i for i in body["input"] if i.get("type") == "reasoning"]
        assert reasoning_items == [{"type": "reasoning", "id": "rs_1", "summary": [{"type": "summary_text", "text": "Need the tool."}]}]
        assert body["tools"] == [{"type": "web_search"}] and body["reasoning"] == {"effort": "low"} and body["max_output_tokens"] == 50

    def test_anthropic_wire_declaration(self) -> None:
        d = PROVIDERS["moonshotai-anthropic"]
        assert d.dialect == "anthropic" and d.bound and d.adapter is AnthropicLM
        assert d.env_keys == ("MOONSHOTAI_API_KEY", "MOONSHOT_API_KEY")
        assert d.base_url == "https://api.moonshot.ai/anthropic/v1"
        assert d.access.auth_scheme == ("bearer",)  # messages--create.md bearerAuth; ANTHROPIC_AUTH_TOKEN in the Claude Code guide
        assert d.access.supports == EndpointSupport(complete=True, stream=True)  # GET /models is 404 on the /anthropic root (live)
        compat = AnthropicCompat.preset("moonshotai")
        assert compat.thinking_format == "effort" and compat.thinking_replay == "unsigned"
        assert compat.sampling_params == "reject" and compat.parallel_tool_calls == "reject"
        assert compat.reasoning_efforts == ("low", "high", "max") and compat.model_prefixes == ("kimi-",)

    def test_anthropic_wire_shapes_and_refusals(self) -> None:
        import json

        from lm15 import Config, Message, Reasoning, Request, ToolChoice
        from lm15.errors import UnsupportedFeatureError
        from lm15.types import TextPart, ThinkingPart

        lm = LMRouter(RouterConfig(env={"MOONSHOTAI_API_KEY": "k"})).lm("moonshotai-anthropic:kimi-k3")

        def body(cfg: Config, msgs=(Message.user("x"),)) -> dict:
            return json.loads(lm.build_request(Request(model="kimi-k3", messages=msgs, config=cfg), stream=False).body)

        # The documented dial alone; off as the honoured disable; absence sends nothing.
        assert body(Config(max_tokens=50, reasoning=Reasoning(effort="low")))["output_config"] == {"effort": "low"}
        assert "thinking" not in body(Config(max_tokens=50, reasoning=Reasoning(effort="low")))
        assert body(Config(max_tokens=50, reasoning=Reasoning(effort="off")))["thinking"] == {"type": "disabled"}
        plain = body(Config(max_tokens=50))
        assert "thinking" not in plain and "output_config" not in plain
        # An unsigned thinking part goes back as a thinking block, not text (signature is "" on this server).
        replay = body(Config(max_tokens=50), msgs=(Message.user("x"), Message.assistant((ThinkingPart(text="hmm"), TextPart(text="ok"))), Message.user("y")))
        assert replay["messages"][1]["content"][0] == {"type": "thinking", "thinking": "hmm"}
        # Silent cells (live 2026-09-03): MAP-13 adapts and records, never sends the ignored value.
        from tests._adapt import adapted, refuses
        for cfg, field, action, wire_key in (
            (Config(max_tokens=50, reasoning=Reasoning(effort="medium")), "config.reasoning.effort", "clamped", None),
            (Config(max_tokens=50, temperature=0.5), "config.temperature", "dropped", "temperature"),
            (Config(max_tokens=50, top_k=1), "config.top_k", "dropped", "top_k"),
            (Config(max_tokens=50, tool_choice=ToolChoice(mode="auto", parallel=False)), "config.tool_choice.parallel", "dropped", None),
        ):
            req = Request(model="kimi-k3", messages=(Message.user("x"),), config=cfg)
            out = adapted(lm, req)
            assert out[field].action == action
            if wire_key:
                assert wire_key not in out["__body__"]
            refuses(lm, req, field)
        assert adapted(lm, Request(model="kimi-k3", messages=(Message.user("x"),), config=Config(max_tokens=50, reasoning=Reasoning(effort="medium"))))["__body__"]["output_config"] == {"effort": "low"}
        # Plain Anthropic is untouched: unsigned thinking still replays as text there, sampling goes out.
        plain_lm = LMRouter(RouterConfig(env={"ANTHROPIC_API_KEY": "k"})).lm("anthropic:claude-sonnet-4-5")
        out = json.loads(plain_lm.build_request(Request(model="claude-sonnet-4-5", messages=(Message.user("x"), Message.assistant((ThinkingPart(text="hmm"),)), Message.user("y")), config=Config(max_tokens=50, temperature=0.5)), stream=False).body)
        assert out["messages"][1]["content"][0] == {"type": "text", "text": "hmm"} and out["temperature"] == 0.5


class TestMeta:
    """Meta Model API: one key, three wires, three provider strings
    (research/providers/meta/README.md; live 2026-09-03)."""

    ENV = {"META_API_KEY": "LLM|1|k"}

    def test_declarations(self) -> None:
        for pid, dialect, cls, url in (
            ("meta", "openai-responses", OpenAILM, "https://api.meta.ai/v1"),
            ("meta-chat", "openai-chat", OpenAIChatLM, "https://api.meta.ai/v1"),
            ("meta-anthropic", "anthropic", AnthropicLM, "https://api.meta.ai/v1"),
        ):
            d = PROVIDERS[pid]
            assert d.dialect == dialect and d.bound and d.compat == "meta" and d.adapter is cls, pid
            assert d.env_keys == ("META_API_KEY",) and d.base_url == url, pid
            assert d.access.auth_header == "bearer", pid
        # The account surfaces ride the Responses entry; the other two are chat wires only.
        assert PROVIDERS["meta"].supports == EndpointSupport(complete=True, stream=True, files=True, images=True, responses_api=True, models=True)
        assert PROVIDERS["meta-chat"].supports == EndpointSupport(complete=True, stream=True, models=True)
        assert PROVIDERS["meta-anthropic"].supports == EndpointSupport(complete=True, stream=True, models=True)

    def test_router_binds_all_three(self) -> None:
        from lm15.providers import AsyncAnthropicLM, AsyncOpenAIChatLM, AsyncOpenAILM
        from lm15.router import AsyncLMRouter

        router = LMRouter(RouterConfig(env=self.ENV))
        for pid, cls, acls in (("meta", OpenAILM, AsyncOpenAILM), ("meta-chat", OpenAIChatLM, AsyncOpenAIChatLM),
                               ("meta-anthropic", AnthropicLM, AsyncAnthropicLM)):
            lm = router.lm(f"{pid}:muse-spark-1.3")
            assert isinstance(lm, cls) and lm.provider == pid and lm.base_url == "https://api.meta.ai/v1"
            alm = AsyncLMRouter(RouterConfig(env=self.ENV)).lm(f"{pid}:muse-spark-1.3")
            assert isinstance(alm, acls) and alm.base_url == lm.base_url and alm.provider == pid
        # Meta's own docs name the variable MODEL_API_KEY; lm15 never reads a
        # vendor-less name (it could be another tool's secret).
        from lm15.router import MissingCredentialError

        with pytest.raises(MissingCredentialError, match="META_API_KEY"):
            LMRouter(RouterConfig(env={"MODEL_API_KEY": "LLM|1|k"})).lm("meta:m")

    def test_responses_wire(self) -> None:
        import json

        from lm15 import CacheConfig, Config, FunctionTool, Message, Reasoning, Request
        from lm15.types import TextPart, ToolCallPart

        lm = LMRouter(RouterConfig(env=self.ENV)).lm("meta:muse-spark-1.3")
        tool = FunctionTool(name="w", description="d", parameters={"type": "object", "properties": {}})

        def body(msgs, cfg=Config(max_tokens=100), tools=(tool,), **kw):
            return json.loads(lm.build_request(Request(model="muse-spark-1.3", messages=msgs, tools=tools, config=cfg, **kw), stream=False).body)

        b = body((Message.user("x"),), Config(max_tokens=100, reasoning=Reasoning(effort="low", summary="auto"), user_id="u"), system="s")
        assert b["max_output_tokens"] == 100 and b["reasoning"] == {"effort": "low", "summary": "auto"}
        assert b["instructions"] == "s" and b["safety_identifier"] == "u"
        # Off reaches the wire as effort none; the server refuses it loudly (live 2026-09-03: HTTP 400).
        assert body((Message.user("x"),), Config(reasoning=Reasoning(effort="off")))["reasoning"] == {"effort": "none"}
        # cache_control="openai_implicit": the two documented fields go; the breakpoint mark (an undocumented
        # field the server swallows silently — probe-cache-breakpoint-mark-raw) is never placed; off sends nothing.
        b = body((Message.user("x"),), Config(max_tokens=100, cache=CacheConfig(key="k", retention="long", prefix_until_index=0)))
        assert b["prompt_cache_key"] == "k" and b["prompt_cache_retention"] == "24h"
        assert "prompt_cache_options" not in b and "prompt_cache_breakpoint" not in json.dumps(b)
        assert "prompt_cache_breakpoint" not in json.dumps(body((Message.user("x"),), Config(max_tokens=100, cache=CacheConfig(prefix="stable")), system="s"))
        assert not [k for k in body((Message.user("x"),), Config(max_tokens=100, cache=CacheConfig(mode="off"))) if "cache" in k]
        # Builtin tools: Meta's own vocabulary; a name Meta lacks goes out verbatim (the server answers 400, live).
        from lm15 import BuiltinTool
        assert body((Message.user("x"),), Config(max_tokens=100), tools=(BuiltinTool(name="web_search"), BuiltinTool(name="code_execution")))["tools"] == [
            {"type": "web_search"}, {"type": "code_execution"}]
        oa_tools = json.loads(OpenAILM(api_key="k").build_request(Request(model="gpt-5", messages=(Message.user("x"),), tools=(BuiltinTool(name="web_search"),)), stream=False).body)["tools"]
        assert oa_tools == [{"type": "web_search_preview"}]
        # Assistant text that precedes a function_call is replayed with phase: commentary (the server stamped it on the way out).
        turn = Message.assistant((TextPart(text="Let me check."), ToolCallPart(id="c1", name="w", input={})))
        items = body((Message.user("x"), turn, Message.tool("c1", "ok")))["input"]
        assert items[1] == {"role": "assistant", "content": [{"type": "output_text", "text": "Let me check."}], "phase": "commentary"}
        assert items[2]["type"] == "function_call" and items[3]["type"] == "function_call_output"
        # A plain final answer carries no phase; OpenAI never gets the field.
        plain = body((Message.user("x"), Message.assistant("Done."), Message.user("y")))["input"]
        assert "phase" not in plain[1]
        oa = json.loads(OpenAILM(api_key="k").build_request(Request(model="gpt-5", messages=(Message.user("x"), turn, Message.tool("c1", "ok")), tools=(tool,)), stream=False).body)
        assert "phase" not in oa["input"][1]

    def test_image_edit_field_is_the_servers(self) -> None:
        from lm15 import ImageGenerationRequest, ImagePart

        req = ImageGenerationRequest(model="muse-image-1.0", prompt="p", images=(ImagePart(media_type="image/png", data="aGk="), ImagePart(media_type="image/png", data="aGk=")))
        meta = LMRouter(RouterConfig(env=self.ENV)).lm("meta:x")._image_generate_request(req).body
        assert b'name="image[0]"' in meta and b'name="image[1]"' in meta and b'name="image[]"' not in meta
        openai = OpenAILM(api_key="k")._image_generate_request(req).body
        assert openai.count(b'name="image[]"') == 2 and b"image[0]" not in openai

    def test_chat_wire(self) -> None:
        import json

        from lm15 import Config, Message, Reasoning, Request

        compat = OpenAIChatCompat.preset("meta")
        assert compat.instruction_role == "developer" and compat.max_tokens_field == "max_completion_tokens"
        assert compat.thinking_format == "reasoning_effort" and compat.user_field == "safety_identifier"
        assert compat.cache_control == "openai_implicit"
        assert compat.forced_tool_choice is None  # the server answers 400 itself (loud, live 2026-09-03)
        lm = LMRouter(RouterConfig(env=self.ENV)).lm("meta-chat:muse-spark-1.3")
        b = json.loads(lm.build_request(Request(model="muse-spark-1.3", system="s", messages=(Message.user("x"),),
                                                config=Config(max_tokens=100, reasoning=Reasoning(effort="low"), user_id="u")), stream=False).body)
        assert b["messages"][0] == {"role": "developer", "content": "s"}
        assert b["max_completion_tokens"] == 100 and b["reasoning_effort"] == "low" and b["safety_identifier"] == "u"
        assert "user" not in b

    def test_anthropic_wire(self) -> None:
        import json

        from lm15 import Config, Message, Reasoning, Request
        from lm15.errors import UnsupportedFeatureError

        compat = AnthropicCompat.preset("meta")
        assert compat.thinking_format == "adaptive" and compat.cache_control == "none"
        lm = LMRouter(RouterConfig(env=self.ENV)).lm("meta-anthropic:muse-spark-1.3")
        say = (Message.user("x"),)

        def body(cfg):
            return json.loads(lm.build_request(Request(model="muse-spark-1.3", messages=say, config=cfg), stream=False).body)

        # Every model is the adaptive class here — no model-name table.
        b = body(Config(max_tokens=100, reasoning=Reasoning(effort="low")))
        assert b["thinking"] == {"type": "adaptive"} and b["output_config"] == {"effort": "low"} and b["max_tokens"] == 100
        # Explicit off reaches the wire so the server refuses it loudly (live 2026-09-03: HTTP 400).
        assert body(Config(max_tokens=100, reasoning=Reasoning(effort="off")))["thinking"] == {"type": "disabled"}
        # Absence stays absence (the server reasons by default either way).
        assert "thinking" not in body(Config(max_tokens=100))
        # minimal goes verbatim (the server answers 400 — live 2026-09-03); a budget is a
        # silent no-op there → MAP-13 drops it and records (effort carries the intent).
        from tests._adapt import adapted
        assert body(Config(max_tokens=100, reasoning=Reasoning(effort="minimal")))["output_config"] == {"effort": "minimal"}
        out = adapted(lm, Request(model="muse-spark-1.3", messages=say, config=Config(max_tokens=100, reasoning=Reasoning(effort="low", thinking_budget=2048))))
        assert out["config.reasoning.thinking_budget"].action == "dropped" and out["__body__"]["output_config"] == {"effort": "low"}
        # The bearer header, not x-api-key.
        headers = {k.lower(): v for k, v in lm.build_request(Request(model="m", messages=say, config=Config(max_tokens=10)), stream=False).headers}
        assert headers["authorization"].startswith("Bearer ") and "x-api-key" not in headers

    def test_responses_compat_preset_table(self) -> None:
        # The Responses dialect binds like the other two: preset name → compat + default base_url.
        lm = OpenAILM(api_key="k", compat="meta")
        assert lm.base_url == "https://api.meta.ai/v1"
        assert OpenAILM(api_key="k", compat="meta", base_url="http://localhost:1/v1").base_url == "http://localhost:1/v1"
        assert OpenAIResponsesCompat.preset("responses") == OpenAIResponsesCompat.preset("openai")
        with pytest.raises(ValueError):
            OpenAIResponsesCompat.preset("nope")
        with pytest.raises(TypeError):
            OpenAILM(api_key="k", compat=42)  # type: ignore[arg-type]
