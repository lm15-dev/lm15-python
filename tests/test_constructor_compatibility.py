"""Constructor layouts pinned to the API before cloud/compat field additions."""

import copy
import dataclasses
import inspect
import pickle

import pytest

from lm15.compat import (
    AnthropicCompat,
    OpenAIChatCompat,
    OpenAIResponsesCompat,
    ResolvedAnthropicCompat,
    ResolvedOpenAIChatCompat,
    ResolvedOpenAIResponsesCompat,
    merge_openai_chat_compat,
    merge_openai_responses_compat,
    resolve_anthropic_compat,
    resolve_openai_chat_compat,
    resolve_openai_responses_compat,
)
from lm15.features import AccessPolicy, EndpointSupport, HostSpec, ProviderManifest


# Deliberately explicit: deriving these lists from current dataclass fields would
# bless the very positional shifts these tests are meant to detect.
ACCESS_NAMES = (
    "provider", "supports", "auth_modes", "enterprise_variants", "env_keys",
    "credential_policy", "auth_header", "headers", "login_hint", "backend",
    "backend_options", "system_prefix", "base_url",
)
ACCESS_VALUES = (
    "review", EndpointSupport(files=True), ("api-key",), ("enterprise",),
    ("REVIEW_KEY",), "key", "x-api-key", (("X-Review", "yes"),),
    "login here", "custom", {"option": "value"}, "prefix", "https://review.invalid",
)
ANTHROPIC_NAMES = (
    "thinking_format", "cache_control", "structured_output", "parallel_tool_calls",
    "model_prefixes", "extensions",
)
ANTHROPIC_VALUES = ("deepseek", "none", "reject", "reject", ("deepseek-",), {"a": 1})
RESPONSES_NAMES = (
    "developer_role", "max_output_tokens_field", "reasoning_format",
    "tool_result_name", "strict_tools", "cache_control", "routing", "extensions",
)
RESPONSES_VALUES = ("system", "max_tokens", "deepseek", "include", "include", "none", {"r": 1}, {"e": 2})
CHAT_NAMES = (
    "instruction_role", "max_tokens_field", "stream_usage", "tool_result_name",
    "assistant_after_tool_result", "thinking_format", "thinking_replay",
    "assistant_reasoning_content", "strict_tools", "builtin_tools", "cache_control",
    "user_field", "forced_tool_choice", "json_schema", "routing", "extensions",
)
CHAT_VALUES = (
    "developer", "max_tokens", "omit", "include", "insert", "deepseek", "native",
    "include_empty", "include", "groq", "none", "user_id", "reject", "reject",
    {"r": 1}, {"e": 2},
)
COMPAT_CASES = [
    (AnthropicCompat, ANTHROPIC_NAMES, ANTHROPIC_VALUES,
     {"thinking_replay": "unsigned", "sampling_params": "reject", "reasoning_efforts": ("low", "high"), "tool_result_media": "reject"}),
    (ResolvedAnthropicCompat, ANTHROPIC_NAMES, ANTHROPIC_VALUES,
     {"thinking_replay": "unsigned", "sampling_params": "reject", "reasoning_efforts": ("low", "high"), "tool_result_media": "reject"}),
    (OpenAIResponsesCompat, RESPONSES_NAMES, RESPONSES_VALUES,
     {"commentary_phase": "tag", "edit_image_field": "indexed", "builtin_tools": "verbatim", "tool_result_media": "images"}),
    (ResolvedOpenAIResponsesCompat, RESPONSES_NAMES, RESPONSES_VALUES,
     {"commentary_phase": "tag", "edit_image_field": "indexed", "builtin_tools": "verbatim", "tool_result_media": "images"}),
    (OpenAIChatCompat, CHAT_NAMES, CHAT_VALUES,
     {"reasoning_efforts": ("low", "high"), "model_overrides": (("review-", {"json_schema": "send"}),), "tool_result_media": "native", "token_scoring": "none"}),
    (ResolvedOpenAIChatCompat, CHAT_NAMES, CHAT_VALUES, {"reasoning_efforts": ("low", "high"), "tool_result_media": "native", "token_scoring": "none"}),
]


def assert_layout(cls, old_names, new_names):
    params = inspect.signature(cls).parameters
    assert tuple(n for n, p in params.items() if p.kind == p.POSITIONAL_OR_KEYWORD) == old_names
    assert {n for n, p in params.items() if p.kind == p.KEYWORD_ONLY} == set(new_names)
    assert cls.__match_args__ == old_names
    assert all(p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD) for p in params.values())


def test_access_policy_old_constructors_and_signature():
    assert ProviderManifest is AccessPolicy
    positional = AccessPolicy(*ACCESS_VALUES)
    keyword = AccessPolicy(**dict(zip(ACCESS_NAMES, ACCESS_VALUES)))
    assert positional == keyword
    for name, value in zip(ACCESS_NAMES, ACCESS_VALUES):
        assert getattr(positional, name) == value
    assert positional.auth_scheme == ("x-api-key",)
    assert AccessPolicy(provider="review", auth_header="x-api-key").auth_scheme == ("x-api-key",)
    assert AccessPolicy("review").auth_header == "bearer"
    assert AccessPolicy("review").auth_scheme == ("bearer",)
    assert_layout(AccessPolicy, ACCESS_NAMES, ("auth_scheme", "host"))
    assert "auth_header" not in {f.name for f in dataclasses.fields(AccessPolicy)}
    assert {f.name for f in dataclasses.fields(AccessPolicy) if f.kw_only} == {"auth_scheme", "host"}
    with pytest.raises(TypeError):
        AccessPolicy(*ACCESS_VALUES, None)


@pytest.mark.parametrize("schemes,header", [
    (("bearer",), "bearer"),
    (("x-api-key",), "x-api-key"),
    (("api-key", "bearer"), "x-api-key"),
    (("query-key",), "bearer"),
    (("sigv4",), "bearer"),
    (("sigv4", "api-key", "bearer"), "x-api-key"),
])
def test_access_new_keywords_copy_and_replace_preserve_canonical_schemes(schemes, header):
    host = HostSpec("https://review.invalid", sigv4_service="review")
    policy = AccessPolicy("review", auth_scheme=schemes, host=host, headers=(("X-Old", "1"),))
    assert policy.auth_scheme == schemes
    assert policy.auth_header == header
    for cloned in (dataclasses.replace(policy), copy.copy(policy), copy.deepcopy(policy),
                   pickle.loads(pickle.dumps(policy))):
        assert cloned == policy
        assert cloned is not policy
        assert cloned.auth_scheme == schemes
    renamed = dataclasses.replace(policy, provider="renamed")
    assert renamed.provider == "renamed"
    assert renamed.auth_scheme == schemes
    updated = policy.with_headers({"x-old": "2", "X-New": "3"})
    assert updated.headers == (("x-old", "2"), ("X-New", "3"))
    assert updated.auth_scheme == schemes
    assert updated.host is host
    assert policy.headers == (("X-Old", "1"),)
    assert dataclasses.replace(policy, auth_scheme=("query-key",)).auth_scheme == ("query-key",)
    for legacy in ("bearer", "x-api-key"):
        changed = dataclasses.replace(policy, auth_header=legacy)
        assert changed.auth_scheme == (legacy,)
        assert changed.auth_header == legacy
    with pytest.raises(dataclasses.FrozenInstanceError):
        policy.auth_scheme = ("bearer",)
    with pytest.raises(dataclasses.FrozenInstanceError):
        del policy.host
    assert not hasattr(policy, "__dict__")


def test_access_legacy_alias_is_explicit_override_not_stored_state():
    policy = AccessPolicy("review", auth_header="x-api-key", auth_scheme=("api-key", "bearer"))
    assert policy.auth_scheme == ("x-api-key",)
    changed = dataclasses.replace(policy, auth_scheme=("api-key", "query-key"))
    assert changed.auth_scheme == ("api-key", "query-key")
    assert changed.auth_header == "x-api-key"


def test_access_normalization_and_independent_defaults():
    options = {"option": "value"}
    policy = AccessPolicy("review", auth_modes=["key"], enterprise_variants=["cloud"],
                          env_keys=["REVIEW_KEY"], headers=[["X-Count", 1]],
                          backend_options=options, auth_scheme="x-api-key")
    options["option"] = "changed"
    assert policy.auth_modes == ("key",)
    assert policy.enterprise_variants == ("cloud",)
    assert policy.env_keys == ("REVIEW_KEY",)
    assert policy.headers == (("X-Count", "1"),)
    assert policy.backend_options == {"option": "value"}
    assert policy.auth_scheme == ("x-api-key",)
    a, b = AccessPolicy("a"), AccessPolicy("b")
    assert a.supports is not b.supports
    assert a.backend_options is not b.backend_options


@pytest.mark.parametrize("kwargs,match", [
    ({"provider": ""}, "provider must be non-empty"),
    ({"credential_policy": "invalid"}, "unknown credential_policy"),
    ({"credential_policy": "oauth", "env_keys": ("REVIEW_KEY",)}, "no env_keys"),
    ({"auth_header": "invalid"}, "unknown auth_header"),
    ({"auth_scheme": ()}, "at least one scheme"),
    ({"auth_scheme": ("invalid",)}, "unknown auth_scheme"),
    ({"auth_scheme": ("sigv4",)}, "sigv4 needs a host"),
    ({"auth_scheme": ("sigv4",), "host": HostSpec("https://review.invalid")}, "sigv4 needs a host"),
    ({"credential_policy": "aws-chain"}, "cloud chain policy needs a host"),
])
def test_access_constructor_and_replace_keep_validations(kwargs, match):
    with pytest.raises(ValueError, match=match):
        AccessPolicy(**{"provider": "review", **kwargs})
    with pytest.raises(ValueError, match=match):
        dataclasses.replace(AccessPolicy("review"), **kwargs)


@pytest.mark.parametrize("cls,names,values,new", COMPAT_CASES, ids=lambda x: x.__name__ if isinstance(x, type) else None)
def test_compat_old_full_layout_and_new_keywords(cls, names, values, new):
    old = dict(zip(names, values))
    positional = cls(*values)
    assert positional == cls(**old)
    for name, value in old.items():
        assert getattr(positional, name) == value
    assert_layout(cls, names, new)
    assert {f.name for f in dataclasses.fields(cls) if f.kw_only} == set(new)
    policy = cls(*values, **new)
    assert policy == cls(**old, **new)
    for name, value in new.items():
        assert getattr(policy, name) == value
    for cloned in (dataclasses.replace(policy), copy.copy(policy), copy.deepcopy(policy),
                   pickle.loads(pickle.dumps(policy))):
        assert cloned == policy
        assert cloned is not policy
    assert dataclasses.replace(positional, **new) == policy
    changed = dataclasses.replace(policy, extensions={"changed": True})
    assert changed.extensions == {"changed": True}
    assert policy.extensions == old["extensions"]
    for name, value in new.items():
        assert getattr(changed, name) == value
    with pytest.raises(TypeError):
        cls(*values, next(iter(new.values())))
    with pytest.raises(dataclasses.FrozenInstanceError):
        policy.extensions = {}
    assert not hasattr(policy, "__dict__")


def test_anthropic_two_positionals_still_mean_format_and_cache():
    policy = AnthropicCompat("deepseek", "none")
    assert policy.thinking_format == "deepseek"
    assert policy.cache_control == "none"
    assert policy.thinking_replay is None


@pytest.mark.parametrize("kwargs,match", [
    ({"thinking_replay": "none"}, "unsupported thinking_replay"),
    ({"sampling_params": "invalid"}, "unsupported sampling_params"),
    ({"reasoning_efforts": ("off",)}, "reasoning_efforts must be a tuple"),
    ({"reasoning_efforts": ["low"]}, "reasoning_efforts must be a tuple"),
    ({"model_prefixes": ()}, "model_prefixes must be"),
    ({"extensions": []}, "must be a JSON object"),
])
def test_anthropic_new_keywords_and_replace_still_validate(kwargs, match):
    with pytest.raises((ValueError, TypeError), match=match):
        AnthropicCompat("deepseek", "none", **kwargs)
    with pytest.raises((ValueError, TypeError), match=match):
        dataclasses.replace(AnthropicCompat("deepseek", "none"), **kwargs)


def test_compat_resolution_merge_and_model_copy_keep_new_fields():
    anthropic = AnthropicCompat(*ANTHROPIC_VALUES, thinking_replay="unsigned",
                                sampling_params="reject", reasoning_efforts=("low", "high"), tool_result_media="reject")
    assert resolve_anthropic_compat(anthropic) == ResolvedAnthropicCompat(**dataclasses.asdict(anthropic))
    responses = OpenAIResponsesCompat(*RESPONSES_VALUES, commentary_phase="tag",
                                      edit_image_field="indexed", builtin_tools="verbatim", tool_result_media="images")
    merged = merge_openai_responses_compat(responses, OpenAIResponsesCompat(extensions={"new": 3}))
    assert merged == dataclasses.replace(responses, extensions={"e": 2, "new": 3})
    assert resolve_openai_responses_compat(responses) == ResolvedOpenAIResponsesCompat(**dataclasses.asdict(responses))
    chat = OpenAIChatCompat(*CHAT_VALUES, reasoning_efforts=("low", "high"), tool_result_media="native", token_scoring="none")
    merged_chat = merge_openai_chat_compat(chat, OpenAIChatCompat(extensions={"new": 3}))
    assert merged_chat == dataclasses.replace(chat, extensions={"e": 2, "new": 3})
    resolved_kwargs = dataclasses.asdict(chat)
    resolved_kwargs.pop("model_overrides")
    assert resolve_openai_chat_compat(chat) == ResolvedOpenAIChatCompat(**resolved_kwargs)
    overridden = dataclasses.replace(chat, model_overrides=(("review-", {"json_schema": "send"}),))
    assert overridden.for_model("review-model") == dataclasses.replace(chat, json_schema="send")
    assert overridden.for_model("other") is overridden


def test_endpoint_support_layout_is_unchanged():
    names = ("complete", "stream", "live", "files", "batches", "images", "speech",
             "video", "responses_api", "models", "caches", "extra")
    values = (False, False, True, True, True, True, True, True, True, True, True, frozenset({"custom"}))
    assert_layout(EndpointSupport, names, ())
    support = EndpointSupport(*values)
    assert support == EndpointSupport(**dict(zip(names, values)))
    assert support.supports_endpoint("custom")
    assert dataclasses.replace(support, complete=True).complete
