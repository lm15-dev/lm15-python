"""Offline regression tests for cloud routing and credential boundaries."""

import os
import datetime as dt
import inspect
import io
import json
import subprocess
import urllib.error
import urllib.request
from types import SimpleNamespace

import pytest

from lm15 import access
from lm15.cloud import chains, sigv4
from lm15.credentials import ApiKey, AwsCredentials, BearerToken, credential_from_dict
from lm15.errors import AuthError, NotConfiguredError, UnsupportedFeatureError
from lm15.providers import AsyncGeminiLM, AsyncOpenAILM, GeminiLM, OpenAILM
from lm15.registry import PROVIDERS
from lm15.router import AsyncLMRouter, LMRouter, RouterConfig
from lm15.types import LiveConfig, Message, Request

NOW = dt.datetime(2026, 9, 3, 12, tzinfo=dt.timezone.utc)
_SRC = chains.CredentialSource(rung="az", label="az account get-access-token")
SECRET = "SECRET-SENTINEL-DO-NOT-PRINT"


def request(model="m"):
    return Request(model=model, messages=[Message.user("hello")])


@pytest.mark.parametrize("router_type", [LMRouter, AsyncLMRouter])
@pytest.mark.parametrize("provider", [p for p, d in PROVIDERS.items() if d.hosted])
def test_every_cloud_door_routes_and_builds(router_type, provider):
    definition = PROVIDERS[provider]
    values = {"region": "us-east-1", "resource": "test", "project": "p", "workspace": "w"}
    settings = {s.name: values[s.name] for s in definition.access.host.settings if s.name in values}
    credential = BearerToken("test") if provider in ("vertex", "vertex-anthropic") else "test"
    router = router_type(RouterConfig(env={}, api_keys={provider: credential}, settings={provider: settings}))
    lm = router.lm(f"{provider}:m")
    assert lm is router.lm(f"{provider}:other")
    inner = getattr(lm, "_inner", lm)
    built = inner.build_request(request(), stream=False)
    assert built.url.startswith("https://")
    assert "test" in str(built.headers) or "key=test" in built.url


@pytest.mark.parametrize("cls", [OpenAILM, AsyncOpenAILM])
def test_openai_old_positional_constructor_is_preserved(cls):
    positional = [n for n, p in inspect.signature(cls).parameters.items()
                  if p.kind == p.POSITIONAL_OR_KEYWORD]
    assert positional == ["api_key", "transport", "base_url", "profile", "access", "credentials_path", "account_id"]
    for name in ("compat", "settings", "clock"):
        assert inspect.signature(cls).parameters[name].kind == inspect.Parameter.KEYWORD_ONLY


def test_router_settings_do_not_shift_transport_position():
    names = list(inspect.signature(RouterConfig).parameters)
    assert names.index("transport") < names.index("settings")


@pytest.mark.parametrize("policy,preset", [(access.META, "meta"), (access.MOONSHOTAI_RESPONSES, "moonshotai")])
def test_direct_responses_policy_uses_registry_compat(policy, preset):
    direct = OpenAILM(api_key="test", access=policy)
    explicit = OpenAILM(api_key="test", access=policy, compat=preset)
    assert direct._compat(request()) == explicit._compat(request())
    assert direct.build_request(request(), False).body == explicit.build_request(request(), False).body


@pytest.mark.parametrize("setting", ["resource", "region", "location"])
@pytest.mark.parametrize("value", ["evil.example/", "user@evil.example", "x?key=", "x#", "x\n"])
def test_host_settings_cannot_replace_the_authority(setting, value):
    from lm15.cloud.hosts import render_base_url

    policy = {"resource": access.AZURE, "region": access.BEDROCK_CHAT, "location": access.VERTEX}[setting]
    with pytest.raises(NotConfiguredError, match="DNS label"):
        render_base_url(policy.host, {setting: value, "project": "p"})


def test_vertex_model_and_project_stay_inside_path_components():
    lm = GeminiLM(api_key=BearerToken("test"), access=access.VERTEX, settings={"project": "p?query#frag"})
    built = lm.build_request(request("models/gemini?query#frag"), False)
    assert "/projects/p%3Fquery%23frag/" in built.url
    assert built.url.endswith("/models/gemini%3Fquery%23frag:generateContent")


@pytest.mark.parametrize("relative", ["@evil.example/creds", "//evil.example/creds", "\n/creds", "/creds#fragment"])
def test_container_relative_uri_cannot_replace_host(relative):
    with pytest.raises(NotConfiguredError):
        chains._container_config(chains.ChainContext(env={"AWS_CONTAINER_CREDENTIALS_RELATIVE_URI": relative}))


@pytest.mark.parametrize("url", ["file://localhost/creds", "ftp://localhost/creds", "https:///creds", "https://user@host/creds"])
def test_container_full_uri_requires_http_without_userinfo(url):
    with pytest.raises(NotConfiguredError):
        chains._container_config(chains.ChainContext(env={"AWS_CONTAINER_CREDENTIALS_FULL_URI": url}))


def test_online_context_uses_the_same_home_as_the_doctor(tmp_path, monkeypatch):
    from pathlib import Path

    monkeypatch.setattr(Path, "home", lambda: pytest.fail("ambient HOME used"))
    ctx = chains.ChainContext.online({"HOME": str(tmp_path)})
    assert ctx.path("~/.aws/credentials") == tmp_path / ".aws/credentials"
    explicit = tmp_path / "explicit"
    assert chains.ChainContext.online({}, home=explicit).home == explicit


@pytest.mark.parametrize("rung", ["environment", "managed-identity", "metadata", "service-account"])
@pytest.mark.parametrize("status", [199, 302, 400])
def test_oauth_parser_requires_success_status(rung, status):
    with pytest.raises(AuthError):
        chains.token_exchange_parse(access.AZURE, rung, status,
                                     {"access_token": SECRET, "expires_in": 3600},
                                     chains.ChainContext(env={}, now=lambda: NOW))


@pytest.mark.parametrize("token", [None, "", True, 123, [], {"secret": SECRET}])
def test_oauth_parser_rejects_non_string_tokens_without_echoing_them(token):
    with pytest.raises(AuthError) as exc:
        chains.token_exchange_parse(access.AZURE, "environment", 200,
                                     {"access_token": token}, chains.ChainContext(env={}))
    assert SECRET not in str(exc.value) + repr(exc.value)


@pytest.mark.parametrize("field", ["expires_on", "expires_in"])
@pytest.mark.parametrize("value", ["inf", "1e300", "-1e300"])
def test_unrepresentable_oauth_expiry_is_unknown_not_cached_forever(field, value):
    credential = chains._bearer_from_oauth({"access_token": "test", field: value}, NOW, "test")
    assert credential.expires_at is None


@pytest.mark.parametrize("winner", ["pwsh", "azd", None])
def test_azure_tries_developer_commands_through_auth_errors(winner):
    calls = []

    def run(argv, timeout):
        command = argv[0]
        calls.append(command)
        if command != winner:
            raise AuthError("command failed")
        key = {"pwsh": "Token", "azd": "token"}[command]
        return json.dumps({key: "test"})

    ctx = chains.ChainContext(env={"PATH": "/bin", "AZURE_TOKEN_CREDENTIALS": "dev"},
                              files={"/bin/az": "stub", "/bin/pwsh": "stub", "/bin/azd": "stub"}, run=run)
    if winner is None:
        with pytest.raises(AuthError, match="Azure developer credentials failed"):
            chains.resolve(access.AZURE, ctx)
    else:
        assert chains.resolve(access.AZURE, ctx)[0].value == "test"
    assert calls == (["az", "pwsh"] if winner == "pwsh" else ["az", "pwsh", "azd"])


def test_azure_deployed_auth_failure_does_not_try_developer_commands():
    ctx = chains.ChainContext(
        env={"AZURE_TENANT_ID": "tenant", "AZURE_CLIENT_ID": "client", "AZURE_CLIENT_SECRET": "test", "PATH": "/bin"},
        files={"/bin/az": "stub"}, http=lambda *a: (401, {}, b"{}"),
        run=lambda *a: pytest.fail("developer command used after deployed credential failure"),
    )
    with pytest.raises(AuthError, match="Entra client credentials: HTTP 401"):
        chains.resolve(access.AZURE, ctx)


@pytest.mark.parametrize("acquire,key", [(chains._az_cli_acquire, "accessToken"),
                                         (chains._pwsh_acquire, "Token"), (chains._azd_acquire, "token")])
def test_azure_commands_reject_non_string_tokens(acquire, key):
    ctx = chains.ChainContext(env={"PATH": "/bin"},
                              files={"/bin/az": "stub", "/bin/pwsh": "stub", "/bin/azd": "stub"},
                              run=lambda *a: json.dumps({key: {"secret": SECRET}}))
    with pytest.raises(AuthError) as exc:
        acquire(ctx)
    assert SECRET not in str(exc.value) + repr(exc.value)


def test_empty_context_path_does_not_search_process_path(monkeypatch):
    import shutil

    monkeypatch.setattr(shutil, "which", lambda *a, **k: pytest.fail("ambient PATH searched"))
    assert chains.ChainContext(env={}).on_path("az") is None


def test_credential_http_disables_redirects_and_closes_errors(monkeypatch):
    raw = io.BytesIO(b"failure")

    def opener(handler):
        assert isinstance(handler, chains._NoRedirect)
        req = urllib.request.Request("https://source.example", headers={"Authorization": SECRET})
        assert handler.redirect_request(req, None, 302, "found", {}, "https://other.example") is None

        def open_request(*args, **kwargs):
            raise urllib.error.HTTPError("https://source.example", 302, "found", {}, raw)

        return SimpleNamespace(open=open_request)

    monkeypatch.setattr(urllib.request, "build_opener", opener)
    assert chains._default_http("POST", "https://source.example", {}, b"token", 1)[0] == 302
    assert raw.closed


@pytest.mark.parametrize("status", [302, 400, 500])
def test_token_exchange_errors_never_echo_response_bodies(status):
    ctx = chains.ChainContext(env={}, http=lambda *a: (status, {}, json.dumps({"error_description": SECRET}).encode()))
    with pytest.raises(AuthError) as exc:
        chains._exchange(ctx, "POST", "https://example.test", {}, None, "token exchange")
    assert SECRET not in str(exc.value) + repr(exc.value)


def test_command_errors_hide_stderr_and_argv(monkeypatch):
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=1, stderr=SECRET))
    with pytest.raises(AuthError) as exc:
        chains._default_run([SECRET], 1)
    assert SECRET not in str(exc.value) + repr(exc.value)


def test_malformed_aws_config_does_not_echo_secret_lines():
    ctx = chains.ChainContext(env={}, files={"~/.aws/config": SECRET})
    with pytest.raises(NotConfiguredError) as exc:
        chains._aws_config(ctx)
    assert SECRET not in str(exc.value) + repr(exc.value)


@pytest.mark.parametrize("body", [{}, {"AccessKeyId": "id"}, {"SecretAccessKey": "key"}, {"AccessKeyId": 1, "SecretAccessKey": 2}])
def test_malformed_aws_response_is_not_a_none_string_credential(body):
    with pytest.raises(AuthError):
        chains._aws_from_response(body)


def test_chain_context_and_signature_reprs_hide_credentials():
    ctx = chains.ChainContext(env={"KEY": SECRET}, files={"/key": SECRET}, settings={"token": SECRET})
    signed = sigv4.sign(method="GET", url="https://example.test", headers={}, payload=b"",
                        credentials=AwsCredentials("id", "key", SECRET), region="us-east-1", service="sts", now=NOW)
    assert SECRET not in repr(ctx) + repr(signed)


def test_sigv4_drops_stale_session_token():
    signed = sigv4.sign(method="GET", url="https://example.test", headers={"X-Amz-Security-Token": "old"},
                        payload=b"", credentials=AwsCredentials("id", "key"), region="us-east-1", service="sts", now=NOW)
    assert "x-amz-security-token" not in signed.headers
    assert "x-amz-security-token" not in signed.canonical_request


def test_cache_rejects_expired_refresh_results(monkeypatch):
    monkeypatch.setattr(chains, "resolve", lambda *a, **k: (BearerToken("old", NOW), _SRC))
    provider = chains.credential_provider(access.AZURE, chains.ChainContext(env={}, now=lambda: NOW))
    with pytest.raises(AuthError, match="expired"):
        provider()


def test_cli_token_without_expiry_is_not_cached_forever(monkeypatch):
    calls = []

    def resolve(*args, **kwargs):
        calls.append(1)
        return BearerToken(str(len(calls))), _SRC

    monkeypatch.setattr(chains, "resolve", resolve)
    provider = chains.credential_provider(access.AZURE, chains.ChainContext(env={}, now=lambda: NOW))
    assert provider().value == "1"
    assert provider().value == "2"


def test_powershell_resource_is_a_quoted_literal():
    commands = []
    ctx = chains.ChainContext(env={"PATH": "/bin"}, files={"/bin/pwsh": "stub"},
                              settings={"scope": "https://resource/'value/.default"},
                              run=lambda argv, timeout: (commands.append(argv), '{"Token":"t"}')[1])
    assert chains._pwsh_acquire(ctx).value == "t"
    assert "-ResourceUrl 'https://resource/''value'" in commands[0][-1]


@pytest.mark.parametrize("rung,status", [("credential_process", 1), ("container", 302), ("imds", 500)])
def test_failed_credential_status_cannot_produce_keys(rung, status):
    with pytest.raises(AuthError):
        chains.token_exchange_parse(access.BEDROCK_CHAT, rung, status,
                                     {"Version": 1, "AccessKeyId": "id", "SecretAccessKey": "key"},
                                     chains.ChainContext(env={}))


@pytest.mark.parametrize("value", [None, 123, [], {}])
@pytest.mark.parametrize("kind", ["api_key", "bearer_token"])
def test_credential_decoder_rejects_non_string_values(kind, value):
    with pytest.raises(ValueError):
        credential_from_dict({"kind": kind, "value": value})


def test_credential_tags_cannot_be_overridden():
    with pytest.raises(TypeError):
        ApiKey("test", kind="aws")
    with pytest.raises(ValueError):
        AwsCredentials("id", 123)


# Azure Arc keeps its challenge token in one folder per platform. Each test
# runs on the platform whose rules it checks: pathlib reads paths the host's
# way, so faking os.name on the other platform tests nothing real.
_WINDOWS = os.name == "nt"
_ARC_DIR = r"C:\ProgramData\AzureConnectedMachineAgent\Tokens" if _WINDOWS else "/var/opt/azcmagent/tokens"
_ARC_ENV = {"IDENTITY_ENDPOINT": "http://localhost/identity", "IMDS_ENDPOINT": "http://localhost",
            **({"PROGRAMDATA": r"C:\ProgramData"} if _WINDOWS else {})}
_ARC_OUTSIDE = (
    [r"C:\private\key.key", _ARC_DIR + r"\..\key.key", _ARC_DIR + r"\key.txt"]
    if _WINDOWS else
    ["/private/key", "/var/opt/azcmagent/tokens/../key", "/var/opt/azcmagent/tokens/key.txt"]
)


@pytest.mark.parametrize("path", _ARC_OUTSIDE)
def test_arc_challenge_cannot_read_arbitrary_files(monkeypatch, path):
    ctx = chains.ChainContext(env=_ARC_ENV, files={path: SECRET},
                              http=lambda *a: (401, {"www-authenticate": f'Basic realm="{path}"'}, b""))
    monkeypatch.setattr(ctx, "read", lambda path: pytest.fail("untrusted challenge file read"))
    with pytest.raises(AuthError, match="invalid challenge file location"):
        chains._azure_msi_acquire(ctx)


def test_arc_valid_challenge_uses_only_the_agent_token():
    calls = []
    path = (_ARC_DIR + "\\token.key") if _WINDOWS else "/var/opt/azcmagent/tokens/token.key"

    def http(method, url, headers, body, timeout):
        calls.append(headers)
        if len(calls) == 1:
            return 401, {"www-authenticate": f'Basic realm="{path}"'}, b""
        return 200, {}, b'{"access_token":"test","expires_in":3600}'

    ctx = chains.ChainContext(env=_ARC_ENV, files={path: SECRET}, http=http, now=lambda: NOW)
    assert chains._azure_msi_acquire(ctx).value == "test"
    assert calls[1]["Authorization"] == f"Basic {SECRET}"


@pytest.mark.parametrize("cls,policy,settings", [(OpenAILM, access.META, {}), (GeminiLM, access.VERTEX, {"project": "p"})])
def test_unsupported_live_fails_before_opening_a_socket(cls, policy, settings):
    lm = cls(api_key=BearerToken("test"), access=policy, settings=settings)
    with pytest.raises(UnsupportedFeatureError):
        lm.live(LiveConfig(model="m"))


@pytest.mark.parametrize("cls,policy,settings", [(AsyncOpenAILM, access.META, {}), (AsyncGeminiLM, access.VERTEX, {"project": "p"})])
async def test_async_unsupported_live_fails_before_opening_a_socket(cls, policy, settings):
    lm = cls(api_key=BearerToken("test"), access=policy, settings=settings)
    with pytest.raises(UnsupportedFeatureError):
        await lm.live(LiveConfig(model="m"))


def test_wire_request_and_loaded_credential_reprs_hide_secrets():
    from lm15.cloud.hosts import FinishedRequest
    from lm15.transports import TransportRequest

    values = [
        access.LoadedCredential(SECRET),
        FinishedRequest(SECRET, {"key": SECRET}, SECRET, {"key": SECRET}),
        TransportRequest("POST", f"https://example.test/?key={SECRET}", [("authorization", SECRET)], SECRET.encode()),
    ]
    for value in values:
        assert SECRET not in repr(value)


@pytest.mark.parametrize("data", [b"\x30", b"\x30\x82\x01", b"\x30\x81"])
def test_truncated_der_length_raises_value_error(data):
    from lm15.cloud import rs256

    with pytest.raises(ValueError, match="truncated length"):
        rs256._der_sequence(data)


def test_online_commands_use_the_context_environment(monkeypatch):
    captured = []

    def run(*args, **kwargs):
        captured.append(kwargs["env"])
        return SimpleNamespace(returncode=0, stdout="test")

    monkeypatch.setattr(subprocess, "run", run)
    ctx = chains.ChainContext.online(env={"PATH": "/configured", "AZURE_TENANT_ID": "tenant"})
    assert ctx.run(["az"], 1) == "test"
    assert captured == [{"PATH": "/configured", "AZURE_TENANT_ID": "tenant"}]


def test_concurrent_cloud_refresh_runs_once(monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    import threading

    started = threading.Event()
    release = threading.Event()
    calls = []

    def resolve(*args, **kwargs):
        calls.append(1)
        started.set()
        assert release.wait(5)
        return BearerToken("fresh", NOW + dt.timedelta(hours=1)), _SRC

    monkeypatch.setattr(chains, "resolve", resolve)
    provider = chains.credential_provider(access.AZURE, chains.ChainContext(env={}, now=lambda: NOW))
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(provider) for _ in range(4)]
        try:
            assert started.wait(5)
        finally:
            release.set()
        assert [f.result().value for f in futures] == ["fresh"] * 4
    assert len(calls) == 1
