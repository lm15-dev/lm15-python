"""Cloud identity and endpoints (spec/auth.md AUTH-1 named credentials and
provenance, AUTH-2 JWT reading, AUTH-10 endpoint override; ratified in
session 2026-09-19, changes/2026-09-19-cloud-identity-and-endpoints.md).

Three developer moments, one contract: a laptop walks the chain and says
what won; a deployment names one identity and never falls through; an
endpoint is a URL root the door completes, with the door's behaviour
attached.  Nothing here touches the network: every chain context is
offline or handed a fake ``http``/``run``.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import pytest

from lm15 import access
from lm15.cloud import chains
from lm15.cloud.chains import ChainContext, CredentialSource, NAMED_RUNGS, credential_provider, named_rungs, resolve
from lm15.cloud.hosts import join_endpoint, render_base_url, resolve_settings
from lm15.credentials import BearerToken
from lm15.doctor import explain_auth
from lm15.errors import AuthError, NotConfiguredError
from lm15.features import NAMED_CREDENTIALS
from lm15.providers import AnthropicLM, AsyncOpenAILM, GeminiLM, OpenAIChatLM, OpenAILM
from lm15.router import ADAPTERS, LMRouter, RouterConfig, _build_planning_lm
from lm15.types import Message, Request

NOW = dt.datetime(2026, 9, 19, 12, 0, tzinfo=dt.timezone.utc)
CONTRACT = Path(__file__).resolve().parents[2] / "lm15-contract"
NAMED_FIXTURE = Path(__file__).resolve().parent.parent / "conformance" / "auth_named_credentials.json"


def _req(model: str) -> Request:
    return Request(model=model, messages=[Message.user("Say ok.")])


def _headers(lm, model: str) -> dict[str, str]:
    return {k.lower(): v for k, v in lm.build_request(_req(model), stream=False).headers}


# ─── AUTH-10: the endpoint is a URL root the door completes ──────────


class TestEndpoint:
    def test_host_template_splits_into_root_and_path(self):
        assert access.AZURE.host.root_template == "https://{resource}.openai.azure.com"
        assert access.AZURE.host.path_template == "/openai/v1"
        assert access.VERTEX.host.root_template == "https://{location_host}"
        assert access.VERTEX.host.path_template.startswith("/v1/projects/{project}/locations/{location}")
        assert access.VERTEX_EXPRESS.host.path_template == "/v1/publishers/google"

    def test_url_only_settings_are_the_ones_an_endpoint_replaces(self):
        assert access.AZURE.host.url_only_settings == {"resource"}
        assert access.AZURE_ANTHROPIC.host.url_only_settings == {"resource"}
        # AWS signs with the region: an endpoint does not make it optional (boto3 agrees)
        assert access.BEDROCK_CHAT.host.url_only_settings == frozenset()
        assert access.AWS_ANTHROPIC.host.url_only_settings == frozenset()
        # Vertex: location is in the path (and derives the host); project is in the path
        assert access.VERTEX.host.url_only_settings == frozenset()

    @pytest.mark.parametrize("endpoint,expected", [
        ("https://acct.services.ai.azure.com", "https://acct.services.ai.azure.com/openai/v1"),
        ("https://acct.services.ai.azure.com/", "https://acct.services.ai.azure.com/openai/v1"),
        ("https://acct.openai.azure.com/openai/v1", "https://acct.openai.azure.com/openai/v1"),
        ("https://acct.openai.azure.com/openai/v1/", "https://acct.openai.azure.com/openai/v1"),
        ("https://acct.services.ai.azure.com/openai", "https://acct.services.ai.azure.com/openai/v1"),
        ("https://ai-gw.corp.example/foundry", "https://ai-gw.corp.example/foundry/openai/v1"),
        ("http://localhost:8080", "http://localhost:8080/openai/v1"),
    ])
    def test_join_endpoint_appends_the_door_path_unless_present(self, endpoint, expected):
        assert join_endpoint(endpoint, "/openai/v1") == expected

    def test_join_endpoint_foundry_claude_accepts_microsofts_spelling(self):
        # Microsoft's examples give `https://acct.services.ai.azure.com/anthropic` as the Anthropic base.
        assert join_endpoint("https://acct.services.ai.azure.com/anthropic", "/anthropic/v1") == "https://acct.services.ai.azure.com/anthropic/v1"
        assert join_endpoint("https://acct.services.ai.azure.com", "/anthropic/v1") == "https://acct.services.ai.azure.com/anthropic/v1"

    @pytest.mark.parametrize("bad", ["acct.services.ai.azure.com", "ftp://x", "https://x?y=1", "https://x#f", "https://u:p@x", "https://"])
    def test_join_endpoint_refuses_non_http_query_fragment_userinfo(self, bad):
        with pytest.raises(NotConfiguredError):
            join_endpoint(bad, "/openai/v1", provider="azure")

    def test_render_base_url_with_endpoint_keeps_path_settings(self):
        settings = {"project": "my proj", "location": "europe-west4"}
        url = render_base_url(access.VERTEX.host, settings, "https://psc.internal.example")
        assert url == "https://psc.internal.example/v1/projects/my%20proj/locations/europe-west4/publishers/google"

    def test_resolve_settings_relaxes_only_url_only_settings(self):
        got = resolve_settings(access.AZURE.host, None, {}, provider="azure", endpoint="https://x.example")
        assert "resource" not in got and got["scope"] == "https://ai.azure.com/.default"
        with pytest.raises(NotConfiguredError, match="region"):
            resolve_settings(access.BEDROCK_CHAT.host, None, {}, provider="bedrock-chat", endpoint="https://vpce.example")

    def test_missing_resource_error_names_the_endpoint_variable(self):
        with pytest.raises(NotConfiguredError) as exc:
            resolve_settings(access.AZURE.host, None, {}, provider="azure")
        assert "AZURE_OPENAI_RESOURCE" in str(exc.value) and "AZURE_OPENAI_ENDPOINT" in str(exc.value)

    def test_bare_adapter_base_url_on_a_cloud_door_is_the_endpoint_root(self):
        lm = OpenAILM(api_key="k", access=access.AZURE, base_url="https://acct.services.ai.azure.com")
        assert lm.base_url == "https://acct.services.ai.azure.com/openai/v1"
        assert lm.provider == "azure" and "resource" not in lm.host_settings
        req = lm.build_request(_req("gpt-5-mini"), stream=False)
        assert req.url == "https://acct.services.ai.azure.com/openai/v1/responses"
        assert {k.lower() for k in dict(req.headers)} >= {"api-key"}  # the door's scheme, not the dialect's
        claude = AnthropicLM(api_key="k", access=access.AZURE_ANTHROPIC, base_url="https://acct.services.ai.azure.com")
        assert claude.base_url == "https://acct.services.ai.azure.com/anthropic/v1"

    def test_bare_adapter_default_base_url_still_renders_the_template(self):
        lm = OpenAILM(api_key="k", access=access.AZURE, settings={"resource": "acct"})
        assert lm.base_url == "https://acct.openai.azure.com/openai/v1"

    def test_bedrock_endpoint_still_needs_region_and_signs_with_it(self):
        with pytest.raises(NotConfiguredError, match="region"):
            OpenAIChatLM(api_key="k", access=access.BEDROCK_CHAT, base_url="https://vpce-1.bedrock-runtime.us-east-1.vpce.amazonaws.com")
        lm = OpenAIChatLM(api_key="k", access=access.BEDROCK_CHAT, settings={"region": "us-east-1"},
                          base_url="https://vpce-1.bedrock-runtime.us-east-1.vpce.amazonaws.com")
        assert lm.base_url == "https://vpce-1.bedrock-runtime.us-east-1.vpce.amazonaws.com/openai/v1"

    def test_router_vendor_variables_in_order(self):
        env = {"AWS_BEARER_TOKEN_BEDROCK": "t", "AWS_REGION": "us-east-1", "AWS_ENDPOINT_URL": "https://generic.example",
               "AWS_ENDPOINT_URL_BEDROCK_RUNTIME": "https://specific.example"}
        router = LMRouter(RouterConfig(env=env))
        lm = router.lm("bedrock-chat:openai.gpt-oss-20b-1:0")
        assert lm.base_url == "https://specific.example/openai/v1"
        router = LMRouter(RouterConfig(env={k: v for k, v in env.items() if k != "AWS_ENDPOINT_URL_BEDROCK_RUNTIME"}))
        assert router.lm("bedrock-chat:openai.gpt-oss-20b-1:0").base_url == "https://generic.example/openai/v1"
        # an explicit base_urls entry beats the variable
        router = LMRouter(RouterConfig(env=env, base_urls={"bedrock-chat": "https://explicit.example"}))
        assert router.lm("bedrock-chat:openai.gpt-oss-20b-1:0").base_url == "https://explicit.example/openai/v1"

    def test_router_foundry_variable_for_claude(self):
        router = LMRouter(RouterConfig(env={"ANTHROPIC_FOUNDRY_API_KEY": "k", "ANTHROPIC_FOUNDRY_BASE_URL": "https://acct.services.ai.azure.com/anthropic"}))
        assert router.lm("azure-anthropic:claude-sonnet-4-5").base_url == "https://acct.services.ai.azure.com/anthropic/v1"

    def test_async_router_completes_the_endpoint(self):
        from lm15.router import AsyncLMRouter

        router = AsyncLMRouter(RouterConfig(env={"AZURE_OPENAI_API_KEY": "k", "AZURE_OPENAI_ENDPOINT": "https://acct.services.ai.azure.com"}))
        lm = router.lm("azure:gpt-5-mini")
        assert isinstance(lm, AsyncOpenAILM) and lm.base_url == "https://acct.services.ai.azure.com/openai/v1"

    def test_plan_uses_the_endpoint(self):
        router = LMRouter(RouterConfig(env={"AZURE_OPENAI_ENDPOINT": "https://acct.services.ai.azure.com"}))
        built = _build_planning_lm(router.resolve("azure:gpt-5-mini"), router.config, ADAPTERS, None)
        assert built.base_url == "https://acct.services.ai.azure.com/openai/v1"
        router.plan(_req("azure:gpt-5-mini"))  # no credential, no resource: the endpoint alone suffices

    def test_every_hosted_door_declares_its_endpoint_variables_or_none_deliberately(self):
        # Vertex has no vendor variable lm15 can cite; base_urls only (stated in access.py).
        for policy in access.CLOUD_HOST_POLICIES:
            if policy.provider.startswith("vertex"):
                assert policy.host.endpoint_env == ()
            else:
                assert policy.host.endpoint_env, policy.provider


# ─── AUTH-1: named credentials ───────────────────────────────────────


class TestNamedCredentials:
    def test_the_four_names_cover_every_chain(self):
        assert NAMED_CREDENTIALS == ("platform", "workload", "environment", "cli")
        for chain in ("aws-chain", "azure-chain", "gcp-chain"):
            assert set(NAMED_RUNGS[chain]) == set(NAMED_CREDENTIALS)

    def test_named_rungs_exist_in_the_chain_and_never_include_the_door_key(self):
        for policy in (access.AZURE, access.AZURE_ANTHROPIC, access.BEDROCK_ANTHROPIC, access.BEDROCK_CHAT, access.VERTEX):
            chain_names = [rung.name for rung in chains.chain_for(policy)]
            for name in NAMED_CREDENTIALS:
                rungs = named_rungs(policy, name)
                assert rungs, (policy.provider, name)
                for rung in rungs:
                    assert rung.name in chain_names
                    assert not rung.name.startswith("env:") or rung.name == "env:AWS_ACCESS_KEY_ID"
            assert chain_names.index(named_rungs(policy, "platform")[0].name) >= 0

    def test_platform_means_the_machines_own_identity(self):
        assert [r.name for r in named_rungs(access.AZURE, "platform")] == ["managed-identity"]
        assert [r.name for r in named_rungs(access.BEDROCK_CHAT, "platform")] == ["container", "imds"]
        assert [r.name for r in named_rungs(access.VERTEX, "platform")] == ["metadata"]

    def test_unknown_name_fails_at_construction(self):
        with pytest.raises(NotConfiguredError, match="unknown named credential"):
            credential_provider(access.AZURE, ChainContext(env={}), named="managed-identity")

    def test_named_credential_absent_is_an_error_not_a_fall_through(self):
        # az is on PATH and would answer; `platform` must not ask it.
        calls = []

        def run(argv, timeout):
            calls.append(argv)
            return json.dumps({"accessToken": "x"})

        def http(method, url, headers, body, timeout):
            assert "169.254.169.254" in url
            return 404, {}, b""

        ctx = ChainContext(env={"PATH": "/bin"}, files={"/bin/az": "stub"}, http=http, run=run, now=lambda: NOW)
        provider = credential_provider(access.AZURE, ctx, named="platform")
        with pytest.raises(NotConfiguredError) as exc:
            provider()
        message = str(exc.value)
        assert 'named credential "platform"' in message and "managed identity" in message.lower()
        assert "will not try" in message
        assert calls == []

    def test_named_credential_resolves_and_carries_provenance(self):
        def http(method, url, headers, body, timeout):
            assert "169.254.169.254" in url and headers.get("Metadata") == "true"
            return 200, {}, json.dumps({"access_token": "SECRET-SENTINEL-DO-NOT-PRINT", "expires_in": 3600}).encode()

        ctx = ChainContext(env={}, files={}, http=http, run=None, now=lambda: NOW)
        provider = credential_provider(access.AZURE, ctx, named="platform")
        token = provider()
        assert isinstance(token, BearerToken)
        assert provider.source == CredentialSource(rung="managed-identity", label="Azure managed identity", named="platform",
                                                   expires_at=NOW + dt.timedelta(hours=1))
        described = provider.source.describe(NOW)
        assert "managed identity" in described and 'named credential "platform"' in described and "expires in 1 h" in described
        assert "SECRET" not in described

    def test_chain_walk_carries_provenance_too(self):
        ctx = ChainContext(env={"AZURE_OPENAI_API_KEY": "SECRET-SENTINEL-DO-NOT-PRINT", "IDENTITY_ENDPOINT": "http://x", "IDENTITY_HEADER": "h"},
                           files={}, now=lambda: NOW)
        value, source = resolve(access.AZURE, ctx)
        assert source.rung == "env:AZURE_OPENAI_API_KEY" and source.named is None
        assert "SECRET" not in source.describe()

    def test_gcp_workload_and_environment_are_told_apart_by_file_type(self):
        sa = json.dumps({"type": "service_account", "client_email": "a@b", "private_key": "x", "token_uri": "https://t"})
        ctx = ChainContext(env={"GOOGLE_APPLICATION_CREDENTIALS": "/creds.json"}, files={"/creds.json": sa}, now=lambda: NOW)
        steps, configured = chains.explain(access.VERTEX, ctx, explicit=False, named="workload")
        assert [s.kind for s in steps] == ["api_keys", "adc-env"]
        assert steps[1].state == "absent" and '"environment"' in steps[1].detail and "service_account" in steps[1].detail
        assert not configured
        steps, configured = chains.explain(access.VERTEX, ctx, explicit=False, named="environment")
        assert steps[1].state == "unprobed" and configured
        # and online the wrong type is refused by name, never exchanged
        online = ChainContext(env=ctx.env, files=ctx.files, http=lambda *a: pytest.fail("token exchange for the wrong type"), now=lambda: NOW)
        with pytest.raises(NotConfiguredError, match='"environment", not "workload"'):
            resolve(access.VERTEX, online, named="workload")

    def test_aws_cli_name_covers_the_profile_rungs_only(self):
        names = [r.name for r in named_rungs(access.BEDROCK_ANTHROPIC, "cli")]
        assert names == ["assume-role", "sso", "shared-credentials-file", "login", "credential_process", "config-file"]
        assert "imds" not in names and "env:AWS_ACCESS_KEY_ID" not in names

    def test_router_credentials_entry_builds_a_named_provider(self):
        router = LMRouter(RouterConfig(env={"AZURE_OPENAI_RESOURCE": "acct"}, credentials={"azure": "platform"}))
        lm = router.lm("azure:gpt-5-mini")
        assert getattr(lm.api_key, "named", None) == "platform"
        assert 'named credential "platform"' in lm.credential_origin()

    def test_router_refuses_a_key_and_a_name_for_one_door(self):
        with pytest.raises(NotConfiguredError, match="both api_keys and credentials"):
            LMRouter(RouterConfig(env={}, api_keys={"azure": "k"}, credentials={"azure": "platform"}))

    def test_router_refuses_names_on_non_cloud_doors_and_unknown_names(self):
        with pytest.raises(NotConfiguredError, match="not a cloud door"):
            LMRouter(RouterConfig(env={}, credentials={"openai": "platform"}))
        with pytest.raises(NotConfiguredError, match="not a named credential"):
            RouterConfig(env={}, credentials={"azure": "managed-identity"})
        with pytest.raises(NotConfiguredError, match="not a provider lm15 routes to"):
            LMRouter(RouterConfig(env={}, credentials={"azur": "platform"}))

    def test_router_accepts_the_underscore_spelling(self):
        router = LMRouter(RouterConfig(env={"ANTHROPIC_FOUNDRY_RESOURCE": "acct"}, credentials={"azure_anthropic": "cli"}))
        assert getattr(router.lm("azure-anthropic:claude-sonnet-4-5").api_key, "named", None) == "cli"

    def test_bare_adapter_credential_keyword(self):
        lm = OpenAILM(access=access.AZURE, credential="platform", base_url="https://acct.services.ai.azure.com")
        assert getattr(lm.api_key, "named", None) == "platform"
        assert lm.base_url == "https://acct.services.ai.azure.com/openai/v1"
        with pytest.raises(NotConfiguredError, match="both api_key= and credential="):
            OpenAILM(api_key="k", access=access.AZURE, credential="platform", settings={"resource": "acct"})
        with pytest.raises(NotConfiguredError, match="not a cloud door"):
            OpenAILM(credential="platform")
        with pytest.raises(NotConfiguredError, match="unknown named credential"):
            GeminiLM(access=access.VERTEX, credential="metadata", settings={"project": "p"})

    def test_async_adapter_credential_keyword(self):
        lm = AsyncOpenAILM(access=access.AZURE, credential="workload", settings={"resource": "acct"})
        assert getattr(lm.api_key, "named", None) == "workload"


# ─── AUTH-1: provenance on every auth error ──────────────────────────


class TestProvenance:
    def _body(self) -> str:
        return json.dumps({"error": {"code": "401", "message": "Access denied due to invalid subscription key or wrong API endpoint."}})

    def test_env_key_is_named(self):
        router = LMRouter(RouterConfig(env={"AZURE_OPENAI_API_KEY": "SECRET-SENTINEL-DO-NOT-PRINT", "AZURE_OPENAI_RESOURCE": "acct"}))
        lm = router.lm("azure:gpt-5-mini")
        assert "azure-chain (not yet resolved)" in lm.credential_origin()
        lm.build_request(_req("gpt-5-mini"), stream=False)  # the wire needs a resolved credential
        err = lm.normalize_error(401, self._body())
        assert isinstance(err, AuthError)
        assert "credential came from: env $AZURE_OPENAI_API_KEY" in str(err)
        assert err.credential_origin.startswith("env $AZURE_OPENAI_API_KEY")
        assert "SECRET" not in str(err)

    def test_callable_is_named_honestly(self):
        lm = OpenAILM(api_key=lambda: BearerToken("SECRET-SENTINEL-DO-NOT-PRINT"), access=access.AZURE, settings={"resource": "acct"})
        err = lm.normalize_error(401, self._body())
        assert "credential came from: an application-supplied callable (identity not inspected by lm15)" in str(err)

    def test_explicit_value_is_named_without_the_value(self):
        lm = OpenAILM(api_key="SECRET-SENTINEL-DO-NOT-PRINT")
        err = lm.normalize_error(401, self._body())
        assert "credential came from: an explicit api_key (value never shown)" in str(err)
        assert "SECRET" not in str(err)

    def test_chain_rung_is_named_after_it_won(self):
        ctx = ChainContext(env={"AZURE_OPENAI_API_KEY": "SECRET-SENTINEL-DO-NOT-PRINT"}, files={}, now=lambda: NOW)
        lm = OpenAILM(api_key=credential_provider(access.AZURE, ctx), access=access.AZURE, settings={"resource": "acct"})
        assert "not yet resolved" in lm.credential_origin()
        lm.build_request(_req("gpt-5-mini"), stream=False)
        err = lm.normalize_error(401, self._body())
        assert "credential came from: env $AZURE_OPENAI_API_KEY" in str(err)
        assert "SECRET" not in str(err)

    def test_provenance_survives_the_login_hint_and_is_added_once(self):
        from lm15.errors import with_credential_origin

        err = AuthError("nope", provider="azure", status=401)
        once = with_credential_origin(err, "env $X")
        twice = with_credential_origin(once, "env $Y")
        assert str(twice).count("credential came from:") == 1 and "env $X" in str(twice)
        assert twice.status == 401

    def test_non_auth_errors_carry_no_provenance(self):
        lm = OpenAILM(api_key="k")
        err = lm.normalize_error(429, json.dumps({"error": {"message": "slow down"}}))
        assert "credential came from" not in str(err)

    def test_async_mirror_names_the_source(self):
        from lm15.router import AsyncLMRouter

        router = AsyncLMRouter(RouterConfig(env={"AZURE_OPENAI_API_KEY": "k", "AZURE_OPENAI_RESOURCE": "acct"}))
        lm = router.lm("azure:gpt-5-mini")
        assert "azure-chain" in lm._inner.credential_origin()
        env_only = AsyncLMRouter(RouterConfig(env={"GOOGLE_API_KEY": "k"})).lm("vertex-express:gemini-3-pro")
        assert "env $GOOGLE_API_KEY" in env_only._inner.credential_origin()


# ─── AUTH-7: the doctor in named mode, and the base URL ──────────────


class TestDoctor:
    def test_named_mode_walks_only_the_named_rungs(self):
        report = explain_auth("azure", env={"AZURE_OPENAI_RESOURCE": "acct", "IDENTITY_ENDPOINT": "http://x", "IDENTITY_HEADER": "h",
                                            "PATH": "/bin"},
                              files={"/bin/az": "stub"}, credential="platform")
        assert [s.kind for s in report.steps] == ["api_keys", "managed-identity"]
        assert report.steps[1].state == "unprobed" and report.configured
        assert report.named == "platform" and "managed identity" in report.named_meaning.lower()
        text = report.describe()
        assert 'named credential "platform"' in text and "the chain is not walked" in text
        assert "az account" not in text

    def test_router_config_supplies_credentials_and_base_urls(self):
        router = LMRouter(RouterConfig(env={"PATH": "/nowhere"}, credentials={"azure": "cli"},
                                       base_urls={"azure": "https://acct.services.ai.azure.com"}))
        report = explain_auth("azure", config=router.config)
        assert [s.kind for s in report.steps] == ["api_keys", "az", "pwsh", "azd"]
        assert report.base_url == "https://acct.services.ai.azure.com/openai/v1" and report.base_url_source == "base_urls"
        assert "resource" not in dict(report.settings)
        assert "base url: https://acct.services.ai.azure.com/openai/v1 (from base_urls)" in report.describe()

    def test_base_url_from_the_vendor_variable_and_from_the_template(self):
        report = explain_auth("azure", env={"AZURE_OPENAI_ENDPOINT": "https://acct.services.ai.azure.com/"})
        assert report.base_url == "https://acct.services.ai.azure.com/openai/v1" and report.base_url_source == "env $AZURE_OPENAI_ENDPOINT"
        report = explain_auth("azure", env={"AZURE_OPENAI_RESOURCE": "acct"})
        assert report.base_url == "https://acct.openai.azure.com/openai/v1" and report.base_url_source == "template"
        report = explain_auth("azure", env={})
        assert report.base_url is None and "AZURE_OPENAI_ENDPOINT" in dict(report.settings)["error"]

    def test_doctor_refuses_a_key_and_a_name_like_the_router(self):
        with pytest.raises(NotConfiguredError, match="both api_keys and credentials"):
            explain_auth("azure", env={}, api_keys={"azure": "k"}, credential="platform")
        with pytest.raises(NotConfiguredError, match="not a cloud door"):
            explain_auth("openai", env={}, credential="platform")

    def test_doctor_never_prints_secrets_in_named_mode(self):
        report = explain_auth("bedrock-chat", env={"AWS_REGION": "us-east-1", "AWS_ACCESS_KEY_ID": "AKID",
                                                   "AWS_SECRET_ACCESS_KEY": "SECRET-SENTINEL-DO-NOT-PRINT"}, credential="environment")
        assert "SECRET-SENTINEL" not in report.describe() and report.steps[1].state == "selected"


# ─── The contract fixture for named credentials ──────────────────────


_FIXTURE = json.loads(NAMED_FIXTURE.read_text(encoding="utf-8"))


def test_named_fixture_is_the_contracts_copy():
    contract_copy = CONTRACT / "auth" / "named-credentials.json"
    if contract_copy.is_file():
        assert json.loads(contract_copy.read_text(encoding="utf-8")) == _FIXTURE


@pytest.mark.parametrize("case", _FIXTURE["cases"], ids=[c["id"] for c in _FIXTURE["cases"]])
def test_named_credentials_contract_case(case: dict, tmp_path: Path) -> None:
    home = tmp_path / "home"
    home.mkdir()
    env = {k: v.replace("~/", f"{home}/") for k, v in case.get("env", {}).items()}
    env["HOME"] = str(home)
    files = {}
    for rel, content in case.get("files", {}).items():
        target = home / rel[2:] if rel.startswith("~/") else Path(rel)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        files[str(target)] = content
    kwargs = {"env": env, "files": files, "home": str(home), "credential": case["credential"]}
    if case.get("api_keys_providers"):
        kwargs["api_keys"] = {p: _FIXTURE["sentinel"] for p in case["api_keys_providers"]}
    if case.get("base_url"):
        kwargs["base_url"] = case["base_url"]
    expect = case["expect"]
    if "error" in expect:
        with pytest.raises(NotConfiguredError, match=expect["error"]) as raised:
            explain_auth(case["provider"], **kwargs)
        # AUTH-5: a refusal is a rendered surface too (2026-10-10).
        assert _FIXTURE["sentinel"] not in str(raised.value)
        assert _FIXTURE["sentinel"] not in repr(raised.value)
        return
    report = explain_auth(case["provider"], **kwargs)
    assert [(s.kind, s.state) for s in report.steps] == [(s["kind"], s["state"]) for s in expect["steps"]]
    assert report.configured == expect["configured"]
    assert report.named == case["credential"]
    if "base_url" in expect:
        assert report.base_url == expect["base_url"]
    if "settings" in expect:
        assert dict(report.settings) == {**dict(report.settings), **expect["settings"]}
    assert _FIXTURE["sentinel"] not in report.describe()


# ─── Google external_account with an AWS source (AUTH-1 gcp-chain) ───────────
#
# The subject token is pinned byte for byte by lm15-contract auth/token-vectors.json
# (gcp-external-account-aws*.build, computed by google-auth itself). These walk the
# whole acquisition: where the AWS identity comes from, then STS, then impersonation.

_AWS_WIF = {
    "type": "external_account",
    "audience": "//iam.googleapis.com/projects/1/locations/global/workloadIdentityPools/p/providers/aws",
    "subject_token_type": "urn:ietf:params:aws:token-type:aws4_request",
    "token_url": "https://sts.googleapis.com/v1/token",
    "credential_source": {
        "environment_id": "aws1",
        "region_url": "http://169.254.169.254/latest/meta-data/placement/availability-zone",
        "url": "http://169.254.169.254/latest/meta-data/iam/security-credentials",
        "regional_cred_verification_url": "https://sts.{region}.amazonaws.com?Action=GetCallerIdentity&Version=2011-06-15",
    },
    "service_account_impersonation_url": "https://iamcredentials.googleapis.com/v1/projects/-/serviceAccounts/sa@p.iam.gserviceaccount.com:generateAccessToken",
}


def _aws_wif_http(calls, answers):
    def http(method, url, headers, body, timeout):
        calls.append((method, url, dict(headers), body))
        for prefix, reply in answers:
            if url.startswith(prefix):
                status, data = reply
                return status, {}, data if isinstance(data, bytes) else json.dumps(data).encode()
        raise AssertionError(f"unexpected request {method} {url}")
    return http


_GOOGLE_ANSWERS = [
    ("https://sts.googleapis.com/", (200, {"access_token": "federated", "expires_in": 3600, "token_type": "Bearer"})),
    ("https://iamcredentials.googleapis.com/", (200, {"accessToken": "sa-token", "expireTime": "2026-09-03T13:00:00Z"})),
]


def _subject(calls):
    import urllib.parse
    sts = next(c for c in calls if c[1].startswith("https://sts.googleapis.com/"))
    body = json.loads(sts[3])
    assert body["subjectTokenType"] == "urn:ietf:params:aws:token-type:aws4_request"
    return json.loads(urllib.parse.unquote(body["subjectToken"]))


class TestGcpExternalAccountAws:
    def test_ecs_container_credentials_sign_the_request(self):
        """On ECS the AWS identity is the task role, served at the container endpoint (lm15's addition)."""
        calls = []
        http = _aws_wif_http(calls, [
            ("http://169.254.170.2/v2/credentials/abc", (200, {"AccessKeyId": "ASIATASK", "SecretAccessKey": "s", "Token": "task-session", "Expiration": "2026-09-03T18:00:00Z"})),
            *_GOOGLE_ANSWERS,
        ])
        ctx = ChainContext(env={"AWS_CONTAINER_CREDENTIALS_RELATIVE_URI": "/v2/credentials/abc", "AWS_REGION": "us-east-2"}, files={}, http=http, now=lambda: NOW)
        token = chains._gcp_from_info(ctx, _AWS_WIF, "wif.json")
        assert token.value == "sa-token"
        assert [c[1].split("?")[0] for c in calls] == ["http://169.254.170.2/v2/credentials/abc", "https://sts.googleapis.com/v1/token",
                                                       _AWS_WIF["service_account_impersonation_url"]]
        signed = _subject(calls)
        assert signed["url"] == "https://sts.us-east-2.amazonaws.com?Action=GetCallerIdentity&Version=2011-06-15"
        headers = {h["key"]: h["value"] for h in signed["headers"]}
        assert headers["Authorization"].startswith("AWS4-HMAC-SHA256 Credential=ASIATASK/20260919/us-east-2/sts/aws4_request")
        assert headers["x-amz-security-token"] == "task-session"
        assert headers["x-goog-cloud-target-resource"] == _AWS_WIF["audience"]
        assert calls[2][2]["authorization"] == "Bearer federated"

    def test_ec2_metadata_with_imdsv2_gives_region_and_role(self):
        calls = []
        source = dict(_AWS_WIF["credential_source"], imdsv2_session_token_url="http://169.254.169.254/latest/api/token")
        http = _aws_wif_http(calls, [
            ("http://169.254.169.254/latest/api/token", (200, b"imds-session")),
            ("http://169.254.169.254/latest/meta-data/placement/availability-zone", (200, b"eu-west-1b")),
            ("http://169.254.169.254/latest/meta-data/iam/security-credentials/web", (200, {"Code": "Success", "AccessKeyId": "ASIAEC2", "SecretAccessKey": "s", "Token": "t"})),
            ("http://169.254.169.254/latest/meta-data/iam/security-credentials", (200, b"web\n")),
            *_GOOGLE_ANSWERS,
        ])
        ctx = ChainContext(env={}, files={}, http=http, now=lambda: NOW)
        assert chains._gcp_from_info(ctx, dict(_AWS_WIF, credential_source=source), "wif.json").value == "sa-token"
        assert all(c[2].get("X-aws-ec2-metadata-token") == "imds-session" for c in calls[1:4])
        assert sum(c[0] == "PUT" for c in calls) == 1
        assert calls[0][2] == {"X-aws-ec2-metadata-token-ttl-seconds": "300"}
        assert _subject(calls)["url"].startswith("https://sts.eu-west-1.amazonaws.com?")

    def test_environment_keys_win_and_no_metadata_is_asked(self):
        calls = []
        http = _aws_wif_http(calls, _GOOGLE_ANSWERS)
        env = {"AWS_ACCESS_KEY_ID": "AKIDENV", "AWS_SECRET_ACCESS_KEY": "s", "AWS_DEFAULT_REGION": "ca-central-1",
               "AWS_CONTAINER_CREDENTIALS_RELATIVE_URI": "/v2/credentials/abc"}
        ctx = ChainContext(env=env, files={}, http=http, now=lambda: NOW)
        assert chains._gcp_from_info(ctx, _AWS_WIF, "wif.json").value == "sa-token"
        headers = {h["key"]: h["value"] for h in _subject(calls)["headers"]}
        assert "x-amz-security-token" not in headers and "AKIDENV/20260919/ca-central-1/sts" in headers["Authorization"]

    @pytest.mark.parametrize("source,match", [
        ({"environment_id": "aws2", "regional_cred_verification_url": "https://sts.{region}.amazonaws.com"}, "only aws1"),
        ({"environment_id": "aws1"}, "regional_cred_verification_url"),
        ({"environment_id": "azure1", "regional_cred_verification_url": "https://x"}, "environment_id"),
    ])
    def test_malformed_aws_source_is_refused(self, source, match):
        ctx = ChainContext(env={"AWS_REGION": "us-east-1", "AWS_ACCESS_KEY_ID": "a", "AWS_SECRET_ACCESS_KEY": "s"}, files={},
                           http=lambda *a: pytest.fail("no request before the source is valid"), now=lambda: NOW)
        with pytest.raises(NotConfiguredError, match=match):
            chains._gcp_from_info(ctx, dict(_AWS_WIF, credential_source=source), "wif.json")

    def test_no_aws_identity_says_what_was_looked_for(self):
        source = {k: v for k, v in _AWS_WIF["credential_source"].items() if k != "url"}
        ctx = ChainContext(env={"AWS_REGION": "us-east-1"}, files={}, http=lambda *a: pytest.fail("nothing to ask"), now=lambda: NOW)
        with pytest.raises(NotConfiguredError, match="AWS_ACCESS_KEY_ID nor a container endpoint"):
            chains._gcp_from_info(ctx, dict(_AWS_WIF, credential_source=source), "wif.json")
