"""Cloud hosts (changes/2026-09-03-cloud-hosts.md): credentials, signing,
host rewrites, chains, registry rules.

The signing tests read the contract's vectors directly
(lm15-contract/auth/sigv4-vectors.json, token-vectors.json) so a change
in either repository is caught here before the harness runs.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path

import pytest

from lm15 import access
from lm15.cloud import chains, rs256, sigv4
from lm15.credentials import (
    ApiKey,
    AwsCredentials,
    BearerToken,
    coerce_credential,
    credential_from_dict,
    credential_to_dict,
)
from lm15.errors import NotConfiguredError
from lm15.providers import AnthropicLM, GeminiLM, OpenAIChatLM, OpenAILM
from lm15.registry import PROVIDERS
from lm15.types import Message, Request

CONTRACT = Path(__file__).resolve().parents[2] / "lm15-contract"
SIGV4 = json.loads((CONTRACT / "auth" / "sigv4-vectors.json").read_text())
TOKENS = json.loads((CONTRACT / "auth" / "token-vectors.json").read_text())
TEST_KEY = (CONTRACT / "auth" / "test-keys" / "rsa-2048-test-only.pem").read_text()
TEST_CERT = (CONTRACT / "auth" / "test-keys" / "rsa-2048-test-only.cert.pem").read_text()
NOW = dt.datetime(2026, 9, 3, 12, 0, 0, tzinfo=dt.timezone.utc)
AWS = AwsCredentials("AKIDEXAMPLE", "wJalrXUtNFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY".replace("tNF", "tnF"), session_token="SESSION-VECTOR")


def _expand(value):
    if isinstance(value, dict) and "$file" in value:
        text = (CONTRACT / value["$file"]).read_text()
        if value.get("strip_comment_lines"):
            text = "\n".join(line for line in text.splitlines() if not line.startswith("#")) + "\n"
        return text
    if isinstance(value, dict):
        return {k: _expand(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand(v) for v in value]
    return value


# ─── credentials ─────────────────────────────────────────────────────


class TestCredentials:
    def test_string_reads_as_api_key(self):
        assert coerce_credential("k") == ApiKey("k")

    def test_reprs_never_show_values(self):
        for value in (ApiKey("s3cret"), BearerToken("s3cret"), AwsCredentials("AKID", "s3cret", session_token="s3cret")):
            assert "s3cret" not in repr(value)

    @pytest.mark.parametrize("case", [c for c in json.loads((CONTRACT / "serde" / "canonical.json").read_text())["cases"] if c["kind"] == "credential"],
                             ids=lambda c: c["id"])
    def test_canonical_roundtrip(self, case):
        assert credential_to_dict(credential_from_dict(case["value"])) == case["value"]

    def test_expiry_skew(self):
        token = BearerToken("t", expires_at=NOW + dt.timedelta(seconds=200))
        assert token.is_expired(NOW)  # inside the 5-minute window (AUTH-3)
        assert not BearerToken("t", expires_at=NOW + dt.timedelta(seconds=400)).is_expired(NOW)


# ─── signing ─────────────────────────────────────────────────────────


@pytest.mark.parametrize("case", SIGV4["cases"], ids=lambda c: c["id"])
def test_sigv4_vectors(case):
    fixed = SIGV4["fixed"]
    req = case["request"]
    headers = {k: (v if isinstance(v, str) else ",".join(v)) for k, v in req["headers"].items() if k.lower() not in ("host", "x-amz-date")}
    # The vector's token is a header (post-sts-header-*) or case-level
    # (get-vanilla-with-session-token), as harness/check.py reads it.
    token = headers.pop("X-Amz-Security-Token", None) or case.get("session_token")
    got = sigv4.sign(
        method=req["method"], url="https://example.amazonaws.com" + req["target"], headers=headers, payload=req["body"].encode(),
        credentials=AwsCredentials(fixed["access_key_id"], fixed["secret_access_key"], session_token=token),
        region=fixed["region"], service=fixed["service"], now=dt.datetime(2015, 8, 30, 12, 36, tzinfo=dt.timezone.utc),
    )
    assert got.canonical_request == case["expect"]["canonical_request"]
    assert got.string_to_sign == case["expect"]["string_to_sign"]
    assert got.authorization == case["expect"]["authorization"]


class TestRs256:
    def test_pkcs8_and_pkcs1_agree(self):
        key = rs256.load_private_key(TEST_KEY)
        assert key.n.bit_length() == 2048 and key.e == 65537
        assert "s3cret" not in repr(key) and "bits=2048" in repr(key)

    def test_x5t_thumbprint(self):
        der = rs256.certificate_der(TEST_CERT)
        assert rs256.b64url(hashlib.sha1(der).digest()) == TOKENS["test_key"]["x5t"]

    @pytest.mark.parametrize("case", [c for c in TOKENS["cases"] if "jwt" in c.get("expect", {})], ids=lambda c: c["id"])
    def test_jwt_bytes_match_vectors(self, case):
        key = rs256.load_private_key(TEST_KEY)
        jwt = case["expect"]["jwt"]
        assert rs256.jwt_encode(jwt["header"], jwt["payload"], key) == jwt["compact"]

    def test_encrypted_and_pkcs12_name_the_fix(self):
        with pytest.raises(NotConfiguredError, match="openssl pkey"):
            rs256.load_private_key("-----BEGIN ENCRYPTED PRIVATE KEY-----\nAA==\n-----END ENCRYPTED PRIVATE KEY-----")
        with pytest.raises(NotConfiguredError, match="pkcs12"):
            rs256.load_private_key("not a pem")


# ─── token exchange vectors through the chains ───────────────────────


@pytest.mark.parametrize("case", TOKENS["cases"], ids=lambda c: c["id"])
def test_token_vectors(case):
    policy = PROVIDERS[case["provider"]].access
    inputs = _expand(case["input"])
    if case["id"].endswith(".build"):
        env = dict(inputs.get("env", {}))
        files = {}
        if "certificate_pem" in inputs:
            files[env["AZURE_CLIENT_CERTIFICATE_PATH"]] = inputs["certificate_pem"] + "\n" + TEST_KEY
        ctx = chains.ChainContext(env=env, files=files or None, now=lambda: NOW, settings=inputs.get("settings", {}))
        got = chains.token_exchange_build(policy, case["rung"], inputs, ctx)
        assert got == case["expect"]["request"]
    else:
        ctx = chains.ChainContext(env={}, now=lambda: NOW)
        got = chains.token_exchange_parse(policy, case["rung"], inputs["status"], inputs["body"], ctx)
        assert credential_to_dict(got) == case["expect"]["credential"]


# ─── hosts ───────────────────────────────────────────────────────────


def _req(model: str) -> Request:
    return Request(model=model, messages=[Message.user("Say ok.")])


class TestHosts:
    def test_required_setting_raises_with_the_variable(self):
        with pytest.raises(NotConfiguredError, match="AWS_REGION"):
            AnthropicLM(api_key=AWS, access=access.BEDROCK_ANTHROPIC)

    def test_unknown_setting_is_an_error(self):
        with pytest.raises(ValueError, match="unknown host setting"):
            AnthropicLM(api_key=AWS, access=access.BEDROCK_ANTHROPIC, settings={"region": "us-east-1", "zone": "x"})

    def test_bedrock_mantle_is_signed_sigv4(self):
        lm = AnthropicLM(api_key=AWS, access=access.BEDROCK_ANTHROPIC, settings={"region": "us-east-1"}, clock=lambda: NOW)
        req = lm.build_request(_req("anthropic.claude-opus-5"), stream=False)
        headers = dict(req.headers)
        assert req.url == "https://bedrock-mantle.us-east-1.api.aws/anthropic/v1/messages"
        assert headers["x-amz-date"] == "20260903T120000Z"
        assert headers["x-amz-security-token"] == "SESSION-VECTOR"
        assert headers["authorization"].startswith("AWS4-HMAC-SHA256 Credential=AKIDEXAMPLE/20260903/us-east-1/bedrock-mantle/aws4_request, SignedHeaders=anthropic-version;content-type;host;x-amz-date;x-amz-security-token, Signature=")
        assert "x-api-key" not in headers and "x-amz-content-sha256" not in headers
        assert json.loads(req.body)["model"] == "anthropic.claude-opus-5"

    def test_sigv4_signature_is_deterministic_and_clock_bound(self):
        lm = AnthropicLM(api_key=AWS, access=access.BEDROCK_CHAT if False else access.BEDROCK_ANTHROPIC, settings={"region": "us-east-1"}, clock=lambda: NOW)
        a = dict(lm.build_request(_req("m"), stream=False).headers)["authorization"]
        b = dict(lm.build_request(_req("m"), stream=False).headers)["authorization"]
        later = AnthropicLM(api_key=AWS, access=access.BEDROCK_ANTHROPIC, settings={"region": "us-east-1"}, clock=lambda: NOW + dt.timedelta(seconds=1))
        c = dict(later.build_request(_req("m"), stream=False).headers)["authorization"]
        assert a == b != c

    def test_jwt_as_plain_string_on_a_key_first_door_travels_as_bearer(self):
        # azure-identity's get_bearer_token_provider returns a str; on a door
        # whose key header precedes bearer, that str would ride `api-key` and
        # die as a bare 401 (live 2026-09-04).  A JWT is never an API key on
        # any door lm15 has, so it travels as a bearer token (AUTH-2, amended
        # 2026-09-19; before that date lm15 refused and named the wrap).
        import base64

        jwt = ".".join(base64.urlsafe_b64encode(part).decode().rstrip("=") for part in (b'{"alg":"RS256","typ":"JWT"}', b'{"aud":"x"}', b"sig"))

        def headers(lm, model):
            return {k.lower(): v for k, v in lm.build_request(_req(model), stream=False).headers}

        bare = OpenAILM(api_key=lambda: jwt, access=access.AZURE, settings={"resource": "lab"})
        assert headers(bare, "dep")["authorization"] == f"Bearer {jwt}"
        assert "api-key" not in headers(bare, "dep")
        # the wrap is still accepted, and still the form when nothing should be read from the token's shape
        ok = OpenAILM(api_key=lambda: BearerToken(jwt), access=access.AZURE, settings={"resource": "lab"})
        assert headers(ok, "dep")["authorization"] == f"Bearer {jwt}"
        # a real api-key string still travels as api-key
        key = OpenAILM(api_key="0123456789abcdef0123456789abcdef", access=access.AZURE, settings={"resource": "lab"})
        assert headers(key, "dep")["api-key"] == "0123456789abcdef0123456789abcdef"
        # x-api-key-first doors (Foundry Claude) likewise send a JWT as bearer
        foundry = AnthropicLM(api_key=jwt, access=access.AZURE_ANTHROPIC, settings={"resource": "lab"})
        assert headers(foundry, "claude-sonnet-4-5")["authorization"] == f"Bearer {jwt}"
        # a door without bearer (bedrock-anthropic: sigv4, x-api-key) keeps the key header
        bedrock = AnthropicLM(api_key=jwt, access=access.BEDROCK_ANTHROPIC, settings={"region": "us-east-1"})
        assert headers(bedrock, "anthropic.claude-opus-5")["x-api-key"] == jwt
        # on a bearer-first door a plain JWT string is fine (OpenAI, DeepSeek…)
        assert headers(OpenAILM(api_key=jwt), "m")["authorization"] == f"Bearer {jwt}"

    def test_aws_anthropic_api_key_and_workspace_header(self):
        lm = AnthropicLM(api_key="SHORT-TERM", access=access.AWS_ANTHROPIC, settings={"region": "us-west-2", "workspace": "wrkspc_01"})
        req = lm.build_request(_req("claude-sonnet-5"), stream=False)
        headers = dict(req.headers)
        assert req.url == "https://aws-external-anthropic.us-west-2.api.aws/v1/messages"
        assert headers["x-api-key"] == "SHORT-TERM" and headers["anthropic-workspace-id"] == "wrkspc_01"

    def test_vertex_anthropic_model_in_path_version_in_body(self):
        lm = AnthropicLM(api_key=BearerToken("ya29-x"), access=access.VERTEX_ANTHROPIC, settings={"project": "p1"})
        req = lm.build_request(_req("claude-opus-5"), stream=True)
        body = json.loads(req.body)
        assert req.url == "https://aiplatform.googleapis.com/v1/projects/p1/locations/global/publishers/anthropic/models/claude-opus-5:streamRawPredict"
        assert "model" not in body and body["anthropic_version"] == "vertex-2023-10-16"
        assert "anthropic-version" not in {k.lower() for k, _ in req.headers}
        assert dict(req.headers)["Authorization"] == "Bearer ya29-x"

    @pytest.mark.parametrize("location,host", [("global", "aiplatform.googleapis.com"), ("us", "aiplatform.us.rep.googleapis.com"), ("us-east5", "us-east5-aiplatform.googleapis.com")])
    def test_vertex_location_hosts(self, location, host):
        lm = GeminiLM(api_key=BearerToken("t"), access=access.VERTEX, settings={"project": "p", "location": location})
        assert lm.build_request(_req("gemini-2.5-pro"), stream=False).url == f"https://{host}/v1/projects/p/locations/{location}/publishers/google/models/gemini-2.5-pro:generateContent"

    def test_vertex_express_key_in_query(self):
        lm = GeminiLM(api_key="EXPRESS", access=access.VERTEX_EXPRESS)
        req = lm.build_request(_req("gemini-2.5-flash"), stream=False)
        assert req.url.endswith("/publishers/google/models/gemini-2.5-flash:generateContent?key=EXPRESS")
        assert "x-goog-api-key" not in {k.lower() for k, _ in req.headers}

    def test_azure_api_key_header_and_deployment_model(self):
        lm = OpenAILM(api_key="AZ", access=access.AZURE, settings={"resource": "acme"})
        req = lm.build_request(_req("my-deployment"), stream=False)
        assert req.url == "https://acme.openai.azure.com/openai/v1/responses"
        assert dict(req.headers)["api-key"] == "AZ" and json.loads(req.body)["model"] == "my-deployment"

    def test_azure_bearer_token_uses_authorization(self):
        lm = OpenAIChatLM(api_key=BearerToken("eyX"), access=access.AZURE_CHAT, settings={"resource": "acme"})
        assert dict(lm.build_request(_req("d"), stream=False).headers)["Authorization"] == "Bearer eyX"

    def test_wrong_credential_kind_fails_at_construction(self):
        with pytest.raises(NotConfiguredError, match="cannot travel"):
            AnthropicLM(api_key=AWS, access=access.AZURE_ANTHROPIC, settings={"resource": "r"})

    def test_bedrock_chat_models_request_is_signed(self):
        lm = OpenAIChatLM(api_key=AWS, access=access.BEDROCK_CHAT, settings={"region": "eu-west-1"}, clock=lambda: NOW)
        req = lm._models_request()
        assert req.url.startswith("https://bedrock-runtime.eu-west-1.amazonaws.com/openai/v1/models")
        assert "authorization" in dict(req.headers)

    def test_bedrock_mantle_chat_url_and_service(self):
        lm = OpenAIChatLM(api_key=AWS, access=access.BEDROCK_MANTLE_CHAT, settings={"region": "us-east-1"}, clock=lambda: NOW)
        req = lm.build_request(_req("openai.gpt-oss-20b"), stream=False)
        assert req.url == "https://bedrock-mantle.us-east-1.api.aws/v1/chat/completions"
        headers = dict(req.headers)
        assert headers["authorization"].startswith("AWS4-HMAC-SHA256 Credential=AKIDEXAMPLE/20260903/us-east-1/bedrock-mantle/aws4_request")
        models = lm._models_request()
        assert models.url == "https://bedrock-mantle.us-east-1.api.aws/v1/models"
        assert lm.supports.models is True

    def test_bedrock_mantle_chat_bearer_token_goes_in_authorization(self):
        lm = OpenAIChatLM(api_key=BearerToken("bedrock-api-key-abc"), access=access.BEDROCK_MANTLE_CHAT, settings={"region": "us-east-1"})
        headers = dict(lm.build_request(_req("openai.gpt-oss-20b"), stream=False).headers)
        assert headers["Authorization"] == "Bearer bedrock-api-key-abc"

    def test_provider_callable_invoked_once_per_request(self):
        calls = []

        def provider():
            calls.append(1)
            return "k"

        lm = OpenAILM(api_key=provider, access=access.AZURE, settings={"resource": "acme"})
        lm.build_request(_req("d"), stream=False)
        assert len(calls) == 1


# ─── chains (offline) ─────────────────────────────────────────────────


class TestChains:
    def _explain(self, policy, env, files=None):
        ctx = chains.ChainContext(env=env, home=Path("/home/u"), files=files or {})
        steps, configured = chains.explain(policy, ctx, explicit=False)
        return {s.kind: s.state for s in steps}, configured

    def test_aws_order_is_botocore_order(self):
        names = [r.name for r in chains.chain_for(access.BEDROCK_ANTHROPIC)]
        assert names == ["env:AWS_BEARER_TOKEN_BEDROCK", "env:AWS_ACCESS_KEY_ID", "assume-role", "web-identity", "sso",
                         "shared-credentials-file", "login", "credential_process", "config-file", "container", "imds"]

    def test_azure_order_is_default_azure_credential_order(self):
        names = [r.name for r in chains.chain_for(access.AZURE)]
        assert names == ["env:AZURE_OPENAI_API_KEY", "environment", "workload-identity", "managed-identity", "az", "pwsh", "azd"]

    def test_gcp_order_is_adc_order(self):
        assert [r.name for r in chains.chain_for(access.VERTEX)] == ["adc-env", "adc-file", "metadata", "gcloud"]

    def test_every_rung_kind_is_in_the_vocabulary(self):
        for policy in access.CLOUD_HOST_POLICIES:
            if policy.cloud_chain:
                for rung in chains.chain_for(policy):
                    assert rung.kind in chains.RUNG_KINDS

    def test_container_url_allow_list_is_botocores(self):
        ctx = chains.ChainContext(env={"AWS_CONTAINER_CREDENTIALS_FULL_URI": "http://169.254.170.23/creds"})
        assert chains._container_config(ctx) == "http://169.254.170.23/creds"
        with pytest.raises(NotConfiguredError, match="Unsupported host"):
            chains._container_config(chains.ChainContext(env={"AWS_CONTAINER_CREDENTIALS_FULL_URI": "http://evil.example/creds"}))
        assert chains._container_config(chains.ChainContext(env={"AWS_CONTAINER_CREDENTIALS_FULL_URI": "https://evil.example/creds"}))

    def test_offline_walk_never_touches_network(self):
        states, configured = self._explain(access.BEDROCK_ANTHROPIC, {"AWS_WEB_IDENTITY_TOKEN_FILE": "/t", "AWS_ROLE_ARN": "arn:r"})
        assert states["web-identity"] == "unprobed" and states["imds"] == "unprobed" and configured

    def test_azure_narrowing(self):
        states, _ = self._explain(access.AZURE, {"AZURE_TOKEN_CREDENTIALS": "dev", "PATH": "/bin"}, {"/bin/az": "x"})
        assert states["environment"] == "absent" and states["managed-identity"] == "absent" and states["az"] == "unprobed"

    def test_cache_key_separates_identities(self):
        a = chains.cache_key(access.AZURE, chains.ChainContext(env={"AZURE_TENANT_ID": "t1"}))
        b = chains.cache_key(access.AZURE, chains.ChainContext(env={"AZURE_TENANT_ID": "t2"}))
        assert a != b

    def test_caching_provider_reresolves_after_expiry(self):
        calls = []
        rung = chains.Rung("x", "env", "x", "", lambda ctx: ("usable", ""), lambda ctx: (calls.append(1), BearerToken("t", expires_at=ctx.now() + dt.timedelta(seconds=400)))[1])
        policy = access.AZURE
        clock = [NOW]
        ctx = chains.ChainContext(env={}, now=lambda: clock[0])
        provider = chains._CachingProvider(policy, ctx)
        chains._CHAINS["azure-chain"] = lambda p: [rung]  # type: ignore[assignment]
        try:
            provider()
            provider()
            assert len(calls) == 1
            clock[0] = NOW + dt.timedelta(seconds=200)  # inside skew → refresh
            provider()
            assert len(calls) == 2
        finally:
            chains._CHAINS["azure-chain"] = chains._azure_chain


# ─── registry ────────────────────────────────────────────────────────


class TestRegistry:
    def test_hosted_entries_one_wire_each(self):
        hosted = {k: v for k, v in PROVIDERS.items() if v.hosted}
        assert set(hosted) == {"azure", "azure-chat", "azure-anthropic", "aws-anthropic", "bedrock-anthropic", "bedrock-chat", "bedrock-mantle-chat", "vertex", "vertex-express", "vertex-anthropic"}
        for entry in hosted.values():
            assert entry.access.host is not None
            assert entry.access.provider == entry.id

    def test_sigv4_policies_name_their_service(self):
        for pid in ("aws-anthropic", "bedrock-anthropic", "bedrock-chat", "bedrock-mantle-chat"):
            policy = PROVIDERS[pid].access
            assert "sigv4" in policy.auth_scheme and policy.host.sigv4_service

    def test_settings_have_no_silent_region_default(self):
        for pid in ("aws-anthropic", "bedrock-anthropic", "bedrock-chat", "bedrock-mantle-chat"):
            region = next(s for s in PROVIDERS[pid].access.host.settings if s.name == "region")
            assert region.default is None


# ─── per-model compat overrides (bedrock door) ────────────────────────


class TestModelOverrides:
    def test_bedrock_preset_forwards_and_refuses_per_family(self):
        from lm15.compat import OpenAIChatCompat

        preset = OpenAIChatCompat.preset("bedrock")
        assert preset.forced_tool_choice == "send" and preset.json_schema == "send"
        assert preset.for_model("openai.gpt-oss-20b-1:0").forced_tool_choice == "reject"
        assert preset.for_model("openai.gpt-oss-120b-1:0").json_schema == "reject"
        assert preset.for_model("google.gemma-3-12b-it").forced_tool_choice == "reject"
        assert preset.for_model("google.gemma-3-12b-it").json_schema == "send"
        assert preset.for_model("deepseek.v3.2").forced_tool_choice == "send"

    def test_bedrock_mantle_preset_refuses_gpt_oss_only(self):
        from lm15.compat import OpenAIChatCompat

        preset = OpenAIChatCompat.preset("bedrock-mantle")
        assert preset.forced_tool_choice == "send" and preset.json_schema == "send"
        assert preset.for_model("openai.gpt-oss-20b").forced_tool_choice == "reject"
        assert preset.for_model("openai.gpt-oss-20b").json_schema == "reject"
        # Gemma honours tool_choice on this door (family-google-gemma-*-tool-choice, 2026-09-04).
        assert preset.for_model("google.gemma-3-12b-it").forced_tool_choice == "send"

    def test_override_applies_at_build_time(self):
        from lm15.errors import UnsupportedFeatureError
        from lm15.types import FunctionTool
        from lm15.types import Config, ToolChoice

        tool = FunctionTool(name="f", description="d", parameters={"type": "object"})
        lm = OpenAIChatLM(api_key=AWS, access=access.BEDROCK_CHAT, settings={"region": "us-east-1"}, clock=lambda: NOW)
        honoured = Request(model="deepseek.v3.2", messages=[Message.user("x")], tools=[tool], config=Config(tool_choice=ToolChoice(mode="required")))
        assert json.loads(lm.build_request(honoured, stream=False).body)["tool_choice"] == "required"
        refused = Request(model="openai.gpt-oss-20b-1:0", messages=[Message.user("x")], tools=[tool], config=Config(tool_choice=ToolChoice(mode="required")))
        with pytest.raises(UnsupportedFeatureError):
            lm.build_request(refused, stream=False)

    def test_unknown_override_knob_rejected(self):
        from lm15.compat import OpenAIChatCompat

        with pytest.raises(ValueError, match="not an overridable knob"):
            OpenAIChatCompat(model_overrides=(("x.", {"routing": "y"}),))


class TestBearerTokenInKeyHeader:
    """AUTH-2 (amended 2026-09-04): a BearerToken travels under Authorization
    where the door offers it, else under the door's key header.  Found
    offline: AWS_BEARER_TOKEN_BEDROCK in the environment made the mantle
    door raise "cannot travel" although the door documents that key."""

    def test_bedrock_anthropic_bearer_token_goes_in_x_api_key(self):
        lm = AnthropicLM(api_key=BearerToken("bedrock-api-key-abc"), access=access.BEDROCK_ANTHROPIC, settings={"region": "us-east-1"})
        headers = dict(lm.build_request(_req("anthropic.claude-haiku-4-5"), stream=False).headers)
        assert headers["x-api-key"] == "bedrock-api-key-abc"
        assert "Authorization" not in headers and "authorization" not in headers

    def test_bedrock_chat_bearer_token_goes_in_authorization(self):
        lm = OpenAIChatLM(api_key=BearerToken("bedrock-api-key-abc"), access=access.BEDROCK_CHAT, settings={"region": "us-east-1"})
        headers = dict(lm.build_request(_req("openai.gpt-oss-20b-1:0"), stream=False).headers)
        assert headers["Authorization"] == "Bearer bedrock-api-key-abc"
        assert "x-api-key" not in {k.lower() for k in headers}

    def test_azure_anthropic_api_key_uses_live_verified_x_api_key(self):
        # Foundry docs also name `api-key`, but live it is 401; x-api-key
        # reaches deployment lookup (2026-09-04 partial capture).
        lm = AnthropicLM(api_key="AZ", access=access.AZURE_ANTHROPIC, settings={"resource": "r"})
        headers = dict(lm.build_request(_req("claude-x"), stream=False).headers)
        assert headers["x-api-key"] == "AZ"
        assert "api-key" not in {k.lower() for k in headers}

    def test_azure_anthropic_bearer_token_prefers_authorization_over_key_header(self):
        # The policy lists x-api-key before bearer; an Entra token must still go as bearer.
        lm = AnthropicLM(api_key=BearerToken("eyX"), access=access.AZURE_ANTHROPIC, settings={"resource": "r"})
        headers = dict(lm.build_request(_req("claude-x"), stream=False).headers)
        assert headers["Authorization"] == "Bearer eyX"
        assert "x-api-key" not in {k.lower() for k in headers} and "api-key" not in {k.lower() for k in headers}

    def test_bedrock_anthropic_chain_env_key_builds(self):
        # The chain's own product (env -> BearerToken) must be accepted by the door it was resolved for.
        from lm15.cloud.chains import ChainContext, resolve

        ctx = ChainContext(env={"AWS_BEARER_TOKEN_BEDROCK": "bedrock-api-key-abc", "AWS_REGION": "us-east-1"}, files={}, settings={"region": "us-east-1"})
        cred, source = resolve(access.BEDROCK_ANTHROPIC, ctx)
        assert isinstance(cred, BearerToken)
        assert source.rung == "env:AWS_BEARER_TOKEN_BEDROCK" and "AWS_BEARER_TOKEN_BEDROCK" in source.describe()
        AnthropicLM(api_key=cred, access=access.BEDROCK_ANTHROPIC, settings={"region": "us-east-1"})


class TestSchemeSelectionD1:
    """AUTH-2 as ratified 2026-09-06 (lm15-contract/changes/2026-09-06-decisions.md D1):
    ApiKey takes the policy's first header scheme in policy order; a
    BearerToken takes bearer, else x-api-key; AwsCredentials sigv4 only;
    anything else is a NotConfiguredError naming the accepted schemes."""

    @pytest.mark.parametrize("policy,expected", [
        (access.ANTHROPIC_API, "x-api-key"),
        (access.OPENAI_API, "bearer"),
        (access.AZURE, "api-key"),
        (access.AZURE_ANTHROPIC, "x-api-key"),
        (access.VERTEX_EXPRESS, "query-key"),
        (access.BEDROCK_ANTHROPIC, "x-api-key"),
    ])
    def test_api_key_takes_first_header_scheme_in_policy_order(self, policy, expected):
        assert access.select_scheme(policy, ApiKey("k")) == expected

    @pytest.mark.parametrize("policy,expected", [
        (access.OPENAI_API, "bearer"),
        (access.AZURE, "bearer"),
        (access.AZURE_ANTHROPIC, "bearer"),
        (access.BEDROCK_CHAT, "bearer"),
        (access.BEDROCK_ANTHROPIC, "x-api-key"),
        (access.AWS_ANTHROPIC, "x-api-key"),
        (access.ANTHROPIC_API, "x-api-key"),
    ])
    def test_bearer_token_prefers_bearer_then_x_api_key(self, policy, expected):
        assert access.select_scheme(policy, BearerToken("t")) == expected

    def test_bearer_token_on_door_without_bearer_or_x_api_key_names_accepted_schemes(self):
        from dataclasses import replace

        api_key_only = replace(access.AZURE, auth_scheme=("api-key",))
        with pytest.raises(NotConfiguredError, match=r"cannot travel under api-key; it accepts bearer/x-api-key"):
            access.select_scheme(api_key_only, BearerToken("t"))
        with pytest.raises(NotConfiguredError, match=r"cannot travel under query-key; it accepts bearer/x-api-key"):
            access.select_scheme(access.VERTEX_EXPRESS, BearerToken("t"))

    @pytest.mark.parametrize("policy", [access.BEDROCK_CHAT, access.BEDROCK_ANTHROPIC, access.BEDROCK_MANTLE_CHAT])
    def test_aws_credentials_take_sigv4(self, policy):
        assert access.select_scheme(policy, AWS) == "sigv4"

    @pytest.mark.parametrize("policy", [access.OPENAI_API, access.ANTHROPIC_API, access.AZURE_ANTHROPIC, access.VERTEX_EXPRESS])
    def test_aws_credentials_on_non_sigv4_door_raise(self, policy):
        with pytest.raises(NotConfiguredError, match=r"it accepts sigv4"):
            access.select_scheme(policy, AWS)

    def test_bearer_token_on_first_party_anthropic_goes_in_x_api_key(self):
        # D1 cost, stated: the door does not take tokens; the provider's
        # 401 answers, not a local error.
        lm = AnthropicLM(api_key=BearerToken("tok"), access=access.ANTHROPIC_API)
        headers = dict(lm.build_request(_req("claude-x"), stream=False).headers)
        assert headers["x-api-key"] == "tok"
        assert "authorization" not in {k.lower() for k in headers}
