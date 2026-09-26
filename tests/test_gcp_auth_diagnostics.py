"""Google Cloud (Vertex) auth failures say what to do, and nothing secret.

Each case here was first met live on 2026-09-26 (project lm15-vertex-live;
see lm15-contract/changes/2026-09-26-vertex-live.md): a stale ADC login, an
impersonation target that did not exist, an expired federation token, an
API key sent to the OAuth door, and a fresh project whose IAM grant had not
propagated.  Before, each surfaced as "HTTP 400" plus API-key advice.

The secrecy boundary is AUTH-5 / AUTH-21: from a failed token exchange only
the status and a fixed-vocabulary OAuth word may be shown; a command's
stderr and argv are never shown.
"""

from __future__ import annotations

import json

import pytest

from lm15 import access
from lm15.cloud import chains
from lm15.credentials import ApiKey, BearerToken
from lm15.errors import AuthError, NotConfiguredError
from lm15.providers import GeminiLM
from lm15.types import Message, Request

SECRET = "SECRET-SENTINEL-DO-NOT-PRINT"


def _ctx(status: int, body: dict, creds: str) -> chains.ChainContext:
    files = {"/creds.json": creds}
    return chains.ChainContext(
        env={"GOOGLE_APPLICATION_CREDENTIALS": "/creds.json", "NO_GCE_CHECK": "1"},
        http=lambda *a: (status, {}, json.dumps(body).encode()),
        files=files,
        run=None,
    )


def _resolve(ctx: chains.ChainContext):
    return chains.resolve(access.VERTEX, ctx)


def _authorized_user() -> str:
    return json.dumps({"type": "authorized_user", "client_id": "c", "client_secret": SECRET, "refresh_token": SECRET})


def test_stale_adc_login_names_the_word_and_the_command_not_the_description():
    ctx = _ctx(400, {"error": "invalid_grant", "error_description": f"reauth related error {SECRET}"},
               **{"creds": _authorized_user()})
    with pytest.raises(AuthError) as exc:
        _resolve(ctx)
    text = str(exc.value) + repr(exc.value)
    assert "HTTP 400 (invalid_grant)" in text
    assert exc.value.provider_code == "invalid_grant"
    assert "gcloud auth application-default login" in text
    assert "/creds.json" in text
    assert SECRET not in text
    assert "API key" not in text


def test_words_outside_the_oauth_vocabulary_are_not_shown():
    ctx = _ctx(400, {"error": SECRET}, **{"creds": _authorized_user()})
    with pytest.raises(AuthError) as exc:
        _resolve(ctx)
    assert SECRET not in str(exc.value) + repr(exc.value)
    assert exc.value.provider_code is None


def test_impersonation_refusal_names_the_file_and_the_role_not_the_account():
    info = {
        "type": "impersonated_service_account",
        "service_account_impersonation_url":
            "https://iamcredentials.googleapis.com/v1/projects/-/serviceAccounts/robot@p.iam.gserviceaccount.com:generateAccessToken",
        "source_credentials": json.loads(_authorized_user()),
    }
    calls = iter([(200, {"access_token": "src", "expires_in": 3600}),
                  (403, {"error": {"code": 403, "status": "PERMISSION_DENIED", "message": SECRET}})])

    def http(*_a):
        status, body = next(calls)
        return status, {}, json.dumps(body).encode()

    ctx = chains.ChainContext(env={"GOOGLE_APPLICATION_CREDENTIALS": "/creds.json"}, http=http,
                              files={"/creds.json": json.dumps(info)}, run=None)
    with pytest.raises(AuthError) as exc:
        _resolve(ctx)
    text = str(exc.value)
    assert "HTTP 403" in text and "roles/iam.serviceAccountTokenCreator" in text and "/creds.json" in text
    assert "robot@" not in text  # an account id is session-sensitive (AUTH-21), the file names it
    assert SECRET not in text + repr(exc.value)


def test_gcloud_rung_failure_points_at_the_command_without_its_output():
    def run(argv, timeout):
        raise AuthError("credential command exited 1")

    ctx = chains.ChainContext(env={"NO_GCE_CHECK": "1", "CLOUDSDK_CONFIG": "/none", "PATH": "/bin"}, http=None,
                              files={"/bin/gcloud": ""}, run=run)
    with pytest.raises(AuthError) as exc:
        _resolve(ctx)
    text = str(exc.value)
    assert "`gcloud auth print-access-token` failed" in text
    assert "run `gcloud auth print-access-token` yourself" in text
    assert "API key" not in text


def test_nothing_found_lists_what_was_probed_and_the_commands():
    ctx = chains.ChainContext(env={"NO_GCE_CHECK": "1", "CLOUDSDK_CONFIG": "/none", "PATH": "/bin"}, http=None,
                              files={}, run=None)
    with pytest.raises(NotConfiguredError) as exc:
        _resolve(ctx)
    text = str(exc.value)
    assert "gcloud auth application-default login" in text
    assert "GOOGLE_APPLICATION_CREDENTIALS" in text
    assert "NO_GCE_CHECK set" in text  # the probe summary says why each rung was skipped


def _vertex(api_key) -> GeminiLM:
    return GeminiLM(api_key=api_key, access=access.VERTEX, settings={"project": "p"})


def _google_error(code: int, status: str) -> str:
    return json.dumps({"error": {"code": code, "status": status, "message": "Permission denied."}})


def test_vertex_403_is_an_iam_question():
    err = _vertex(BearerToken(SECRET)).normalize_error(403, _google_error(403, "PERMISSION_DENIED"))
    assert isinstance(err, AuthError)
    text = str(err)
    assert "roles/aiplatform.user" in text and "few minutes" in text
    assert "Check that your API key" not in text
    assert text.count("credential came from:") == 1
    assert SECRET not in text


def test_vertex_401_for_a_token_says_it_expired():
    err = _vertex(BearerToken(SECRET)).normalize_error(401, _google_error(401, "UNAUTHENTICATED"))
    text = str(err)
    assert "access token" in text and "gcloud auth application-default login" in text
    assert "Check that your API key" not in text


def test_vertex_401_for_a_key_names_vertex_keys_and_the_bearer_wrap():
    err = _vertex("AQ.not-a-real-key").normalize_error(401, _google_error(401, "UNAUTHENTICATED"))
    text = str(err)
    assert "Vertex AI key" in text and "BearerToken(value)" in text
    assert "not-a-real-key" not in text


def test_vertex_express_keeps_the_api_key_guidance():
    lm = GeminiLM(api_key=ApiKey(SECRET), access=access.VERTEX_EXPRESS)
    err = lm.normalize_error(403, _google_error(403, "PERMISSION_DENIED"))
    assert isinstance(err, AuthError)
    assert "roles/aiplatform.user" not in str(err)


# ─── API keys on the vertex door (AUTH-2/AUTH-10, amended 2026-09-26) ─


def _auth_headers(lm: GeminiLM) -> dict:
    built = lm.build_request(Request(model="gemini-2.5-flash", messages=[Message.user("hi")]), stream=False)
    headers = dict(built.headers)
    return {k.lower(): v for k, v in headers.items() if k.lower() in ("authorization", "x-goog-api-key")}


@pytest.mark.parametrize("key", ["AQ.Ab8RN6-test", "AIzaSyTestKey", "test-key-123"])
def test_a_string_on_vertex_is_a_key(key):
    assert _auth_headers(_vertex(key)) == {"x-goog-api-key": key}


@pytest.mark.parametrize("token", ["ya29.a0-test", "eyJhbGciOiJSUzI1NiJ9.eyJzdWIiOiJ4In0.c2ln"])
def test_a_token_shaped_string_on_vertex_is_a_bearer_token(token):
    assert _auth_headers(_vertex(token)) == {"authorization": f"Bearer {token}"}


def test_a_bearer_wrap_is_never_read_as_a_key():
    assert _auth_headers(_vertex(BearerToken("opaque-token"))) == {"authorization": "Bearer opaque-token"}


def test_vertex_anthropic_takes_no_key_header():
    assert access.VERTEX_ANTHROPIC.auth_scheme == ("bearer",)


def test_vertex_reads_no_ambient_google_api_key():
    assert access.VERTEX.env_keys == ()


# ─── Where the project comes from (AUTH-10, amended 2026-09-26) ───────


def _project(env: dict, files: dict, http=None):
    from lm15.cloud.hosts import resolve_settings

    ctx = chains.ChainContext(env={"HOME": "/h", **env}, home=__import__("pathlib").Path("/h"), files=files, http=http)
    sources: dict = {}
    try:
        value = resolve_settings(access.VERTEX.host, None, ctx.env, provider="vertex",
                                 profile=chains.profile_settings(access.VERTEX, ctx), sources=sources,
                                 unprobed_ok=http is None).get("project")
    except NotConfiguredError:
        value = None
    return value, sources.get("project")


CONFIG = "[core]\naccount = a@example.com\nproject = from-gcloud\n"
ADC = json.dumps({"type": "authorized_user", "client_id": "c", "client_secret": "s", "refresh_token": "r",
                  "quota_project_id": "from-adc-quota"})
SA = json.dumps({"type": "service_account", "project_id": "from-sa", "client_email": "x@y", "private_key": "k"})


def test_project_env_wins():
    assert _project({"GOOGLE_CLOUD_PROJECT": "from-env"}, {"~/.config/gcloud/configurations/config_default": CONFIG}) == \
        ("from-env", "env:GOOGLE_CLOUD_PROJECT")


def test_project_service_account_file_before_gcloud():
    files = {"/sa.json": SA, "~/.config/gcloud/configurations/config_default": CONFIG}
    assert _project({"GOOGLE_APPLICATION_CREDENTIALS": "/sa.json"}, files) == ("from-sa", "adc-env")


def test_project_gcloud_config_before_adc_quota_project():
    files = {"~/.config/gcloud/application_default_credentials.json": ADC,
             "~/.config/gcloud/configurations/config_default": CONFIG}
    assert _project({"NO_GCE_CHECK": "1"}, files) == ("from-gcloud", "gcloud-config")


def test_project_active_named_configuration():
    files = {"~/.config/gcloud/active_config": "work\n",
             "~/.config/gcloud/configurations/config_default": CONFIG,
             "~/.config/gcloud/configurations/config_work": "[core]\nproject = from-work\n"}
    assert _project({}, files) == ("from-work", "gcloud-config")
    assert _project({"CLOUDSDK_ACTIVE_CONFIG_NAME": "default"}, files) == ("from-gcloud", "gcloud-config")


def test_project_cloudsdk_core_project_and_config_dir():
    assert _project({"CLOUDSDK_CORE_PROJECT": "from-core-env"}, {}) == ("from-core-env", "env:CLOUDSDK_CORE_PROJECT")
    files = {"/cfg/configurations/config_default": CONFIG}
    assert _project({"CLOUDSDK_CONFIG": "/cfg"}, files) == ("from-gcloud", "gcloud-config")


def test_project_bad_config_name_never_leaves_the_directory():
    files = {"~/.config/gcloud/configurations/config_../../x": CONFIG}
    assert _project({"CLOUDSDK_ACTIVE_CONFIG_NAME": "../../x", "NO_GCE_CHECK": "1"}, files) == (None, "missing")


def test_project_adc_quota_project_last_before_metadata():
    files = {"~/.config/gcloud/application_default_credentials.json": ADC}
    assert _project({}, files) == ("from-adc-quota", "adc-file")


def test_project_metadata_offline_is_unprobed_and_online_is_asked():
    assert _project({}, {}) == (None, "unprobed:metadata")
    seen = []

    def http(method, url, headers, body, timeout):
        seen.append((url, headers.get("Metadata-Flavor")))
        return 200, {}, b"from-metadata"

    assert _project({}, {}, http=http) == ("from-metadata", "metadata")
    assert seen == [("http://metadata.google.internal/computeMetadata/v1/project/project-id", "Google")]
    assert _project({"NO_GCE_CHECK": "1"}, {}, http=http) == (None, "missing")
