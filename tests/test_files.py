"""Files: account-scoped storage (upload / get / list / delete / download).

Synthetic wire bodies below mirror live captures from 2026-08-31
(lm15-dev/curl-fixtures/files-2026-08-31/): OpenAI /v1/files (multipart,
purpose form field, epoch times, `after` cursor), Anthropic /v1/files
(multipart GA, `next_page` cursor, `downloadable` verbatim), Gemini
/upload/v1beta/files (multipart/related with display_name, wrapped
upload body, URI addressing, `pageToken` cursor, `:download` refusal).
"""
from __future__ import annotations

import json

import pytest

from lm15 import (
    FileInfo,
    FilePage,
    FileUploadRequest,
    InvalidRequestError,
    UnsupportedFeatureError,
)
from lm15.providers import AnthropicLM, GeminiLM, OpenAILM
from lm15.serde import (
    file_info_from_dict,
    file_info_to_dict,
    file_page_from_dict,
    file_page_to_dict,
    file_upload_request_from_dict,
    file_upload_request_to_dict,
)
from lm15.testing import FakeResponse, FakeTransport
from lm15.types import FILE_READINESS_VALUES


def wire(payload: dict | str | bytes, status: int = 200) -> FakeResponse:
    if isinstance(payload, bytes):
        return FakeResponse(status=status, body=payload)
    body = payload if isinstance(payload, str) else json.dumps(payload)
    return FakeResponse(status=status, body=body.encode("utf-8"))


# ─── Live-captured wire bodies (2026-08-31) ─────────────────────────

OPENAI_FILE = {
    "object": "file", "id": "file-DHLYq8rMTixWCBz1GF8auM",
    "purpose": "user_data", "filename": "sample.txt", "bytes": 76,
    "created_at": 1788215944, "expires_at": None,
    "status": "processed", "status_details": None,
}

ANTHROPIC_FILE = {
    "type": "file", "id": "file_01BMcX8RJhdSVQw4ns12MxHF", "size_bytes": 76,
    "created_at": "2026-08-31T22:39:04.248542Z", "expires_at": None,
    "filename": "sample.txt", "mime_type": "text/plain", "downloadable": False,
}

GEMINI_FILE = {
    "name": "files/n9i36fxppl2j", "displayName": "g3.txt",
    "mimeType": "text/plain", "sizeBytes": "76",
    "createTime": "2026-08-31T22:39:04.789662Z",
    "updateTime": "2026-08-31T22:39:04.789662Z",
    "expirationTime": "2026-09-02T22:39:04.608592208Z",
    "uri": "https://generativelanguage.googleapis.com/v1beta/files/n9i36fxppl2j",
    "state": "ACTIVE", "source": "UPLOADED",
}


# ─── Vocabulary / type invariants ────────────────────────────────────

def test_readiness_vocabulary_shape() -> None:
    assert FILE_READINESS_VALUES == {"pending", "ready", "failed"}


def test_file_info_invariants() -> None:
    with pytest.raises(ValueError, match="readiness"):
        FileInfo(id="f1", readiness="active")  # provider wire word, not canonical
    with pytest.raises(TypeError, match="downloadable"):
        FileInfo(id="f1", downloadable="yes")
    with pytest.raises(ValueError):
        FileInfo(id="")
    assert FileInfo(id="f1").ready is True
    assert FileInfo(id="f1", readiness="pending").ready is False


def test_upload_request_has_no_model_field() -> None:
    # Files are account-scoped on every provider; a model field would
    # imply a scoping no wire has.
    assert "model" not in {f for f in FileUploadRequest.__dataclass_fields__}


def test_upload_request_exactly_one_source() -> None:
    with pytest.raises(TypeError):
        FileUploadRequest(filename="a.txt")
    with pytest.raises(ValueError):
        FileUploadRequest(filename="a.txt", bytes_data=b"x", path="/tmp/a.txt")


def test_file_page_items_typed() -> None:
    with pytest.raises(TypeError):
        FilePage(items=("not-a-fileinfo",))


# ─── OpenAI ─────────────────────────────────────────────────────────

def test_openai_upload_defaults_to_user_data() -> None:
    transport = FakeTransport([wire(OPENAI_FILE)])
    lm = OpenAILM(api_key="k", transport=transport)
    info = lm.file_upload(FileUploadRequest(filename="sample.txt", bytes_data=b"x" * 76, media_type="text/plain"))
    body = transport.requests[0].body
    assert b'name="purpose"' in body and b"user_data" in body
    assert b'filename="sample.txt"' in body
    assert info.id == "file-DHLYq8rMTixWCBz1GF8auM"
    assert info.readiness == "ready"
    assert info.media_type is None  # OpenAI reports no MIME type
    assert info.size_bytes == 76
    assert info.created_at == "2026-08-31T22:39:04Z"  # epoch normalized to ISO UTC
    assert info.downloadable is None  # purpose policy, not reported per file
    assert info.provider_data == OPENAI_FILE


def test_openai_purpose_extension_override() -> None:
    transport = FakeTransport([wire(OPENAI_FILE)])
    lm = OpenAILM(api_key="k", transport=transport)
    lm.file_upload(FileUploadRequest(
        filename="a.png", bytes_data=b"x", media_type="image/png",
        extensions={"purpose": "vision"},
    ))
    assert b"vision" in transport.requests[0].body


def test_openai_readiness_folds() -> None:
    lm = OpenAILM(api_key="k", transport=FakeTransport([]))
    assert lm._file_info({**OPENAI_FILE, "status": "processed"}).readiness == "ready"
    assert lm._file_info({**OPENAI_FILE, "status": "uploaded"}).readiness == "pending"
    assert lm._file_info({**OPENAI_FILE, "status": "pending"}).readiness == "pending"  # Azure OpenAI v1
    assert lm._file_info({**OPENAI_FILE, "status": "error"}).readiness == "failed"
    assert lm._file_info({**OPENAI_FILE, "status": "failed"}).readiness == "failed"
    assert lm._file_info({**OPENAI_FILE, "status": "unknown-word"}).readiness == "ready"
    assert lm._file_info({k: v for k, v in OPENAI_FILE.items() if k != "status"}).readiness == "ready"


def test_openai_shaped_readiness_fold_table() -> None:
    # spec/vocabularies.md FileReadiness, ratified 2026-09-06 (D6): one
    # table shared by every OpenAI-shaped file object (openai, azure, meta).
    from lm15.providers.common import openai_file_readiness

    assert openai_file_readiness("uploaded") == "pending"
    assert openai_file_readiness("pending") == "pending"
    assert openai_file_readiness("error") == "failed"
    assert openai_file_readiness("failed") == "failed"
    assert openai_file_readiness("processed") == "ready"
    assert openai_file_readiness(None) == "ready"
    assert openai_file_readiness("") == "ready"
    assert openai_file_readiness("something-new") == "ready"
    assert openai_file_readiness(7) == "ready"


def test_openai_waits_through_azure_pending_state() -> None:
    transport = FakeTransport([
        wire({**OPENAI_FILE, "status": "pending"}),
        wire({**OPENAI_FILE, "status": "processed"}),
    ])
    lm = OpenAILM(api_key="k", transport=transport)
    info = lm.file_wait_ready(OPENAI_FILE["id"], poll_every=0, timeout=1)
    assert info.readiness == "ready"
    assert len(transport.requests) == 2


def test_openai_expiring_file() -> None:
    lm = OpenAILM(api_key="k", transport=FakeTransport([]))
    # Live fact: batch-purpose files carry an epoch expires_at.
    info = lm._file_info({**OPENAI_FILE, "expires_at": 1790796140})
    assert info.expires_at == "2026-09-30T19:22:20Z"


def test_openai_list_cursor_is_last_id() -> None:
    page1 = {
        "object": "list",
        "data": [OPENAI_FILE, {**OPENAI_FILE, "id": "file-second"}],
        "has_more": True,
        "first_id": OPENAI_FILE["id"], "last_id": "file-second",
    }
    page2 = {"object": "list", "data": [{**OPENAI_FILE, "id": "file-third"}], "has_more": False}
    transport = FakeTransport([wire(page1), wire(page2)])
    lm = OpenAILM(api_key="k", transport=transport)
    page = lm.file_list(limit=2)
    assert [f.id for f in page.items] == ["file-DHLYq8rMTixWCBz1GF8auM", "file-second"]
    assert page.next_cursor == "file-second"
    page = lm.file_list(limit=2, cursor=page.next_cursor)
    assert "after=file-second" in transport.requests[1].url
    assert page.next_cursor is None


def test_openai_get_delete_download() -> None:
    transport = FakeTransport([wire(OPENAI_FILE), wire({"object": "file", "deleted": True, "id": "f"}), wire(b"raw bytes")])
    lm = OpenAILM(api_key="k", transport=transport)
    info = lm.file_get("file-DHLYq8rMTixWCBz1GF8auM")
    assert info.filename == "sample.txt"
    assert lm.file_delete("file-DHLYq8rMTixWCBz1GF8auM") is None
    assert lm.file_download("file-abc") == b"raw bytes"
    urls = [r.url for r in transport.requests]
    assert urls[0].endswith("/files/file-DHLYq8rMTixWCBz1GF8auM")
    assert transport.requests[1].method == "DELETE"
    assert urls[2].endswith("/files/file-abc/content")


def test_openai_download_refusal_maps_typed() -> None:
    # Live capture: 400 "Not allowed to download files of purpose: user_data".
    error = {"error": {"message": "Not allowed to download files of purpose: user_data",
                       "type": "invalid_request_error", "param": None, "code": None}}
    lm = OpenAILM(api_key="k", transport=FakeTransport([wire(error, status=400)]))
    with pytest.raises(InvalidRequestError, match="Not allowed to download"):
        lm.file_download("file-DHLYq8rMTixWCBz1GF8auM")


# ─── Anthropic ──────────────────────────────────────────────────────

def test_anthropic_upload_is_multipart_ga() -> None:
    transport = FakeTransport([wire(ANTHROPIC_FILE)])
    lm = AnthropicLM(api_key="k", transport=transport)
    info = lm.file_upload(FileUploadRequest(filename="sample.txt", bytes_data=b"x" * 76, media_type="text/plain"))
    req = transport.requests[0]
    headers = dict((k.lower(), v) for k, v in req.headers)
    assert headers["content-type"].startswith("multipart/form-data")
    assert "anthropic-beta" not in headers  # GA: no beta header (verified live)
    assert b'name="file"' in req.body and b'filename="sample.txt"' in req.body
    assert info.id == "file_01BMcX8RJhdSVQw4ns12MxHF"
    assert info.media_type == "text/plain"
    assert info.size_bytes == 76
    assert info.created_at == "2026-08-31T22:39:04Z"
    assert info.expires_at is None
    assert info.readiness == "ready"
    assert info.downloadable is False  # verbatim from the wire


def test_anthropic_list_next_page_token() -> None:
    token = "page_eyJhIjoiNmI2OWJhMGYifQ"
    transport = FakeTransport([
        wire({"data": [ANTHROPIC_FILE], "next_page": token}),
        wire({"data": [], "next_page": None}),
    ])
    lm = AnthropicLM(api_key="k", transport=transport)
    page = lm.file_list(limit=1)
    assert page.next_cursor == token
    page = lm.file_list(limit=1, cursor=page.next_cursor)
    assert f"page={token}" in transport.requests[1].url
    assert page.items == ()
    assert page.next_cursor is None


def test_anthropic_download_refusal_maps_typed() -> None:
    # Live capture: 400 file_not_downloadable for user uploads.
    error = {"type": "error", "error": {"type": "invalid_request_error",
             "message": "File `file_x` is not downloadable. Only files generated by a tool "
                        "(for example, the code execution tool) can be downloaded.",
             "details": {"error_code": "file_not_downloadable"}}}
    lm = AnthropicLM(api_key="k", transport=FakeTransport([wire(error, status=400)]))
    with pytest.raises(InvalidRequestError, match="not downloadable"):
        lm.file_download("file_x")


def test_anthropic_delete_returns_none() -> None:
    transport = FakeTransport([wire({"id": "file_x", "type": "file_deleted"})])
    lm = AnthropicLM(api_key="k", transport=transport)
    assert lm.file_delete("file_x") is None
    assert transport.requests[0].method == "DELETE"


# ─── Gemini ─────────────────────────────────────────────────────────

def test_gemini_upload_multipart_carries_display_name() -> None:
    transport = FakeTransport([wire({"file": GEMINI_FILE})])
    lm = GeminiLM(api_key="k", transport=transport)
    info = lm.file_upload(FileUploadRequest(filename="g3.txt", bytes_data=b"x", media_type="text/plain"))
    req = transport.requests[0]
    headers = dict((k.lower(), v) for k, v in req.headers)
    assert "/upload/v1beta/files" in req.url
    assert headers["x-goog-upload-protocol"] == "multipart"
    assert headers["content-type"].startswith("multipart/related")
    assert b'"display_name": "g3.txt"' in req.body or b'"display_name":"g3.txt"' in req.body
    # The canonical id is the URI — what the frozen chat mapping places
    # into fileData.fileUri when a Part carries file_id.
    assert info.id == "https://generativelanguage.googleapis.com/v1beta/files/n9i36fxppl2j"
    assert info.filename == "g3.txt"
    assert info.size_bytes == 76  # wire int64-as-string, normalized
    assert info.expires_at == "2026-09-02T22:39:04Z"
    assert info.readiness == "ready"
    assert info.downloadable is False  # source UPLOADED: server refuses download


def test_gemini_file_id_round_trips_into_chat_wire() -> None:
    # The loop that matters: upload → id → Part.file_id → fileData.fileUri.
    from lm15.types import DocumentPart, Message, Request

    transport = FakeTransport([wire({"file": GEMINI_FILE})])
    lm = GeminiLM(api_key="k", transport=transport)
    info = lm.file_upload(FileUploadRequest(filename="g3.txt", bytes_data=b"x", media_type="text/plain"))
    request = Request(model="gemini-2.5-flash", messages=(
        Message(role="user", parts=(DocumentPart(media_type="text/plain", file_id=info.id),)),
    ))
    payload = lm._payload(request)
    part = payload["contents"][0]["parts"][0]
    assert part["fileData"]["fileUri"] == info.id


def test_gemini_resource_derivation() -> None:
    assert GeminiLM._file_resource("https://generativelanguage.googleapis.com/v1beta/files/abc") == "files/abc"
    assert GeminiLM._file_resource("files/abc") == "files/abc"
    assert GeminiLM._file_resource("abc") == "files/abc"


def test_gemini_get_unwrapped_and_states() -> None:
    transport = FakeTransport([wire(GEMINI_FILE)])  # get returns the bare object
    lm = GeminiLM(api_key="k", transport=transport)
    info = lm.file_get("https://generativelanguage.googleapis.com/v1beta/files/n9i36fxppl2j")
    assert transport.requests[0].url.endswith("/v1beta/files/n9i36fxppl2j")
    assert info.readiness == "ready"
    assert lm._file_info({**GEMINI_FILE, "state": "PROCESSING"}).readiness == "pending"
    assert lm._file_info({**GEMINI_FILE, "state": "FAILED"}).readiness == "failed"
    # Wire vocabulary drift precedent (BATCH_STATE_* vs JOB_STATE_*):
    # match on the suffix, never the documented prefix.
    assert lm._file_info({**GEMINI_FILE, "state": "FILE_STATE_PROCESSING"}).readiness == "pending"


def test_gemini_wait_ready_polls_until_active() -> None:
    transport = FakeTransport([
        wire({**GEMINI_FILE, "state": "PROCESSING"}),
        wire({**GEMINI_FILE, "state": "ACTIVE"}),
    ])
    lm = GeminiLM(api_key="k", transport=transport)
    info = lm.file_wait_ready(GEMINI_FILE["uri"], poll_every=0.0)
    assert info.readiness == "ready"
    assert len(transport.requests) == 2


def test_gemini_wait_ready_returns_failed_snapshot() -> None:
    # Mirrors BatchJob.wait: terminal means return, the caller inspects.
    transport = FakeTransport([wire({**GEMINI_FILE, "state": "FAILED"})])
    lm = GeminiLM(api_key="k", transport=transport)
    assert lm.file_wait_ready(GEMINI_FILE["uri"]).readiness == "failed"


def test_gemini_wait_ready_timeout() -> None:
    transport = FakeTransport([wire({**GEMINI_FILE, "state": "PROCESSING"}) for _ in range(9)])
    lm = GeminiLM(api_key="k", transport=transport)
    with pytest.raises(TimeoutError, match="still pending"):
        lm.file_wait_ready(GEMINI_FILE["uri"], poll_every=0.0, timeout=0.0)


def test_gemini_list_page_token() -> None:
    transport = FakeTransport([
        wire({"files": [GEMINI_FILE], "nextPageToken": "tok1"}),
        wire({}),  # an empty listing is a bare {}
    ])
    lm = GeminiLM(api_key="k", transport=transport)
    page = lm.file_list(limit=1)
    assert page.next_cursor == "tok1"
    page = lm.file_list(limit=1, cursor="tok1")
    assert "pageToken=tok1" in transport.requests[1].url
    assert page.items == () and page.next_cursor is None


def test_gemini_download_endpoint_shape() -> None:
    transport = FakeTransport([wire(b"generated bytes")])
    lm = GeminiLM(api_key="k", transport=transport)
    assert lm.file_download("files/gen1") == b"generated bytes"
    assert transport.requests[0].url.endswith("/files/gen1:download?alt=media")


# ─── Unsupported surfaces stay honest ────────────────────────────────

def test_openai_chat_files_unsupported() -> None:
    from lm15.providers.openai_chat import OpenAIChatLM

    lm = OpenAIChatLM(api_key="k", transport=FakeTransport([]))
    for call in (
        lambda: lm.file_upload(FileUploadRequest(filename="a", bytes_data=b"x")),
        lambda: lm.file_get("f"),
        lambda: lm.file_list(),
        lambda: lm.file_delete("f"),
        lambda: lm.file_download("f"),
    ):
        with pytest.raises(UnsupportedFeatureError, match="files not supported"):
            call()


def test_subscription_adapters_block_files() -> None:
    # The dialect implements files; the access policy does not carry them
    # (a subscription token has no files API). The base drivers gate on the
    # bound policy, so every verb raises before any hook or network.
    from lm15.providers.claude_code import ClaudeCodeLM
    from lm15.providers.openai_codex import OpenAICodexLM

    for lm in (
        ClaudeCodeLM(api_key="tok", transport=FakeTransport([])),
        OpenAICodexLM(api_key="tok", account_id="acct", transport=FakeTransport([])),
    ):
        assert lm.supports.files is False
        for call in (
            lambda: lm.file_upload(FileUploadRequest(filename="a", bytes_data=b"x")),
            lambda: lm.file_get("f"),
            lambda: lm.file_list(),
            lambda: lm.file_delete("f"),
            lambda: lm.file_download("f"),
        ):
            with pytest.raises(UnsupportedFeatureError, match="files not supported"):
                call()


# ─── Async twin ─────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_async_file_lifecycle() -> None:
    from lm15.providers.async_base import AsyncAnthropicLM
    from lm15.transports import AsyncTransportResponse

    class ScriptedAsyncTransport:
        """Replays scripted bodies over the real AsyncTransportResponse shape."""

        def __init__(self, responses):
            self.responses = list(responses)
            self.requests = []

        def stream(self, request):
            self.requests.append(request)
            fake = self.responses.pop(0)

            async def chunks():
                yield fake.body

            async def release(body_consumed: bool) -> None:
                return None

            return AsyncTransportResponse(
                status=fake.status, reason="OK",
                headers=[("content-type", "application/json")],
                http_version="HTTP/1.1", chunks=chunks(), release=release,
            )

    transport = ScriptedAsyncTransport([
        wire(ANTHROPIC_FILE),
        wire(ANTHROPIC_FILE),
        wire({"data": [ANTHROPIC_FILE], "next_page": None}),
        wire({"id": "f", "type": "file_deleted"}),
    ])
    lm = AsyncAnthropicLM(api_key="k", transport=transport)
    info = await lm.file_upload(FileUploadRequest(filename="sample.txt", bytes_data=b"x", media_type="text/plain"))
    assert info.id == ANTHROPIC_FILE["id"]
    info = await lm.file_wait_ready(info.id)
    assert info.ready
    page = await lm.file_list()
    assert page.items[0].media_type == "text/plain"
    assert await lm.file_delete(info.id) is None


# ─── Serde round trips ──────────────────────────────────────────────

def test_serde_upload_request_bytes_roundtrip() -> None:
    r = FileUploadRequest(filename="notes.txt", bytes_data=b"hello", media_type="text/plain")
    d = file_upload_request_to_dict(r)
    assert d["bytes_data"] == "aGVsbG8="  # base64, the media-part precedent
    back = file_upload_request_from_dict(d)
    assert back == r


def test_serde_upload_request_path_roundtrip() -> None:
    r = FileUploadRequest(filename="clip.mp4", path="/data/clip.mp4", media_type="video/mp4")
    d = file_upload_request_to_dict(r)
    assert d["path"] == "/data/clip.mp4" and "bytes_data" not in d
    assert file_upload_request_from_dict(d) == r


def test_serde_file_info_roundtrip_keeps_false() -> None:
    info = FileInfo(id="f", downloadable=False, readiness="pending")
    d = file_info_to_dict(info)
    assert d["downloadable"] is False  # False is data, not emptiness
    assert file_info_from_dict(d) == info


def test_serde_file_page_roundtrip() -> None:
    page = FilePage(items=(FileInfo(id="f1"),), next_cursor="tok")
    assert file_page_from_dict(file_page_to_dict(page)) == page


def test_a_local_path_serializes_with_forward_slashes_on_every_os():
    """spec/types.md (clarified 2026-09-24): the same request serializes the
    same on Windows, where str(Path("/data/clip.mp4")) is "\\\\data\\\\clip.mp4"."""
    from pathlib import PurePosixPath, PureWindowsPath

    from lm15.serde import _wire_path

    assert _wire_path(PureWindowsPath("/data/clip.mp4")) == "/data/clip.mp4"
    assert _wire_path(PureWindowsPath(r"C:\Users\me\clip.mp4")) == "C:/Users/me/clip.mp4"
    assert _wire_path(PurePosixPath("/data/clip.mp4")) == "/data/clip.mp4"
    assert _wire_path("/data/clip.mp4") == "/data/clip.mp4"
    assert _wire_path(PureWindowsPath(r"\\server\share\clip.mp4")) == "//server/share/clip.mp4"
    # On POSIX a backslash is part of a name, never a separator.
    assert _wire_path(PurePosixPath("odd\\name.mp4")) == "odd\\name.mp4"
