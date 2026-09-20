"""Provider defaults must not override the caller's connection budget."""
import pytest

from lm15 import (
    BatchRequest, Config, FileUploadRequest, ImageGenerationRequest, ImagePart, Message,
    Request, SpeechGenerationRequest, VideoGenerationRequest, judgments, yes_no,
)
from lm15.providers import AnthropicLM, GeminiLM, OpenAILM, OpenAIChatLM, TypeSafeLM, XaiLM
from lm15.transports import StdlibTransport


PROVIDERS = [OpenAILM, OpenAIChatLM, AnthropicLM, GeminiLM, TypeSafeLM, XaiLM]


def inherited(wire):
    assert wire.connect_timeout is None
    assert wire.read_timeout is None
    assert wire.write_timeout is None


@pytest.mark.parametrize("provider", PROVIDERS)
@pytest.mark.parametrize("seconds", [0.2, 1800])
def test_models_inherit_configured_budget(provider, seconds):
    with StdlibTransport(read_timeout=seconds) as transport:
        lm = provider(api_key="k", transport=transport)
        inherited(lm._models_request())
        assert transport._read_timeout == seconds


@pytest.mark.parametrize("seconds", [0.2, 1800])
def test_typesafe_and_token_scoring_inherit_configured_budget(seconds):
    with StdlibTransport(read_timeout=seconds) as transport:
        lm = TypeSafeLM(api_key="k", transport=transport)
        req = Request(model="jev-latest", messages=(Message.user("x"),), config=Config(response_format=judgments(ok=yes_no("OK?"))))
        inherited(lm.build_request(req, stream=False))
        chat = OpenAIChatLM(api_key="k", transport=transport, compat="vllm")
        inherited(chat._judgment_tokenize_request("m", [{"role": "user", "content": "x"}], continue_final=True))
        inherited(chat._judgment_score_request("m", [[1]], [2]))
        assert transport._read_timeout == seconds


@pytest.mark.parametrize("provider", [OpenAILM, AnthropicLM, GeminiLM])
def test_file_endpoints_inherit_budget(provider):
    with StdlibTransport(read_timeout=1800) as transport:
        lm = provider(api_key="k", transport=transport)
        inherited(lm._file_upload_request(FileUploadRequest(filename="a.txt", bytes_data=b"a", media_type="text/plain")))
        inherited(lm._file_get_request("files/f" if provider is GeminiLM else "f"))
        inherited(lm._file_list_request(2, None))
        inherited(lm._file_delete_request("files/f" if provider is GeminiLM else "f"))
        inherited(lm._file_download_request("files/f" if provider is GeminiLM else "f"))


@pytest.mark.parametrize("provider", [OpenAILM, AnthropicLM, GeminiLM])
def test_batch_endpoints_inherit_budget(provider):
    with StdlibTransport(read_timeout=1800) as transport:
        lm = provider(api_key="k", transport=transport)
        model = "claude-haiku-4-5" if provider is AnthropicLM else "gemini-2.5-flash" if provider is GeminiLM else "gpt-4.1"
        batch = BatchRequest(requests=(Request(model=model, messages=(Message.user("x"),)),))
        if provider is OpenAILM:
            inherited(lm._batch_upload_request(batch))
        inherited(lm._batch_submit_request(batch, {"id": "file-1"} if provider is OpenAILM else None))
        inherited(lm._batch_status_request("batches/b" if provider is GeminiLM else "b"))
        inherited(lm._batch_cancel_request("batches/b" if provider is GeminiLM else "b"))
        inherited(lm._batch_list_request(2))
        if provider is OpenAILM:
            for wire in lm._batch_result_fetches({"output_file_id": "out", "error_file_id": "err"}):
                inherited(wire)
        elif provider is AnthropicLM:
            for wire in lm._batch_result_fetches({"results_url": "https://api.anthropic.com/v1/messages/batches/b/results"}):
                inherited(wire)


def test_cache_endpoints_inherit_budget():
    with StdlibTransport(read_timeout=1800) as transport:
        lm = GeminiLM(api_key="k", transport=transport)
        req = Request(model="gemini-2.5-flash", messages=(Message.user("x"),))
        inherited(lm._cache_create_request(req, 60, None))
        inherited(lm._cache_get_request("cachedContents/c"))
        inherited(lm._cache_list_request(2, None))
        inherited(lm._cache_delete_request("cachedContents/c"))
        inherited(lm._cache_update_request("cachedContents/c", 60))


@pytest.mark.parametrize("provider,model", [(OpenAILM, "sora-2"), (GeminiLM, "veo-3.1-lite-generate-preview"), (XaiLM, "grok-imagine-video")])
def test_video_endpoints_inherit_budget(provider, model):
    with StdlibTransport(read_timeout=1800) as transport:
        lm = provider(api_key="k", transport=transport)
        inherited(lm._video_submit_request(VideoGenerationRequest(model=model, prompt="x")))
        inherited(lm._video_status_request("models/m/operations/v" if provider is GeminiLM else "v"))
        if provider is not XaiLM:
            inherited(lm._video_list_request(2, model))
        if provider is OpenAILM:
            inherited(lm._video_result_fetch({"id": "v"}))
        elif provider is GeminiLM:
            inherited(lm._video_result_fetch({"response": {"generateVideoResponse": {"generatedSamples": [{"video": {
                "uri": "https://generativelanguage.googleapis.com/v1beta/files/f:download?alt=media"}}]}}}))


@pytest.mark.parametrize("provider,model", [(OpenAILM, "gpt-image-1-mini"), (GeminiLM, "gemini-2.5-flash-image"), (XaiLM, "grok-imagine-image")])
def test_image_endpoints_inherit_budget(provider, model):
    with StdlibTransport(read_timeout=1800) as transport:
        lm = provider(api_key="k", transport=transport)
        inherited(lm._image_generate_request(ImageGenerationRequest(model=model, prompt="x")))


@pytest.mark.parametrize("provider,model", [(OpenAILM, "gpt-image-1-mini"), (XaiLM, "grok-imagine-image")])
def test_image_edit_endpoints_inherit_budget(provider, model):
    with StdlibTransport(read_timeout=1800) as transport:
        lm = provider(api_key="k", transport=transport)
        inherited(lm._image_generate_request(ImageGenerationRequest(
            model=model, prompt="x", images=(ImagePart(media_type="image/png", data="eA=="),))))


@pytest.mark.parametrize("provider,model", [(OpenAILM, "gpt-4o-mini-tts"), (GeminiLM, "gemini-2.5-flash-preview-tts")])
def test_speech_endpoints_inherit_budget(provider, model):
    with StdlibTransport(read_timeout=1800) as transport:
        lm = provider(api_key="k", transport=transport)
        inherited(lm._speech_generate_request(SpeechGenerationRequest(model=model, prompt="x", voice="alloy" if provider is OpenAILM else "Kore")))
