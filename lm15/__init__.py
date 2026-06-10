"""lm15 — a provider-neutral, low-level foundation for talking to LLM APIs.

lm15 is deliberately NOT a user-facing convenience layer. It is the dependency
for libraries that want to build their own take on the right Python API for
AI systems: one canonical representation (Request/Response/Message/Part,
stream events, errors), exact serde for it, and adapters that translate it to
and from each provider's wire format — stdlib-only, with its own HTTP
transport. Build your DSL on top; let lm15 handle the providers.

Conformance to the canonical representation is pinned by the lm15-contract
corpus; this package is the reference implementation, not the spec.

Quick tour:

    from lm15 import AnthropicLM, Request, Message

    lm = AnthropicLM(api_key="...")
    response = lm.complete(Request(model="claude-sonnet-4-5",
                                   messages=(Message.user("hello"),)))
    response.text          # canonical accessors
    for event in lm.stream(request): ...   # canonical stream events

Subpackages kept off the top level on purpose: `lm15.transports` (HTTP
plumbing; its Request/TransportError are transport-level, not canonical),
`lm15.live` (realtime sessions; optional `websockets` dependency),
`lm15.vet` (the conformance shim CLI: `python -m lm15.vet`).
"""

from importlib.metadata import PackageNotFoundError, version as _version

try:
    __version__ = _version("lm15")
except PackageNotFoundError:  # running from a source checkout
    __version__ = "0.0.0"

# ── Canonical types ──────────────────────────────────────────────────
from .types import (
    # request/response core
    Request,
    Response,
    Message,
    Usage,
    Config,
    CacheConfig,
    Reasoning,
    ToolChoice,
    ErrorDetail,
    ContinuationState,
    # parts
    TextPart,
    ThinkingPart,
    RefusalPart,
    CitationPart,
    ImagePart,
    AudioPart,
    VideoPart,
    DocumentPart,
    BinaryPart,
    ToolCallPart,
    ToolResultPart,
    # part factory helpers
    text,
    thinking,
    refusal,
    citation,
    image,
    audio,
    video,
    document,
    binary,
    tool_call,
    tool_result,
    # tools
    FunctionTool,
    BuiltinTool,
    ToolCallInfo,
    # streaming
    StreamStartEvent,
    StreamDeltaEvent,
    StreamEndEvent,
    StreamErrorEvent,
    TextDelta,
    ThinkingDelta,
    AudioDelta,
    ImageDelta,
    ToolCallDelta,
    CitationDelta,
    ContinuationDelta,
    # auxiliary endpoints
    EmbeddingRequest,
    EmbeddingResponse,
    FileUploadRequest,
    FileUploadResponse,
    BatchRequest,
    BatchResponse,
    ImageGenerationRequest,
    ImageGenerationResponse,
    AudioGenerationRequest,
    AudioGenerationResponse,
    AudioFormat,
    # live session types
    LiveConfig,
    # vocabulary aliases + constants
    Role,
    PartType,
    FinishReason,
    ReasoningEffort,
    ErrorCode,
    StreamEventType,
    ROLE_VALUES,
    FINISH_REASONS,
    ERROR_CODES,
)

# ── Canonical JSON serde ─────────────────────────────────────────────
from .serde import (
    part_to_dict,
    part_from_dict,
    message_to_dict,
    message_from_dict,
    messages_to_json,
    messages_from_json,
    tool_to_dict,
    tool_from_dict,
    tool_choice_to_dict,
    tool_choice_from_dict,
    reasoning_to_dict,
    reasoning_from_dict,
    config_to_dict,
    config_from_dict,
    usage_to_dict,
    usage_from_dict,
    error_detail_to_dict,
    error_detail_from_dict,
    delta_to_dict,
    delta_from_dict,
    stream_event_to_dict,
    stream_event_from_dict,
    request_to_dict,
    request_from_dict,
    response_to_dict,
    response_from_dict,
    model_info_to_dict,
    model_info_from_dict,
)

# ── Errors ───────────────────────────────────────────────────────────
from .errors import (
    LM15Error,
    TransportError,
    ConfigurationError,
    CapabilityError,
    ProviderError,
    AuthError,
    BillingError,
    RateLimitError,
    InvalidRequestError,
    ContextLengthError,
    TimeoutError,
    ServerError,
    UnsupportedModelError,
    UnsupportedFeatureError,
    NotConfiguredError,
    map_http_error,
    canonical_error_code,
    error_class_for_code,
)

# ── Providers ────────────────────────────────────────────────────────
from .providers import (
    OpenAILM,
    OpenAIChatLM,
    AnthropicLM,
    GeminiLM,
    ClaudeCodeLM,
    OpenAICodexLM,
    BaseProviderLM,
    ProviderLM,
    HttpResponse,
    SyncTransport,
)

# ── Stream assembly ──────────────────────────────────────────────────
from .result import (
    Result,
    AsyncResult,
    StreamChunk,
    materialize_response,
    response_to_events,
)

# ── Profiles, compat, model metadata ─────────────────────────────────
from .profiles import EndpointProfile, ProviderProfile
from .compat import OpenAIChatCompat, OpenAIResponsesCompat
from .models import ModelInfo, ModelRegistry
from .protocols import Capabilities, LiveSession
from .features import EndpointSupport, ProviderManifest
from .sse import SSEEvent, parse_sse

__all__ = [
    "__version__",
    # core
    "Request", "Response", "Message", "Usage", "Config", "CacheConfig",
    "Reasoning", "ToolChoice", "ErrorDetail", "ContinuationState",
    # parts
    "TextPart", "ThinkingPart", "RefusalPart", "CitationPart", "ImagePart",
    "AudioPart", "VideoPart", "DocumentPart", "BinaryPart", "ToolCallPart",
    "ToolResultPart",
    # part factories
    "text", "thinking", "refusal", "citation", "image", "audio", "video",
    "document", "binary", "tool_call", "tool_result",
    # tools
    "FunctionTool", "BuiltinTool", "ToolCallInfo",
    # streaming
    "StreamStartEvent", "StreamDeltaEvent", "StreamEndEvent",
    "StreamErrorEvent", "TextDelta", "ThinkingDelta", "AudioDelta",
    "ImageDelta", "ToolCallDelta", "CitationDelta", "ContinuationDelta",
    # auxiliary endpoints
    "EmbeddingRequest", "EmbeddingResponse", "FileUploadRequest",
    "FileUploadResponse", "BatchRequest", "BatchResponse",
    "ImageGenerationRequest", "ImageGenerationResponse",
    "AudioGenerationRequest", "AudioGenerationResponse", "AudioFormat",
    "LiveConfig",
    # vocabularies
    "Role", "PartType", "FinishReason", "ReasoningEffort", "ErrorCode",
    "StreamEventType", "ROLE_VALUES", "FINISH_REASONS", "ERROR_CODES",
    # serde
    "part_to_dict", "part_from_dict", "message_to_dict", "message_from_dict",
    "messages_to_json", "messages_from_json", "tool_to_dict", "tool_from_dict",
    "tool_choice_to_dict", "tool_choice_from_dict", "reasoning_to_dict",
    "reasoning_from_dict", "config_to_dict", "config_from_dict",
    "usage_to_dict", "usage_from_dict", "error_detail_to_dict",
    "error_detail_from_dict", "delta_to_dict", "delta_from_dict",
    "stream_event_to_dict", "stream_event_from_dict", "request_to_dict",
    "request_from_dict", "response_to_dict", "response_from_dict",
    "model_info_to_dict", "model_info_from_dict",
    # errors
    "LM15Error", "TransportError", "ConfigurationError", "CapabilityError",
    "ProviderError", "AuthError", "BillingError", "RateLimitError",
    "InvalidRequestError", "ContextLengthError", "TimeoutError",
    "ServerError", "UnsupportedModelError", "UnsupportedFeatureError",
    "NotConfiguredError", "map_http_error", "canonical_error_code",
    "error_class_for_code",
    # providers
    "OpenAILM", "OpenAIChatLM", "AnthropicLM", "GeminiLM", "ClaudeCodeLM", "OpenAICodexLM",
    "BaseProviderLM", "ProviderLM", "HttpResponse", "SyncTransport",
    # stream assembly
    "Result", "AsyncResult", "StreamChunk", "materialize_response",
    "response_to_events",
    # profiles/compat/models
    "EndpointProfile", "ProviderProfile", "OpenAIChatCompat",
    "OpenAIResponsesCompat", "ModelInfo", "ModelRegistry", "Capabilities",
    "LiveSession", "EndpointSupport", "ProviderManifest",
    # sse
    "SSEEvent", "parse_sse",
]
