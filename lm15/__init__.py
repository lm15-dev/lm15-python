__version__ = "0.1.0"

from .capabilities import hydrate_with_specs
from .client import UniversalLM
from .middleware import with_cache, with_history, with_retries
from .model_catalog import build_provider_model_index, fetch_models_dev
from .plugins import discover_provider_entry_points, load_plugins
from .transports.base import TransportPolicy
from .types import (
    AudioFormat,
    AudioGenerationRequest,
    AudioGenerationResponse,
    BatchRequest,
    BatchResponse,
    Config,
    DataSource,
    EmbeddingRequest,
    EmbeddingResponse,
    FileUploadRequest,
    FileUploadResponse,
    ImageGenerationRequest,
    ImageGenerationResponse,
    LMRequest,
    LMResponse,
    LiveClientEvent,
    LiveConfig,
    LiveServerEvent,
    Message,
    Part,
    PartDelta,
    StreamEvent,
    Tool,
    ToolConfig,
    Usage,
)


def build_default(
    use_pycurl: bool = True,
    policy: TransportPolicy | None = None,
    hydrate_models_dev_catalog: bool = False,
    discover_plugins: bool = True,
):
    from .factory import build_default as _build_default

    return _build_default(
        use_pycurl=use_pycurl,
        policy=policy,
        hydrate_models_dev=hydrate_models_dev_catalog,
        discover_plugins=discover_plugins,
    )


__all__ = [
    "__version__",
    "UniversalLM",
    "build_default",
    "TransportPolicy",
    "with_cache",
    "with_history",
    "with_retries",
    "hydrate_with_specs",
    "load_plugins",
    "discover_provider_entry_points",
    "fetch_models_dev",
    "build_provider_model_index",
    "Config",
    "DataSource",
    "LMRequest",
    "LMResponse",
    "Message",
    "Part",
    "PartDelta",
    "StreamEvent",
    "EmbeddingRequest",
    "EmbeddingResponse",
    "FileUploadRequest",
    "FileUploadResponse",
    "BatchRequest",
    "BatchResponse",
    "ImageGenerationRequest",
    "ImageGenerationResponse",
    "AudioGenerationRequest",
    "AudioGenerationResponse",
    "LiveConfig",
    "LiveClientEvent",
    "LiveServerEvent",
    "AudioFormat",
    "Tool",
    "ToolConfig",
    "Usage",
]
