# Changelog

## 0.2.0 (2026-04-08)

### Added

- **v2 API surface** — `lm15.complete()`, `lm15.stream()`, `lm15.model()`, `lm15.upload()`.
- **`Model` object** — callable with bound config, conversation history, `submit_tools()`, `with_*` derivation.
- **`Stream` object** — high-level `StreamChunk` events, `.text` iterator, materialized `.response`.
- **Callable tools** — pass Python functions as tools. Schema inferred from type hints/docstring, auto-executed on tool_call.
- **Built-in tool shorthand** — `tools=["web_search"]` expands to `Tool(name="web_search", type="builtin")`.
- **Prompt caching** — `prompt_caching=True` on model objects. Anthropic: automatic `cache_control` breakpoints. Gemini: `CachedContent` creation/reuse. OpenAI: no-op (automatic).
- **Per-part cache hints** — `Part.image(url=..., cache=True)` flows through `Part.metadata` to provider adapters.
- **`Part` convenience constructors** — `Part.image()`, `Part.audio()`, `Part.video()`, `Part.document()`, `Part.thinking()` with `url`/`data`/`file_id` kwargs and optional `cache` hint.
- **`Message.assistant()`** constructor.
- **`LMResponse` convenience properties** — `.text`, `.image`, `.images`, `.audio`, `.tool_calls`, `.thinking`, `.citations`.
- **`PartDelta` dataclass** — typed streaming deltas replacing `dict[str, Any]`.
- **`StreamEvent.delta_text`** property — extracts text from delta events (works with both `PartDelta` and legacy dicts).
- **`MiddlewarePipeline.add()`** method.
- **`Tool.type` defaults to `"function"`** — `name` is now the first positional argument.
- **`prefill=` kwarg** — seeds the assistant response.
- **`reasoning=` kwarg** — `True` or `{"effort": "high", "budget": 10000}`.
- **`output=` kwarg** — hints desired output modality (`"image"`, `"audio"`).
- **Cookbooks v2** — 10 progressive examples covering the new API surface.
- **API v2 spec** — `docs/API_SPEC_V2.md`.
- **Cold-start benchmark** — `benchmarks/cold_start.sh`.
- **Dynamic `__version__`** — reads from package metadata, single source of truth in `pyproject.toml`.

### Changed

- README rewritten around v2 API.
- `Config.stream` field removed (dead weight — streaming controlled by method choice).
- Provider adapters filter `prompt_caching` from `config.provider` passthrough.

### Fixed

- Anthropic adapter now applies `cache_control` on system prompt and conversation prefix when `prompt_caching=True`.

## 0.1.0 (2026-04-07)

Initial release.

- Universal LM contract: `LMRequest` / `LMResponse` / `StreamEvent` / `Part` / `Message`.
- Provider adapters: OpenAI (Responses API), Anthropic, Gemini.
- Transport: `urllib` (default), `pycurl` (optional).
- Middleware: retries, cache, history.
- Plugin discovery via entry points.
- models.dev catalog hydration.
- Completeness test harness.
