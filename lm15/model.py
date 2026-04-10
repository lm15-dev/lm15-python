from __future__ import annotations

import asyncio
import inspect
from dataclasses import dataclass
from typing import Any, Callable

from .capabilities import resolve_provider
from .client import UniversalLM
from .errors import RateLimitError, ServerError, TimeoutError, TransportError
from .live import AsyncLiveSession
from .result import AsyncResult, Result, response_to_events
from .types import AudioFormat, Config, FileUploadRequest, LMRequest, LMResponse, LiveConfig, Message, Part, Tool


class _Unset:
    pass


UNSET = _Unset()


def _py_type_to_json_schema(t: Any) -> dict[str, Any]:
    origin = getattr(t, "__origin__", None)
    if origin in (list, tuple, set):
        return {"type": "array"}
    if origin is dict:
        return {"type": "object"}
    if t in (str,):
        return {"type": "string"}
    if t in (int,):
        return {"type": "integer"}
    if t in (float,):
        return {"type": "number"}
    if t in (bool,):
        return {"type": "boolean"}
    return {"type": "string"}


def callable_to_tool(fn: Callable[..., Any]) -> Tool:
    sig = inspect.signature(fn)
    hints = inspect.get_annotations(fn, eval_str=True)
    properties: dict[str, Any] = {}
    required: list[str] = []
    for name, param in sig.parameters.items():
        if param.kind not in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY):
            continue
        ann = hints.get(name, str)
        properties[name] = _py_type_to_json_schema(ann)
        if param.default is inspect.Parameter.empty:
            required.append(name)

    schema = {"type": "object", "properties": properties}
    if required:
        schema["required"] = required

    return Tool(
        name=fn.__name__,
        type="function",
        description=(inspect.getdoc(fn) or "").strip() or None,
        parameters=schema,
    )


@dataclass(slots=True, frozen=True)
class HistoryEntry:
    request: LMRequest
    response: LMResponse


class _History(list[HistoryEntry]):
    def __init__(self, on_clear: Callable[[], None]):
        super().__init__()
        self._on_clear = on_clear

    def clear(self) -> None:
        super().clear()
        self._on_clear()


class Model:
    def __init__(
        self,
        *,
        lm: UniversalLM,
        model: str,
        system: str | None = None,
        tools: list[Tool | Callable[..., Any] | str] | None = None,
        on_tool_call: Callable[..., Any] | None = None,
        provider: str | None = None,
        retries: int = 0,
        cache: bool | dict = False,
        prompt_caching: bool = False,
        temperature: float | None = None,
        max_tokens: int | None = None,
        max_tool_rounds: int = 8,
    ) -> None:
        self._lm = lm
        self.model = model
        self.system = system
        self.provider = provider
        self.retries = retries
        self.cache = cache
        self.prompt_caching = prompt_caching
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_tool_rounds = max_tool_rounds
        self.on_tool_call = on_tool_call
        self._bound_tools = list(tools or [])
        self._conversation: list[Message] = []
        self.history = _History(self._reset_conversation)
        self._local_cache: dict[str, LMResponse] | None = {} if cache is True else (dict(cache) if isinstance(cache, dict) else None)
        self._pending_tool_calls: list[Part] = []

    def _reset_conversation(self) -> None:
        self._conversation = []
        self._pending_tool_calls = []

    def copy(
        self,
        *,
        model: str | _Unset = UNSET,
        system: str | None | _Unset = UNSET,
        tools: list[Tool | Callable[..., Any] | str] | _Unset = UNSET,
        on_tool_call: Callable[..., Any] | None | _Unset = UNSET,
        provider: str | None | _Unset = UNSET,
        retries: int | _Unset = UNSET,
        cache: bool | dict | _Unset = UNSET,
        prompt_caching: bool | _Unset = UNSET,
        temperature: float | None | _Unset = UNSET,
        max_tokens: int | None | _Unset = UNSET,
        max_tool_rounds: int | _Unset = UNSET,
        history: bool = True,
    ) -> "Model":
        cache_value: bool | dict
        if cache is UNSET:
            if self._local_cache is not None:
                cache_value = dict(self._local_cache)
            else:
                cache_value = self.cache
        elif isinstance(cache, dict):
            cache_value = dict(cache)
        else:
            cache_value = cache

        out = Model(
            lm=self._lm,
            model=self.model if model is UNSET else model,
            system=self.system if system is UNSET else system,
            tools=list(self._bound_tools) if tools is UNSET else list(tools),
            on_tool_call=self.on_tool_call if on_tool_call is UNSET else on_tool_call,
            provider=self.provider if provider is UNSET else provider,
            retries=self.retries if retries is UNSET else retries,
            cache=cache_value,
            prompt_caching=self.prompt_caching if prompt_caching is UNSET else prompt_caching,
            temperature=self.temperature if temperature is UNSET else temperature,
            max_tokens=self.max_tokens if max_tokens is UNSET else max_tokens,
            max_tool_rounds=self.max_tool_rounds if max_tool_rounds is UNSET else max_tool_rounds,
        )
        if history:
            out._conversation = list(self._conversation)
            out.history.extend(self.history)
            out._pending_tool_calls = list(self._pending_tool_calls)
        return out

    # Compatibility shims
    def with_model(self, name: str) -> "Model":
        return self.copy(model=name)

    def with_system(self, system: str) -> "Model":
        return self.copy(system=system)

    def with_tools(self, tools: list[Tool | Callable[..., Any] | str]) -> "Model":
        return self.copy(tools=tools)

    def with_provider(self, provider: str, base_url: str | None = None) -> "Model":
        _ = base_url
        return self.copy(provider=provider)

    def upload(self, path: str | bytes, *, media_type: str | None = None) -> Part:
        if isinstance(path, bytes):
            data = path
            filename = "file.bin"
        else:
            from pathlib import Path

            p = Path(path)
            data = p.read_bytes()
            filename = p.name
            if media_type is None:
                import mimetypes

                media_type = mimetypes.guess_type(str(p))[0]
        media_type = media_type or "application/octet-stream"
        req = FileUploadRequest(model=self.model, filename=filename, bytes_data=data, media_type=media_type)
        f = self._lm.file_upload(req, provider=self.provider or resolve_provider(self.model))

        if media_type.startswith("image/"):
            return Part.image(file_id=f.id, media_type=media_type)
        if media_type.startswith("audio/"):
            return Part.audio(file_id=f.id, media_type=media_type)
        if media_type.startswith("video/"):
            return Part.video(file_id=f.id, media_type=media_type)
        return Part.document(file_id=f.id, media_type=media_type)

    def prepare(
        self,
        prompt: str | list[str | Part] | None = None,
        *,
        messages: list[Message] | None = None,
        tools: list[Tool | Callable[..., Any] | str] | None = None,
        reasoning: bool | dict[str, Any] | None = None,
        prefill: str | None = None,
        output: str | None = None,
        prompt_caching: bool | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        top_p: float | None = None,
        stop: list[str] | None = None,
    ) -> LMRequest:
        request, _, _ = self._build_request(
            prompt=prompt,
            messages=messages,
            tools=tools,
            reasoning=reasoning,
            prefill=prefill,
            output=output,
            prompt_caching=prompt_caching,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            stop=stop,
        )
        return request

    def call(
        self,
        prompt: str | list[str | Part] | None = None,
        *,
        messages: list[Message] | None = None,
        tools: list[Tool | Callable[..., Any] | str] | None = None,
        on_tool_call: Callable[..., Any] | None = None,
        reasoning: bool | dict[str, Any] | None = None,
        prefill: str | None = None,
        output: str | None = None,
        prompt_caching: bool | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        top_p: float | None = None,
        stop: list[str] | None = None,
        max_tool_rounds: int | None = None,
        provider: str | None = None,
    ) -> Result:
        request, callable_registry, update_conversation = self._build_request(
            prompt=prompt,
            messages=messages,
            tools=tools,
            reasoning=reasoning,
            prefill=prefill,
            output=output,
            prompt_caching=prompt_caching,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            stop=stop,
        )

        resolved_provider = provider or self.provider

        def start_stream(req: LMRequest):
            cached = self._cache_lookup(req, resolved_provider)
            if cached is not None:
                return response_to_events(cached, req)
            return self._lm.stream(req, provider=resolved_provider)

        def on_finished(final_request: LMRequest, resp: LMResponse) -> None:
            self._cache_store(final_request, resp, resolved_provider)
            self.history.append(HistoryEntry(request=final_request, response=resp))
            self._pending_tool_calls = resp.tool_calls
            if update_conversation:
                self._conversation = list(final_request.messages) + [resp.message]

        return Result(
            request=request,
            start_stream=start_stream,
            on_finished=on_finished,
            callable_registry=callable_registry,
            on_tool_call=on_tool_call if on_tool_call is not None else self.on_tool_call,
            max_tool_rounds=max_tool_rounds if max_tool_rounds is not None else self.max_tool_rounds,
            retries=self.retries,
        )

    def __call__(self, prompt: str | list[str | Part] | None = None, **kwargs) -> Result:
        return self.call(prompt, **kwargs)

    def stream(self, prompt: str | list[str | Part] | None = None, **kwargs) -> Result:
        return self.call(prompt, **kwargs)

    def acall(self, prompt: str | list[str | Part] | None = None, **kwargs) -> AsyncResult:
        return AsyncResult(self.call, prompt, **kwargs)

    def live(
        self,
        *,
        tools: list[Tool | Callable[..., Any] | str] | None = None,
        on_tool_call: Callable[..., Any] | None = None,
        voice: str | None = None,
        input_format: AudioFormat | None = None,
        output_format: AudioFormat | None = None,
        provider: str | None = None,
    ):
        resolved_provider = provider or self.provider
        tool_defs, _ = self._normalize_tools(tools if tools is not None else self._bound_tools)

        config = LiveConfig(
            model=self.model,
            system=self.system,
            tools=tool_defs,
            voice=voice,
            input_format=input_format,
            output_format=output_format,
        )
        session = self._lm.live(config, provider=resolved_provider)
        callback = on_tool_call if on_tool_call is not None else self.on_tool_call
        if hasattr(session, "set_on_tool_call"):
            session.set_on_tool_call(callback)
        return session

    async def alive(self, **kwargs) -> AsyncLiveSession:
        session = await asyncio.to_thread(self.live, **kwargs)
        return AsyncLiveSession(session)

    def submit_tools(self, results: dict[str, Any], *, provider: str | None = None) -> Result:
        if not self._pending_tool_calls:
            raise ValueError(
                "no pending tool calls\n\n"
                "  submit_tools() continues a tool-call conversation.\n"
                "  It requires a prior call that returned finish_reason='tool_call'.\n\n"
                "  Common causes:\n"
                "    - The previous call wasn't consumed yet (access resp.finish_reason, resp.text, or resp.response first)\n"
                "    - The previous call didn't include tools=\n"
                "    - The model chose to answer directly instead of calling a tool\n"
                "    - history.clear() was called between the tool call and submit_tools()\n"
            )

        parts: list[Part] = []
        for tc in self._pending_tool_calls:
            if not tc.id or tc.id not in results:
                continue
            out = results[tc.id]
            if isinstance(out, Part):
                content = [out]
            elif isinstance(out, list) and all(isinstance(x, Part) for x in out):
                content = out
            else:
                content = [Part.text_part(str(out))]
            parts.append(Part.tool_result(tc.id, content, name=tc.name))

        if not parts:
            pending_ids = [tc.id for tc in self._pending_tool_calls]
            provided_ids = list(results.keys())
            raise ValueError(
                "no tool results matched pending calls\n\n"
                f"  Pending tool call IDs: {pending_ids}\n"
                f"  Provided result IDs:   {provided_ids}\n\n"
                "  Each key in the results dict must match a tool call ID.\n"
                "  Use: results = {tc.id: result for tc in resp.tool_calls}\n"
            )

        tool_message = Message(role="tool", parts=tuple(parts))
        follow_messages = tuple(self._conversation) + (tool_message,)
        request, callable_registry, _ = self._build_request(
            prompt=None,
            messages=list(follow_messages),
            tools=None,
            reasoning=None,
            prefill=None,
            output=None,
            prompt_caching=None,
            temperature=None,
            max_tokens=None,
            top_p=None,
            stop=None,
        )

        resolved_provider = provider or self.provider

        def start_stream(req: LMRequest):
            cached = self._cache_lookup(req, resolved_provider)
            if cached is not None:
                return response_to_events(cached, req)
            return self._lm.stream(req, provider=resolved_provider)

        def on_finished(final_request: LMRequest, resp: LMResponse) -> None:
            self._cache_store(final_request, resp, resolved_provider)
            self.history.append(HistoryEntry(request=final_request, response=resp))
            self._pending_tool_calls = resp.tool_calls
            self._conversation = list(final_request.messages) + [resp.message]

        return Result(
            request=request,
            start_stream=start_stream,
            on_finished=on_finished,
            callable_registry=callable_registry,
            on_tool_call=self.on_tool_call,
            max_tool_rounds=self.max_tool_rounds,
            retries=self.retries,
        )

    def _build_request(
        self,
        *,
        prompt: str | list[str | Part] | None,
        messages: list[Message] | None,
        tools: list[Tool | Callable[..., Any] | str] | None,
        reasoning: bool | dict[str, Any] | None,
        prefill: str | None,
        output: str | None,
        prompt_caching: bool | None,
        temperature: float | None,
        max_tokens: int | None,
        top_p: float | None,
        stop: list[str] | None,
    ) -> tuple[LMRequest, dict[str, Callable[..., Any]], bool]:
        if prompt is not None and messages is not None:
            raise ValueError(
                "prompt and messages are mutually exclusive\n\n"
                "  Use prompt= for a single question:\n"
                "    lm15.call(model, 'What is TCP?')\n\n"
                "  Use messages= for a conversation history:\n"
                "    lm15.call(model, messages=[Message.user('Hi'), ...])\n"
            )

        update_conversation = False
        if messages is not None:
            turn_messages = tuple(messages)
        else:
            if prompt is None:
                raise ValueError(
                    "either prompt or messages is required\n\n"
                    "  Provide a prompt string:\n"
                    "    model.call('What is TCP?')\n\n"
                    "  Or a messages list:\n"
                    "    model.call(messages=[Message.user('What is TCP?')])\n"
                )
            if isinstance(prompt, str):
                turn_messages = (Message.user(prompt),)
            else:
                parts = [Part.text_part(item) if isinstance(item, str) else item for item in prompt]
                turn_messages = (Message(role="user", parts=tuple(parts)),)
            if prefill:
                turn_messages = turn_messages + (Message.assistant(prefill),)
            update_conversation = True

        final_messages = tuple(self._conversation) + turn_messages if update_conversation else turn_messages
        tool_defs, callable_registry = self._normalize_tools(tools if tools is not None else self._bound_tools)

        provider_cfg: dict[str, Any] = {}
        if self.prompt_caching if prompt_caching is None else prompt_caching:
            provider_cfg["prompt_caching"] = True
        if output == "image":
            provider_cfg["output"] = "image"
        elif output == "audio":
            provider_cfg["output"] = "audio"

        reasoning_cfg: dict[str, Any] | None = None
        if reasoning is True:
            reasoning_cfg = {"enabled": True}
        elif isinstance(reasoning, dict):
            reasoning_cfg = {"enabled": True, **reasoning}

        cfg = Config(
            max_tokens=max_tokens if max_tokens is not None else self.max_tokens,
            temperature=temperature if temperature is not None else self.temperature,
            top_p=top_p,
            stop=tuple(stop or ()),
            reasoning=reasoning_cfg,
            provider=provider_cfg or None,
        )

        request = LMRequest(
            model=self.model,
            messages=final_messages,
            system=self.system,
            tools=tool_defs,
            config=cfg,
        )
        return request, callable_registry, update_conversation

    def _normalize_tools(self, tools: list[Tool | Callable[..., Any] | str]) -> tuple[tuple[Tool, ...], dict[str, Callable[..., Any]]]:
        out: list[Tool] = []
        registry: dict[str, Callable[..., Any]] = {}
        for t in tools:
            if isinstance(t, Tool):
                out.append(t)
                if t.fn is not None and callable(t.fn):
                    registry[t.name] = t.fn
            elif isinstance(t, str):
                out.append(Tool(name=t, type="builtin"))
            elif callable(t):
                tool = callable_to_tool(t)
                out.append(tool)
                registry[tool.name] = t
            else:
                raise TypeError(f"unsupported tool type: {type(t)}")
        return tuple(out), registry

    def _cache_lookup(self, request: LMRequest, provider: str | None) -> LMResponse | None:
        if self._local_cache is None:
            return None
        return self._local_cache.get(self._cache_key(request, provider))

    def _cache_store(self, request: LMRequest, response: LMResponse, provider: str | None) -> None:
        if self._local_cache is None:
            return
        self._local_cache[self._cache_key(request, provider)] = response

    def _cache_key(self, request: LMRequest, provider: str | None) -> str:
        return str((provider or resolve_provider(request.model), request))

    @staticmethod
    def _is_retryable_error(exc: Exception) -> bool:
        return isinstance(exc, (RateLimitError, TimeoutError, ServerError, TransportError))
