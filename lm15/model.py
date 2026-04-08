from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Callable

from .capabilities import resolve_provider
from .client import UniversalLM
from .stream import Stream
from .types import Config, FileUploadRequest, LMRequest, LMResponse, Message, Part, Tool


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
        provider: str | None = None,
        retries: int = 0,
        cache: bool | dict = False,
        prompt_caching: bool = False,
        temperature: float | None = None,
        max_tokens: int | None = None,
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
        self._bound_tools = list(tools or [])
        self._conversation: list[Message] = []
        self.history = _History(self._reset_conversation)
        self._local_cache: dict[str, LMResponse] | None = {} if cache is True else (cache if isinstance(cache, dict) else None)
        self._pending_tool_calls: list[Part] = []

    def _reset_conversation(self) -> None:
        self._conversation = []
        self._pending_tool_calls = []

    def with_model(self, name: str) -> "Model":
        return Model(
            lm=self._lm,
            model=name,
            system=self.system,
            tools=self._bound_tools,
            provider=self.provider,
            retries=self.retries,
            cache=self._local_cache if self._local_cache is not None else self.cache,
            prompt_caching=self.prompt_caching,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

    def with_system(self, system: str) -> "Model":
        return Model(
            lm=self._lm,
            model=self.model,
            system=system,
            tools=self._bound_tools,
            provider=self.provider,
            retries=self.retries,
            cache=self._local_cache if self._local_cache is not None else self.cache,
            prompt_caching=self.prompt_caching,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

    def with_tools(self, tools: list[Tool | Callable[..., Any] | str]) -> "Model":
        return Model(
            lm=self._lm,
            model=self.model,
            system=self.system,
            tools=tools,
            provider=self.provider,
            retries=self.retries,
            cache=self._local_cache if self._local_cache is not None else self.cache,
            prompt_caching=self.prompt_caching,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

    def with_provider(self, provider: str, base_url: str | None = None) -> "Model":
        _ = base_url  # reserved for future use
        return Model(
            lm=self._lm,
            model=self.model,
            system=self.system,
            tools=self._bound_tools,
            provider=provider,
            retries=self.retries,
            cache=self._local_cache if self._local_cache is not None else self.cache,
            prompt_caching=self.prompt_caching,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

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

    def __call__(
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
        provider: str | None = None,
    ) -> LMResponse:
        request, callable_registry, turn_messages, update_conversation = self._build_request(
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

        resp = self._complete_with_cache(request, provider=provider or self.provider)
        final_request = request
        max_tool_hops = 8
        hops = 0
        while callable_registry and resp.tool_calls and hops < max_tool_hops:
            hops += 1
            tool_parts: list[Part] = []
            handled = False
            for tc in resp.tool_calls:
                fn = callable_registry.get(tc.name or "")
                if fn is None:
                    continue
                handled = True
                out = self._invoke_tool(fn, tc.input or {})
                if isinstance(out, Part):
                    content = [out]
                elif isinstance(out, list) and all(isinstance(x, Part) for x in out):
                    content = out
                else:
                    content = [Part.text_part(str(out))]
                tool_parts.append(Part.tool_result(tc.id or "", content, name=tc.name))

            if not handled:
                break

            tool_msg = Message(role="tool", parts=tuple(tool_parts))
            follow_messages = final_request.messages + (resp.message, tool_msg)
            final_request = LMRequest(
                model=final_request.model,
                messages=follow_messages,
                system=final_request.system,
                tools=final_request.tools,
                config=final_request.config,
            )
            resp = self._complete_with_cache(final_request, provider=provider or self.provider)

        self._pending_tool_calls = resp.tool_calls
        self.history.append(HistoryEntry(request=final_request, response=resp))

        if update_conversation:
            self._conversation = list(final_request.messages) + [resp.message]

        return resp

    def stream(
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
        provider: str | None = None,
    ) -> Stream:
        request, _, _, update_conversation = self._build_request(
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

        def on_finished(req: LMRequest, resp: LMResponse) -> None:
            self.history.append(HistoryEntry(request=req, response=resp))
            if update_conversation:
                self._conversation = list(req.messages) + [resp.message]
            self._pending_tool_calls = resp.tool_calls

        events = self._lm.stream(request, provider=provider or self.provider)
        return Stream(events=events, request=request, on_finished=on_finished)

    def submit_tools(self, results: dict[str, Any], *, provider: str | None = None) -> LMResponse:
        if not self._pending_tool_calls:
            raise ValueError("no pending tool calls")

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
            raise ValueError("no tool results matched pending calls")

        return self(messages=[Message(role="tool", parts=tuple(parts))], provider=provider)

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
    ) -> tuple[LMRequest, dict[str, Callable[..., Any]], tuple[Message, ...], bool]:
        if prompt is not None and messages is not None:
            raise ValueError("prompt and messages are mutually exclusive")

        turn_messages: tuple[Message, ...]
        update_conversation = False
        if messages is not None:
            turn_messages = tuple(messages)
        else:
            if prompt is None:
                raise ValueError("either prompt or messages is required")
            if isinstance(prompt, str):
                turn_messages = (Message.user(prompt),)
            else:
                parts: list[Part] = []
                for item in prompt:
                    parts.append(Part.text_part(item) if isinstance(item, str) else item)
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
        return request, callable_registry, turn_messages, update_conversation

    def _normalize_tools(self, tools: list[Tool | Callable[..., Any] | str]) -> tuple[tuple[Tool, ...], dict[str, Callable[..., Any]]]:
        out: list[Tool] = []
        registry: dict[str, Callable[..., Any]] = {}
        for t in tools:
            if isinstance(t, Tool):
                out.append(t)
            elif isinstance(t, str):
                out.append(Tool(name=t, type="builtin"))
            elif callable(t):
                tool = callable_to_tool(t)
                out.append(tool)
                registry[tool.name] = t
            else:
                raise TypeError(f"unsupported tool type: {type(t)}")
        return tuple(out), registry

    def _complete_with_cache(self, request: LMRequest, *, provider: str | None) -> LMResponse:
        if self._local_cache is not None:
            key = str((provider, request))
            if key in self._local_cache:
                return self._local_cache[key]
            resp = self._lm.complete(request, provider=provider)
            self._local_cache[key] = resp
            return resp
        return self._lm.complete(request, provider=provider)

    @staticmethod
    def _invoke_tool(fn: Callable[..., Any], payload: dict[str, Any]) -> Any:
        try:
            return fn(**payload)
        except TypeError:
            return fn(payload)
