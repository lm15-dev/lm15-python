from __future__ import annotations

from dataclasses import dataclass, field

from .types import LMResponse, Message, Part


@dataclass(slots=True)
class Conversation:
    system: str | None = None
    _messages: list[Message] = field(default_factory=list, init=False, repr=False)

    def user(self, content: str | list[str | Part]) -> None:
        if isinstance(content, str):
            self._messages.append(Message.user(content))
            return
        parts = [Part.text_part(item) if isinstance(item, str) else item for item in content]
        self._messages.append(Message(role="user", parts=tuple(parts)))

    def assistant(self, response: LMResponse) -> None:
        self._messages.append(response.message)

    def tool_results(self, results: dict[str, str | Part | list[Part]]) -> None:
        self._messages.append(Message.tool_results(results))

    def prefill(self, text: str) -> None:
        self._messages.append(Message.assistant(text))

    @property
    def messages(self) -> tuple[Message, ...]:
        return tuple(self._messages)

    def clear(self) -> None:
        self._messages.clear()
