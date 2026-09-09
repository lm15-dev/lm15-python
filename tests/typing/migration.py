"""Run with mypy --follow-imports=silent --python-version 3.10 tests/typing/migration.py."""
from lm15 import AsyncLMRouter, AsyncResponseStream, LMRouter, Response, ResponseStream
from lm15.doctor import AuthReport, explain_auth

messages: list[dict[str, str]] = [{"role": "user", "content": "Hello"}]


def sync(router: LMRouter, flag: bool) -> None:
    plain: Response = router.complete_from_openai_chat("gpt-4o-mini", messages)
    explicit: Response = router.complete_from_openai_chat("gpt-4o-mini", messages, stream=False)
    streamed: ResponseStream = router.complete_from_openai_chat("gpt-4o-mini", messages, stream=True)
    dynamic: Response | ResponseStream = router.complete_from_openai_chat("gpt-4o-mini", messages, stream=flag)
    report: AuthReport = explain_auth(router.resolve_openai_chat("gpt-4o-mini"), config=router.config)
    with streamed:
        for text in streamed:
            fragment: str = text
    assembled: Response = streamed.response


async def asynchronous(router: AsyncLMRouter, flag: bool) -> None:
    plain: Response = await router.complete_from_openai_chat("gpt-4o-mini", messages)
    explicit: Response = await router.complete_from_openai_chat("gpt-4o-mini", messages, stream=False)
    streamed: AsyncResponseStream = await router.complete_from_openai_chat("gpt-4o-mini", messages, stream=True)
    dynamic: Response | AsyncResponseStream = await router.complete_from_openai_chat("gpt-4o-mini", messages, stream=flag)
    async with streamed:
        async for text in streamed:
            fragment: str = text
    assembled: Response = await streamed.response()
