from __future__ import annotations

import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lm15 import LMRequest, Message, Part, Tool, build_default


def main() -> None:
    if os.getenv("LM15_EXAMPLES_SKIP_LIVE") == "1":
        print("SKIP: LM15_EXAMPLES_SKIP_LIVE=1")
        return

    if not os.getenv("OPENAI_API_KEY"):
        print("SKIP: OPENAI_API_KEY not set")
        return

    lm = build_default(use_pycurl=True)
    tools = (
        Tool(
            name="get_weather",
            description="Get weather by city",
            parameters={"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
        ),
    )
    req = LMRequest(model="gpt-4.1-mini", messages=(Message.user("What's weather in Montreal?"),), tools=tools)
    resp = lm.complete(req, provider="openai")
    tool_calls = [p for p in resp.message.parts if p.type == "tool_call"]
    print(f"tool_calls={len(tool_calls)}")


if __name__ == "__main__":
    main()
