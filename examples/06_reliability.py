from __future__ import annotations

import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lm15 import LMRequest, Message, Part, TransportPolicy, build_default, with_cache, with_history, with_retries


def main() -> None:
    if os.getenv("LM15_EXAMPLES_SKIP_LIVE") == "1":
        print("SKIP: LM15_EXAMPLES_SKIP_LIVE=1")
        return

    if not os.getenv("OPENAI_API_KEY"):
        print("SKIP: OPENAI_API_KEY not set")
        return

    policy = TransportPolicy(timeout=30.0, connect_timeout=10.0, read_timeout=30.0, max_retries=1)
    lm = build_default(use_pycurl=True, policy=policy)

    history = []
    cache = {}
    lm.middleware.add(with_cache(cache))
    lm.middleware.add(with_history(history))
    lm.middleware.add(with_retries(max_retries=2))

    req = LMRequest(model="gpt-4.1-mini", messages=(Message(role="user", parts=(Part.text_part("Reply with ok"),)),))
    _ = lm.complete(req, provider="openai")
    _ = lm.complete(req, provider="openai")
    print("history:", len(history), "cache:", len(cache))


if __name__ == "__main__":
    main()
