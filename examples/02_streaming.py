from __future__ import annotations

import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lm15 import LMRequest, Message, Part, build_default


def main() -> None:
    if os.getenv("LM15_EXAMPLES_SKIP_LIVE") == "1":
        print("SKIP: LM15_EXAMPLES_SKIP_LIVE=1")
        return

    if not os.getenv("OPENAI_API_KEY"):
        print("SKIP: OPENAI_API_KEY not set")
        return

    lm = build_default(use_pycurl=True)
    req = LMRequest(model="gpt-4.1-mini", messages=(Message(role="user", parts=(Part.text_part("Write 'ok' and stop."),)),))
    for event in lm.stream(req, provider="openai"):
        if event.delta_text is not None:
            print(event.delta_text, end="")
    print()


if __name__ == "__main__":
    main()
