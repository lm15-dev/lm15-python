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

    if not any(os.getenv(k) for k in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY")):
        print("SKIP: no provider key set")
        return

    lm = build_default(use_pycurl=True)
    req = LMRequest(model="gpt-4.1-mini", messages=(Message(role="user", parts=(Part.text_part("Reply with exactly: ok"),)),))
    resp = lm.complete(req)
    print(resp.text or "")


if __name__ == "__main__":
    main()
