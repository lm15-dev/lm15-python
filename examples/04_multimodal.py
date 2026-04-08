from __future__ import annotations

import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lm15 import DataSource, LMRequest, Message, Part, build_default


def main() -> None:
    if os.getenv("LM15_EXAMPLES_SKIP_LIVE") == "1":
        print("SKIP: LM15_EXAMPLES_SKIP_LIVE=1")
        return

    if not (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")):
        print("SKIP: GEMINI_API_KEY/GOOGLE_API_KEY not set")
        return

    lm = build_default(use_pycurl=True)
    image = Part(type="image", source=DataSource(type="url", url="https://example.com/cat.jpg", media_type="image/jpeg", detail="low"))
    req = LMRequest(
        model="gemini-2.0-flash-lite",
        messages=(Message(role="user", parts=(image, Part.text_part("Describe this image in one sentence."))),),
    )
    resp = lm.complete(req, provider="gemini")
    print(resp.text or "")


if __name__ == "__main__":
    main()
