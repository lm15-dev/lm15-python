from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lm15.client import UniversalLM
from lm15.features import EndpointSupport, ProviderManifest
from lm15.middleware import with_cache, with_history
from lm15.types import LMRequest, LMResponse, Message, Part, Usage


class EchoAdapter:
    provider = "echo"
    capabilities = None
    supports = EndpointSupport(complete=True, stream=True)
    manifest = ProviderManifest(provider="echo", supports=supports)

    def __init__(self):
        self.calls = 0

    def complete(self, request: LMRequest) -> LMResponse:
        self.calls += 1
        return LMResponse(
            id="e1",
            model=request.model,
            message=Message(role="assistant", parts=(Part.text_part("ok"),)),
            finish_reason="stop",
            usage=Usage(),
        )

    def stream(self, request: LMRequest):
        yield from ()


class MiddlewareTests(unittest.TestCase):
    def test_history_and_cache(self):
        lm = UniversalLM()
        adapter = EchoAdapter()
        lm.register(adapter)

        hist: list[dict] = []
        cache: dict = {}
        lm.middleware.add(with_cache(cache))
        lm.middleware.add(with_history(hist))

        req = LMRequest(model="echo-model", messages=(Message.user("hi"),))
        _ = lm.complete(req, provider="echo")
        _ = lm.complete(req, provider="echo")

        self.assertEqual(adapter.calls, 1)
        self.assertEqual(len(hist), 1)


if __name__ == "__main__":
    unittest.main()
