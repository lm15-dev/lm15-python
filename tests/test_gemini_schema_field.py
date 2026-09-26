"""MAP-16: the field a schema reaches Gemini in, on every Gemini surface.

The harness's mapping direction grades generateContent and cached prefixes
through the vet shim; this also covers the Live setup frame, which the
harness cannot build without a transcript.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from lm15 import Config, FunctionTool, Message, Request
from lm15.providers.gemini import gemini_openapi_schema
from lm15.types import LiveConfig
from lm15.vet import adapter_for_provider

VECTORS = Path(__file__).resolve().parents[2] / "lm15-contract" / "mapping" / "gemini-schema-field.json"


def vectors():
    if not VECTORS.exists():
        return [pytest.param(None, marks=pytest.mark.skip(reason="sibling contract mapping vectors unavailable"))]
    return [pytest.param(c, id=c["id"]) for c in json.loads(VECTORS.read_text(encoding="utf-8"))["cases"]]


@pytest.mark.parametrize("vector", vectors())
def test_every_surface_follows_the_rule(vector):
    schema, openapi = vector["schema"], vector["openapi"]
    assert gemini_openapi_schema(schema) is openapi
    lm = adapter_for_provider("gemini", "k")
    tool = FunctionTool(name="f", parameters=schema)
    body = json.loads(lm.build_request(Request(model="gemini-2.5-flash", messages=(Message.user("x"),), tools=(tool,)), stream=False).body)
    decl = body["tools"][0]["functionDeclarations"][0]
    want, other = ("parameters", "parametersJsonSchema") if openapi else ("parametersJsonSchema", "parameters")
    assert other not in decl and json.dumps(decl[want]) == json.dumps(schema)
    live = lm._live_setup_payload(LiveConfig(model="gemini-3.1-flash-live-preview", tools=(tool,)))
    decl = live["setup"]["tools"][0]["functionDeclarations"][0]
    assert other not in decl and json.dumps(decl[want]) == json.dumps(schema)
    fmt = {"type": "json_schema", "schema": schema}
    body = json.loads(lm.build_request(Request(model="gemini-2.5-flash", messages=(Message.user("x"),), config=Config(response_format=fmt)), stream=False).body)
    gen = body["generationConfig"]
    want, other = ("responseSchema", "responseJsonSchema") if openapi else ("responseJsonSchema", "responseSchema")
    assert other not in gen and json.dumps(gen[want]) == json.dumps(schema)
