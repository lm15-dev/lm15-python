from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from lm15.transports.base import HttpRequest, HttpResponse


@dataclass(slots=True)
class ProbeResult:
    status: str  # pass | fail | skip
    details: str


class FakeTransport:
    def __init__(self, payload: dict | None = None, stream_lines: Iterable[bytes] | None = None, status: int = 200):
        self.payload = payload or {}
        self.stream_lines = list(stream_lines or [])
        self.status = status
        self.last_request: HttpRequest | None = None

    def request(self, req: HttpRequest) -> HttpResponse:
        self.last_request = req
        body = json.dumps(self.payload).encode("utf-8")
        return HttpResponse(status=self.status, headers={"content-type": "application/json"}, body=body)

    def stream(self, req: HttpRequest):
        self.last_request = req
        for line in self.stream_lines:
            yield line


def fixture_path(root: Path, name: str) -> Path:
    return root / "tests" / "fixtures" / name


def load_json_fixture(root: Path, name: str) -> dict:
    return json.loads(fixture_path(root, name).read_text())


def portability_fixture_path(root: Path, name: str) -> Path:
    return root / "spec" / "fixtures" / "v1" / name


def load_portability_fixture(root: Path, name: str) -> dict:
    return json.loads(portability_fixture_path(root, name).read_text())


def to_bytes_lines(lines: Iterable[str]) -> list[bytes]:
    return [line.encode("utf-8") for line in lines]
