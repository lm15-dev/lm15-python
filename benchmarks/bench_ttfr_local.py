"""Time-to-first-request against a local loopback HTTP server.

Measures, per run in a fresh Python process:
    process_start → import     (cold-import time)
    import → client             (client/transport construction)
    client → request done       (pure client overhead for one round-trip)
    total                       (everything combined)

The server runs in the same process on an ephemeral port, so the request
path is purely loopback — no DNS, no TLS, no external latency.  This
isolates the library's own CPU cost.

Usage:
    python bench_ttfr_local.py <client>
where <client> is one of: lm15-sync lm15-async httpx-sync httpx-async requests

Results are printed as one JSON line to stdout.
"""

from __future__ import annotations

import http.server
import json
import socketserver
import sys
import threading
import time


# ─── In-process HTTP server ──────────────────────────────────────────


class _Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/plain")
        self.send_header("Content-Length", "2")
        self.end_headers()
        self.wfile.write(b"ok")

    def log_message(self, *args: object) -> None:
        pass


class _TCPServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True


def _start_server() -> tuple[_TCPServer, str]:
    srv = _TCPServer(("127.0.0.1", 0), _Handler)
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}/"


# ─── Per-client benchmark functions ──────────────────────────────────


def _bench_lm15_sync(url: str) -> tuple[float, float, float, float]:
    t0 = time.perf_counter()
    from lm15.transports import TransportRequest, StdlibTransport
    t1 = time.perf_counter()
    tr = StdlibTransport()
    t2 = time.perf_counter()
    with tr.stream(TransportRequest(method="GET", url=url)) as resp:
        body = resp.read()
    t3 = time.perf_counter()
    tr.close()
    assert resp.status == 200 and body == b"ok"
    return t0, t1, t2, t3


def _bench_lm15_async(url: str) -> tuple[float, float, float, float]:
    t0 = time.perf_counter()
    from lm15.transports import TransportRequest, StdlibAsyncTransport
    import asyncio
    t1 = time.perf_counter()

    async def main() -> tuple[float, float, int, bytes]:
        tr = StdlibAsyncTransport()
        ta = time.perf_counter()
        async with tr.stream(TransportRequest(method="GET", url=url)) as resp:
            out = b""
            async for chunk in resp:
                out += chunk
        tb = time.perf_counter()
        await tr.aclose()
        return ta, tb, resp.status, out

    ta, tb, status, body = asyncio.run(main())
    assert status == 200 and body == b"ok"
    return t0, t1, ta, tb


def _bench_httpx_sync(url: str) -> tuple[float, float, float, float]:
    t0 = time.perf_counter()
    import httpx
    t1 = time.perf_counter()
    with httpx.Client() as c:
        t2 = time.perf_counter()
        r = c.get(url)
        t3 = time.perf_counter()
    assert r.status_code == 200 and r.content == b"ok"
    return t0, t1, t2, t3


def _bench_httpx_async(url: str) -> tuple[float, float, float, float]:
    t0 = time.perf_counter()
    import httpx
    import asyncio
    t1 = time.perf_counter()

    async def main() -> tuple[float, float, int, bytes]:
        async with httpx.AsyncClient() as c:
            ta = time.perf_counter()
            r = await c.get(url)
            tb = time.perf_counter()
        return ta, tb, r.status_code, r.content

    ta, tb, status, body = asyncio.run(main())
    assert status == 200 and body == b"ok"
    return t0, t1, ta, tb


def _bench_requests(url: str) -> tuple[float, float, float, float]:
    t0 = time.perf_counter()
    import requests
    t1 = time.perf_counter()
    s = requests.Session()
    t2 = time.perf_counter()
    r = s.get(url)
    t3 = time.perf_counter()
    assert r.status_code == 200 and r.content == b"ok"
    return t0, t1, t2, t3


def _bench_urllib3(url: str) -> tuple[float, float, float, float]:
    t0 = time.perf_counter()
    import urllib3
    t1 = time.perf_counter()
    pm = urllib3.PoolManager()
    t2 = time.perf_counter()
    r = pm.request("GET", url)
    t3 = time.perf_counter()
    assert r.status == 200 and r.data == b"ok"
    return t0, t1, t2, t3


_BENCHES = {
    "lm15-sync": _bench_lm15_sync,
    "lm15-async": _bench_lm15_async,
    "httpx-sync": _bench_httpx_sync,
    "httpx-async": _bench_httpx_async,
    "requests": _bench_requests,
    "urllib3": _bench_urllib3,
}


# ─── Entry point ─────────────────────────────────────────────────────


def main() -> None:
    if len(sys.argv) != 2 or sys.argv[1] not in _BENCHES:
        print(f"usage: {sys.argv[0]} <{ '|'.join(_BENCHES) }>", file=sys.stderr)
        sys.exit(1)

    client = sys.argv[1]
    srv, url = _start_server()
    try:
        t0, t1, t2, t3 = _BENCHES[client](url)
    finally:
        srv.shutdown()

    ms = lambda a, b: (b - a) * 1000.0  # noqa: E731
    print(json.dumps({
        "lib": client,
        "scenario": "local-loopback",
        "import_ms": ms(t0, t1),
        "client_ms": ms(t1, t2),
        "request_ms": ms(t2, t3),
        "total_ms": ms(t0, t3),
    }))


if __name__ == "__main__":
    main()
