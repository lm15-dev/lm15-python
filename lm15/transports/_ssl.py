"""SSL context factory.

We rely on the stdlib `ssl` module's `create_default_context`, which on
Python 3.10+ loads the system trust store correctly on Linux/macOS/Windows.
No certifi bundle is shipped — set SSL_CERT_FILE if your system store is
broken, or pass an explicit ca_bundle= to the transport.

Under Pyodide the `ssl` module is absent (there is no socket to wrap);
importing lm15 must still work there — the fetch transport carries the
wire — so the import is optional and the socket transports refuse by
name at connect time, not at import time.
"""
from __future__ import annotations

try:
    import ssl
except ImportError:  # pragma: no cover - Pyodide
    ssl = None  # type: ignore[assignment]

if ssl is not None:
    SSLError = ssl.SSLError
else:  # pragma: no cover - Pyodide

    class SSLError(OSError):
        """Never raised where there is no ssl; keeps the socket transports' except clauses valid."""


def make_ssl_context(
    *, verify: bool = True, ca_bundle: str | None = None
) -> "ssl.SSLContext":
    if ssl is None:  # pragma: no cover - Pyodide
        from ._exceptions import ConnectError

        raise ConnectError(
            "this Python has no ssl module (Pyodide?), so the socket transports cannot open TLS; "
            "use lm15.transports.FetchTransport, the host's fetch"
        )
    if not verify:
        ctx = ssl._create_unverified_context()
        return ctx
    ctx = ssl.create_default_context()
    if ca_bundle:
        ctx.load_verify_locations(cafile=ca_bundle)
    return ctx
