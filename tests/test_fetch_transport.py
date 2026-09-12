"""FetchTransport: the host's fetch under Pyodide; a named refusal on CPython."""

import pytest

from lm15.transports import FetchTransport, TransportError


def test_fetch_transport_names_pyodide_on_cpython():
    with pytest.raises(TransportError, match="Pyodide"):
        FetchTransport()


def test_fetch_transport_is_the_async_protocol_shape():
    # The full behaviour runs under Pyodide in lm15-ts/tests/pyodide.test.ts
    # (Node hosts Pyodide there); here only the surface every AsyncTransport has.
    assert callable(getattr(FetchTransport, "stream"))
    assert callable(getattr(FetchTransport, "aclose"))
