# Cross-SDK adapters

This directory contains the logical request cases and dump adapters used by the
conformance runner.

At the moment only `dump_request.py` is active. It is the lm15-python reference
implementation: it accepts one logical case JSON value and emits the normalized
provider HTTP request that lm15-python would send.

Future ports should add equivalent dump adapters here or expose compatible CLIs
that `conformance/check_request_fixtures.py` can call.
