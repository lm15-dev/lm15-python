#!/usr/bin/env bash
# Run a command inside a no-network sandbox. Default: the offline conformance suite.
#
# Usage:
#   conformance/run_offline.sh                 # run_all.py --strict, offline
#   conformance/run_offline.sh <cmd> [args...] # arbitrary command, offline
#
# Prefers a network namespace (unshare -rn). Falls back to stripping API keys
# and pointing HTTP(S)_PROXY at an unroutable address when user namespaces are
# unavailable. Any code path that tries to reach the network inside this
# wrapper must fail loudly — that is the point.
set -u

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$HERE")"

PYBIN="$REPO/.venv/bin/python"
[ -x "$PYBIN" ] || PYBIN="$(command -v python3)"

if [ "$#" -gt 0 ]; then
  CMD=("$@")
else
  CMD=("$PYBIN" "$HERE/run_all.py" "--strict")
fi

if unshare -rn true 2>/dev/null; then
  exec unshare -rn -- "${CMD[@]}"
fi

exec env -u OPENAI_API_KEY -u ANTHROPIC_API_KEY -u GEMINI_API_KEY -u GOOGLE_API_KEY \
  HTTPS_PROXY=http://127.0.0.1:9 HTTP_PROXY=http://127.0.0.1:9 NO_PROXY='' \
  "${CMD[@]}"
