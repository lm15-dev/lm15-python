#!/bin/bash
# Cold-start benchmark: fresh venv → install → import → client → API call.
# Measures end-to-end latency for a single completion against Gemini.
#
# Requirements: uv, GEMINI_API_KEY set.
# Usage: bash benchmarks/cold_start.sh [RUNS]
set -e

RUNS=${1:-10}
MODEL="gemini-3.1-flash-lite-preview"
PROMPT="Reply with exactly: ok"
WORKDIR=$(mktemp -d)
CSVFILE="benchmarks/cold_start_results.csv"

trap "rm -rf $WORKDIR" EXIT

bench_one() {
    local label=$1 pkg=$2 script=$3

    for i in $(seq 1 $RUNS); do
        rm -rf "$WORKDIR/venv"

        t_start=$(python3 -c "import time; print(time.perf_counter())")

        uv venv "$WORKDIR/venv" -q 2>/dev/null
        uv pip install -q $pkg --python "$WORKDIR/venv/bin/python" 2>/dev/null

        t_installed=$(python3 -c "import time; print(time.perf_counter())")

        result=$("$WORKDIR/venv/bin/python" -c "$script" 2>&1)

        install_ms=$(python3 -c "print(f'{1000*($t_installed - $t_start):.0f}')")
        rest_ms=$(echo "$result" | grep TIMINGS | sed 's/.*import=\([0-9]*\)ms client=\([0-9]*\)ms call=\([0-9]*\)ms total=\([0-9]*\)ms/\1 \2 \3 \4/')
        import_ms=$(echo $rest_ms | cut -d' ' -f1)
        client_ms=$(echo $rest_ms | cut -d' ' -f2)
        call_ms=$(echo $rest_ms | cut -d' ' -f3)
        code_total=$(echo $rest_ms | cut -d' ' -f4)

        echo "$label,$i,$install_ms,$import_ms,$client_ms,$call_ms,$code_total"
    done
}

LM15_SCRIPT="
import time, sys, os
t0 = time.perf_counter()
from lm15 import LMRequest, Message, Part, build_default
t_import = time.perf_counter()
lm = build_default(use_pycurl=False)
t_client = time.perf_counter()
req = LMRequest(model='$MODEL', messages=(Message(role='user', parts=(Part.text_part('$PROMPT'),)),))
resp = lm.complete(req)
t_call = time.perf_counter()
print(f'TIMINGS import={1000*(t_import-t0):.0f}ms client={1000*(t_client-t_import):.0f}ms call={1000*(t_call-t_client):.0f}ms total={1000*(t_call-t0):.0f}ms', file=sys.stderr)
"

LITELLM_SCRIPT="
import time, sys, os
t0 = time.perf_counter()
import litellm
t_import = time.perf_counter()
t_client = time.perf_counter()
resp = litellm.completion(model='gemini/$MODEL', messages=[{'role': 'user', 'content': '$PROMPT'}])
t_call = time.perf_counter()
print(f'TIMINGS import={1000*(t_import-t0):.0f}ms client={1000*(t_client-t_import):.0f}ms call={1000*(t_call-t_client):.0f}ms total={1000*(t_call-t0):.0f}ms', file=sys.stderr)
"

GENAI_SCRIPT="
import time, sys, os
t0 = time.perf_counter()
from google import genai
t_import = time.perf_counter()
client = genai.Client(api_key=os.environ['GEMINI_API_KEY'])
t_client = time.perf_counter()
resp = client.models.generate_content(model='$MODEL', contents='$PROMPT')
t_call = time.perf_counter()
print(f'TIMINGS import={1000*(t_import-t0):.0f}ms client={1000*(t_client-t_import):.0f}ms call={1000*(t_call-t_client):.0f}ms total={1000*(t_call-t0):.0f}ms', file=sys.stderr)
"

echo "library,run,install_ms,import_ms,client_ms,call_ms,code_total_ms" | tee "$CSVFILE"

bench_one "lm15" "lm15" "$LM15_SCRIPT" | tee -a "$CSVFILE"
bench_one "litellm" "litellm" "$LITELLM_SCRIPT" | tee -a "$CSVFILE"
bench_one "google-genai" "google-genai" "$GENAI_SCRIPT" | tee -a "$CSVFILE"

echo ""
echo "=== Medians ==="
python3 -c "
import csv
from collections import defaultdict

rows = defaultdict(list)
with open('$CSVFILE') as f:
    for r in csv.DictReader(f):
        lib = r['library']
        rows[lib].append({k: int(v) for k, v in r.items() if k not in ('library', 'run')})

def median(vals):
    s = sorted(vals)
    n = len(s)
    return s[n//2] if n % 2 else (s[n//2-1] + s[n//2]) // 2

print(f'| {\"\":15s} | {\"install\":>9s} | {\"import\":>9s} | {\"client\":>9s} | {\"call\":>9s} | {\"total\":>9s} |')
print(f'|{\"\":-<17s}|{\"\":-<11s}|{\"\":-<11s}|{\"\":-<11s}|{\"\":-<11s}|{\"\":-<11s}|')
for lib in ['lm15', 'google-genai', 'litellm']:
    data = rows[lib]
    install = median([d['install_ms'] for d in data])
    imp = median([d['import_ms'] for d in data])
    cli = median([d['client_ms'] for d in data])
    call = median([d['call_ms'] for d in data])
    total = median([d['code_total_ms'] for d in data])
    print(f'| {lib:15s} | {install:7d}ms | {imp:7d}ms | {cli:7d}ms | {call:7d}ms | {total:7d}ms |')
"
