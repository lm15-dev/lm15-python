#!/usr/bin/env python3
"""
Validate curl fixtures against live APIs and persist machine-readable results.

Outputs:
- results/latest.json
- results/history.jsonl
- results/bodies/<case-id>/<timestamp>.txt
"""

import copy
import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).parent
CASES_DIR = ROOT / "cases"
RESULTS_DIR = ROOT / "results"
BODIES_DIR = RESULTS_DIR / "bodies"
LATEST_JSON = RESULTS_DIR / "latest.json"
HISTORY_JSONL = RESULTS_DIR / "history.jsonl"


def _find_dotenv() -> Path:
    for parent in [ROOT, *ROOT.parents]:
        candidate = parent / ".env"
        if candidate.exists():
            return candidate
    return ROOT.parent / ".env"


DOTENV_PATH = _find_dotenv()

KEY_MAP = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "gemini": "GEMINI_API_KEY",
}


def load_dotenv(path: Path = DOTENV_PATH):
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export "):].strip()
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if value and ((value[0] == value[-1]) and value[0] in {'"', "'"}):
            value = value[1:-1]
        if not os.environ.get(key):
            os.environ[key] = value

_COST_INDEX = None
_ADDITIVE_CACHE_PROVIDERS = {"anthropic"}
_SEPARATE_REASONING_PROVIDERS = {"gemini", "google"}


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def build_request(case: dict, api_key: str, redact: bool = False):
    req = case["request"]
    url = req["url"]
    params = req.get("params", {})
    if params:
        qs = "&".join(f"{k}={v}" for k, v in params.items())
        url += "?" + qs

    headers = {}
    for k, v in req.get("headers", {}).items():
        if isinstance(v, str) and "$" in v:
            for env_var in KEY_MAP.values():
                v = v.replace(f"${env_var}", api_key if not redact else "REDACTED")
        headers[k] = v

    return url, headers, req.get("body")


def build_curl_display(case: dict, api_key: str) -> str:
    url, headers, body = build_request(case, api_key, redact=True)
    parts = [f"curl -X {case['request']['method']} '{url}'"]
    for k, v in headers.items():
        parts.append(f"  -H '{k}: {v}'")
    if body:
        parts.append(f"  -d '{json.dumps(body)}'")
    return " \\\n".join(parts)


def _curl_json(method: str, url: str, headers: dict[str, str], body: dict | None = None, timeout: int = 45):
    cmd = ["curl", "-sS", "-X", method, url]
    for k, v in headers.items():
        cmd.extend(["-H", f"{k}: {v}"])
    if body is not None:
        cmd.extend(["-d", json.dumps(body)])
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, encoding="utf-8")
    return json.loads(proc.stdout)


def _curl_multipart(url: str, headers: dict[str, str], form_parts: list[tuple[str, str]], timeout: int = 45):
    cmd = ["curl", "-sS", url]
    for k, v in headers.items():
        cmd.extend(["-H", f"{k}: {v}"])
    for k, v in form_parts:
        cmd.extend(["-F", f"{k}={v}"])
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, encoding="utf-8")
    return json.loads(proc.stdout)


def prepare_case(provider: str, case: dict, api_key: str):
    case = copy.deepcopy(case)
    if provider != "openai":
        return case

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    body = case.get("request", {}).get("body", {}) or {}

    # previous_response_id setup
    if body.get("previous_response_id") == "$AUTO_PREVIOUS_RESPONSE_ID":
        seed = {
            "model": body.get("model", "gpt-4.1-mini"),
            "input": [{"role": "user", "content": [{"type": "input_text", "text": "Remember the codeword: maple."}]}],
            "stream": False,
        }
        resp = _curl_json("POST", "https://api.openai.com/v1/responses", headers, seed)
        case["request"]["body"]["previous_response_id"] = resp["id"]

    # conversation setup
    conv = body.get("conversation")
    if isinstance(conv, dict) and conv.get("id") == "$AUTO_CONVERSATION_ID":
        conv_obj = _curl_json("POST", "https://api.openai.com/v1/conversations", headers, {})
        case["request"]["body"]["conversation"] = {"id": conv_obj["id"]}

    # file_search setup
    tools = body.get("tools") or []
    for tool in tools:
        if tool.get("type") == "file_search" and "$AUTO_VECTOR_STORE_ID" in (tool.get("vector_store_ids") or []):
            vs = _curl_json("POST", "https://api.openai.com/v1/vector_stores", headers, {"name": "curl-fixtures-temp-vs"})
            tool["vector_store_ids"] = [vs["id"]]

    return case


def run_curl(case: dict, api_key: str, timeout: int = 45):
    case = prepare_case(case.get("provider"), case, api_key)
    url, headers, body = build_request(case, api_key, redact=False)
    body_file = RESULTS_DIR / ".tmp_body.txt"
    cmd = [
        "curl", "-sS", "-o", str(body_file), "-w", "%{http_code}",
        "-X", case["request"]["method"], url,
    ]
    for k, v in headers.items():
        cmd.extend(["-H", f"{k}: {v}"])
    if body is not None:
        cmd.extend(["-d", json.dumps(body)])

    started = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, encoding="utf-8")
    ended = time.time()
    http_code = proc.stdout.strip()
    raw_body = body_file.read_text(encoding="utf-8") if body_file.exists() else ""
    if body_file.exists():
        body_file.unlink()
    return {
        "http_code": http_code,
        "raw_body": raw_body,
        "stderr": proc.stderr,
        "duration_seconds": round(ended - started, 3),
    }


def fetch_models_dev(timeout: float = 20.0):
    req = urllib.request.Request("https://models.dev/api.json", headers={"User-Agent": "lm15"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = json.loads(r.read())
    out = {}
    providers = data.get("providers") or data
    for provider_id, provider_payload in providers.items():
        if not isinstance(provider_payload, dict) or "models" not in provider_payload:
            continue
        for model_id, m in (provider_payload.get("models") or {}).items():
            if isinstance(m, dict) and m.get("cost"):
                out[model_id] = {"provider": provider_id, "raw": m}
    return out


def get_cost_index():
    global _COST_INDEX
    if _COST_INDEX is None:
        _COST_INDEX = fetch_models_dev()
    return _COST_INDEX


def infer_model(case: dict):
    body = case.get("request", {}).get("body", {})
    if isinstance(body, dict) and body.get("model"):
        return body["model"]
    url = case.get("request", {}).get("url", "")
    if "/models/" in url:
        frag = url.split("/models/", 1)[1]
        return frag.split(":", 1)[0]
    return None


def _extract_json_from_sse(raw_body: str, provider: str):
    lines = [line for line in raw_body.splitlines() if line.startswith("data: ")]
    payloads = []
    for line in lines:
        frag = line[6:]
        if frag.strip() == "[DONE]":
            continue
        try:
            payloads.append(json.loads(frag))
        except Exception:
            pass
    if not payloads:
        return None
    if provider == "openai":
        for p in reversed(payloads):
            if p.get("type") == "response.completed":
                return p.get("response")
    if provider == "anthropic":
        for p in reversed(payloads):
            if p.get("type") == "message_delta" and p.get("usage"):
                return {"usage": p.get("usage")}
            if p.get("type") == "message_stop":
                # no usage on stop itself usually
                continue
    if provider == "gemini":
        return payloads[-1]
    return payloads[-1]


def extract_usage(provider: str, raw_body: str):
    try:
        data = json.loads(raw_body)
    except Exception:
        data = _extract_json_from_sse(raw_body, provider)
        if data is None:
            return None

    if provider == "openai":
        u = data.get("usage") or {}
        u_in = u.get("input_tokens_details") or {}
        u_out = u.get("output_tokens_details") or {}
        return {
            "input_tokens": u.get("input_tokens", 0),
            "output_tokens": u.get("output_tokens", 0),
            "total_tokens": u.get("total_tokens", 0),
            "reasoning_tokens": u_out.get("reasoning_tokens"),
            "cache_read_tokens": u_in.get("cached_tokens"),
            "input_audio_tokens": u_in.get("audio_tokens"),
            "output_audio_tokens": u_out.get("audio_tokens"),
        }
    if provider == "anthropic":
        u = data.get("usage") or {}
        return {
            "input_tokens": u.get("input_tokens", 0),
            "output_tokens": u.get("output_tokens", 0),
            "total_tokens": (u.get("input_tokens", 0) or 0) + (u.get("output_tokens", 0) or 0),
            "cache_read_tokens": u.get("cache_read_input_tokens"),
            "cache_write_tokens": u.get("cache_creation_input_tokens"),
        }
    if provider == "gemini":
        u = data.get("usageMetadata") or {}
        return {
            "input_tokens": u.get("promptTokenCount", 0),
            "output_tokens": u.get("candidatesTokenCount", 0),
            "total_tokens": u.get("totalTokenCount", 0),
            "cache_read_tokens": u.get("cachedContentTokenCount"),
            "reasoning_tokens": u.get("thoughtsTokenCount"),
        }
    return None


def estimate_cost_from_usage(usage: dict, spec: dict, provider: str):
    cost = ((spec.get("raw") or {}).get("cost") or {})
    def per_token(rate):
        return (rate or 0) / 1_000_000
    r_input = per_token(cost.get("input"))
    r_output = per_token(cost.get("output"))
    r_cache_read = per_token(cost.get("cache_read"))
    r_cache_write = per_token(cost.get("cache_write"))
    r_reasoning = per_token(cost.get("reasoning"))
    r_input_audio = per_token(cost.get("input_audio"))
    r_output_audio = per_token(cost.get("output_audio"))

    cache_read = usage.get("cache_read_tokens") or 0
    cache_write = usage.get("cache_write_tokens") or 0
    reasoning = usage.get("reasoning_tokens") or 0
    input_audio = usage.get("input_audio_tokens") or 0
    output_audio = usage.get("output_audio_tokens") or 0

    if provider in _ADDITIVE_CACHE_PROVIDERS:
        text_input = (usage.get("input_tokens") or 0) - input_audio
    else:
        text_input = (usage.get("input_tokens") or 0) - cache_read - cache_write - input_audio
    text_input = max(text_input, 0)

    if provider in _SEPARATE_REASONING_PROVIDERS:
        text_output = (usage.get("output_tokens") or 0) - output_audio
    else:
        text_output = (usage.get("output_tokens") or 0) - reasoning - output_audio
    text_output = max(text_output, 0)

    total = (
        text_input * r_input +
        text_output * r_output +
        cache_read * r_cache_read +
        cache_write * r_cache_write +
        reasoning * r_reasoning +
        input_audio * r_input_audio +
        output_audio * r_output_audio
    )
    return round(total, 8)


def estimate_cost_usd(provider: str, case: dict, raw_body: str):
    usage = extract_usage(provider, raw_body)
    if not usage:
        return None
    model = infer_model(case)
    if not model:
        return None
    spec = get_cost_index().get(model)
    if not spec:
        return None
    return estimate_cost_from_usage(usage, spec, provider)


def is_streaming_case(case: dict) -> bool:
    body = case.get("request", {}).get("body", {}) or {}
    params = case.get("request", {}).get("params", {}) or {}
    url = case.get("request", {}).get("url", "")
    return bool(body.get("stream") is True or params.get("alt") == "sse" or "streamGenerateContent" in url)


def validate_response(provider: str, case: dict, raw_body: str):
    expect = case.get("expect", {}) or {}
    checks = []

    if is_streaming_case(case):
        if provider == "openai":
            if "event: response.completed" not in raw_body:
                checks.append("missing response.completed event")
            if extract_usage(provider, raw_body) is None:
                checks.append("missing usage in streaming response")
        elif provider == "anthropic":
            if "event:" not in raw_body:
                checks.append("missing SSE events")
            if "message_start" not in raw_body and "message_stop" not in raw_body:
                checks.append("missing Anthropic message lifecycle events")
        elif provider == "gemini":
            if "data: " not in raw_body:
                checks.append("missing SSE data events")
            if extract_usage(provider, raw_body) is None:
                checks.append("missing usageMetadata in streaming response")
    else:
        try:
            data = json.loads(raw_body)
        except Exception:
            return False, ["response body is not valid JSON"]

        if provider == "openai":
            if not data.get("id"):
                checks.append("missing id")
            if data.get("object") != "response":
                checks.append("missing/invalid object=response")
            if extract_usage(provider, raw_body) is None:
                checks.append("missing usage")
        elif provider == "anthropic":
            if not data.get("id"):
                checks.append("missing id")
            if "content" not in data:
                checks.append("missing content")
            if extract_usage(provider, raw_body) is None:
                checks.append("missing usage")
        elif provider == "gemini":
            req_url = case.get("request", {}).get("url", "")
            if "/cachedContents" in req_url:
                if not data.get("name"):
                    checks.append("missing cache resource name")
                if extract_usage(provider, raw_body) is None:
                    checks.append("missing usageMetadata")
            else:
                if not data.get("candidates"):
                    checks.append("missing candidates")
                if extract_usage(provider, raw_body) is None:
                    checks.append("missing usageMetadata")

        body_contains = expect.get("body_contains") or []
        for s in body_contains:
            if s not in raw_body:
                checks.append(f"body missing expected substring: {s}")

    return len(checks) == 0, checks


def save_body(case_id: str, tested_at: str, raw_body: str) -> str | None:
    if raw_body is None:
        return None
    safe_case = case_id.replace("/", "_")
    stamp = tested_at.replace(":", "-")
    out_dir = BODIES_DIR / safe_case
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{stamp}.txt"
    path.write_text(raw_body, encoding="utf-8")
    return str(path.relative_to(ROOT))


def load_latest():
    if not LATEST_JSON.exists():
        return {"generated_at": None, "cases": {}}
    with open(LATEST_JSON, encoding="utf-8") as f:
        return json.load(f)


def persist_results(run_results: dict):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    latest = load_latest()
    latest["generated_at"] = now_iso()
    latest.setdefault("cases", {}).update(run_results)
    with open(LATEST_JSON, "w", encoding="utf-8") as f:
        json.dump(latest, f, indent=2, sort_keys=True)
    with open(HISTORY_JSONL, "a", encoding="utf-8") as f:
        for result in run_results.values():
            f.write(json.dumps(result, sort_keys=True) + "\n")


def main():
    load_dotenv()

    filter_provider = None
    filter_task = None
    dry_run = False

    args = sys.argv[1:]
    i = 0
    while i < len(args):
        arg = args[i]
        if arg == "--dry-run":
            dry_run = True
        elif arg == "--task":
            if i + 1 >= len(args):
                print("--task requires a value", file=sys.stderr)
                return 2
            filter_task = args[i + 1]
            i += 1
        elif arg in KEY_MAP:
            filter_provider = arg
        i += 1

    passed = failed = skipped = 0
    run_results = {}

    for provider_dir in sorted(CASES_DIR.iterdir()):
        if not provider_dir.is_dir():
            continue
        provider = provider_dir.name
        if filter_provider and provider != filter_provider:
            continue

        key_var = KEY_MAP.get(provider)
        api_key = os.environ.get(key_var, "") if key_var else ""

        for case_file in sorted(provider_dir.glob("*.json")):
            feature = case_file.stem
            case_id = f"{provider}.{feature}"
            if filter_task and case_id != filter_task:
                continue
            case = json.loads(case_file.read_text(encoding="utf-8"))
            tested_at = now_iso()

            if dry_run:
                if not api_key:
                    print(f"  ⚠️  {case_id}: {key_var} not set, skipping")
                else:
                    print(f"--- {case_id} ---")
                    print(build_curl_display(case, api_key))
                    print()
                continue

            if not api_key:
                print(f"  ⚠️  {case_id}: {key_var} not set, skipping")
                rec = {
                    "case_id": case_id,
                    "provider": provider,
                    "feature": feature,
                    "tested_at": tested_at,
                    "result": "skipped",
                    "http_code": None,
                    "duration_seconds": None,
                    "estimated_cost_usd": None,
                    "body_path": None,
                    "reason": f"{key_var} not set",
                }
                run_results[case_id] = rec
                skipped += 1
                continue

            try:
                result = run_curl(case, api_key)
                expected = str(case.get("expect", {}).get("status", 200))
                semantic_ok, semantic_errors = validate_response(provider, case, result["raw_body"]) if result["http_code"] == expected else (False, [])
                outcome = "pass" if result["http_code"] == expected and semantic_ok else "fail"
                body_path = save_body(case_id, tested_at, result["raw_body"])
                rec = {
                    "case_id": case_id,
                    "provider": provider,
                    "feature": feature,
                    "tested_at": tested_at,
                    "result": outcome,
                    "http_code": int(result["http_code"]) if result["http_code"].isdigit() else result["http_code"],
                    "duration_seconds": result["duration_seconds"],
                    "estimated_cost_usd": estimate_cost_usd(provider, case, result["raw_body"]),
                    "body_path": body_path,
                    "stderr": result["stderr"] or None,
                    "semantic_errors": semantic_errors or None,
                }
                run_results[case_id] = rec
                if outcome == "pass":
                    print(f"  ✅ {case_id} (HTTP {result['http_code']}, {result['duration_seconds']}s)")
                    passed += 1
                else:
                    preview = result['raw_body'][:200].replace("\n", " ")
                    if result['http_code'] != expected:
                        print(f"  ❌ {case_id} (expected {expected}, got {result['http_code']}, {result['duration_seconds']}s)")
                    else:
                        print(f"  ❌ {case_id} (semantic validation failed, {result['duration_seconds']}s)")
                        if semantic_errors:
                            print(f"     {'; '.join(semantic_errors[:3])}")
                    if preview:
                        print(f"     {preview}")
                    failed += 1
            except subprocess.TimeoutExpired:
                rec = {
                    "case_id": case_id,
                    "provider": provider,
                    "feature": feature,
                    "tested_at": tested_at,
                    "result": "fail",
                    "http_code": None,
                    "duration_seconds": None,
                    "estimated_cost_usd": None,
                    "body_path": None,
                    "reason": "timeout",
                }
                run_results[case_id] = rec
                print(f"  ❌ {case_id} (timeout)")
                failed += 1
            except Exception as e:
                rec = {
                    "case_id": case_id,
                    "provider": provider,
                    "feature": feature,
                    "tested_at": tested_at,
                    "result": "fail",
                    "http_code": None,
                    "duration_seconds": None,
                    "estimated_cost_usd": None,
                    "body_path": None,
                    "reason": str(e),
                }
                run_results[case_id] = rec
                print(f"  ❌ {case_id} ({e})")
                failed += 1

    if not dry_run:
        persist_results(run_results)
        print(f"\n=== Summary ===")
        print(f"Pass: {passed}")
        print(f"Fail: {failed}")
        print(f"Skip: {skipped}")
        if failed > 0:
            sys.exit(1)


if __name__ == "__main__":
    main()
