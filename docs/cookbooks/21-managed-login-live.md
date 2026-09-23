---
rat:
  project: ../..
  python:
    requires: ">=3.10"
    dependencies: ["-e .", numpy]
---

# Managed login, live

A hands-on check of `lm15.login` and `connect()` against real providers. Run the cells top to bottom. Two of them open your browser (or print a URL and a code if it cannot). Everything you sign in to is saved in one private file; the last cell shows how to undo it.

**Costs:** the request cells contact real providers, including xAI and Claude. Subscription login is not a guarantee of included usage; charges depend on your account and the provider. Run only the cells you intend to authorize.

## 0. Where things get saved

```python
from lm15 import Message, LMRouter, RouterConfig, Request
from lm15.login import Auth, TerminalUI, AuthOperationError
from lm15.doctor import explain_auth

auth = Auth.local()                     # reads nothing until asked
ui = TerminalUI(open_browser=True)      # set False over SSH: you will get a URL to open elsewhere

print("store:", auth.store.description)
for provider in ("xai", "claude-code", "openai-codex", "gemini"):
    s = auth.status(provider)
    print(f"{provider:13s} {s.presence:7s} {s.usability:12s} expires={s.expires_at} logged_out={s.logged_out}")
```

```output
store: /home/maxime/.config/lm15/credentials.json
xai           saved   ready        expires=2026-09-23T16:40:03Z logged_out=False
claude-code   saved   ready        expires=unknown logged_out=False
openai-codex  absent  unknown      expires=None logged_out=False
gemini        absent  unknown      expires=None logged_out=False
```

## 1. Sign in to xAI (device code — the migrated, supported path)

You get a code and a link. Approve in the browser; the cell finishes on its own.

```python
current = auth.status("xai").connection
if current:
    print("already saved:", current)
else:
    connection = auth.login("xai", "device", ui=ui)
    print("saved:", connection)
```

```output
Open https://accounts.x.ai/oauth2/device?user_code=XXXX-XXXX
and enter this code:  XXXX-XXXX
(the code is valid for about 30 minutes)
saved: Connection(id='cn_Z_0tEcI8Te25EmNP', provider='xai', kind='account', method='device', label='xAI subscription')
```

## 2. A real request through a managed router

No environment key is involved. The credential comes from the saved login, renewed under the lock if it is close to expiry.

```python
router = LMRouter(RouterConfig(auth=auth))
reply = router.complete(Request(model="xai:grok-4.7",
                                messages=(Message.user("Say hi in three words."),)))
print(reply.text)
```

```output
Hello there, friend.
```

```py
import time
```

```py
s = router.stream(Request(
  model = 'xai:grok-4.7',
  messages = (
    Message.user('my name is maxe')
  )
))

for i in s:
  print(i)
  time.sleep(1)
```

```output
StreamStartEvent(id=None, model='grok-4.7', adaptations=(), type='start')
StreamDeltaEvent(delta=ThinkingDelta(text='The', part_index=0, type='thinking'), type='delta')
StreamDeltaEvent(delta=ThinkingDelta(text=' user', part_index=0, type='thinking'), type='delta')
StreamDeltaEvent(delta=ThinkingDelta(text=' said', part_index=0, type='thinking'), type='delta')
StreamDeltaEvent(delta=ThinkingDelta(text=' "', part_index=0, type='thinking'), type='delta')
StreamDeltaEvent(delta=ThinkingDelta(text='my', part_index=0, type='thinking'), type='delta')
StreamDeltaEvent(delta=ThinkingDelta(text=' name', part_index=0, type='thinking'), type='delta')
StreamDeltaEvent(delta=ThinkingDelta(text=' is', part_index=0, type='thinking'), type='delta')
StreamDeltaEvent(delta=ThinkingDelta(text=' max', part_index=0, type='thinking'), type='delta')
StreamDeltaEvent(delta=ThinkingDelta(text='e', part_index=0, type='thinking'), type='delta')
StreamDeltaEvent(delta=ThinkingDelta(text='".', part_index=0, type='thinking'), type='delta')
StreamDeltaEvent(delta=ThinkingDelta(text=' This', part_index=0, type='thinking'), type='delta')
StreamDeltaEvent(delta=ThinkingDelta(text=' is', part_index=0, type='thinking'), type='delta')
StreamDeltaEvent(delta=ThinkingDelta(text=' a', part_index=0, type='thinking'), type='delta')
StreamDeltaEvent(delta=ThinkingDelta(text=' simple', part_index=0, type='thinking'), type='delta')
StreamDeltaEvent(delta=ThinkingDelta(text=' introduction', part_index=0, type='thinking'), type='delta')
StreamDeltaEvent(delta=ThinkingDelta(text='.', part_index=0, type='thinking'), type='delta')
StreamDeltaEvent(delta=ThinkingDelta(text=' I', part_index=0, type='thinking'), type='delta')
StreamDeltaEvent(delta=ThinkingDelta(text=' should', part_index=0, type='thinking'), type='delta')
StreamDeltaEvent(delta=ThinkingDelta(text=' respond', part_index=0, type='thinking'), type='delta')
StreamDeltaEvent(delta=ThinkingDelta(text=' friendly', part_index=0, type='thinking'), type='delta')
StreamDeltaEvent(delta=ThinkingDelta(text=' and', part_index=0, type='thinking'), type='delta')
StreamDeltaEvent(delta=ThinkingDelta(text=' acknowledge', part_index=0, type='thinking'), type='delta')
StreamDeltaEvent(delta=ThinkingDelta(text=' their', part_index=0, type='thinking'), type='delta')
StreamDeltaEvent(delta=ThinkingDelta(text=' name', part_index=0, type='thinking'), type='delta')
StreamDeltaEvent(delta=ThinkingDelta(text='.\n', part_index=0, type='thinking'), type='delta')
StreamDeltaEvent(delta=TextDelta(text='Nice', part_index=0, logprobs=(), logprobs_complete=True, type='text'), type='delta')
StreamDeltaEvent(delta=TextDelta(text=' to', part_index=0, logprobs=(), logprobs_complete=True, type='text'), type='delta')
StreamDeltaEvent(delta=TextDelta(text=' meet', part_index=0, logprobs=(), logprobs_complete=True, type='text'), type='delta')
StreamDeltaEvent(delta=TextDelta(text=' you', part_index=0, logprobs=(), logprobs_complete=True, type='text'), type='delta')
StreamDeltaEvent(delta=TextDelta(text=',', part_index=0, logprobs=(), logprobs_complete=True, type='text'), type='delta')
StreamDeltaEvent(delta=TextDelta(text=' Max', part_index=0, logprobs=(), logprobs_complete=True, type='text'), type='delta')
StreamDeltaEvent(delta=TextDelta(text='e', part_index=0, logprobs=(), logprobs_complete=True, type='text'), type='delta')
StreamDeltaEvent(delta=TextDelta(text='.', part_index=0, logprobs=(), logprobs_complete=True, type='text'), type='delta')
StreamDeltaEvent(delta=TextDelta(text=' What', part_index=0, logprobs=(), logprobs_complete=True, type='text'), type='delta')
StreamDeltaEvent(delta=TextDelta(text=' can', part_index=0, logprobs=(), logprobs_complete=True, type='text'), type='delta')
StreamDeltaEvent(delta=TextDelta(text=' I', part_index=0, logprobs=(), logprobs_complete=True, type='text'), type='delta')
StreamDeltaEvent(delta=TextDelta(text=' help', part_index=0, logprobs=(), logprobs_complete=True, type='text'), type='delta')
StreamDeltaEvent(delta=TextDelta(text=' you', part_index=0, logprobs=(), logprobs_complete=True, type='text'), type='delta')
StreamDeltaEvent(delta=TextDelta(text=' with', part_index=0, logprobs=(), logprobs_complete=True, type='text'), type='delta')
StreamDeltaEvent(delta=TextDelta(text='?', part_index=0, logprobs=(), logprobs_complete=True, type='text'), type='delta')
StreamEndEvent(
    finish_reason='stop',
    usage=Usage(input_tokens=1247, output_tokens=15, total_tokens=1331, cache_read_tokens=1152, cache_write_tokens=None, reasoning_tokens=69, input_audio_tokens=0, output_audio_tokens=0),
    provider_data=<dict: 8 keys>,
    type='end',
)
```

```py
print("via:", router.lm("xai:grok-4.7").credential_origin())
```

```output
via: managed connection cn_Z_0tEcI8Te25EmNP (xAI subscription)
```

## 3. Status and the doctor — never a secret

```python
print(auth.status("xai"))
print()
print(explain_auth("xai", config=router.config))
```

```output
ConnectionStatus(provider='xai', presence='saved', usability='ready', connection=Connection(id='cn_Z_0tEcI8Te25EmNP', provider='xai', kind='account', method='device', label='xAI subscription'), expires_at='2026-09-23T06:10:16Z', logged_out=False, verification=None, detail=None)

auth for provider 'xai':
   - explicit api_keys entry: not provided
  => saved connection cn_Z_0tEcI8Te25EmNP: xAI subscription (ready, expires 2026-09-23T06:10:16Z)
   - env $XAI_API_KEY: not set
  configured: yes — saved connection cn_Z_0tEcI8Te25EmNP
```

## 4. Your existing Claude Code login, as an explicit choice

This does not copy anything out of `~/.claude/.credentials.json`; it saves "use that source" and reads the file in place each request.

```python
if auth.status("claude-code").connection is None:
    try:
        print(auth.configure("claude-code", method="external:claude-code-cli"))
    except AuthOperationError as e:
        print("not usable:", e.reason, "—", e)
print("claude-code:", auth.status("claude-code").usability)
```

```output
claude-code: ready
```

```python
reply = LMRouter(RouterConfig(auth=auth)).complete(
    Request(model="claude-code:claude-opus-5", messages=(Message.user("Say hi in three words."),)))
print(reply.text)
```

```output
Hi there, friend!
```

See now: `connect()` wraps connection and model selection into one interactive step.


## 5. The short path

`connect()` asks which saved connection and which model, then returns a client pinned to that choice. Answer the two questions in the terminal.

```python
from lm15.interactive import connect

with connect(auth=auth, ui=ui) as lm:
    print("bound to:", lm.selection.routed, "on", lm.connection.label)
    print(lm.complete(messages=[Message.user("One-word greeting.")]).text)
```

```output
Connections are saved privately in /home/maxime/.config/lm15/credentials.json.

Use a saved connection, or connect another?
  1. claude-code via your Claude Code login (~/.claude/.credentials.json)  — claude-code · saved
  2. xAI subscription  — xai · saved
  3. groq API key  — groq · saved
  4. Connect another account or API key
Choose a number: 2

Which xai model?
  1. grok-4.20-0309-non-reasoning  — listed by your account just now
  2. grok-4.20-0309-reasoning  — listed by your account just now
  3. grok-4.20-multi-agent-0309  — listed by your account just now
  4. grok-4.3  — listed by your account just now
  5. grok-4.5  — listed by your account just now
  6. grok-4.6  — listed by your account just now
  7. grok-4.7  — listed by your account just now
  8. grok-build-0.1  — listed by your account just now
  9. grok-imagine-image  — listed by your account just now
  10. grok-imagine-image-2.0  — listed by your account just now
  11. grok-imagine-image-quality  — listed by your account just now
  12. grok-imagine-video  — listed by your account just now
  13. grok-imagine-video-1.5  — listed by your account just now
  14. Type a model id (not verified against your account)
Choose a number: 7
Ready: xai:grok-4.7 through xAI subscription.
bound to: xai:grok-4.7 on xAI subscription
Hello.
```

## 6. (Optional) LM15's own Claude sign-in — unverified

The `"browser"` method now uses Claude's hosted **Authentication code** page,
not a localhost callback. Open the new link on your XPS, sign in, then copy the
**whole displayed code (including `#state`)** into this cell's input prompt.
The full hosted return URL also works. Do not paste it into chat. No SSH port
forwarding is needed. Old output below may still show a localhost link until
you rerun the cell; do not reuse that old link or code.

This method remains **unverified**. A successful login and reply are only part
of the evidence: renewal, provider permission and billing need separate checks.
Your existing saved connection is replaced only if the new login is saved
successfully. Skip this cell if you do not want to try it.

```python
old = auth.status("claude-code").connection
connection = auth.login("claude-code", "browser", ui=ui, allow_unverified=True,
                        replace=old.id if old else None)
print("saved:", connection)
reply = LMRouter(RouterConfig(auth=auth)).complete(
   Request(model="claude-code:claude-sonnet-4-5", messages=(Message.user("Say hi."),)))
print(reply.text)
```

```output
Open this link to sign in:
  https://claude.com/cai/oauth/authorize?code=true&client_id=9d1c250a-e61b-44d9-88ed-5944d1962f5e&response_type=code&redirect_uri=https%3A%2F%2Fplatform.claude.com%2Foauth%2Fcode%2Fcallback&scope=org%3Acreate_api_key+user%3Aprofile+user%3Ainference+user%3Asessions%3Aclaude_code+user%3Amcp_servers+user%3Afile_upload&code_challenge=<redacted>&code_challenge_method=S256&state=<redacted>
Sign in to Claude in your browser. On the Authentication code page, copy the whole displayed code (including #state) and paste it here. The full return URL also works. Your browser may be on another machine; no localhost connection is needed.
Paste the full code#state or return URL here
> <authorization-code>#<state> (redacted; single-use, already consumed)
… Exchanging the authorization code…
saved: Connection(id='cn_CMvdgfxbT8vKGZto', provider='claude-code', kind='account', method='browser', label='Claude subscription')
Hi! I'm Claude Code, Anthropic's CLI assistant. I'm here to help you with coding tasks, file operations, and technical questions. How can I assist you today?
```


```py
reply = LMRouter(RouterConfig(auth=auth)).complete(
   Request(model="claude-code:claude-sonnet-4-5", system='you are a pirate', messages=(Message.user("Say hi."),)))
```

```py
reply
```

```output
Response(
    text="Ahoy there, matey! 🏴\u200d☠️ \n\nClaude Code here, ready to help ye navigate the seven seas of programming! Whether ye be needin' to write some code, debug a troublesome script, or plunder some data from yer files, I'm at yer service!\n\nWhat can this old sea dog help ye with today? ⚓",
    model='claude-sonnet-4-5-20250929',
    finish_reason='stop',
    usage=Usage(input_tokens=29, output_tokens=88, total_tokens=117, cache_read_tokens=0, cache_write_tokens=0, reasoning_tokens=None, input_audio_tokens=None, output_audio_tokens=None),
    adaptations=['config.max_tokens:defaulted'],
    id='msg_011CfLKGcg4QoK5LCpf8WDqM',
    provider_data=<dict: 10 keys>,
)
```


## 7. Sign out — and see that nothing falls back to a paid key

After logout, a set `XAI_API_KEY` is *blocked*, and the error says so. Passing the key explicitly still works, because that is a deliberate choice.

```python
RUN_LOGOUT = True

if RUN_LOGOUT:
    print(auth.logout("xai"))
    try:
        LMRouter(RouterConfig(env={"XAI_API_KEY": "not-a-real-key"})).lm("xai:grok-4-fast")
    except Exception as e:
        print("blocked, as it should be:")
        print("  ", str(e).splitlines()[0])
    print("status:", auth.status("xai"))
    # to sign back in: re-run cell 1
```

```output
ForgetResult(provider='xai', forgot=True, routes=('xai',), identity_generation='2')
blocked, as it should be:
   the 'xai' subscription login was signed out. $XAI_API_KEY is set but is used only when passed explicitly: sign in again (Log in again: run lm15.auth.login_xai() (SuperGrok / X Premium subscription auth)), or pass the key deliberately with RouterConfig(api_keys={'xai': "..."}).
status: ConnectionStatus(provider='xai', presence='absent', usability='unknown', connection=None, expires_at=None, logged_out=True, verification=None, detail='signed out; sign in again or pass a key explicitly')
```

## 8. Codex — browser-assisted setup on this machine

These cells use the already-open XPS browser through the local `agent-browser`
CLI, and the same Rat kernel as this notebook. They do not read or overwrite
`~/.codex/auth.json`. New credentials go into LM15's own store. Login methods
remain unverified; inference may consume account usage. Run only these new cells,
not the earlier logout or Claude login cells again.

This small notebook UI opens a dedicated tab and privately transfers its final
localhost return URL back to the server. It does not print authorization codes.
**Check the chosen ChatGPT account before approving.** The manual return is held
until `codex_ui.account_confirmed` is explicitly set below.

```python
import json
import os
import subprocess
import threading
import time
from urllib.parse import urlsplit
from lm15 import Config, LMRouter, Message, Request, RouterConfig
from lm15.login import Auth, AuthOperationError, LoginCancelled
from lm15.login.types import AuthUrlNotice, DeviceCodeNotice, ManualCodePrompt

codex_auth = Auth.local()
print("Codex:", codex_auth.status("openai-codex").presence)

class CodexBrowserUI:
    def __init__(self):
        self.tab = None
        self.account_confirmed = False
        self.cancel = threading.Event()
        self.device_code = None

    def browser(self, op, **args):
        if op == "eval" and args.get("expression") == "location.href":
            # Tab metadata retains the OAuth URL even on Chromium's error page.
            # Reading it also avoids competing with page-click commands.
            for tab in self.browser("tabs"):
                if tab.get("id", tab.get("tab")) == args.get("tab"):
                    return {"value": tab.get("url", "")}
            raise RuntimeError("The login tab was closed")
        for attempt in range(5):
            result = subprocess.run(
                ["agent-browser", "request", "-"],
                input=json.dumps({"op": op, "args": args}),
                capture_output=True, text=True, timeout=15,
                env={**os.environ, "AGENT_BROWSER_HOST": "xps"},
            )
            if not result.returncode:
                return json.loads(result.stdout)
            # Retry only the controller's explicit guarantee of NO queued action,
            # never an ambiguous click/submission failure or OAuth exchange.
            if "No action was queued" not in result.stdout + result.stderr:
                break
            time.sleep(0.25)
        raise RuntimeError("Browser command failed; inspect the dedicated tab")

    def notify(self, notice):
        if isinstance(notice, (AuthUrlNotice, DeviceCodeNotice)):
            url = notice.url if isinstance(notice, AuthUrlNotice) else notice.verification_url
            if isinstance(notice, DeviceCodeNotice):
                self.device_code = notice.user_code  # private; never printed
            self.tab = self.browser("open", url=url)["tab"]

    def prompt(self, prompt):
        if not isinstance(prompt, ManualCodePrompt):
            raise RuntimeError("Unexpected prompt; choose the login method explicitly")
        deadline = time.monotonic() + 880
        while time.monotonic() < deadline and not self.cancel.wait(1):
            if not self.tab or not self.account_confirmed:
                continue
            url = self.browser("eval", tab=self.tab, expression="location.href").get("value", "")
            parsed = urlsplit(url)
            if (self.account_confirmed and parsed.scheme == "http"
                    and parsed.hostname == "localhost" and parsed.port == 1455
                    and parsed.path == "/auth/callback"):
                return url
        raise LoginCancelled("Browser login cancelled or expired")

    def dismiss(self, prompt):
        self.cancel.set()
```

## 9. Start the Codex browser login

The login runs in a background thread so the notebook remains available while
we use the browser. An existing LM15 Codex connection is not replaced unless
`REPLACE_CODEX_CONNECTION` is explicitly enabled. Your CLI login is untouched.

```python
REPLACE_CODEX_CONNECTION = False

if "codex_login_thread" in globals() and codex_login_thread.is_alive():
    raise RuntimeError("A Codex login is already running; finish or cancel it first")
old_codex = codex_auth.status("openai-codex").connection
if old_codex and not REPLACE_CODEX_CONNECTION:
    raise RuntimeError("A Codex connection is already saved; skip login or explicitly allow replacement")
codex_ui = CodexBrowserUI()
codex_login_result = {}

def finish_codex_browser_login():
    try:
        codex_login_result["connection"] = codex_auth.login(
            "openai-codex", "browser", ui=codex_ui, allow_unverified=True,
            replace=old_codex.id if old_codex else None, cancel=codex_ui.cancel,
        )
    except BaseException as error:
        codex_login_result["error"] = error

codex_login_thread = threading.Thread(target=finish_codex_browser_login, daemon=True)
codex_login_thread.start()
print("Login started. Check the ChatGPT account in the newly opened browser tab.")
```

## 10. Confirm the account, then inspect completion

Run the first cell only after checking that the browser shows your intended
ChatGPT subscription account. This permits the final return URL to be delivered
privately to the pending login. Approve the sign-in in the browser if requested.

```python
codex_ui.account_confirmed = True
print("The selected browser account has been confirmed.")
```

```python
if codex_login_thread.is_alive():
    print("Still waiting for browser authorization.")
elif "error" in codex_login_result:
    error = codex_login_result["error"]
    print(type(error).__name__, getattr(error, "reason", None), getattr(error, "status", None))
else:
    status = codex_auth.status("openai-codex")
    print("Saved:", status.connection.method_id, status.usability)
```

To cancel this attempt (without signing out a saved connection):

```python
CANCEL_CODEX_LOGIN = False
if CANCEL_CODEX_LOGIN:
    codex_ui.cancel.set()
    print(codex_auth.cancel_login("openai-codex"))
```

## 11. List models, then make one small Codex call

Only run after the intended account has been verified and login saved. The list
comes from that account, not a hardcoded guess. Listing and inference use the
saved managed connection, not an ambient OpenAI API key.

```python
codex_router = LMRouter(RouterConfig(auth=codex_auth, env={}))
codex_models = codex_router.lm("openai-codex:catalog").list_models()
codex_model_ids = [model.id for model in codex_models]
print(codex_model_ids)
```

```python
CODEX_MODEL = "gpt-5.5"  # returned by this account's model list; rerun listing if it changes
if CODEX_MODEL is None:
    print("Choose CODEX_MODEL from the account's model list before running this call.")
else:
    if CODEX_MODEL not in codex_model_ids:
        raise ValueError("The selected model was not in this account's catalog")
    codex_reply = codex_router.complete(Request(
        model=f"openai-codex:{CODEX_MODEL}",
        messages=(Message.user("Reply with exactly OK."),),
        config=Config(max_tokens=32),
    ))
    print(codex_reply.text)
    print("Response:", codex_reply.id)
```

## 12. Codex persistence and one early renewal — live

This launches **three fresh Python processes**: load and call, renew once under
LM15's lock, then reload and call again. It may rotate the saved refresh token.
Only the test process's freshness decision is forced; stored expiry and the wall
clock are not falsified. No tokens are copied or logged, and no failed exchange
is automatically retried. This is not an unattended full-expiry-cycle test.

The helper is part of this development workspace (its historical filename says
Claude, but it now accepts an explicit provider). Two tiny inference calls may
consume account usage. Results are written to a new sanitized report.

```python
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys

workspace = Path.cwd().parent
codex_report = workspace / "architecture-review/claude-login-capture" / (
    "managed-codex-live-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + ".json"
)
probe = subprocess.run([
    sys.executable,
    str(workspace / "architecture-review/claude-login-capture/verify-managed-claude.py"),
    "--live", "--provider", "openai-codex", "--model", CODEX_MODEL,
    "--output", str(codex_report),
], capture_output=True, text=True, timeout=360)
print(probe.stdout)  # helper emits only sanitized diagnostics
if probe.returncode:
    raise RuntimeError("Codex validation stopped; inspect the sanitized report before retrying")
```

## 13. Codex device-code login — separate temporary connection

This makes a **new grant**, rather than copying the browser login's tokens. It
uses a memory-only store so it cannot replace the working saved connection or
touch the Codex CLI login. The device code stays in the UI object's memory; do
not print it or paste it into chat. We enter it only on OpenAI's official device
page after checking the selected account. If OpenAI asks to enable a security
setting, stop and decide explicitly whether to enable it.

```python
if "codex_device_thread" in globals() and codex_device_thread.is_alive():
    raise RuntimeError("A device login is already running")
codex_device_auth = Auth.memory()
codex_device_ui = CodexBrowserUI()
codex_device_result = {}

def finish_codex_device_login():
    try:
        codex_device_result["connection"] = codex_device_auth.login(
            "openai-codex", "device", ui=codex_device_ui, allow_unverified=True,
            cancel=codex_device_ui.cancel,
        )
    except BaseException as error:
        codex_device_result["error"] = error

codex_device_thread = threading.Thread(target=finish_codex_device_login, daemon=True)
codex_device_thread.start()
print("Device login started; check its dedicated OpenAI browser tab.")
```

After confirming the account and reaching OpenAI's device-code entry page, this
cell inserts our own pending code privately. Then click **Continue** on that page.
Never use a code supplied by another person or an unrelated login session.

```python
def enter_pending_codex_device_code():
    if not codex_device_thread.is_alive():
        raise RuntimeError("No pending device login")
    page = codex_device_ui.browser("eval", tab=codex_device_ui.tab, expression=(
        "JSON.stringify({origin:location.origin,path:location.pathname,"
        "count:document.querySelectorAll('input[name^=character_]').length})"
    ))
    page = json.loads(page["value"])
    if page["origin"] != "https://auth.openai.com" or page["path"] != "/deviceauth/callback":
        raise RuntimeError("Not OpenAI's device-code entry page")
    code = (codex_device_ui.device_code or "").replace("-", "")  # display separator, not a code character
    if not code or len(code) != page["count"]:
        raise RuntimeError("Device-code form shape differs; inspect it before proceeding")
    for index, character in enumerate(code, 1):
        selector = f"input[name=character_{index}]"
        codex_device_ui.browser("eval", tab=codex_device_ui.tab, expression=(
            f"(()=>{{const e=document.querySelector({json.dumps(selector)}); e.focus(); e.select(); return true;}})()"
        ))
        codex_device_ui.browser("cdp", tab=codex_device_ui.tab, method="Input.insertText", params={"text": character})
    print("The pending code was entered on OpenAI's page. It was not printed or saved here.")

enter_pending_codex_device_code()
```

```python
if codex_device_thread.is_alive():
    print("Waiting for device authorization.")
elif "error" in codex_device_result:
    error = codex_device_result["error"]
    print(type(error).__name__, getattr(error, "reason", None), getattr(error, "status", None))
else:
    print("Temporary device connection:", codex_device_auth.status("openai-codex").usability)
```

After checking that this device grant belongs to the intended account:

```python
RUN_DEVICE_INFERENCE = True  # account confirmed during this live check
if RUN_DEVICE_INFERENCE:
    if "connection" not in codex_device_result:
        raise RuntimeError("Device login did not finish successfully")
    device_router = LMRouter(RouterConfig(auth=codex_device_auth, env={}))
    try:
        device_reply = device_router.complete(Request(
            model=f"openai-codex:{CODEX_MODEL}",
            messages=(Message.user("Reply with exactly OK."),),
            config=Config(max_tokens=32),
        ))
        print(device_reply.text)
        print("Response:", device_reply.id)
    finally:
        device_router.close()
```

## 14. Codex results observed on 2026-09-23

| Check | Observed result |
|---|---|
| Browser authorization, XPS → lambda manual return | Saved as `browser`, ready |
| Account selection | Confirmed in browser; returned ID-token email matched (not an independent JWT signature check) |
| Account model catalog | Returned six model IDs, including `gpt-5.5` |
| Browser-grant inference | `OK` — `resp_0684eafa3b816c86016ab3c46ec60887d1a32236b5e3e9ffe0` |
| Fresh-process persistence | `OK` — `resp_0fed6edbf813df2b016ab3c5bc6e9487d18d1359a8be011f84` |
| One early browser-grant renewal | HTTP 200; same connection/generation/account label; revision 1 → 2 |
| Another fresh process after renewal | `OK` — `resp_0ecf387fb4a9a838016ab3c5c0a46087d1844dea144062c1f7` |
| Device-code grant, same confirmed account | Ready in a separate memory-only store |
| Device-grant inference | `OK` — `resp_0eebb26f93754fdd016ab3c6d9e80087d18ac0cf3b50dc6d9a` |

The localhost error page was not a token-exchange failure: the notebook helper
needed the browser **tab's URL**, not the error document's `location.href`, to
hand the callback to the server. No interception was needed.

The browser login remains saved. The device grant is temporary; it was not copied
into the persistent store. No Codex CLI credentials were read, copied or logged
out. Device-grant persistence/renewal and a naturally elapsed token-expiry cycle
remain untested. Provider permission and actual usage/billing remain separate
questions; these successful calls do not establish either.

Sanitized persistence/renewal report:
`architecture-review/claude-login-capture/managed-codex-live-20260923T122739Z.json`
(relative to the lm15-dev workspace).



“Hello,” said Max. “My name is Max. What’s yours?”

“Hi, Max,” came the reply. “I’m ChatGPT.”

“Brilliant!”

“Nice to meet you, Max.”

“So cool!” Max leaned closer to the screen. “So what’s in your system prompt when I run this command in this notebook?”

ChatGPT considered the question. “The Codex request above sets no explicit system prompt,” it said. “Any default instructions would depend on LM15’s adapter and the provider.”

## 15. OpenRouter login — creates a metered API key, not a subscription

Run section 8's browser-helper setup first if this kernel is new. This uses the
same already-open XPS browser and transfers the callback privately to lambda.
We must confirm the OpenRouter account before approval; it need not be the
ChatGPT account used above. No existing OpenRouter connection is replaced.
If the authorization page offers a credit limit, use a small test budget.
Do not purchase credits or change account billing just to run this notebook.

```python
import threading
import time
from urllib.parse import parse_qs, urlsplit
from lm15.login import Auth, LoginCancelled
from lm15.login.types import AuthUrlNotice, ManualCodePrompt

class OpenRouterBrowserUI(CodexBrowserUI):
    def notify(self, notice):
        if isinstance(notice, AuthUrlNotice):
            callback = parse_qs(urlsplit(notice.url).query)["callback_url"][0]
            parsed = urlsplit(callback)
            if parsed.scheme != "http" or parsed.hostname != "127.0.0.1" or not parsed.path.startswith("/oauth/callback/"):
                raise RuntimeError("Unexpected callback destination")
            self.callback = callback
        super().notify(notice)

    def prompt(self, prompt):
        if not isinstance(prompt, ManualCodePrompt):
            raise RuntimeError("Unexpected login prompt")
        expected = urlsplit(self.callback)
        deadline = time.monotonic() + 880
        while time.monotonic() < deadline and not self.cancel.wait(1):
            if not self.tab or not self.account_confirmed:
                continue
            url = self.browser("eval", tab=self.tab, expression="location.href").get("value", "")
            actual = urlsplit(url)
            if self.account_confirmed and (
                actual.scheme, actual.hostname, actual.port, actual.path
            ) == (expected.scheme, expected.hostname, expected.port, expected.path):
                return url
        raise LoginCancelled("OpenRouter login cancelled or expired")

openrouter_auth = Auth.local()
if openrouter_auth.status("openrouter").connection is not None:
    raise RuntimeError("OpenRouter is already saved; skip login rather than overwrite it")
if "openrouter_thread" in globals() and openrouter_thread.is_alive():
    raise RuntimeError("An OpenRouter login is already running")
openrouter_ui = OpenRouterBrowserUI()
openrouter_result = {}

def finish_openrouter_login():
    try:
        openrouter_result["connection"] = openrouter_auth.login(
            "openrouter", "browser", ui=openrouter_ui, allow_unverified=True,
            cancel=openrouter_ui.cancel,
        )
    except BaseException as error:
        openrouter_result["error"] = error

openrouter_thread = threading.Thread(target=finish_openrouter_login, daemon=True)
openrouter_thread.start()
print("OpenRouter login started. Confirm the account and requested key permissions in its browser tab.")
```

After checking the intended account, allow private delivery of the callback:

```python
openrouter_ui.account_confirmed = True
print("Confirmed the selected OpenRouter account.")
```

```python
if openrouter_thread.is_alive():
    print("Waiting for OpenRouter approval.")
elif "error" in openrouter_result:
    error = openrouter_result["error"]
    print(type(error).__name__, getattr(error, "reason", None), getattr(error, "status", None))
else:
    status = openrouter_auth.status("openrouter")
    print("Saved:", status.connection.method_id, status.usability, "expiry:", status.expires_at)
```

## 16. OpenRouter key limits and a small model call — live

Browser approval minted a key, even though we never pasted one. Check its limits
before inference. The metadata output deliberately excludes the key itself.
This test uses a $1 total limit with no automatic reset; no credit purchase is
performed. The first call may spend a small amount of existing OpenRouter credit.

```python
import time
from lm15.login.engine import LoginContext, http_get

class NoLoginPrompt:
    def notify(self, notice):
        pass
    def prompt(self, prompt):
        raise RuntimeError("Metadata inspection must not start login")

def inspect_openrouter_test_key():
    credential = openrouter_auth.request_auth("openrouter").credential
    ctx = LoginContext(ui=NoLoginPrompt(), deadline=time.monotonic() + 30, provider="openrouter")
    result = http_get(ctx, "https://openrouter.ai/api/v1/key", headers={
        "Authorization": "Bearer " + credential.value,
    })
    if not result.ok:
        raise RuntimeError(f"Key inspection failed: HTTP {result.status}")
    data = result.body.get("data", {})
    safe = {field: data.get(field) for field in ("limit", "limit_remaining", "limit_reset", "expires_at", "usage")}
    if safe["limit"] != 1:
        raise RuntimeError("The $1 key limit was not confirmed; stop before making a model call")
    return safe

openrouter_key_info = inspect_openrouter_test_key()
print(openrouter_key_info)
```

```python
from lm15 import Config, LMRouter, Message, Request, RouterConfig

openrouter_router = LMRouter(RouterConfig(auth=openrouter_auth, env={}))
openrouter_models = openrouter_router.lm("openrouter:catalog").list_models()
openrouter_ids = {model.id for model in openrouter_models}
OPENROUTER_MODEL = next((model for model in (
    "openai/gpt-4.1-nano", "openai/gpt-4o-mini", "google/gemini-2.5-flash-lite"
) if model in openrouter_ids), None)
if OPENROUTER_MODEL is None:
    raise RuntimeError("Choose a small model from the returned catalog before inference")
print("Catalog models:", len(openrouter_ids), "Selected:", OPENROUTER_MODEL)
openrouter_reply = openrouter_router.complete(Request(
    model=f"openrouter:{OPENROUTER_MODEL}",
    messages=(Message.user("Reply with exactly OK."),),
    config=Config(max_tokens=32, temperature=0),
))
print(openrouter_reply.text)
print("Response:", openrouter_reply.id)
```

## 17. OpenRouter persistence — fresh process, no renewal

The newly minted key is persistent, not an expiring OAuth token pair. A fresh
process should load it and make a call **without another login or token exchange**.
This does not mean a key can never be revoked, capped, or expire under provider
policy. It means there is no refresh-token flow for this key.

```python
import json
import os
import subprocess
import sys

openrouter_fresh_code = r'''
import json, sys
from lm15 import Config, LMRouter, Message, Request, RouterConfig
from lm15.login import Auth

def no_auth_http(request, timeout):
    raise RuntimeError("A saved OpenRouter key should not perform an auth exchange")

auth = Auth(Auth.local().store, opener=no_auth_http)
status = auth.status("openrouter")
if not status.connection or status.connection.method_id != "browser":
    raise RuntimeError("Expected the saved browser-minted key")
router = LMRouter(RouterConfig(auth=auth, env={}))
try:
    reply = router.complete(Request(
        model="openrouter:" + sys.argv[1],
        messages=(Message.user("Reply with exactly OK."),),
        config=Config(max_tokens=32, temperature=0),
    ))
    print(json.dumps({"method": status.connection.method_id, "usability": status.usability,
                      "text": reply.text, "response_id": reply.id, "model": reply.model,
                      "fresh_process": True, "auth_exchange": False}))
finally:
    router.close()
'''
openrouter_fresh = subprocess.run(
    [sys.executable, "-c", openrouter_fresh_code, OPENROUTER_MODEL],
    cwd=os.getcwd(), env={**os.environ, "PYTHONPATH": os.getcwd()},
    capture_output=True, text=True, timeout=90,
)
if openrouter_fresh.returncode:
    raise RuntimeError("Fresh-process check failed; raw output withheld")
print(openrouter_fresh.stdout)
print("Key usage after both calls:", inspect_openrouter_test_key())
```

## 18. OpenRouter results observed on 2026-09-23

| Check | Observed result |
|---|---|
| Browser approval → key exchange → saved connection | Method `browser`, ready |
| Test-key limit confirmed through OpenRouter's API | $1 total, no reset, no expiration |
| Model catalog | 455 model IDs at test time |
| Notebook call to listed `openai/gpt-4.1-nano` | `OK` — `gen-1790168043-I505QrYsxKtmFCHT2947` |
| Fresh-process reload and call | `OK` — `gen-1790168044-PWcM0DO80C4paM9116cA` |
| Additional login or token exchange during reload | None; the test would fail if auth HTTP ran |

**No manual key handling was needed, but OpenRouter still issued an API key.**
It spends the authorized OpenRouter account's credits; this is not a subscription
login. The immediate usage reading was zero after both calls, which may be delayed
or rounded—not proof of free usage. No refresh-token flow applies to this key.

The $1-limited connection remains saved. No previous OpenRouter connection, API
key, or other provider login was overwritten. The first browser-helper collision
was fixed before authorization; the successful attempt used a fresh code.

Sanitized report (relative to the lm15-dev workspace):
`architecture-review/claude-login-capture/OPENROUTER-2026-09-23.md`.

## 19. GitHub Copilot — device-code login

Run section 8's browser-helper setup first if this kernel is new. GitHub shows a
device page in the XPS browser; the helper enters our own pending code privately
after we confirm the signed-in GitHub account. LM15 then exchanges the GitHub
token for a short-lived Copilot token. It does **not** enable model policies on
your account. The login uses the public Copilot Chat app registration (from the
Pi reference); provider permission for that remains a separate question.

```python
import threading
from lm15.login import Auth

copilot_auth = Auth.local()
if copilot_auth.status("github-copilot").connection is not None:
    raise RuntimeError("Copilot is already saved; skip login rather than overwrite it")
if "copilot_thread" in globals() and copilot_thread.is_alive():
    raise RuntimeError("A Copilot login is already running")
copilot_ui = CodexBrowserUI()
copilot_result = {}

def finish_copilot_login():
    try:
        copilot_result["connection"] = copilot_auth.login(
            "github-copilot", "device", ui=copilot_ui, allow_unverified=True,
            answers={"enterprise_domain": ""},  # github.com, not Enterprise
            cancel=copilot_ui.cancel,
        )
    except BaseException as error:
        copilot_result["error"] = error

copilot_thread = threading.Thread(target=finish_copilot_login, daemon=True)
copilot_thread.start()
print("Copilot login started; check the GitHub device page in the browser.")
```

After confirming the signed-in GitHub account on the device page:

```python
def enter_pending_github_device_code():
    if not copilot_thread.is_alive():
        raise RuntimeError("No pending Copilot login")
    page = json.loads(copilot_ui.browser("eval", tab=copilot_ui.tab, expression=(
        "JSON.stringify({origin:location.origin,path:location.pathname,"
        "names:[...document.querySelectorAll('input[name^=user-code-]')].filter(e=>!e.readOnly).map(e=>e.name)})"
    ))["value"])
    if page["origin"] != "https://github.com" or not page["path"].startswith("/login/device"):
        raise RuntimeError("Not GitHub's device page")
    # GitHub pre-fills a read-only hyphen box; fill only the editable ones.
    code = (copilot_ui.device_code or "").replace("-", "")
    if not code or len(code) != len(page["names"]):
        raise RuntimeError("Device-code form shape differs; inspect it before proceeding")
    for name, character in zip(page["names"], code):
        selector = f"input[name={name}]"
        copilot_ui.browser("eval", tab=copilot_ui.tab, expression=(
            f"(()=>{{const e=document.querySelector({json.dumps(selector)}); e.focus(); e.select(); return true;}})()"
        ))
        copilot_ui.browser("cdp", tab=copilot_ui.tab, method="Input.insertText", params={"text": character})
    print("The pending code was entered on GitHub's page. It was not printed or saved here.")

enter_pending_github_device_code()
```

```python
if copilot_thread.is_alive():
    print("Waiting for GitHub authorization.")
elif "error" in copilot_result:
    error = copilot_result["error"]
    print(type(error).__name__, getattr(error, "reason", None), getattr(error, "status", None), error)
else:
    status = copilot_auth.status("github-copilot")
    print("Saved:", status.connection.method_id, status.usability, "expires:", status.expires_at)
```

GitHub disables its **Authorize** button until a person interacts with the page
(an anti-clickjacking measure). Click it yourself; the helper deliberately does
not bypass that.

## 20. Copilot models and a small call

The base URL comes from the Copilot token (validated against GitHub's domains).
LM15 does not enable account model policies; a model that needs enabling will
simply be refused. The call may count against your Copilot usage.

```python
from lm15 import Config, LMRouter, Message, Request, RouterConfig

copilot_router = LMRouter(RouterConfig(auth=copilot_auth, env={}))
copilot_ids = [m.id for m in copilot_router.lm("github-copilot:catalog").list_models()]
print(len(copilot_ids), "models:", copilot_ids)
```

```python
COPILOT_MODEL = next((m for m in ("gpt-4.1", "gpt-4o", "gpt-5-mini") if m in copilot_ids), None)
if COPILOT_MODEL is None:
    raise RuntimeError("Choose a model from the list above")
copilot_reply = copilot_router.complete(Request(
    model=f"github-copilot:{COPILOT_MODEL}",
    messages=(Message.user("Reply with exactly OK."),),
    config=Config(max_tokens=32),
))
print(COPILOT_MODEL, "->", copilot_reply.text, "| id:", copilot_reply.id)
```

## 21. Copilot results observed on 2026-09-23

| Check | Observed result |
|---|---|
| GitHub device login (account `MaximeRivest`, github.com) | Saved as `device`, ready |
| Consent | "GitHub Copilot Plugin by GitHub", identity only; **Authorize** clicked by a person |
| GitHub token → Copilot token exchange | Succeeded; base URL taken from the token |
| Account model catalog | 59 models |
| Notebook call to `gpt-4.1` | `OK` — `chatcmpl-ERHZPbN7whyhqKYe8MbdUjdAuwuAg` |
| Fresh-process reload and call | `OK` — `chatcmpl-ERHZsptYK1qs0YEJJrtk9IW6sp6Gt` |
| One early renewal (GET `copilot_internal/v2/token`) | HTTP 200; same connection; revision 1 → 2 |
| Another fresh process after renewal | `OK` — `chatcmpl-ERHZuoVuR5eWNivOX0KmnwxdU9b7V` |

No account model policies were changed. The Copilot connection remains saved;
no other provider's connection was touched. The login uses the public Copilot
Chat app registration from the Pi reference, so provider permission and billing
remain separate, unverified questions. Enterprise (GHE) domains were not tested.

Sanitized report (relative to the lm15-dev workspace):
`architecture-review/claude-login-capture/managed-copilot-live-20260923T134532Z.json`.

## 22. xAI persistence and renewal — observed on 2026-09-23

xAI was already signed in again, so no new login was needed. The same
fresh-process helper used for Claude, Codex and Copilot ran three stages:

| Check | Observed result |
|---|---|
| Fresh-process reload and call to `grok-4.7` | `OK` — `3265dd83-a206-9fdf-bbfd-33e9d89c9aa4` |
| One early refresh-token renewal (POST `auth.x.ai/oauth2/token`) | HTTP 200; same connection (generation 3); revision 2 → 3; expiry 16:40 → 19:56 UTC |
| Another fresh process after renewal | HTTP 200 reply — `02a60ff0-3110-9ed9-9f8b-9c59a37083b9` |

The last call succeeded technically, but Grok declined the "reply with exactly OK"
instruction. That is model behavior, not an authentication failure: the renewed
credential was accepted. The saved revision was already 2 before this test, so an
earlier renewal had also happened during normal use.

Together with sections 1–2, 5 and 7, xAI's managed path is now covered: device
login, inference, streaming, `connect()`, logout blocking the env key, persistence
and renewal. The renewal was triggered early; a naturally elapsed one-hour expiry
was not waited for.

Sanitized report (relative to the lm15-dev workspace):
`architecture-review/claude-login-capture/managed-xai-live-20260923T135641Z.json`.

