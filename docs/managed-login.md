# Sign in once, use everywhere (managed login)

`lm15.login` and `lm15.interactive` implement the ratified managed-authentication
contract (lm15-contract `spec/auth.md` AUTH-12–26, 2026-09-22). Python is the
first implementation; TypeScript follows.

**What it gives you.** One place that remembers how you connect to each
provider — a subscription login, a pasted key, "use `$GROQ_API_KEY`", "use my
Claude Code login" — and one rule for which identity a request uses. Nothing is
picked silently; nothing falls back to a paid key behind your back.

**What it is not.** Not a framework, not an agent loop, not a global "current
account". It never starts a login during an ordinary request.

## The short path

```python
from lm15 import Message
from lm15.interactive import connect

with connect() as lm:                      # asks: which connection? which model?
    answer = lm.complete(messages=[Message.user("Explain drought stress.")])
    print(answer.text)
```

`connect()` shows where connections are saved (`~/.config/lm15/credentials.json`
by default), offers saved connections first, then "connect another". For a new
connection it lists providers, then how to connect — subscriptions first, then
keys. If a key is already set in your environment it is *offered* as an explicit
choice, never taken automatically. After a login it fetches the account's model
list and asks which model. It returns a client pinned to that connection and
model.

It refuses to run without a person: on a server with no terminal, attach an
`Auth` to a router instead (below).

## The explicit pieces underneath

```python
from lm15.login import Auth, TerminalUI

auth = Auth.local()                              # reads nothing yet
auth.providers()                                 # what can be connected
auth.methods("xai")                              # how; each says supported / unverified / unavailable

auth.login("xai", "device", ui=TerminalUI())     # device code shown in the terminal; saved on success
auth.set_api_key("groq", "gsk-…")                # a literal key, no verification
auth.configure("gemini", method="env", answers={"name": "GEMINI_API_KEY"})   # use $GEMINI_API_KEY at request time
auth.configure("claude-code", method="external:claude-code-cli")            # use your Claude Code login, read in place

auth.status("xai")                               # saved? ready / renewal_due / needs_login; expiry; no secrets
auth.connections()
auth.verify("xai")                               # explicit non-inference check (lists models); may be metered
auth.logout("xai")                               # local forgetting, remembered across restarts
```

Then either attach it to a router:

```python
from lm15 import LMRouter, RouterConfig

router = LMRouter(RouterConfig(auth=auth))
router.complete(Request(model="xai:grok-4", messages=[Message.user("hi")]))
```

or bind one model:

```python
from lm15.login import BoundClient, ModelSelection

c = auth.status("xai").connection
lm = BoundClient(auth, ModelSelection("xai", "grok-4", c.id, c.identity_generation))
```

## Which identity a request uses

With `RouterConfig(auth=...)` attached, in this order (AUTH-15):

1. An explicit `api_keys` entry for the provider — deliberate authority, always wins.
2. An explicit named cloud identity (`credentials={"azure": "platform"}`).
3. The saved connection for that provider in this scope, renewed if due.
4. For keyless local servers only: the placeholder key.

**Never:** an environment variable, another tool's login file, or the machine's
cloud identity. A missing, expired-unrenewable, rejected or signed-out connection
is an `AuthOperationError` with a `reason` (`login_required`,
`credential_rejected`, `indeterminate`, …) — not a silent switch to a metered key.

Without an `Auth` attached, nothing changes for API-key and cloud users, except
one thing that also changed for the old xAI login path (R3): a subscription that
has failed or been signed out now **blocks** `XAI_API_KEY` instead of quietly
using it, and the error names the login and says the key is used only when
passed explicitly.

## A bound client stays bound

The client `connect()` returns is pinned to one connection *id* and one model.
It follows that connection's token renewals. If you later replace the account or
sign out, the old client fails with `connection_changed` / `login_required`; it
never follows the new identity. Call `connect()` again to use the new one.

## Renewal, precisely

- A token is renewed when it is within `min(5 minutes, lifetime / 10)` of its
  actual expiry — five minutes for an hour-long token, six seconds for a
  one-minute one.
- Renewal runs under the store's cross-process lock. The process that wins
  re-reads the file first; if a sibling already renewed, it uses that result
  instead of spending the refresh token twice.
- A durable "renewal in flight" marker is written before the exchange. If the
  process dies mid-exchange, the next reader sees the marker and reports
  `indeterminate` — you sign in again — rather than replaying a one-use token.
- A provider's definite rejection marks the connection `needs_login` and clears
  the unusable material. A server error or a refused connection keeps everything
  and just fails this request.

## What is saved, and where

One private file (`0600`), `~/.config/lm15/credentials.json` or
`$LM15_CREDENTIALS_PATH`. Provider entries keep the same shape the legacy xAI
login and Pi already use (`{"type": "oauth", "access", "refresh", "expires"}`),
so an xAI login made here is the very entry a plain `LMRouter()` reads — one
copy, one owner. A `_lm15` block holds only non-secret bookkeeping: identity
generations, credential revisions, the renewal marker, the logout marker.

`env` and `external:*` connections store a variable *name* or a source *name*,
never a value. Your Claude Code / Codex CLI files are read and renewed in place
through the same locked loaders the CLI adapters use; nothing is copied out of
them.

`Auth.memory()` keeps everything in the process; `Auth(store)` takes your own
store.

## Which logins are proven

`auth.methods(provider)` says so per method:

| Provider | Method | Status today |
|---|---|---|
| xAI | device code | **supported**: managed device login, inference, streaming, `connect()`, logout blocking, fresh-process persistence and one early renewal observed live (2026-09-22/23) |
| Claude / Codex | your existing CLI login (`external:*`) | **supported** (the proven path since 2026-08-31) |
| Claude | LM15-owned hosted browser login | Login, inference, fresh-process persistence and one early renewal observed **2026-09-23**. Still **unverified** for provider permission and billing; explicit opt-in remains required. |
| ChatGPT / Codex | LM15-owned browser login | Login, catalog, inference, fresh-process persistence and one early renewal observed **2026-09-23**. Provider permission and billing remain **unverified**. |
| ChatGPT / Codex | LM15-owned device login | Separate memory-only login and inference observed **2026-09-23**. Device persistence/renewal were not separately tested; permission and billing remain **unverified**. |
| OpenRouter | Browser approval → API key | Key issuance, $1 limit inspection, model discovery, inference and fresh-process reuse observed **2026-09-23**. No refresh-token flow; broader support review remains pending. |
| GitHub Copilot | Device login → Copilot token | github.com login, catalog, inference, fresh-process persistence and one early renewal observed **2026-09-23**. Uses the public Copilot Chat app registration; permission and billing remain **unverified**. |
| Claude (loopback), Meta, Kimi Code | Other LM15-owned login methods | **unverified**: implemented, no live LM15 receipt; `login(..., allow_unverified=True)` to try |
| Radius | — | **unavailable**: its model protocol is not in lm15-python |

An unverified method is never offered by the default picker. Promotion to
*supported* needs a redacted login → inference → renewal receipt per provider,
recorded in lm15-contract. Login success alone does not prove a subscription
entitles you to anything; that is the provider's decision and your account's.

### Claude login with a browser on another machine

```python
current = auth.status("claude-code").connection
connection = auth.login(
    "claude-code", "browser", ui=TerminalUI(), allow_unverified=True,
    replace=current.id if current else None,
)
```

`browser` opens Claude's hosted code page. Copy its complete `code#state` back
into the waiting prompt; a full hosted return URL is also accepted and checked
against the exact registered destination. No localhost listener is opened, so
Python may be on a server while the browser is on your laptop. A new attempt
needs a new code; it never replays a failed exchange automatically.

The older localhost flow remains available as method `loopback` and still lacks
live LM15 verification. The hosted method has now completed real LM15 login and
inference. Separate fresh Python processes confirmed loading the saved credential,
one successful renewal, and inference after reloading the renewed credential.
Renewal was triggered early with a process-local freshness override; saved expiry
and the wall clock were not falsified. This does not prove an unattended full
expiry cycle, provider permission or included billing. Both methods still require
`allow_unverified=True`, and the existing CLI credential route remains available.

### OpenRouter: no manual key handling, but still a key

```python
auth.login("openrouter", "browser", ui=TerminalUI(), allow_unverified=True)
```

OpenRouter's approval page creates a user-controlled API key for the application.
LM15 exchanges the authorization code and saves the key, so you do not need to
create or paste one yourself. Requests spend OpenRouter credits; this is not
subscription access. The approval page can set a spending limit and expiration.
The observed live test used a $1 total limit, no automatic reset and no expiration.

The saved key worked from a fresh Python process without another authorization
or refresh request. Revoking the key or exhausting its limit still stops access.
Immediate usage reporting may lag, so a zero usage value does not prove calls
were free. Broader conformance review is still pending; explicit opt-in remains.

## Errors

`AuthOperationError` (`code="auth_operation"`) is root-level and never
auto-retried. Match on `.reason`; read `.commit_state` (`committed`,
`not_committed`, `unknown`) before deciding whether the store changed;
`.recovery` says what a person can do. Provider text is never copied into it.

For a failed Claude token exchange, `.status` retains the HTTP status and
`.provider_code` contains only a recognized OAuth error code (otherwise `None`).
The message identifies JSON, HTML, empty or invalid responses. A security
challenge is reported only when the response explicitly marks one; HTML or
HTTP 403 alone does not establish the cause. Unknown provider messages are
intentionally withheld rather than risking credentials in logs. These clues
are not proof of registration permission or subscription entitlement, and do
not authorize retrying a one-use code.

Auth HTTP requests identify this SDK as `lm15/<version>` instead of inheriting
Python urllib's default User-Agent. Provider-specific identifiers explicitly
required by a flow are preserved; LM15 does not retry failed token exchanges
with guessed headers or impersonate another HTTP library.

## Async

`AsyncAuth` wraps the same manager for `async` code: logins are long and
interactive, so each operation runs in a worker thread and is awaited.
`AsyncLMRouter(RouterConfig(auth=async_auth.sync))` uses the same store.
