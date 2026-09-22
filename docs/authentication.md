# Authentication

Every way lm15 takes credentials, from zero-setup to rotating tokens.
Find yourself first — most people need exactly one section:

- **Just trying things out?** Export one env var and you're done —
  [first section](#environment-variables-the-default).
- **Building an app or service?** Pass keys explicitly, from wherever
  you keep secrets — [explicit keys](#explicit-keys).
- **Behind Azure or an enterprise setup where tokens expire?**
  [Rotating credentials](#rotating-credentials-token-providers).
- **On Azure, AWS or Google Cloud?** The three lines that cover a
  laptop, a deployment, and your own credential —
  [Cloud hosts](cloud-hosts.md).
- **On a Claude, ChatGPT, or SuperGrok plan, no API account?**
  [Subscriptions](#subscriptions-claude-code-codex-cli-xai).
- **Want to sign in once and have lm15 remember how you connect to
  each provider?** [Managed login](managed-login.md) — `connect()`,
  `Auth`, and the rule for which identity a request uses.
- **Everything local?** [No key at all](#keyless-local-servers).

One rule sits behind all of them, and it explains everything else on
this page: **lm15 never picks an identity silently. It either receives
a credential and puts it on the wire in the provider's dialect, or it
says which one it picked.** It depends on no auth SDKs —
[the design rationale](design-rationale.md#why-api_key-accepts-a-callable-and-lm15-still-has-no-auth-dependencies)
explains why that restraint is the feature. Where lm15 does obtain a
token itself — the subscription adapters refresh their own local OAuth
credential, and the cloud doors can walk the cloud's own credential
chain or run one named identity — the doctor shows the choice before
any request and every auth error names it afterwards.

## Environment variables (the default)

The router reads each provider's standard variable. The adapter
classes themselves **never** touch the environment — env pickup happens
only in the router, explicitly and inspectably.

| provider string | env var | get a key at |
|---|---|---|
| `openai`, `openai-chat` | `OPENAI_API_KEY` | platform.openai.com/api-keys |
| `anthropic` | `ANTHROPIC_API_KEY` | console.anthropic.com |
| `gemini` | `GEMINI_API_KEY`, then `GOOGLE_API_KEY` | aistudio.google.com/apikey |
| `xai` | `XAI_API_KEY` (used only when no subscription login is stored) | console.x.ai |
| `groq` | `GROQ_API_KEY` | console.groq.com/keys |
| `openrouter` | `OPENROUTER_API_KEY` | openrouter.ai/keys |
| `deepseek`, `deepseek-anthropic` | `DEEPSEEK_API_KEY` | platform.deepseek.com/api_keys |
| `zai` | `ZAI_API_KEY` | z.ai/manage-apikey/apikey-list |
| `moonshotai`, `moonshotai-responses`, `moonshotai-anthropic` | `MOONSHOTAI_API_KEY`, then `MOONSHOT_API_KEY` (the name Moonshot's docs use; both are read, the first wins) | platform.kimi.ai/console/api-keys |
| `meta`, `meta-chat`, `meta-anthropic` | `META_API_KEY` (Meta's docs call it `MODEL_API_KEY`; lm15 reads only the vendor-named one) | dev.meta.ai |
| `ollama`, `vllm`, `sglang` | — (keyless, placeholder sent) | — |
| `claude-code`, `openai-codex` | — (local CLI credential) | — |

`resolve()` records *which* variable would be read, never the value:

```python
print(LMRouter().resolve("claude-haiku-4-5"))
```

```output
'claude-haiku-4-5' -> provider 'anthropic' (AnthropicLM); via built-in rule prefix='claude-' — Anthropic Claude family; wire model 'claude-haiku-4-5'; key from $ANTHROPIC_API_KEY.
```

## Explicit keys

When keys live in a secrets manager rather than the shell, pass them
yourself — an explicit key always beats the environment, and passing
`env={}` as well guarantees the environment is never even looked at
(useful in tests, and it is how lm15's own test suite runs):

```python
from lm15 import AnthropicLM, LMRouter, RouterConfig

router = LMRouter(RouterConfig(env={}, api_keys={"anthropic": "sk-ant-..."}))

lm = AnthropicLM(api_key="sk-ant-...")   # or skip the router entirely
```

`api_keys` is repr-suppressed, like every credential field in lm15.

## Rotating credentials (token providers)

Some credentials aren't static strings. Azure Entra tokens expire
after about an hour; enterprises rotate keys on a schedule; OAuth
tokens refresh themselves. If your process runs for days, a key read
once at startup *will* eventually be stale — and the failure shows up
as a confusing 401 at 3 a.m.

lm15's answer: anywhere a key string goes, a **zero-argument callable
returning a string** works too. The adapter calls it when it builds
each request, so whatever your callable returns *now* is what goes on
the wire *now*. Refreshing and caching stay your callable's business —
which means the ecosystem's existing token providers plug in
unchanged:

```python
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from lm15 import LMRouter, Request
from lm15.credentials import BearerToken
from lm15.router import RouterConfig

provider = get_bearer_token_provider(          # returns () -> str, cached and refreshed inside
    DefaultAzureCredential(),
    "https://cognitiveservices.azure.com/.default",
)
router = LMRouter(RouterConfig(
    api_keys={"azure": lambda: BearerToken(provider())},
    settings={"azure": {"resource": "YOUR-RESOURCE"}},
))
router.complete(Request(model="azure:YOUR-DEPLOYMENT", messages=[...]))
```

The `BearerToken(...)` wrap is optional here. A plain string means "API
key", and the Azure doors put an API key in the `api-key` header — but a
JWT is never an API key on any door lm15 has, so a token-provider that
returns a bare JWT string (`azure.identity.get_bearer_token_provider`
does) is sent as a bearer token, and the doctor says so
(`api_keys={"azure": provider}` is enough). Keep the wrap when nothing
should be read from a token's shape. The scope every Azure door defaults
to is `https://ai.azure.com/.default`; the classic
`https://cognitiveservices.azure.com/.default` is also accepted.

You often do not need a provider at all on Azure, AWS or Google Cloud:
with no key set, the router walks the cloud's own default chain (`az
login`, managed identity, a service principal from `AZURE_*` variables;
boto3's and google-auth's orders likewise) with no extra dependency, and
for a deployment `RouterConfig(credentials={"azure": "platform"})` names
one identity and never falls through. See [cloud hosts](cloud-hosts.md).

Or your own logic — anything callable:

```python
def credential() -> str:
    return read_key_from_your_vault()

router = LMRouter(RouterConfig(api_keys={"anthropic": credential}))
```

## Subscriptions (Claude Code / Codex CLI / xAI)

If you use Claude Code or the Codex CLI, you already have working
credentials on disk — no API platform account needed. These adapters
use that login directly, and usage bills to your existing subscription
plan rather than pay-per-token:

```python
from lm15 import ClaudeCodeLM, OpenAICodexLM

lm = ClaudeCodeLM()      # ~/.claude/.credentials.json
lm = OpenAICodexLM()     # ~/.codex/auth.json
```

The credential is validated at construction (missing or unrefreshable
credentials raise typed errors that say which CLI login to run), then
re-resolved on every request: tokens refreshed on disk — by the CLI, by
another process, or by lm15's own expiry refresh — are picked up
without rebuilding the client. Both adapters are also routable
(`claude-code:...`, `openai-codex:...`).

xAI sells subscription access too (SuperGrok / X Premium), but ships
no CLI credential file — so lm15 runs the device-code login itself,
once, and stores the credential locally:

```python
from lm15.auth import login

login("xai")             # prints a URL + code; finishes in the browser
```

`login()` is the one door for every provider. It runs the flow lm15 owns
(today: xAI) and, for everyone else, raises a typed error naming the
exact fix — `claude` `/login` or `codex login` for the CLI-owned
subscriptions, or the console URL where the provider's API key is
created. (`login_xai()` remains as the concrete flow underneath.)

After that, `XaiLM()` (and `grok-*` model strings through the router)
work with no key, and the stored login **outranks** `XAI_API_KEY`: the
subscription spends no money per token, and normal inference must never
unexpectedly spend money. Only truly explicit configuration — an
`api_key=` argument or a `RouterConfig(api_keys={"xai": ...})` entry —
beats the stored login, because an instruction you write in this process
must always win. Refreshed tokens are written back atomically with
owner-only permissions.

**Stated trade-off:** while a subscription login is stored, a set
`XAI_API_KEY` is silently ignored (unusual — most SDKs let env vars win).
If you need that key's account — team billing, its rate limits — pass it
explicitly. When in doubt, run `lm15.doctor.explain_auth("xai")`: it
shows exactly which credential rung won and which were shadowed.

**And after the login fails or is signed out** (ratified 2026-09-22,
R3): the stored login *blocks* `XAI_API_KEY` rather than falling back to
it. An expired login with no refresh token, or a login you signed out of
with `Auth.logout("xai")`, makes `xai:` requests fail with an error that
names the login and says the key is used only when passed explicitly.
Never having logged in leaves the key path exactly as it was. The
managed side of all this — `connect()`, saved keys, "use my Claude Code
login" as an explicit choice — is on [Managed login](managed-login.md).

### What a subscription adapter is

`ClaudeCodeLM` and `OpenAICodexLM` are names, not separate implementations.
Each is a dialect adapter bound to an **access policy**: a plain value that
says which credential travels in which header, which static headers ride
along, which endpoint surfaces the login carries, and which backend
variant the dialect must switch on. The same wire comes out of the long
form:

```python
from lm15 import AnthropicLM, OpenAILM
from lm15.access import CLAUDE_CODE, OPENAI_CODEX

lm = AnthropicLM(access=CLAUDE_CODE)            # == ClaudeCodeLM()
lm = OpenAILM(access=OPENAI_CODEX)              # == OpenAICodexLM()
print(lm.access.provider, lm.supports.files)    # 'openai-codex' False
```

This is what lets a Go or Rust port carry the same facts as a table
instead of a class hierarchy. The policy table and its consult points are
normative (contract `spec/auth.md` AUTH-10). Custom policies are ordinary
values too — `CLAUDE_CODE.with_headers({...})` is how `claude_code_version`
is applied.

## Keyless local servers

`ollama:`, `vllm:`, and `sglang:` model strings need no configuration —
the router sends the placeholder these servers expect (`"ollama"` /
`"EMPTY"`). If your local server *is* locked down, set the key
explicitly via `api_keys`.

## What lm15 guarantees

Worth knowing before a security review asks:

- **Your key is never printed.** `print()` an adapter, log an error,
  render a traceback — credential material appears in none of them,
  whichever form it takes.
- **Nothing is discovered behind your back.** Adapters read only what
  you passed. Env pickup happens in exactly one place (the router),
  `resolve()` tells you which variable it would use, and on a cloud
  door `explain_auth()` shows the walk (or the one named identity)
  before any request. Every auth error from the wire names where the
  credential came from — the rung, the variable, "an explicit
  api_key", or "an application-supplied callable" — never its value.
- **The right header, every time.** Where each credential goes on the
  wire (`Authorization: Bearer`, `x-api-key`, `x-goog-api-key`, …) is
  contract-tested per provider —
  [How lm15 is specified](how-lm15-is-specified.md).
- **Zero auth dependencies, forever.** For Azure/Bedrock/Vertex-style
  delegated auth you bring the token provider (`azure.identity`,
  `boto3`, `google-auth` plug into `api_key=` unchanged), or name one
  identity and let lm15 speak the metadata server and token endpoints
  directly with the standard library — either way lm15 provides the
  seam, not a second identity framework.

## When it goes wrong

Credential problems surface early and loudly: a missing key fails when
you *build* the client, not twenty minutes into a batch run. The
errors are typed (`MissingCredentialError` from the router,
`NotConfiguredError` / `AuthError` from adapters — all catchable under
`LM15Error`), and each one states its own fix:

```output
MissingCredentialError: no API key found for provider 'anthropic'.
Set ANTHROPIC_API_KEY in the environment, or pass
RouterConfig(api_keys={'anthropic': "..."}).
```

Subscription errors carry the re-login hint instead (`run \`claude\`
and use /login`), because there is no env var to set.
