# Cloud hosts: AWS, Azure, Google Cloud

Status: implemented 2026-09-03 from provider documentation and the cloud
SDKs' own resolver sources (spec/auth.md AUTH-1, AUTH-10, AUTH-11;
`changes/2026-09-03-cloud-hosts.md`); named credentials, provenance and
endpoint overrides added 2026-09-19
(`changes/2026-09-19-cloud-identity-and-endpoints.md`). Live status is per
door: Azure OpenAI and Bedrock Chat have receipts; Claude on Foundry was
verified independently by Pamela Fox on 2026-09-18 with an explicit
`azure.identity` credential; `vertex` and `vertex-express` were verified on
2026-09-26 through every Google identity lm15 reads (see [Google Cloud live
status](#google-cloud-live-status-2026-09-26)); other rows remain
documentation-evidenced until their own change entry says otherwise.

A cloud host is a door to a model you already know. The wire is the same
(Anthropic Messages, OpenAI Responses or Chat Completions, Gemini); the
door changes the URL, the signing, a few body rewrites, and the
identity. You send the same `Request` and read the same `Response`.

One rule sits behind the identity half: **lm15 never picks an identity
silently. It either receives a credential, or it says which one it
picked.** Every auth error names where the credential came from; the
doctor says it before any request.

## Pick the line that matches where this code runs

```python
from lm15 import LMRouter, Request, Message
from lm15.router import RouterConfig

router = LMRouter()                                                      # laptop: your az login / aws sso login / gcloud auth
router = LMRouter(RouterConfig(credentials={"azure": "platform"}))       # deployed: the machine's own identity, or fail
router = LMRouter(RouterConfig(api_keys={"azure": my_token_provider}))   # your own credential (azure.identity, boto3, a vault)
# endpoint: AZURE_OPENAI_ENDPOINT=https://<account>.services.ai.azure.com, or base_urls={"azure": ...}

router.complete(Request(model="azure:gpt-5-mini", messages=[Message.user("hi")]))
```

Replace `azure` with `bedrock-anthropic` (needs `AWS_REGION`) or `vertex`
(needs a Google Cloud project, which lm15 finds the way Google's own
libraries do — see [Google Cloud, start to finish](#google-cloud-start-to-finish))
and the three lines do not change. The
request code never changes between the laptop and production; only the
connection setup does.

## The doors

| Provider string | Wire | Needs | Credential |
|---|---|---|---|
| `azure:<deployment>` | OpenAI Responses | `AZURE_OPENAI_RESOURCE`, or the endpoint `AZURE_OPENAI_ENDPOINT` | `AZURE_OPENAI_API_KEY` or the Azure chain |
| `azure-chat:<deployment>` | OpenAI Chat Completions | same | same |
| `azure-anthropic:<model>` | Anthropic Messages (Foundry) | `ANTHROPIC_FOUNDRY_RESOURCE`, or the endpoint `ANTHROPIC_FOUNDRY_BASE_URL` | `ANTHROPIC_FOUNDRY_API_KEY` or the Azure chain |
| `aws-anthropic:<model>` | Anthropic Messages (Claude Platform on AWS) | `AWS_REGION`, `ANTHROPIC_AWS_WORKSPACE_ID` | `ANTHROPIC_AWS_API_KEY` or the AWS chain (SigV4) |
| `bedrock-anthropic:anthropic.<model>` | Anthropic Messages (Bedrock, Opus 4.7+) | `AWS_REGION` | `AWS_BEARER_TOKEN_BEDROCK` or the AWS chain (SigV4) |
| `bedrock-chat:<model id>` | OpenAI Chat Completions (Bedrock runtime; versioned ids) | `AWS_REGION` | same |
| `bedrock-mantle-chat:<model id>` | OpenAI Chat Completions (Bedrock mantle; un-versioned ids; `list_models()` works) | `AWS_REGION` | same |
| `vertex:<model>` | Gemini | a project: `GOOGLE_CLOUD_PROJECT`, else gcloud's, the credential file's or the metadata server's (`GOOGLE_CLOUD_LOCATION`, default `global`) | the Google chain, or an explicit Vertex API key |
| `vertex-anthropic:<model>` | Anthropic Messages (rawPredict) | same | the Google chain (Claude on Vertex takes no API keys) |
| `vertex-express:<model>` | Gemini | nothing | `GOOGLE_API_KEY` |

On Azure the model string is the **deployment name**; an unknown one is
`UnsupportedModelError` (HTTP 404 `DeploymentNotFound`). `list_models()`
on `azure`/`azure-chat` returns the resource's model *catalog* (every
model Azure could deploy there), not the deployments you can call. `region`
and `resource` have no default: a wrong-region default is a residency bug,
so lm15 raises `NotConfiguredError` naming the variable (and, on Azure,
the endpoint variable that makes `resource` unnecessary).

## Day one: the laptop

You already ran `az login` (or `aws sso login`, or `gcloud auth
application-default login`) for other work. Nothing to configure:

```python
router = LMRouter()   # walks the cloud's own chain
print(router.complete(Request(model="azure:gpt-5-mini", messages=[Message.user("hi")])).text)
```

The router walks boto3's, `DefaultAzureCredential`'s and google-auth's
default chains in their exact order, with an explicit `api_keys` entry
first. This is convenient for experiments: it searches several
credential sources and does not promise which one wins. When it works,
nothing is printed. When it fails, the error says which one it picked:

```output
AuthError: azure: HTTP 401 ...
  credential came from: az account get-access-token, expires in 52 min
```

Before any request, offline:

```python
from lm15.doctor import explain_auth
print(explain_auth("azure"))
```

## Week one: the endpoint

The Foundry console shows `https://<account>.services.ai.azure.com`.
Paste it; the door appends its own path (`/openai/v1`, `/anthropic/v1`)
and keeps everything Azure-specific — the `api-key` header scheme, the
`DeploymentNotFound` and content-filter error mapping, the doctor:

```python
router = LMRouter(RouterConfig(base_urls={"azure": "https://my-account.services.ai.azure.com"}))
router.complete(Request(model="azure:gpt-5-mini", ...))
router.complete(Request(model="azure-chat:Kimi-K2.6", ...))
router.complete(Request(model="azure-anthropic:claude-sonnet-4-5", ...))   # its own variable: ANTHROPIC_FOUNDRY_BASE_URL
```

Or no code at all: the vendor's own variable, `AZURE_OPENAI_ENDPOINT`.
With an endpoint, `resource` is not needed. The endpoint is a root or the
full base — `https://acct.services.ai.azure.com`,
`https://acct.services.ai.azure.com/openai/v1` and Microsoft's
`https://acct.services.ai.azure.com/anthropic` all mean the same door;
the door appends only what is missing.

Which host to paste, on Azure: `services.ai.azure.com` is what the
Foundry console shows and it serves OpenAI and non-OpenAI deployments
alike (Kimi, DeepSeek, Claude); the classic alias
`{resource}.openai.azure.com` saves one internal hop for OpenAI models
today. lm15's *template* stays on the alias because it is the only host
every resource kind answers on — a classic `OpenAI`-kind resource has no
`services.ai.azure.com` name at all, while a Foundry resource has both.
Paste the Foundry root whenever the console gives you one.

AWS: `AWS_ENDPOINT_URL_BEDROCK_RUNTIME` (PrivateLink, FIPS), then the
generic `AWS_ENDPOINT_URL`, exactly as the AWS SDK reads them; the region
is still required because the signature names it. Google: no vendor
endpoint variable lm15 can cite; `base_urls={"vertex": "https://europe-west4-aiplatform.googleapis.com"}`
(or a Private Service Connect host) replaces the host, and project and
location stay in the path.

An endpoint is trusted configuration: it decides where credentials and
data go. Never take it from untrusted input, and a corporate gateway must
actually speak the door's dialect.

## Week two: deployed

The app goes to Azure Container Apps, ECS or Cloud Run. No laptop login
may be in the picture, and a missing identity should fail at startup:

```python
router = LMRouter(RouterConfig(
    credentials={"azure": "platform"},   # one identity; the chain is not walked
    base_urls={"azure": os.environ["AZURE_OPENAI_ENDPOINT"]},
))
```

The same word on every cloud: `credentials={"bedrock-anthropic": "platform"}`
on ECS, `credentials={"vertex": "platform"}` on Cloud Run. A user-assigned
Azure identity is still the vendor's variable (`AZURE_CLIENT_ID`).

| name | Azure | AWS | Google |
|---|---|---|---|
| `"platform"` | managed identity (IMDS / App Service / Arc / Cloud Shell / Azure ML) | the ECS/EKS container endpoint, else the EC2 instance role (IMDSv2) | the attached service account via the metadata server (incl. GKE Workload Identity) |
| `"workload"` | federated token file (AKS workload identity) | `AWS_WEB_IDENTITY_TOKEN_FILE` + role (EKS IRSA) | `GOOGLE_APPLICATION_CREDENTIALS` of type `external_account` (workload identity federation) |
| `"environment"` | service principal from `AZURE_TENANT_ID` / `AZURE_CLIENT_ID` + secret or certificate | `AWS_ACCESS_KEY_ID` + `AWS_SECRET_ACCESS_KEY` | `GOOGLE_APPLICATION_CREDENTIALS` of type `service_account` |
| `"cli"` | `az` / `azd` / `pwsh` | the active `aws` profile (assume-role, SSO, shared files, `aws login`, `credential_process`) | the ADC file (`gcloud auth application-default login`) or `gcloud auth print-access-token` |

A named credential that is absent is an error, never a fall-through:

```output
NotConfiguredError: azure: named credential "platform" — Azure managed identity — answered nothing
  (Azure managed identity: managed identity (imds) probed at request time). This door was told to use
  that identity only; it will not try the rest of the azure-chain chain.
```

Two honest wrinkles, stated: `"platform"` on AWS covers two rungs (the
container endpoint, then IMDS) — both are the machine's own identity,
boto3 tries them in that order, and the doctor says which answered. On
Google, `"workload"` and `"environment"` are the same file told apart by
its `type`; the wrong type is refused by name, never read as the other.

Bare adapters take the same names: `OpenAILM(access=AZURE, credential="platform", base_url=...)`.
A named credential and an `api_key` on one door is refused — two answers
to "who am I".

## Any day: your own credential

A team whose policy says "only `azure.identity`, and only this class".
lm15 does one thing here: put the credential on the wire in the door's
dialect. Refresh and caching belong to the callable.

```python
from azure.identity import ManagedIdentityCredential, get_bearer_token_provider

provider = get_bearer_token_provider(ManagedIdentityCredential(), "https://ai.azure.com/.default")
router = LMRouter(RouterConfig(
    api_keys={"azure": provider},            # a returned JWT travels as a bearer token
    base_urls={"azure": os.environ["AZURE_OPENAI_ENDPOINT"]},
))
```

`api_keys={"azure": lambda: BearerToken(provider())}` is the same thing
spelled out, and the form to use when nothing should be read from a
token's shape. A JWT is never an API key on any door lm15 has, so a
plain JWT string is sent as a bearer token even on the doors whose key
header comes first (`api-key` on Azure OpenAI, `x-api-key` on Foundry);
before 2026-09-19 lm15 refused it and named the wrap.

The Entra scope, per door: `https://ai.azure.com/.default` is what every
Azure door here defaults to (the built-in chain requests it; live on the
OpenAI doors 2026-09-04 and on Foundry Claude by Pamela Fox 2026-09-18);
`https://cognitiveservices.azure.com/.default` is the classic Azure
OpenAI audience and is also accepted. Set it with
`settings={"azure": {"scope": ...}}` when your tenant policy names one.

AWS with boto3's own resolver, and Google with google-auth:

```python
import boto3
from lm15.credentials import AwsCredentials

session = boto3.Session()
def creds():
    c = session.get_credentials().get_frozen_credentials()
    return AwsCredentials(c.access_key, c.secret_key, session_token=c.token)

router = LMRouter(RouterConfig(api_keys={"bedrock-anthropic": creds},
                               settings={"bedrock-anthropic": {"region": "us-east-1"}}))
```

```python
import google.auth, google.auth.transport.requests
creds, project = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
def token():
    creds.refresh(google.auth.transport.requests.Request())
    return creds.token

router = LMRouter(RouterConfig(api_keys={"vertex": token}, settings={"vertex": {"project": project}}))
```

## Month three: gateway, private link, sovereign cloud

```python
router = LMRouter(RouterConfig(
    credentials={"azure": "platform"},
    base_urls={"azure": "https://ai-gw.corp.example/foundry"},     # the door still speaks Azure
    settings={"azure": {"authority_host": "https://login.microsoftonline.us",
                        "scope": "https://cognitiveservices.azure.us/.default"}},
))
```

Everything Azure-specific survives the URL change, because the door and
the URL are separate facts.

## The 3 a.m. page

An auth error from the wire carries where the credential came from — the
rung of a chain, the env variable, "an explicit api_key", or "an
application-supplied callable (identity not inspected by lm15)" — never
the value:

```output
AuthError: bedrock-anthropic: HTTP 403 AccessDeniedException
  credential came from: EC2 instance metadata (IMDSv2), expires in 4 h 12 min
```

And the *predicted* choice on a machine you cannot send requests from:

```python
print(explain_auth("bedrock-anthropic", config=router.config))   # this router's keys, names, endpoint and settings
```

```output
auth for provider 'bedrock-anthropic':
  named credential "platform": the ECS/EKS container endpoint, else the EC2 instance role (IMDSv2) — the chain is not walked
   - explicit api_keys entry: not provided
   - container credentials endpoint: no AWS_CONTAINER_CREDENTIALS_* URI
   ? EC2 instance metadata (IMDSv2): instance metadata probed at request time
  configured: probably — EC2 instance metadata (IMDSv2) (unprobed offline)
  setting region: us-east-1
  base url: https://bedrock-mantle.us-east-1.api.aws/anthropic/v1 (from template)
```

`?` marks a rung the offline doctor cannot decide: it needs the network
or a subprocess, and its configuration is present. Such a rung runs
first at request time and may win over a later `=>` rung. For a
credential you supplied as a callable the doctor is honest rather than
clever: it reports the callable and says the underlying identity is not
inspected.

The chains, rung by rung, are in `spec/auth.md` AUTH-1. Rungs that need
signing use the standard library only: SigV4 is HMAC-SHA256; the Google
service-account and Entra certificate assertions are RS256 signed by a
pure-Python signer (about 75 ms per signature, once per one-hour token).

## Azure live status and exact pending work (2026-09-04)

`azure` (Responses) is live-verified for complete, stream, reasoning
(`gpt-5-mini`, including nonzero reasoning tokens), prompt-cache hits,
content-filter errors and completion stops, model listing, Files, Batch,
text-to-speech, and Realtime text WebSockets. `azure-chat` is live-
verified for complete, stream, reasoning, caching, filtering and models.
Sync and async complete/stream both ran live. The Azure chain ran live by
client secret, certificate, token-provider callable, and its `az` rung.
See `lm15-contract/changes/2026-09-04-azure-live.md` and
`2026-09-04-azure-chat-live.md`. On 2026-09-18 Pamela Fox (Microsoft)
ran `azure-anthropic` (Claude Sonnet 4.5) and the OpenAI doors (GPT,
Kimi K2.6, DeepSeek V4 Flash) against a Foundry account with an explicit
`AzureDeveloperCliCredential` — the "your own credential" path — from
`pamelafox/python-stack-foundry-models` (`examples/lm15_request.py`,
`examples/lm15_router.py`); an independent receipt that lm15 has not
reproduced in its own lab (Claude quota is still 0 there).

Two quota-gated proofs remain:

| Pending proof | Blocker | After Azure removes it |
|---|---|---|
| `azure` image generation | `gpt-image-1-mini` GlobalStandard Requests-Per-Minute quota is 0 in eastus2; quota request submitted | Rerun `research/cloud-hosts/azure/provision.sh`; it creates deployment `gpt-image-1-mini`. Run `python3 research/providers/azure/capture.py --only image --force`, review the image case, then set `AZURE.supports.images` and the support-matrix cell true. |
| `azure-anthropic` successful inference | Claude Haiku 4.5 quota is 0. Microsoft denied eastus2 because that region has no capacity. Host, `x-api-key`, both Entra scopes, secret/certificate auth, `/models` refusal, and error mapping are live-proven; no 200 inference exists. | Obtain 10 capacity units in any listed region. Run the provision command below with that region, then run the full `azure-anthropic` capture, draft/review goldens, and run every harness direction. |

Post-quota Claude command (use the region Azure grants):

```bash
FDY_LOC=westus3 \
LM15_LAB_ORG='lm15-dev (open-source project, Maxime Rivest)' \
LM15_LAB_INDUSTRY='Software & Internet' \
LM15_LAB_COUNTRY=CA \
bash research/cloud-hosts/azure/provision.sh

python3 research/providers/azure-anthropic/capture.py --force
```

Sora/video is deliberately excluded, not pending: the user chose to skip
`sora-2` because the product is going away and video is not a priority.
Azure embeddings answered on `/openai/v1`; transcription answered on
Azure's deployment-scoped `?api-version=2025-04-01-preview` route (the v1
route returned 404). lm15 has no canonical embedding or transcription
surface; both are outside the current library API rather than incomplete
Azure implementations.

## Google Cloud, start to finish

Three doors reach Google's models: `vertex` (Gemini in your project),
`vertex-anthropic` (Claude in your project) and `vertex-express` (Gemini
with only an API key, no project). This section is the whole path, from
an empty account to production.

**Once per project.** In the Cloud console (or with `gcloud`): create a
project with billing, enable the Vertex AI API, and give whoever will
call it the *Vertex AI User* role. A new project, role or service
account answers `403 PERMISSION_DENIED` for a few minutes while Google
applies it; that 403 says so.

```bash
gcloud projects create my-project && gcloud billing projects link my-project --billing-account=<id>
gcloud services enable aiplatform.googleapis.com --project my-project
```

**On your laptop.** Sign in once; lm15 finds both the identity and the
project on its own:

```bash
gcloud auth application-default login
gcloud config set project my-project
```

```python
router = LMRouter()
router.complete(Request(model="vertex:gemini-2.5-flash", messages=[Message.user("hi")]))
```

**Deployed on Google Cloud** (Cloud Run, GKE, a VM, Cloud Functions):
attach a service account with the Vertex AI User role and set nothing.
The identity and the project both come from the metadata server. Name it
to fail at startup if it is missing:

```python
router = LMRouter(RouterConfig(credentials={"vertex": "platform"}))
```

**Deployed elsewhere** (another cloud, GitHub Actions, on-premises): use
workload identity federation, so no long-lived key exists. `gcloud iam
workload-identity-pools create-cred-config ...` writes a file; point
`GOOGLE_APPLICATION_CREDENTIALS` at it and name it
(`credentials={"vertex": "workload"}`). A service-account key file works
the same way (`"environment"`), but it is a secret that never expires:
prefer federation.

**With an API key.** Create a key restricted to the Vertex AI API (Cloud
console > APIs & Services > Credentials; Google now issues keys bound to
a service account, beginning `AQ.`). Two doors take it:

```python
# no project, Google chooses where it runs
router = LMRouter()                        # GOOGLE_API_KEY=... in the environment
router.complete(Request(model="vertex-express:gemini-2.5-flash", ...))

# your project and a region you choose (data residency)
router = LMRouter(RouterConfig(
    api_keys={"vertex": os.environ["MY_VERTEX_KEY"]},
    settings={"vertex": {"location": "europe-west4"}},
))
```

On `vertex` a string is sent as a key (`x-goog-api-key`) unless it looks
like a sign-in token (`ya29.…` or a JWT), which is sent as a token, so the
access token google-auth hands you still works as a plain string. An
access token of any other shape: pass `BearerToken(value)`. `vertex`
never reads `GOOGLE_API_KEY` from the environment: that variable usually
belongs to the Gemini API, and picking it up would silently change who
pays. Claude on Vertex refuses keys.

**Where the project comes from**, first found wins — Google's own order
(google-auth and gcloud):

1. `settings={"vertex": {"project": ...}}`
2. `GOOGLE_CLOUD_PROJECT`, then `GCLOUD_PROJECT`
3. the `GOOGLE_APPLICATION_CREDENTIALS` file's `project_id`
4. gcloud's active configuration: `CLOUDSDK_CORE_PROJECT`, else the
   project `gcloud config set project` saved (named configurations and
   `CLOUDSDK_ACTIVE_CONFIG_NAME` included)
5. the ADC file's `quota_project_id`
6. the metadata server, when running on Google Cloud (asked only if
   nothing above answered; `NO_GCE_CHECK=1` skips it)

The doctor prints which one answered:

```output
  setting project: my-project (from gcloud's active configuration)
  setting location: global (from default)
```

**Claude on Vertex needs quota first.** A new project has 0 requests per
minute for every Claude model: the request signs in and routes, then
Google answers 429 "Quota exceeded". Enable the model in Model Garden and
request quota (IAM & Admin > Quotas). Use a location the model lists; in
our test project `global` routed and `us-east5` said "not found".

**When sign-in fails, the error names the fix**, never API-key advice:

```output
AuthError: Google OAuth refresh (~/.config/gcloud/application_default_credentials.json): HTTP 400 (invalid_grant)

  To fix:
    - the saved Google login in ~/.config/gcloud/application_default_credentials.json has expired
      or was revoked; run `gcloud auth application-default login` (Google ends these sessions on its own schedule)
```

Only the status and a standard OAuth error word come from Google's reply
(AUTH-5, AUTH-21): a reply's free text can echo the request, which holds
the refresh token or a signed key. To see gcloud's own reason, run the
command the error names.

### Google Cloud live status (2026-09-26)

Verified against a fresh project (`lm15-vertex-live`) in Python,
TypeScript, Go and Rust; evidence and recorded cases in
`lm15-contract/changes/2026-09-26-vertex-live.md`. Each row answered a
real `gemini-2.5-flash` request.

| Identity | Result |
|---|---|
| gcloud login (ADC file), project from `gcloud config` | works, complete and stream |
| `gcloud auth print-access-token` alone | works |
| service-account key file (`"environment"`), project from the key | works |
| impersonated service account | works |
| workload identity federation (`"workload"`), direct and via a service account | works |
| attached service account on a VM (`"platform"` and the default chain), no project set anywhere | works in all four languages |
| a plain access-token string | works |
| Vertex API key on `vertex` (global and `europe-west4`, complete and stream) and on `vertex-express` | works |
| `vertex-anthropic` | signs in and routes; answers need Claude quota |

`us` and `eu` multi-region locations route to the documented hosts;
Google answered "model not found" for `gemini-2.5-flash` there, which is
Google's model list, not a host fact.

## Credentials are values, not strings

`api_key=` accepts a string (an API key), an `lm15.credentials` value,
or a zero-argument callable returning either:

```python
from lm15.credentials import AwsCredentials, BearerToken
from lm15.providers import AnthropicLM
from lm15 import access

lm = AnthropicLM(
    api_key=AwsCredentials("AKIA…", "…", session_token="…"),
    access=access.BEDROCK_ANTHROPIC,
    settings={"region": "us-east-1"},
)
```

A door lists the schemes it accepts; the credential kind picks one. AWS
credentials always travel as a SigV4 signature; a bearer token under
`Authorization`, or under the door's key header on the two doors that
carry their tokens there (`bedrock-anthropic`, `aws-anthropic`: a
Bedrock short-term key goes as `x-api-key`); an API key under the door's
key header (`x-api-key`, `api-key`) or, for Vertex express, as `?key=`.
A plain string that is a JWT is read as a bearer token, not an API key
(AUTH-2, amended 2026-09-19). The wrong kind for a door fails at
construction.

A Bedrock short-term API key (`AWS_BEARER_TOKEN_BEDROCK`) works on both
Bedrock doors (live 2026-09-04 on `bedrock-chat`). lm15 does not mint
one: with AWS credentials it signs each request directly, which is what
the key would have done for you. Mint one only for a tool that speaks
bearer only (AWS's `aws-bedrock-token-generator`, or
`lm15-contract/research/providers/_aws_bearer.py`).

## What is not supported, and what to do instead

Each is a `NotConfiguredError` with the fix in the message; none falls
through silently.

- `aws login` sessions are read from their cache while fresh; refresh
  needs a DPoP-bound key that lm15 does not implement — run `aws login`.
- Windows-only and interactive Azure rungs (Visual Studio shared cache,
  VS Code, browser, broker) and username/password — run `az login`.
- Azure Service Fabric managed identity (TLS thumbprint pinning) — use a
  certificate, a secret, or another managed-identity host.
- Encrypted PEM keys and PKCS#12 certificates — `openssl pkey` /
  `openssl pkcs12 -nodes` first.
- Google `external_account` with an AWS credential source, and the
  `external_account_authorized_user` / `gdch_service_account` types —
  use a file, URL or executable source, or a service account.
- Bedrock's native Converse wire and the binary event-stream framing
  (Nova, Llama, Mistral on Bedrock) — phase 2; `bedrock:` names nothing
  yet.

## Security and lifecycle limits

Credential files, endpoint overrides, and subprocess configuration are trusted
inputs. Do not accept them from an untrusted tenant. Credential HTTP requests
do not follow redirects. Error messages omit command output and token response
bodies. Request and credential reprs hide secret-bearing fields.

Cloud chains cache expiring credentials until the five-minute refresh window.
A refreshed credential that is already expired raises `AuthError`. CLI bearer
tokens without expiry metadata are resolved again on the next request, rather
than cached forever. No cloud credential cache is written to disk.

Cloud credential providers are synchronous. The async adapters call a
credential provider through `asyncio.to_thread`, so a chain refresh does not
block the event loop (ratified 2026-09-06). A static credential is read
inline; no thread is used.

The stdlib RSA signer uses variable-time Python integer arithmetic, without
blinding. It is not hardened against timing attacks. For high-value keys, the
mitigation is an external token provider: pass `api_key=lambda:
BearerToken(token)` where the token comes from your own signer or from the
cloud CLI (`gcloud auth print-access-token`, `az account get-access-token`),
so the private key never enters this process. Encrypted-key support and a
hardened crypto backend need a separate dependency decision.

## Per-door refusals

A feature the door does not carry raises rather than being dropped
(MAP-8). The documented lists are in `spec/auth.md` AUTH-10; they are
confirmed cell by cell when each door is captured live.
