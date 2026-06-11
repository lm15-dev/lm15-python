# Router & tool-derivation portability spec

!!! warning "PROPOSAL"
    This page is a **proposal** for the lm15-contract, written for the
    Rust, Go, TypeScript, and Julia maintainers. Nothing here is frozen.
    Adoption requires maintainer ratification per
    [AUTHORITY.md](https://github.com/lm15-dev/lm15-contract/blob/main/AUTHORITY.md).
    Until ratified, the Python implementation (`lm15/router.py`,
    `lm15/tools.py`) is reference behavior, not normative.

Two additive surfaces shipped in lm15-python 1.0.0a: a model-string
**router** and **tool derivation from native callables**. Both sit
*outside* the frozen chat core: they produce ordinary canonical values
(`Request` in, provider LM out; `FunctionTool` out) and must never alter
any existing wire format. This page specifies the behavior
language-neutrally so each port can implement it idiomatically.

## Part 1 — the router

### Required behavior

1. **Model-string grammar.** Split the input on the FIRST `:`. If the
   head is a known provider string (a key of *that implementation's*
   adapter table — a port lacking an adapter treats its prefix as part
   of a bare id) AND the remainder is non-empty, the remainder is the
   wire model id. Otherwise the entire string (colons included) is a
   bare model id; in particular `"openai:"` (empty remainder) is the
   bare id `"openai:"` and falls through to the catalog and rule rungs.
   An empty or non-string input is an unknown-model error. Consequence
   to preserve: a fine-tune id like `ft:gpt-4.1:org` requires the
   explicit `openai:ft:gpt-4.1:org`.
2. **Provider strings** are the existing canonical ones: `openai`,
   `openai_chat`, `anthropic`, `gemini`, `claude-code`, `openai-codex`.
   The adapter table is hardcoded, exported, and inspectable data.
3. **Resolution algorithm** — exactly three rungs, fixed order, first
   match wins, order not configurable:
      1. *prefix* — explicit `provider:` form.
      2. *catalog* — only when the caller supplied a model registry.
         Match against `ModelInfo.id` and `aliases`; an exact-id match
         takes precedence over any alias match; an alias resolves to the
         canonical id. Multiple distinct providers matching → **error**
         (ambiguity is never auto-resolved); the error must carry the
         candidate provider list and suggest the explicit form. After
         exact-id precedence is applied, multiple matching entries from
         the same provider are also an ambiguity **error** — never
         resolved by registry iteration order, which is unspecified.
         A catalog match naming a provider with no adapter → unknown-model
         error pointing at direct LM construction.
      3. *rule* — a flat, ordered prefix→provider table; ships with
         built-in defaults (`claude-`→anthropic; `gpt-`, `o1`, `o3`,
         `o4`→openai; `gemini-`→gemini), caller-replaceable as plain data.
         No callbacks, no regexes, no plugins. A rule is `prefix` +
         `provider` plus an optional implementation-defined human `note`
         (rendered in `describe()`, excluded from fixtures: rules and
         `rules_tried` payloads serialize as `prefix`/`provider` only).
         A matching rule whose provider has no adapter in this
         implementation → unknown-model error at match time (the rule is
         NOT skipped; later rules are not consulted).
   No rung matched → unknown-model error recording the rules tried and
   whether a catalog was searched.
4. **Resolution is self-explaining.** `resolve()` performs no network
   I/O and reads no secret *values*. It returns a `Resolution` record:
   `requested`, `model` (wire id), `provider`, `adapter`, `source`
   (`"prefix" | "catalog" | "rule"`), the matched rule, `env_key`
   (which variable *would* be read — name only), catalog `model_info`.
   `adapter` is an implementation-defined display string (the LM class
   name in Python, e.g. `"AnthropicLM"`); it is excluded from
   conformance fixtures — ports key behavior on `provider`. A
   human-readable one-paragraph rendering is required (`describe()` /
   `Display` / `String()`). There is no separate `explain()`.
   `env_key` selection: `None` when the explicit per-provider key map
   has an entry for the provider, or when the provider declares no env
   keys (OAuth). Otherwise it is the first `env_keys` entry whose value
   in the env mapping is set AND non-empty (an empty-string variable
   counts as unset — ports using `LookupEnv`-style APIs must apply the
   same non-empty test), falling back to `env_keys[0]` when none
   qualify. This means `resolve()` reads env-var *presence*, never
   secret values; env handling is unfixtured.
5. **Credential resolution** (in `lm()` only): explicit per-provider
   key map (display-suppressed) beats environment; environment lookup
   uses the provider's existing `ProviderManifest.env_keys` in order,
   first set wins. No new env vars. OAuth providers (`claude-code`,
   `openai-codex`) take no key and use their self-resolving
   constructors. The env mapping must be injectable for hermetic tests.
6. **LM construction and caching**: at most one LM per provider, built
   lazily, reused. `lm()` returns the *ordinary* provider LM type — the
   escape hatch to direct configuration is the return value itself.
7. **Request routing**: `complete`/`stream` resolve `request.model`,
   replace it with the wire id when it differs (a pure copy-with;
   the input request is never mutated), and delegate. No retries, no
   fallbacks, no policy.
8. **Sync + async mirrors** where the language distinguishes them
   (`LMRouter` / `AsyncLMRouter` in Python and TS; a single type is fine
   in Go and Rust per local idiom; Julia per its task model). `resolve`
   stays synchronous everywhere — it performs no network I/O.
9. **Catalog degradation**: with no registry, rungs 1 and 3 fully work.
   Catalog hydration follows the existing entry-point/discovery protocol
   in [model-hydration.md](model-hydration.md) where the language has
   one; otherwise registries are constructed explicitly from canonical
   ModelInfo JSON. Catalog data stays advisory: it never changes
   `build_request` output.

### Error taxonomy additions

All subordinate to the existing `LM15Error` root:

| error | code | required payload |
|---|---|---|
| `RouterError` (base) | `router` | — |
| `UnknownModelError` | `unknown_model` | `model`, `rules_tried`, `catalog_searched` |
| `AmbiguousModelError` | `ambiguous_model` | `model`, `providers` (full candidate list) |
| `MissingCredentialError` | `not_configured` | `provider`, `env_keys` |

`MissingCredentialError` deliberately reuses the existing
`not_configured` code and (where the language has subtyping) the
existing `NotConfiguredError` type, so current handlers keep working.
Error messages must state the concrete fix (which env var to set, which
explicit prefix to use).

### Per-language notes (router)

- **Rust**: `RouterConfig` as a plain struct (builder optional);
  `Resolution` derives `Display`. Errors as variants of the existing
  error enum with the payload fields above. The per-provider LM cache
  implies interior mutability or `&mut self` — either is acceptable;
  trait-object return (`Box<dyn Lm>`) or an adapter enum per local
  precedent.
- **Go**: one `Router` type (no sync/async split); `Resolve` returns
  `(Resolution, error)`; sentinel/typed errors matching the taxonomy,
  `errors.As`-able to the existing not-configured type. Rule table as
  an exported slice `DefaultRules`.
- **TypeScript**: async-only is fine (`lm()`/`complete()` return
  promises) but `resolve()` must stay synchronous. `process.env` is the
  default env mapping, injectable for tests. `Resolution.describe()`
  plus `toString()`.
- **Julia**: a single router type; multiple dispatch on the LM is
  natural; keep the rule table as a `Vector{RouteRule}` constant.

## Part 2 — tools from native callables / types

### Required behavior

1. **Output is a plain canonical `FunctionTool`.** The `parameters`
   field remains opaque JSON Schema and is always emitted (INV-033).
   The wire format does not change in any way. Hand-written
   `FunctionTool` stays the primary, canonical path.
2. **Nothing is executed, registered, or wrapped.** Derivation produces
   schema; dispatch is the caller's code. No port may add an auto-loop.
3. **Eager, conservative derivation.** Errors at derivation time, never
   at request time. Unsupported types fail loudly (`tool_derivation`
   error code) rather than guess; the error names the function, the
   parameter, the offending type, and both escape hatches (per-parameter
   schema override; hand-written FunctionTool).
4. **Semantic invariants** every port must preserve, whatever the
   source-of-truth mechanism:
      - required iff the parameter has no default *in languages whose
        derivation source carries defaults* (Python, Julia). Where the
        mechanism has no defaults, the port must define an explicit
        optionality marker that is independent of nullability: in Go,
        every struct field is required unless tagged optional (e.g. an
        `optional` tag key — pointer-ness alone must NOT imply optional,
        since pointers express nullability); in TS, where the schema is
        accepted as a value, the schema's own `required` list is taken
        as-is and this invariant does not apply;
      - nullability (`Option`/`?`/`| null`) is orthogonal to
        required-ness and maps to `anyOf` with `{"type": "null"}` (or an
        equivalent the provider accepts);
      - string-keyed maps only — emitted as
        `{"type": "object", "additionalProperties": <value schema>}`;
        non-string keys are a derivation error; sequences → `"array"`
        with `"items"`; sets → array + `uniqueItems`; homogeneous
        variadic tuples (`tuple[X, ...]`) → array; fixed-length
        heterogeneous tuples are a derivation error; bare/unparameterized
        containers emit the untyped `{"type": "array"}` /
        `{"type": "object"}`; an explicit "any" type maps to the empty
        schema `{}`;
      - enums → `"enum"` of JSON-compatible values, plus a sibling
        `"type"` when all enum values share one primitive JSON type;
        nested structs inline (no `$ref` in v1, so adding `$ref` later
        is an extension, not a behavior change);
      - JSON-compatible defaults emitted as informational `"default"`,
        never affecting required-ness; non-JSON defaults (including
        non-finite floats — NaN/Infinity) are silently skipped;
      - prose (doc comments) is best-effort; types are strict.
5. **Explainability**: an inspectable account of how each parameter was
   derived is recommended where the mechanism allows it
   (`derive()`/`ToolDerivation` in Python). It is NOT part of the
   proposed contract — derivation diagnostics are per-language.

### Per-language notes (tools)

The *mechanism* is expected to differ per language; only the invariants
above are shared:

- **Rust**: a derive macro (`#[derive(LmTool)]` on an args struct, or a
  fn-attribute macro) generating the schema at compile time; doc
  comments become descriptions. Compile-time failure replaces
  `ToolDerivationError` where possible.
- **Go**: no runtime docstrings — derive from a tagged args struct via
  reflection (`json`/`jsonschema`-style tags supply descriptions);
  return `(FunctionTool, error)`.
- **TypeScript**: types are erased at runtime, so derivation cannot read
  annotations. Idiomatic options: accept a schema value from an
  ecosystem validator the user already has (structural acceptance, not
  a dependency — lm15 stays zero-dep), or a small builder. Do not ship
  a TS-transform.
- **Julia**: derive from a method signature's positional/keyword types
  via reflection; `@doc` strings for descriptions.

### Conformance intent

If ratified, the proposal adds contract fixtures only for the **pure,
deterministic** parts: resolution-algorithm cases (grammar, rung order,
ambiguity, alias canonicalization, error codes/payloads) and
schema-invariant cases for derivation (required/nullable orthogonality,
container/enum mappings) expressed as canonical-JSON expectations.
Docstring/doc-comment extraction and diagnostic surfaces stay
implementation-defined and unfixtured.
