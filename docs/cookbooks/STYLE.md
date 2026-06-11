# Cookbook style guide

The editorial constitution for the 16-recipe lm15 cookbook. Every recipe
page MUST conform. When in doubt, this file wins over precedent in
older docs (the retired monolithic cookbooks were raw material to mine,
not style to copy).

Brand voice in one line: **terse, honest, runnable.**

## The page skeleton

Every recipe page has exactly these sections, in this order:

```markdown
# <Recipe title>            ← H1, matches index.md entry

**Problem** — 2–3 sentences. What the reader is trying to do and why it
is non-obvious. No throat-clearing, no "In this recipe we will…".

## Recipe

Small runnable blocks. Each ```python block is immediately followed by a
```output block containing curated REAL output. Interleave one or two
sentences of connective prose between block pairs — never a wall of code.

## How it works

Prose. Explain the mechanism: which types are involved, what goes on the
wire, what lm15 deliberately does NOT do. Link to reference docs
(`../using-the-router.md`, `../tools-from-functions.md`, etc.) instead of
re-explaining them.

## Variations

Bullet list or short sub-blocks: the async mirror (`AsyncLMRouter`,
async iterators), provider differences (OpenAI vs Anthropic vs Gemini
behavior for this feature), and notable knobs. Variations may show code
without output if the only difference is `async`/`await` syntax; any
behavioral difference must show real output.

## See also

3–5 links: adjacent recipes (relative links like `06-function-tools.md`)
and the relevant deep-dive docs (`../docs/*.md` pages).
```

No other top-level sections. No "Conclusion", no "Summary", no "Next
steps" prose paragraphs — that is what **See also** is for.

## Output curation rules

1. **Never fabricate.** Every ```output block is pasted from an actual
   run of the exact code block above it. Run it, paste it, then curate.
   If you change the code, re-run and re-capture.
2. **Curate, don't dump.** Trim to the lines that teach. Maximum ~10
   lines per output block; if the raw output is longer, keep the
   shape-revealing lines and elide the rest.
3. **The elision marker is `…`** (single character, on its own line or
   inline mid-line). Use it for: base64 payloads, long ids
   (`resp_68ab…`), token-by-token stream noise, usage/metadata dumps,
   repeated list items. Never use `...` or `[snip]`.
4. **Truncate values, not structure.** When showing a dataclass/dict,
   keep the field names visible and elide the values:
   `Usage(input_tokens=17, output_tokens=9, …)`.
5. Model text output varies run to run; that is fine and honest. Do not
   "tidy" model prose. Do not cherry-pick reruns to get a cuter answer.
6. Output blocks are fenced as ```output (not ```text, not bare).
7. Errors shown deliberately (recipe 15) follow the same rules: real
   traceback, trimmed to the exception line plus the 1–3 frames that
   teach, `…` for the rest.

Canonical example of the pair (this output is a real capture):

````markdown
```python
res = LMRouter().resolve("gpt-4.1-mini")
print(res)
```
```output
'gpt-4.1-mini' -> provider 'openai' (OpenAILM); via built-in rule
prefix='gpt-' — OpenAI GPT family …; key from $OPENAI_API_KEY.
```
````

## Code rules

- **Front door is the router.** Recipes use `LMRouter` /
  `AsyncLMRouter` by default. Direct LM construction appears only where
  it is the documented path: custom `base_url`/compat (recipe 14),
  passthrough specifics, library-embedding notes in Variations.
- **Tools come from `lm15.tool()`.** Recipe 06 leads with
  `tool(fn)`/`derive(fn)`; hand-written `FunctionTool` is shown once as
  the canonical escape hatch, not the default.
- **Imports shown once per page**, in the first code block, complete.
  Later blocks on the same page assume them. Never re-import mid-page.
- **Env keys are explained ONCE, in recipe 01.** Recipe 01 shows the
  `.env` loading idiom (search cwd and parents) plus the shell one-liner
  `export $(grep -v "^#" ../.env | xargs)`. Every other page starts with
  a single line: *Keys loaded as in [recipe 01](01-first-request.md).*
  Never read or print a key value; the router's `MissingCredentialError`
  is the teaching surface for missing keys.
- **Models.** Default to cheap: `gpt-4.1-mini` (OpenAI),
  `gemini-3-flash-preview` (Gemini), `claude-sonnet-4-5` (Anthropic)
  only where the recipe needs Anthropic-specific behavior. Local ollama
  appears only in recipe 14 and is guarded by a reachability note, not
  try/except.
- **Tuples idiom.** lm15 surfaces are frozen and tuple-valued; write
  them that way: `messages=(Message.user("Hi"),)` — trailing comma on
  1-tuples, `tools=(weather,)`, never lists.
- **No try/except around demo code**, unless the recipe is *teaching*
  the error (recipe 15, ambiguity in recipe 16's edge notes). A recipe
  that needs defensive wrapping is a recipe with a broken example.
- No type annotations on demo locals; full hints on functions passed to
  `tool()` (they are the input). No `if __name__ == "__main__"`. No
  helper-function scaffolding unless it is the lesson.
- Code blocks are runnable top-to-bottom within a page: a reader who
  pastes every block in order into one file gets the shown outputs.

## Tone rules

- **Terse.** Sentences carry one idea. Cut adverbs, cut "simply",
  "just", "easily", "powerful". If a sentence survives without a word,
  delete the word.
- **Honest.** Name the limits: what lm15 doesn't do (no tool execution,
  no retry policy, no agent loop), where providers diverge, what costs
  money. "Gemini ignores this field" is a sentence we publish.
- **No marketing.** No exclamation points, no "seamless", no
  comparisons to other libraries, no feature counting. The code sells
  itself or the page fails.
- Second person ("you hold the functions"), present tense, active
  voice. O'Reilly register: a competent peer explaining, not a brand
  speaking.

## Length

150–300 lines per page, including code and output blocks. Under 150
means the recipe is too thin to be a page — merge or deepen. Over 300
means it is two recipes — split a Variation out or cut.

## File layout

- Pages live in `docs/cookbooks/`, named `NN-slug.md` (see
  [index.md](index.md) for the canonical 16).
- `index.md` lists all recipes grouped Essentials / Tools / Modalities /
  Beyond chat / Production, one-line hook each, ≤ 90 chars per hook.
- Cross-links between recipes are relative (`05-streaming.md`); links
  to deep-dive docs go up one level (`../using-the-router.md`).
