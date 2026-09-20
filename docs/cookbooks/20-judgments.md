# Judgments: declared answers with probabilities

**Problem** — You want a model to *decide*, not write: which of these categories, how good on a scale you defined, yes or no — for many items, with a probability per option you can put in a table. TypeSafe's Jev does exactly that natively; ordinary models can be made to answer the same questions. You want one request shape for both, and honesty about which numbers were measured.

## Recipe

Describe the answers as a `json_schema` whose properties are `enum`s, booleans, or ordered levels. The helpers emit that schema for you, the way `tool(fn)` emits a tool schema.

```python
from lm15 import LMRouter, Request, Message, Config, judgments, choice, score, yes_no

note = "Ripe blackberry and cassis, toasty oak, firm tannins. Long finish; will reward a decade in the cellar."

answers = judgments(
    quality=score("How good is this wine, according to the note?", {
        "faulty": "Faulty or unpleasant", "simple": "Simple and sound", "good": "Good, well made",
        "excellent": "Excellent - complex and structured", "profound": "Profound - the note treats it as exceptional"}),
    style=choice("What is the dominant style described?", {"fruit": "Fruit-forward", "oak": "Oak-driven", "other": None}),
    ageing=yes_no("Does the note say the wine will improve with age?"),
)

router = LMRouter()
r = router.complete(Request(model="jev-latest", messages=[Message.user("Tasting note:\n" + note)],
                            config=Config(response_format=answers, probabilities="if_available")))
print(r.data)                      # {'quality': 3, 'style': 'fruit', 'ageing': True}
print(r.probabilities["style"])    # {'fruit': 1.0, 'oak': 0.0, 'other': 0.0}
print(r.expected("quality"))       # 3.02  — position on the 0..4 scale, Σ p·i
print(r.method)                    # 'provider_classification'
```

`r.data` is a plain dict; an ordered judgment answers with its level index (`3` = `excellent`), a choice with its key, a yes/no with a bool. `r.probabilities` is one distribution per judgment over the declared keys, or `None` when nothing was measured — never a made-up one.

## Same program, other models

```python
r = router.complete(Request(model="gpt-5-mini", ...))          # structured output: the pick; probabilities None
print([(a.field, a.action) for a in r.adaptations])          # [('config.probabilities', 'dropped')]

r = router.complete(Request(model="vllm:LiquidAI/LFM2.5-2.6B", ...))   # a vLLM ≥ 0.29 server
print(r.method, r.provider_data["coverage"])                 # 'candidate_sequence_likelihood' {...}
```

| provider | pick | probabilities | how |
|---|---|---|---|
| `typesafe` (Jev) | native | native, `provider_classification` | every judgment is one Jev question over your messages |
| `openai`, `openai-chat`, `anthropic`, `gemini` | native structured output | absent | the schema goes as-is (Anthropic and Gemini get the equivalent form their wire honours) |
| `openai-chat` on vLLM ≥ 0.29 | from the distribution | exact, `candidate_sequence_likelihood` | every key is scored as a token path in one batched call; `coverage` says how much probability the model put on the listed keys at all |

`probabilities="required"` refuses up front on a wire that cannot measure them (`UnsupportedFeatureError`, `feature="config.probabilities"`). The `method` travels with the numbers: a Jev distribution and a token-likelihood distribution have the same shape and are not the same measurement, and neither is calibrated on your data until you check.

## Into a table

```python
import pandas as pd
rows = [router.complete(Request(model="jev-latest", messages=[Message.user(n)], config=cfg)) for n in notes]
df = pd.DataFrame([r.data | {"quality_expected": r.expected("quality")} for r in rows])
df["quality"] = pd.Categorical(df.quality, categories=range(5), ordered=True)
```

## What to know

- A beginner's `{"enum": ["a", "b"]}` is already a judgment; descriptions (`choice`/`score` helpers) are optional and help the model.
- Jev's own `confidence` and `score` are kept verbatim in `provider_data["typesafe"]["answers"]`; the expected level is computed, never stored. These opaque provider statistics are not substituted for missing probabilities.
- Every Jev answer must match its declared kind and keys, with a complete distribution of finite numbers in `[0, 1]`. Missing/malformed measurements or undeclared choices raise non-retryable `ProviderError`, not a partial answer or fabricated zero. Absent usage counters stay `None`; reported zeros stay zero.
- **INV-052 deliberately does not validate distribution totals** (providers round), either in the adapter or in `DataPart`. This was not a DataPart defect. Numbers are not normalized or rewritten; even an unusual total is preserved. A noul's complement `1-p` is the specified boolean mapping, not a normalization.
- Candidate-sequence likelihood is different: it normalizes measured log-likelihoods once over the key set as MAP-14 requires. A key set whose every likelihood is zero cannot be normalized and raises `ProviderError`. Malformed token/scoring responses are provider faults; missing requested token ids still trigger the documented `if_available` fallback or `required` refusal.
- Structured input: `Message.user(data({...}))` sends a JSON value as the state, verbatim (Jev reads it as such; a text-only wire gets it as compact JSON).
- Jev's state is the one user part and nothing else: no `system` (put context in the state as a named key, or in the question), no second message (put a transcript in the state as your own object). Both are refused with the native place named — `lm15-contract/changes/2026-09-19-jev-state.md`.
- Contract: `lm15-contract/changes/2026-09-17-judgments.md`, MAP-14.
