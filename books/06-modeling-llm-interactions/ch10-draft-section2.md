## The Deprecation Clock

Every hardcoded model name is a ticking clock. The clock ticks at different speeds — a flagship model like `claude-sonnet-4-5` might last 12-18 months before its successor arrives. An experimental model like `o1-preview` might last weeks. A dated model like `claude-3-haiku-20240307` has the date of its death written into its name.

The most obvious solution is discovery: don't hardcode, query.

```python
models = lm15.models(supports={"tools"}, provider="anthropic", env=".env")
model_id = models[0].id  # whatever exists today
resp = lm15.complete(model_id, "Hello.", env=".env")
```

The code doesn't name a model. It describes what it needs (tools, from Anthropic) and picks from what's available. When `claude-sonnet-4-5` is deprecated and `claude-sonnet-5-0` takes its place, the code picks the new model automatically. The deprecation clock doesn't tick for this code, because the code doesn't commit to a specific model.

But discovery trades one instability for another. The list-models endpoint might change (Anthropic's didn't exist until recently). The capability metadata is often incomplete — OpenAI's model list returns IDs with no capability information, so filtering by `supports={"tools"}` might miss models or include non-tool-supporting ones. The models.dev fallback catalog might be stale. The discovery mechanism is itself a moving target.

And discovery changes what "the same program" means. If you run the program today and it picks `claude-sonnet-4-5`, and you run it tomorrow and it picks `claude-sonnet-5-0`, the two runs use different models with potentially different behavior, different pricing, and different capabilities. The program is "the same" in the sense that the code didn't change. It's "different" in the sense that the model — the most important variable in the system — changed. Whether this is a feature (automatic upgrade) or a bug (unpredictable behavior) depends on the context. A research benchmark needs reproducibility — discovery is a bug. A production assistant needs the best available model — discovery is a feature.

The deeper issue is that model names serve two purposes that are in tension: **identification** (which specific model is this?) and **capability description** (what can this model do?). `claude-sonnet-4-5` identifies a specific model version. `tools=True, reasoning=True, provider="anthropic"` describes a capability requirement. The deprecation clock ticks on identifiers, not on descriptions. A library designed around descriptions — "give me an Anthropic model with tools and reasoning" — is more resilient than one designed around identifiers — "give me claude-sonnet-4-5." But descriptions are imprecise (multiple models might match) and identifiers are exact (only one model matches). Precision and resilience are in tension, and the right balance depends on whether you're building for reproducibility or for adaptability.
