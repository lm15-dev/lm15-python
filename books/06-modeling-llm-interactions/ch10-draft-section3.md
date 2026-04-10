## Escape Hatches as Bets on the Future

The `provider` dict on `Config`. The `provider` dict on `LMResponse`. The `metadata` dict on `Part`. The `raw` dict on `ModelSpec`. The `parameters` dict on `Tool`. The `builtin_config` dict on `Tool`.

Six untyped dicts, scattered through lm15's type system. Each one is an admission: "I don't know what will be here tomorrow."

A typed field says "I know what this is." `temperature: float | None` is a declaration: this field is a float, it controls sampling randomness, it exists on every provider. The type checker enforces it. The documentation describes it. The IDE auto-completes it. The knowledge is encoded in the structure.

An escape hatch says "something goes here." `provider: dict[str, Any] | None` is a question: what will providers need that I can't predict? The type checker can't help — the keys are strings, the values are `Any`. The documentation says "see provider docs." The IDE shows nothing. The ignorance is encoded in the structure.

Both are honest. The typed field is honest about what's known. The escape hatch is honest about what isn't. The dishonest option — the one lm15 avoids — is to pretend everything is known by defining typed fields for today's features and providing no mechanism for tomorrow's. That approach produces a type system that's complete today and broken tomorrow.

The tradeoff is concrete. Each escape hatch sacrifices developer experience — no autocompletion, no type checking, no documentation — in exchange for forward compatibility. Each typed field provides developer experience and bets that the field's meaning won't change. `temperature` is a safe bet — every provider has it, the semantics are universal, the type is stable. `reasoning` is a riskier bet — the semantics differ between providers (Chapter 4), and the parameterization (token budget vs effort level) might converge or diverge further. The riskier the bet, the more the field should look like an escape hatch (a dict) rather than a typed parameter (a named field).

How many escape hatches is the right number? The answer is proportional to the rate of domain change. A library modeling SQL — a domain that changes on decadal timescales — needs zero escape hatches. The domain is fully known. Every feature can be a typed field. A library modeling LLM interactions — a domain that changes on monthly timescales — needs several. The domain is partially known and actively expanding. Some features haven't been invented yet, and the escape hatches are where they'll land when they arrive.

lm15 has six escape hatches. Two years ago, when the library was designed, features like citations, refusals, audio content, and structured output didn't exist. They arrived and landed — some as new Part types (the open type set absorbed them), some as new Config fields (reasoning, prompt caching), and some in escape hatches (provider-specific parameters that don't have universal equivalents yet). The escape hatches that were populated in 2024 have partly been promoted to typed fields in 2025. The escape hatches that are populated today will partly be promoted tomorrow. The hatches are a staging area — temporary homes for features that are too new to model, too provider-specific to universalize, or too unstable to commit to.

The lifecycle of a feature in lm15's type system:

1. **Invention.** A provider adds a capability. It's accessible only through the escape hatch: `config.provider = {"new_feature": True}`.
2. **Adoption.** A second provider adds a similar capability. The feature graduates from the escape hatch to a universal parameter: `new_feature=True` on `Config`. The adapter maps it to each provider's mechanism.
3. **Stabilization.** All providers support the feature with converging semantics. The parameter is stable. The escape hatch for this feature empties.

Not every feature completes this lifecycle. Some stay at stage 1 forever — provider-unique capabilities that never generalize. Some reach stage 2 but never stabilize — features where the semantics keep diverging. The escape hatch is the permanent home for stage 1 features and the temporary home for stage 2 features. Stage 3 features leave the hatch entirely and become part of the universal type system.

This lifecycle mirrors the convergence trajectory from Chapter 4. The escape hatch empties as providers converge. It fills as providers innovate. The net flow — whether more features are entering the hatch or leaving it — is a measure of whether the domain is converging or diverging. Right now, the net flow is outward: more features are graduating from escape hatches to typed fields than the reverse. The convergence is winning. The escape hatches are slowly emptying. Whether that trend continues is the convergence bet from the next section.
