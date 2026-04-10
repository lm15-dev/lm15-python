## The Accidental Layer

Not all layers are designed. Some emerge.

A function gets long, so you extract a helper. The helper gets its own file. The file becomes a module. A module boundary exists — import paths, a namespace, an implicit interface. Nobody drew this boundary on a whiteboard. It grew from the code's shape. The question: is the accidental boundary in the right place?

lm15's `factory.py` is an accidental layer. It started as `build_default()` — a convenience function that created a `UniversalLM` with the standard adapters. Then it absorbed env file parsing. Then API key resolution. Then transport selection. Then plugin discovery. Then models.dev hydration. Each addition was natural — "where should API key resolution live? Near the code that needs API keys — the factory." — and the result is a 130-line function that knows about every layer in the system.

`build_default()` is simultaneously the most coupled module (it imports adapters, transports, auth, plugins, capabilities) and the most useful module (it's the one-line setup that makes lm15 work). The coupling is necessary — the factory's job is to wire the system, and wiring requires knowing every piece. But the coupling means that changes anywhere in the system might require changes to the factory. A new adapter? The factory imports it. A new auth mechanism? The factory handles it. A new env var format? The factory parses it.

Is this an accidental boundary in the right place?

Apply the substitution test. Can you replace the factory without changing the layers it wires? Yes — you can write your own setup code, construct `UniversalLM` manually, register adapters yourself, and never call `build_default()`. The factory is optional. The layers beneath it don't know it exists.

Can you replace a layer without changing the factory? Mostly. Swapping `urllib` for `pycurl` is handled inside the factory (it tries to import `pycurl` and falls back). Adding an adapter requires adding an import and a registration line. These are small changes — the factory changes, but only in the lines specific to the swapped component.

The accidental boundary is in an acceptable place. It's not elegant — the function does six things. But each thing is a wiring concern, and wiring concerns naturally aggregate at the point where everything connects. A more deliberate design might split `build_default()` into sub-functions — `_resolve_keys()`, `_create_transport()`, `_register_adapters()`, `_load_plugins()` — and lm15 does some of this internally. But the module-level boundary is where it is, and it works.

The deeper lesson: **accidental boundaries should be evaluated the same way as deliberate ones.** Apply the substitution test. Check for leaks. Ask whether the two sides change independently. If the answers are favorable, the accident was wise — the code's natural structure produced a useful boundary. If the answers are unfavorable, the accident is tech debt — a boundary that adds complexity without adding isolation, and that should be redesigned or removed.

Most codebases have more accidental layers than deliberate ones. The file that grew too large and was split. The class that accumulated too many methods and was factored. The utility module that became a dependency of everything. Each split created a boundary. Each boundary carries a cost (indirection, imports, interface maintenance) and a benefit (isolation, substitutability, comprehensibility). The cost-benefit ratio depends on whether the boundary is in the right place — and "the right place" is defined by the substitution test: can you change one side without changing the other?

In a young codebase like lm15 — 2,408 lines, 18 months old — the accidental layers are few and mild. In a mature codebase — 200,000 lines, five years old — accidental layers accumulate like geological strata, and evaluating them is the primary work of architectural review. The substitution test scales: it works on a 30-file library and a 3,000-file application. The only thing that changes is the number of boundaries to test.

This chapter has examined boundaries as the fundamental unit of library architecture: where to draw them (the substitution test), how many to have (one per independent change vector), where they leak (and what leaks mean), what the LLM-specific change pattern implies (adapter boundary is load-bearing), and how accidental boundaries compare to deliberate ones. The next chapter examines what flows across the boundaries — specifically, what the library chooses to show the user and what it chooses to hide. A boundary is a wall. Chapter 9 is about the windows and doors.
