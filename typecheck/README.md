# Type-checking gate

`python tools/typecheck.py` (CI job `typecheck`) runs two checks with the
mypy pinned in `requirements/typecheck.txt`:

- **`examples/`: real programs, zero errors.** The Python programs the
  documentation site shows, as a reader copies them, checked with `--strict`
  (except the rules about annotating the example's own functions). An error
  here is a type a user's checker rejects in code the docs told them to write.
  Refresh from a website checkout with
  `python tools/typecheck.py --refresh-examples ../website`.
- **`baseline.txt`: lm15's own findings, only going down.** mypy's findings
  inside `lm15/`, without line numbers. A new finding fails; so does a
  recorded one that no longer occurs (record the fix with `--update`).

Constructor signatures: lm15's value types store tuples but accept lists.
`tools/gen_init_signatures.py` writes the `if TYPE_CHECKING: def __init__`
blocks that tell type checkers so; `tests/test_init_signatures.py` keeps them
matching the dataclasses and what the constructors really accept.
