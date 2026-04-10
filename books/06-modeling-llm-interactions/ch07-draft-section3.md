## What Dependencies Cost

The costs of dependencies are well-known in the abstract — "version conflicts," "security risks," "bloat." The abstract is easy to dismiss. The concrete is harder.

### Coupling

When `httpx` pins `httpcore>=1.0.0,<2.0.0`, your library inherits that constraint. Your users inherit it. Every other package in the user's environment must be compatible with that range. If any other package requires `httpcore>=2.0.0` — because it needs a feature or a bug fix from the new major version — the environment is broken. The user cannot install both packages.

This isn't hypothetical. It's the most common support issue for libraries with deep dependency trees. The library's issue tracker fills with reports that look like bugs but are conflicts: "I can't install your library alongside X." The library author responds: "That's a conflict between our httpcore pin and X's httpcore pin, please contact X's maintainers." The user, who has never heard of httpcore and doesn't know what it does, is now mediating between two libraries' dependency decisions. They didn't sign up for this.

The combinatorial nature of the problem is worth dwelling on. A library with one dependency has one potential conflict surface. A library with 10 has 10. A library with 55 has 55 — and each of those 55 packages has its own dependencies, creating a tree of hundreds of potential conflict points. The probability that a user's environment contains no conflicts with any node in the tree decreases with every node added. For a sufficiently deep tree, conflicts are not a risk — they're a certainty for some fraction of users.

### Security

In January 2022, the maintainer of the npm packages `colors` and `faker` intentionally corrupted them — pushing versions that broke every project that depended on them. Millions of builds failed overnight. In October 2021, `ua-parser-js` was compromised with cryptocurrency-mining malware, affecting projects that had no idea they depended on it transitively. In Python, the `ctx` package on PyPI was found to contain code that exfiltrated environment variables — including, potentially, API keys.

Each incident follows the same pattern: a package deep in the dependency tree is compromised, and every downstream project inherits the compromise. The user who installed an LLM library didn't choose to trust the maintainer of `h11` or `sniffio`. But their code loads those packages, runs their code, and is exposed to their vulnerabilities. Every transitive dependency is an attack surface that the user didn't opt into.

Zero dependencies means zero transitive attack surface. The only code that runs is lm15's code and the Python standard library. The standard library is maintained by the Python core team with a formal security process. lm15's code is readable in an afternoon. The audit surface is bounded and tractable. For a library that handles API keys — strings that grant access to paid services and, in production, to sensitive data — a small audit surface is not an optimization. It's a security posture.

### Cold Start

The numbers bear repeating, because they're the most concrete cost and the easiest to measure.

| | lm15 | google-genai | litellm |
|---|---:|---:|---:|
| **Install** | 72ms | 137ms | 184ms |
| **Import** | 95ms | 2,656ms | 4,534ms |
| **Total cold start** | 167ms | 2,793ms | 4,718ms |
| Dependencies | 0 | 25 | 55 |
| Disk footprint | 408K | 41M | 155M |

The import time difference — 47x between lm15 and litellm — is almost entirely explained by transitive imports. `litellm` imports `httpx`, which imports `httpcore`, which imports `h11`, which imports `anyio`. Each import executes module-level code, registers handlers, and initializes state. At 55 packages, the initialization cascade takes 4.5 seconds.

For context: 4.5 seconds is longer than most LLM API calls. A simple `complete("gpt-4.1-mini", "Hello.")` returns in 0.5-1.5 seconds. With litellm, the import takes longer than the API call. The overhead is paid once per process, but in environments that start processes frequently — serverless, CLI tools, test suites, notebooks — "once" is dozens or hundreds of times per day.

The disk footprint tells a similar story. 408K versus 155M is a factor of 380. A Docker image for a serverless function has a practical size limit — larger images take longer to pull on cold start, which compounds the import time overhead. A library that contributes 155M to the image size is a library that makes deployment measurably slower.

### The Asymmetry

The deepest cost isn't any individual issue — it's the asymmetry between who decides and who pays.

The library author adds `pydantic` to `install_requires`. This costs the author nothing — pip resolves it, the CI passes, the feature works. The author moves on.

The user installs the library. They get `pydantic`, plus `pydantic-core` (a compiled Rust extension), plus `annotated-types`, plus version constraints on all three. Their Docker image grows by 15MB. Their cold start increases by 200ms. Their Alpine Linux CI environment fails to build `pydantic-core` because the Rust toolchain isn't installed. They spend an hour switching to a different base image.

The author made a five-second decision. The user spent an hour on its consequences. This asymmetry is the fundamental market failure of dependency management: the cost is externalized to the party with the least information and the least ability to act. The user can't remove the dependency. They can't pin it differently without forking the library. They can only absorb the cost or switch to a different library.

Every dependency is a decision made by someone who doesn't pay for it, imposed on someone who can't refuse it. The fewer dependencies a library has, the fewer of these imposed decisions its users must absorb.
