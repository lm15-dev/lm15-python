# Chapter 7: The Dependency Question

You're adding LLM support to your application. You pick a library — it looks good, the README is clean, the API is sensible. You run `pip install`. It takes eight seconds. You watch the output scroll:

```
Collecting httpx>=0.25.0
  Downloading httpx-0.28.1-py3-none-any.whl (73 kB)
Collecting pydantic>=2.0
  Downloading pydantic-2.11.1-py3-none-any.whl (443 kB)
Collecting anyio>=3.0
  Downloading anyio-4.9.0-py3-none-any.whl (100 kB)
Collecting h11>=0.13
Collecting sniffio
Collecting certifi
Collecting idna
Collecting httpcore>=1.0.0
Collecting annotated-types>=0.6.0
Collecting pydantic-core==2.33.0
  Downloading pydantic_core-2.33.0-cp312-cp312-manylinux_2_17_x86_64.whl (2.1 MB)
```

Eleven packages. You wanted to call `complete("gpt-4.1-mini", "Hello.")`. You got an HTTP client, a validation framework, an async runtime, a TLS certificate bundle, an HTTP/1.1 protocol implementation, an internationalized domain name library, an async I/O sniffing library, and a Rust-compiled validation core. Two point one megabytes of compiled binary for a validation library you didn't ask for.

This is fine. These are good packages, maintained by skilled people, solving real problems. You probably won't think about them again.

Until six months later, when you add another package to your project — an unrelated tool, maybe a monitoring library — and `pip install` fails:

```
ERROR: Cannot install monitoring-lib and your-llm-lib because these package
versions have conflicting dependencies.
  your-llm-lib requires httpx>=0.25.0,<0.29.0
  monitoring-lib requires httpx>=0.29.0
```

You've never imported `httpx`. You don't know what it does. But two packages you *do* use disagree about which version of it should exist, and `pip` won't let either of them win. You spend 30 minutes reading GitHub issues, pinning versions, and testing combinations. You solve it. You add a comment to `requirements.txt`: `# pinned due to conflict between your-llm-lib and monitoring-lib`. The comment will outlive both libraries in your project.

This is the dependency question. Not "are dependencies good or bad?" — they're both, always. The question is: what is the cost of each link in the chain, who pays it, and when does the cost exceed the benefit?

## The Supply Chain

A dependency tree is a supply chain. When you write `import lm15`, you're not just importing lm15 — you're importing everything lm15 depends on, and everything those packages depend on, transitively, all the way down to the C extensions and the standard library. Each link in the chain is a trust decision you made implicitly. You trust the package author. You trust their release practices — that they won't push a breaking change in a minor version, that they'll respond to security vulnerabilities, that they'll maintain backward compatibility. You trust their transitive choices — that the packages *they* depend on are equally well-maintained. You trust the entire tree.

For lm15, the tree has one node: lm15 itself. The supply chain is one link long. The only trust decision is whether you trust lm15's authors.

For a typical LLM library with 25-55 dependencies, the tree has dozens of nodes. Each node is a maintainer you've never met, a release schedule you don't control, a set of version constraints that interact with every other package in your environment. The supply chain is 25-55 links long, and the failure of any one link — a breaking release, a security vulnerability, a maintainer who abandons the project — becomes your problem.

The supply chain metaphor isn't decorative. It's precise. Supply chains in the physical world have the same property: the more links, the more fragile. A car manufacturer with 500 suppliers is more exposed to disruption than one with 50. Not because any individual supplier is bad, but because the probability that *at least one* supplier has a problem at any given time increases with the count. The same math applies to dependency trees. A library with 55 dependencies has 55 chances for a version conflict, a security issue, or a breaking change. A library with 0 has none.

The costs of the supply chain are externalities — paid by the user, not the library author. The library author adds `httpx` to their dependencies and gets a better HTTP client. The user gets `httpx` plus `httpcore` plus `h11` plus `anyio` plus `sniffio` plus `certifi` plus version constraints on all six. The author made one decision. The user inherited six packages and the interactions between them. This asymmetry — authors decide, users pay — is why dependency count tends to grow: each addition is cheap for the author and expensive for the user, and the author doesn't feel the expense.

For LLM libraries specifically, the expense is felt in three places that matter more than in most software.

**Cold start.** A serverless function that handles LLM requests imports the library on every cold start. lm15 imports in 95 milliseconds. `google-genai` takes 2,656ms. `litellm` takes 4,534ms. For a function that's invoked hundreds of times a day and cold-starts on every invocation (as many serverless platforms do for infrequent functions), 4.5 seconds of import overhead is 4.5 seconds the user waits before anything happens. It's also 4.5 seconds of billed compute on platforms that charge per millisecond.

**Notebooks.** Jupyter restarts the kernel frequently — after crashes, after installs, sometimes deliberately. Each restart reimports everything. A 95ms import is invisible. A 4.5-second import is a context switch — long enough to check your phone, lose your train of thought, break the flow of exploration that notebooks are designed to enable.

**Install reliability.** Packages with C extensions (like `pydantic-core`, which is compiled from Rust) can fail to install on unusual platforms — Alpine Linux in Docker containers, older ARM processors, Windows environments without build tools. A library with zero compiled dependencies installs everywhere Python runs. A library with one compiled dependency installs everywhere that dependency's wheel is available — which is most places, but not all, and the gaps are discovered at deployment time, not development time.

These three costs — cold start, restart overhead, install reliability — are specific to the environments where LLM libraries live. A web framework that runs on a server and starts once can absorb a 4-second import without anyone noticing. A CLI tool that the developer invokes fifty times a day cannot. The deployment context determines whether the dependency cost is negligible or dominant, and the deployment contexts for LLM libraries are disproportionately the ones where the cost is dominant.
