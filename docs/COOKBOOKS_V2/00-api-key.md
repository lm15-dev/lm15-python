# Cookbook 00 — Setting Up Your API Keys

Before lm15 can talk to any AI provider, it needs your **API keys** — secret passwords that prove you have an account with OpenAI, Anthropic, or Google.

---

## Which keys does lm15 look for?

| Provider      | Variable name                            | Where to get one                                                       |
|---------------|------------------------------------------|------------------------------------------------------------------------|
| OpenAI        | `OPENAI_API_KEY`                         | [platform.openai.com/api-keys](https://platform.openai.com/api-keys)  |
| Anthropic     | `ANTHROPIC_API_KEY`                      | [console.anthropic.com/settings/keys](https://console.anthropic.com/settings/keys) |
| Google Gemini | `GEMINI_API_KEY` **or** `GOOGLE_API_KEY` | [aistudio.google.com/apikey](https://aistudio.google.com/apikey)       |

You only need the key(s) for the provider(s) you plan to use.  
One key is enough to get started.

Not sure what's available? Ask lm15:

```python
import lm15
print(lm15.providers())
# {'openai': ('OPENAI_API_KEY',), 'anthropic': ('ANTHROPIC_API_KEY',), 'gemini': ('GEMINI_API_KEY', 'GOOGLE_API_KEY')}
```

---

## The recommended way: a `.env` file

Create a file called `.env` in your project folder:

```
OPENAI_API_KEY=sk-proj-abc123...
ANTHROPIC_API_KEY=sk-ant-abc123...
GEMINI_API_KEY=AIza...
```

Then pass `env=".env"` when you create a model:

```python
import lm15

gpt = lm15.model("gpt-4.1-mini", env=".env")
resp = gpt("Hello!")
print(resp.text)
```

That's it. No extra packages to install. lm15 reads the file, finds the keys it recognises, and uses them.

**Every cookbook after this one assumes you have a `.env` file and uses `env=".env"` on the first `lm15.model()` call.**

### Keep your `.env` out of git

Your `.env` file contains secrets. If you push it to GitHub, anyone can use your keys and run up your bill.

Add `.env` to your `.gitignore`:

```bash
echo ".env" >> .gitignore
```

If `.env` was already committed before you added the rule, untrack it first:

```bash
git rm --cached .env
```

---

## Other ways

### Pass the key directly

For quick experiments or notebooks:

```python
resp = lm15.complete("gpt-4.1-mini", "Hello!", api_key="sk-proj-abc123...")
```

Multiple providers:

```python
resp = lm15.complete("claude-sonnet-4-5", "Hello!", api_key={
    "openai": "sk-proj-...",
    "anthropic": "sk-ant-...",
})
```

### Environment variables

Set them in your terminal before running your script:

```bash
export OPENAI_API_KEY="sk-proj-abc123..."
python my_script.py
```

Then your script needs no `env=` or `api_key=` at all:

```python
import lm15
resp = lm15.complete("gpt-4.1-mini", "Hello!")
```

### Shell config files

lm15 can also read `~/.bashrc`, `~/.zshrc`, or any file with `KEY=VALUE` / `export KEY=VALUE` lines:

```python
gpt = lm15.model("gpt-4.1-mini", env="~/.zshrc")
```

---

## Priority

If you use multiple methods at once, lm15 picks the first key it finds:

1. `api_key=` parameter
2. `env=` file
3. Environment variables

---

## What happens if a key is missing?

lm15 skips providers with no key. If you call a model from an unconfigured provider:

```python
resp = lm15.complete("claude-sonnet-4-5", "Hello!")
# → error: no provider registered for this model
```

The fix: provide the key using any of the ways above.

---

## FAQ

**Q: Do I need all three keys?**  
No. One is enough.

**Q: How do I know the valid names for `api_key={...}`?**  
`lm15.providers()` returns them.

**Q: I'm writing a plugin — how do I declare my env var?**  
Set `env_keys` on your `ProviderManifest`:

```python
from lm15.features import EndpointSupport, ProviderManifest

manifest = ProviderManifest(
    provider="mistral",
    supports=EndpointSupport(),
    env_keys=("MISTRAL_API_KEY",),
)
```

When a user passes `env=`, lm15 sets all key-value pairs from the file into `os.environ`, so your plugin picks them up automatically.
