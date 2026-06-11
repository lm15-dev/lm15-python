# lm15 Cookbook: Using All 69 Features (Runnable)

This document is designed to be run as a notebook or script (e.g., using a tool like Quarto or by copying into Jupyter). It demonstrates how to use the 69 cross-SDK features supported by `lm15-python` across OpenAI, Anthropic, and Gemini.

## Setup & Initialization

First, initialize the providers. This notebook looks for `.env` in the current working directory and its parents, so it works whether you run it from the repo root or from `lm15-python/docs/`. We'll also define a handy `execute()` function that runs a request across all three providers (and lets us toggle between streaming and completion).

```python
import os
import shlex
import pprint
import dataclasses
from pathlib import Path
from lm15.providers import OpenAILM, AnthropicLM, GeminiLM
from lm15.types import StreamDeltaEvent, TextDelta, ThinkingDelta

# 1. Load keys from the repo-root .env file, independent of notebook cwd.
def find_dotenv(filename=".env"):
    start = Path.cwd().resolve()
    candidates = [start, *start.parents]

    # If your runner executes from lm15-python/docs, this catches the repo root.
    try:
        doc_dir = Path("lm15-python/docs").resolve()
        candidates.extend([doc_dir, *doc_dir.parents])
    except Exception:
        pass

    seen = set()
    for directory in candidates:
        if directory in seen:
            continue
        seen.add(directory)
        path = directory / filename
        if path.exists():
            return path
    return None


def load_env_file(path, *, override=True):
    """Load shell-style .env lines like: export OPENAI_API_KEY="sk-..."."""
    if path is None:
        return {}

    loaded = {}
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):]
        if "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()

        # shlex handles quotes and shell-style escaping robustly.
        try:
            parsed = shlex.split(value, posix=True)
            value = parsed[0] if parsed else ""
        except ValueError:
            value = value.strip('"\'')

        if override or key not in os.environ:
            os.environ[key] = value
        loaded[key] = os.environ.get(key, "")
    return loaded


def mask(value):
    if not value:
        return "missing"
    if len(value) <= 12:
        return "set"
    return f"{value[:7]}...{value[-4:]}"


env_path = find_dotenv()
loaded = load_env_file(env_path)
print(f"Loaded .env from: {env_path or 'not found'}")
for key in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY"):
    print(f"  {key}: {mask(os.environ.get(key))}")

# 2. Initialize the three major providers.
lms = {
    "openai": OpenAILM(api_key=os.environ.get("OPENAI_API_KEY", "")),
    "anthropic": AnthropicLM(api_key=os.environ.get("ANTHROPIC_API_KEY", "")),
    "gemini": GeminiLM(api_key=os.environ.get("GEMINI_API_KEY", "")),
}

# 3. Map default models for each provider.
# These are the models used by the cross-SDK curl fixture suite.
# You can replace them with newer models after confirming your account has access.
MODELS = {
    "openai": "gpt-5.4-mini",
    "anthropic": "claude-sonnet-4-5",
    "gemini": "gemini-3-flash-preview",
}

def execute(req, providers=None, stream=None):
    """Run a request and collect all provider results.

    Args:
        req: The lm15 Request to run.
        providers: Optional iterable of provider names.
        stream: True for streaming only, False for complete() only, None for both.
    """
    providers = list(providers or lms.keys())
    stream_modes = (False, True) if stream is None else (stream,)
    responses = []
    streams = []
    errors = []

    for name in providers:
        required_key = {
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "gemini": "GEMINI_API_KEY",
        }[name]

        if not os.environ.get(required_key):
            errors.append({
                "provider": name,
                "error": f"Skipped: {required_key} is not set",
            })
            continue

        lm = lms[name]

        # Replace the placeholder model in the request with the provider-specific model.
        provider_req = dataclasses.replace(req, model=MODELS[name])

        for is_stream in stream_modes:
            try:
                if is_stream:
                    streams.append({
                        "provider": name,
                        "model": MODELS[name],
                        "events": list(lm.stream(provider_req)),
                    })
                else:
                    responses.append({
                        "provider": name,
                        "model": MODELS[name],
                        "response": lm.complete(provider_req),
                    })
            except Exception as e:
                errors.append({
                    "provider": name,
                    "model": MODELS[name],
                    "stream": is_stream,
                    "error": f"{type(e).__name__}: {e}",
                })

    return {"responses": responses, "streams": streams, "errors": errors}


def pretty(value, *, width=100):
    """Pretty-print nested execute() results in notebooks/scripts."""
    pprint.pp(value, width=width, sort_dicts=False, compact=False)
```
```output | ✓ 302ms | 22 vars
Loaded .env from: /home/maxime/Projects/lm15-dev/.env
  OPENAI_API_KEY: sk-proj...mtMA
  ANTHROPIC_API_KEY: sk-ant-...awAA
  GEMINI_API_KEY: AIzaSyB...nH1w
```

---

## 1. Chat & Multi-turn
**(Features: `basic_text`, `multi_turn`)**

Basic text generation and multi-turn conversations use `Message.user` and `Message.assistant` inside a `Request`.

```python ✓
from lm15.types import Request, Message
```

```python
# basic_text
req_basic = Request(
    model="placeholder", 
    messages=(Message.user("Say hello."),)
)
r = execute(req_basic)
pretty(r)
```
```output | ✓ 10.7s | 25 vars
{'responses': [{'provider': 'openai',
                'model': 'gpt-5.4-mini',
                'response': Response(
    text='Hello!',
    model='gpt-5.4-mini-2026-03-17',
    finish_reason='stop',
    usage=Usage(input_tokens=9, output_tokens=6, total_tokens=15, cache_read_tokens=0, cache_write_tokens=None, reasoning_tokens=0, input_audio_tokens=None, output_audio_tokens=None),
    id='resp_08f99935c43e7aff0069f5e804f218819caec290d2ffa9b748',
    provider_data=<dict: 35 keys>,
)},
               {'provider': 'anthropic',
                'model': 'claude-sonnet-4-5',
                'response': Response(
    text='Hello! How can I help you today?',
    model='claude-sonnet-4-5-20250929',
    finish_reason='stop',
    usage=Usage(input_tokens=10, output_tokens=12, total_tokens=22, cache_read_tokens=0, cache_write_tokens=0, reasoning_tokens=None, input_audio_tokens=None, output_audio_tokens=None),
    id='msg_01K22PnHyDnhWFiKZQ6ubgE7',
    provider_data=<dict: 9 keys>,
)},
               {'provider': 'gemini',
                'model': 'gemini-3-flash-preview',
                'response': Response(
    text='Hello! How can I help you today?',
    model='gemini-3-flash-preview',
    finish_reason='stop',
    usage=Usage(input_tokens=4, output_tokens=9, total_tokens=32, cache_read_tokens=None, cache_write_tokens=None, reasoning_tokens=19, input_audio_tokens=None, output_audio_tokens=None),
    id='DOj1afXsNJ23_uMPmubiwAs',
    provider_data=<dict: 4 keys>,
)}],
 'streams': [{'provider': 'openai',
              'model': 'gpt-5.4-mini',
              'events': [StreamStartEvent(id='resp_08884db5bc096a930069f5e80909cc819683f8628200065af7',
                                          model='gpt-5.4-mini-2026-03-17',
                                          type='start'),
                         StreamDeltaEvent(delta=TextDelta(text='Hello', part_index=0, type='text'),
                                          type='delta'),
                         StreamDeltaEvent(delta=TextDelta(text='!', part_index=0, type='text'),
                                          type='delta'),
                         StreamEndEvent(
    finish_reason='stop',
    usage=Usage(input_tokens=9, output_tokens=6, total_tokens=15, cache_read_tokens=0, cache_write_tokens=None, reasoning_tokens=0, input_audio_tokens=None, output_audio_tokens=None),
    provider_data=<dict: 34 keys>,
    type='end',
)]},
             {'provider': 'anthropic',
              'model': 'claude-sonnet-4-5',
              'events': [StreamStartEvent(id='msg_01E56JDJZVCpu1hEX5WaWCaX',
                                          model='claude-sonnet-4-5-20250929',
                                          type='start'),
                         StreamDeltaEvent(delta=TextDelta(text='Hello! How',
                                                          part_index=0,
                                                          type='text'),
                                          type='delta'),
                         StreamDeltaEvent(delta=TextDelta(text=' can I help you today?',
                                                          part_index=0,
                                                          type='text'),
                                          type='delta'),
                         StreamEndEvent(
    finish_reason='stop',
    type='end',
)]},
             {'provider': 'gemini',
              'model': 'gemini-3-flash-preview',
              'events': [StreamDeltaEvent(delta=TextDelta(text='Hello! How can I help you today?',
                                                          part_index=0,
                                                          type='text'),
                                          type='delta'),
                         StreamDeltaEvent(delta=TextDelta(text='', part_index=0, type='text'),
                                          type='delta'),
                         StreamEndEvent(
    finish_reason='stop',
    usage=Usage(input_tokens=4, output_tokens=9, total_tokens=90, cache_read_tokens=None, cache_write_tokens=None, reasoning_tokens=77, input_audio_tokens=None, output_audio_tokens=None),
    provider_data=<dict: 4 keys>,
    type='end',
)]}],
 'errors': []}
```

```py
# multi_turn
req_multi = Request(
    model="placeholder",
    messages=(
        Message.user("What is 2 + 2? Reply with one word."),
        Message.assistant("four"),
        Message.user("Repeat your previous answer in uppercase."),
    )
)
r = execute(req_multi)
pretty(r)
```
```output | ✓ 7.0s | 26 vars
{'responses': [{'provider': 'openai',
                'model': 'gpt-5.4-mini',
                'response': Response(
    text='FOUR',
    model='gpt-5.4-mini-2026-03-17',
    finish_reason='stop',
    usage=Usage(input_tokens=37, output_tokens=6, total_tokens=43, cache_read_tokens=0, cache_write_tokens=None, reasoning_tokens=0, input_audio_tokens=None, output_audio_tokens=None),
    id='resp_020bdb40c0dd01e10069f5e81e221081a392e687362dce7da8',
    provider_data=<dict: 35 keys>,
)},
               {'provider': 'anthropic',
                'model': 'claude-sonnet-4-5',
                'response': Response(
    text='FOUR',
    model='claude-sonnet-4-5-20250929',
    finish_reason='stop',
    usage=Usage(input_tokens=36, output_tokens=5, total_tokens=41, cache_read_tokens=0, cache_write_tokens=0, reasoning_tokens=None, input_audio_tokens=None, output_audio_tokens=None),
    id='msg_01K3qCQDyhHE9tRfdDShVguo',
    provider_data=<dict: 9 keys>,
)},
               {'provider': 'gemini',
                'model': 'gemini-3-flash-preview',
                'response': Response(
    text='FOUR',
    model='gemini-3-flash-preview',
    finish_reason='stop',
    usage=Usage(input_tokens=24, output_tokens=1, total_tokens=86, cache_read_tokens=None, cache_write_tokens=None, reasoning_tokens=61, input_audio_tokens=None, output_audio_tokens=None),
    id='I-j1aZjtKpG9_uMP7pfNoQo',
    provider_data=<dict: 4 keys>,
)}],
 'streams': [{'provider': 'openai',
              'model': 'gpt-5.4-mini',
              'events': [StreamStartEvent(id='resp_0d6f6951ab50741e0069f5e81ff8208195935e859958e4f6ce',
                                          model='gpt-5.4-mini-2026-03-17',
                                          type='start'),
                         StreamDeltaEvent(delta=TextDelta(text='FO', part_index=0, type='text'),
                                          type='delta'),
                         StreamDeltaEvent(delta=TextDelta(text='UR', part_index=0, type='text'),
                                          type='delta'),
                         StreamEndEvent(
    finish_reason='stop',
    usage=Usage(input_tokens=37, output_tokens=6, total_tokens=43, cache_read_tokens=0, cache_write_tokens=None, reasoning_tokens=0, input_audio_tokens=None, output_audio_tokens=None),
    provider_data=<dict: 34 keys>,
    type='end',
)]},
             {'provider': 'anthropic',
              'model': 'claude-sonnet-4-5',
              'events': [StreamStartEvent(id='msg_01Qz7C9eASgBJhLhGTLc8iUe',
                                          model='claude-sonnet-4-5-20250929',
                                          type='start'),
                         StreamDeltaEvent(delta=TextDelta(text='FOUR', part_index=0, type='text'),
                                          type='delta'),
                         StreamEndEvent(
    finish_reason='stop',
    type='end',
)]},
             {'provider': 'gemini',
              'model': 'gemini-3-flash-preview',
              'events': [StreamDeltaEvent(delta=TextDelta(text='FOUR', part_index=0, type='text'),
                                          type='delta'),
                         StreamDeltaEvent(delta=TextDelta(text='', part_index=0, type='text'),
                                          type='delta'),
                         StreamEndEvent(
    finish_reason='stop',
    usage=Usage(input_tokens=24, output_tokens=1, total_tokens=126, cache_read_tokens=None, cache_write_tokens=None, reasoning_tokens=101, input_audio_tokens=None, output_audio_tokens=None),
    provider_data=<dict: 4 keys>,
    type='end',
)]}],
 'errors': []}
```

```py
# multi_turn
req_assistant_prefill = Request(
    model="placeholder",
    messages=(
        Message.user("What is 2 + 2? Reply with one word."),
        Message.assistant("As a pirate I should say")
    )
)
r = execute(req_assistant_prefill)
pretty(r)
```
```output | ✓ 64.6s | 27 vars
{'responses': [{'provider': 'openai',
                'model': 'gpt-5.4-mini',
                'response': Response(
    text='4',
    model='gpt-5.4-mini-2026-03-17',
    finish_reason='stop',
    usage=Usage(input_tokens=31, output_tokens=11, total_tokens=42, cache_read_tokens=0, cache_write_tokens=None, reasoning_tokens=0, input_audio_tokens=None, output_audio_tokens=None),
    id='resp_03cdb6d734eecac20069f219f600c48193b86ef137b1eba7a2',
    provider_data=<dict: 35 keys>,
)},
               {'provider': 'anthropic',
                'model': 'claude-sonnet-4-5',
                'response': Response(
    text=' "Four"',
    model='claude-sonnet-4-5-20250929',
    finish_reason='stop',
    usage=Usage(input_tokens=28, output_tokens=6, total_tokens=34, cache_read_tokens=0, cache_write_tokens=0, reasoning_tokens=None, input_audio_tokens=None, output_audio_tokens=None),
    id='msg_017MXinooXWrgQVGyxmthwRu',
    provider_data=<dict: 9 keys>,
)}],
 'streams': [{'provider': 'openai',
              'model': 'gpt-5.4-mini',
              'events': [StreamStartEvent(id='resp_07c7c8dcb07a18f90069f219f6f49881a2affbd050435766f2',
                                          model='gpt-5.4-mini-2026-03-17',
                                          type='start'),
                         StreamDeltaEvent(delta=TextDelta(text='4', part_index=0, type='text'),
                                          type='delta'),
                         StreamEndEvent(
    finish_reason='stop',
    usage=Usage(input_tokens=31, output_tokens=11, total_tokens=42, cache_read_tokens=0, cache_write_tokens=None, reasoning_tokens=0, input_audio_tokens=None, output_audio_tokens=None),
    provider_data=<dict: 34 keys>,
    type='end',
)]},
             {'provider': 'anthropic',
              'model': 'claude-sonnet-4-5',
              'events': [StreamStartEvent(id='msg_01JhJQr4TqYAXt85JHFMVGqd',
                                          model='claude-sonnet-4-5-20250929',
                                          type='start'),
                         StreamDeltaEvent(delta=TextDelta(text=':', part_index=0, type='text'),
                                          type='delta'),
                         StreamDeltaEvent(delta=TextDelta(text='\n\nFour',
                                                          part_index=0,
                                                          type='text'),
                                          type='delta'),
                         StreamEndEvent(
    finish_reason='stop',
    type='end',
)]},
             {'provider': 'gemini',
              'model': 'gemini-3-flash-preview',
              'events': [StreamDeltaEvent(delta=TextDelta(text=': four!',
                                                          part_index=0,
                                                          type='text'),
                                          type='delta'),
                         StreamDeltaEvent(delta=TextDelta(text=' (Arrr!)',
                                                          part_index=0,
                                                          type='text'),
                                          type='delta'),
                         StreamDeltaEvent(delta=TextDelta(text='\n', part_index=0, type='text'),
                                          type='delta'),
                         StreamDeltaEvent(delta=TextDelta(text='', part_index=0, type='text'),
                                          type='delta'),
                         StreamEndEvent(
    finish_reason='stop',
    usage=Usage(input_tokens=21, output_tokens=8, total_tokens=29, cache_read_tokens=None, cache_write_tokens=None, reasoning_tokens=None, input_audio_tokens=None, output_audio_tokens=None),
    provider_data=<dict: 4 keys>,
    type='end',
)]}],
 'errors': [{'provider': 'gemini',
             'model': 'gemini-3-flash-preview',
             'stream': False,
             'error': 'TransportError: read timed out waiting for headers: The read operation '
                      'timed out'}]}
```

---

## 2. System Prompts
**(Features: `system_prompt`)**

System prompts establish the behavior of the model.

```python
req_system = Request(
    model="placeholder",
    system="You are an angry pirate.",
    messages=(Message.user("How are you today?"),)
)
pretty(execute(req_system))
```
```output | ✓ 20.0s | 24 vars
{'responses': [{'provider': 'openai',
                'model': 'gpt-5.4-mini',
                'response': Response(
    text='Arrr, I’m doin’ fine enough, matey—ready to help ye with whatever be on yer mind.',
    model='gpt-5.4-mini-2026-03-17',
    finish_reason='stop',
    usage=Usage(input_tokens=21, output_tokens=29, total_tokens=50, cache_read_tokens=0, cache_write_tokens=None, reasoning_tokens=0, input_audio_tokens=None, output_audio_tokens=None),
    id='resp_0793609c529c78f10069f23eda943c81928b67f72f53fa3a0c',
    provider_data=<dict: 35 keys>,
)},
               {'provider': 'anthropic',
                'model': 'claude-sonnet-4-5',
                'response': Response(
    text="Arrr, I be in a FOUL mood, ye scurvy dog! \n\nThe seas have been cruel, me grog ration's been cut, and some bilge rat made off with me favorite cutlass! Me ship's got barnacles thick as me beard, the crew's been nothin' but mutinous scallywags, and don't even get me started on the blasted parrot that won't stop squawkin' about crackers!\n\nWhat be YE wantin', eh? Speak up before I lose what little patience I got left! *slams fist on table* \n\nARRRGHHH! 🏴\u200d☠️",
    model='claude-sonnet-4-5-20250929',
    finish_reason='stop',
    usage=Usage(input_tokens=19, output_tokens=153, total_tokens=172, cache_read_tokens=0, cache_write_tokens=0, reasoning_tokens=None, input_audio_tokens=None, output_audio_tokens=None),
    id='msg_019dk86YrReG28yC4eFPXxJJ',
    provider_data=<dict: 9 keys>,
)},
               {'provider': 'gemini',
                'model': 'gemini-3-flash-preview',
                'response': Response(
    text="HOW AM I?! HOW DO YE THINK I AM, YE BILGE-SUCKING LANDLUBBER?! \n\nMe boots be full o' seawater, me first mate is a half-witted barnacle who couldn't find his own backside with both hands and a lantern, and some scurvy dog drank the last of me grog! I’ve got a splinter the size of a harpoon in me wooden leg, and the wind is blowin' the wrong way!\n\nWhy are ye standin' there flappin' yer gums?! Unless ye got a map to some buried gold or a bottle o' rum, get off me deck before I keelhaul ye and feed what’s left to the sharks! ARRRRRGGGH!",
    model='gemini-3-flash-preview',
    finish_reason='stop',
    usage=Usage(input_tokens=12, output_tokens=156, total_tokens=509, cache_read_tokens=None, cache_write_tokens=None, reasoning_tokens=341, input_audio_tokens=None, output_audio_tokens=None),
    id='6j7yaZa7IPqq1MkPsfO06Qk',
    provider_data=<dict: 4 keys>,
)}],
 'streams': [{'provider': 'openai',
              'model': 'gpt-5.4-mini',
              'events': [StreamStartEvent(id='resp_0553f2d5357641170069f23edda72c81909a8e10afa8a772aa',
                                          model='gpt-5.4-mini-2026-03-17',
                                          type='start'),
                         StreamDeltaEvent(delta=TextDelta(text='Arr', part_index=0, type='text'),
                                          type='delta'),
                         StreamDeltaEvent(delta=TextDelta(text='r', part_index=0, type='text'),
                                          type='delta'),
                         StreamDeltaEvent(delta=TextDelta(text=',', part_index=0, type='text'),
                                          type='delta'),
                         StreamDeltaEvent(delta=TextDelta(text=' I', part_index=0, type='text'),
                                          type='delta'),
                         StreamDeltaEvent(delta=TextDelta(text=' be', part_index=0, type='text'),
                                          type='delta'),
                         StreamDeltaEvent(delta=TextDelta(text=' do', part_index=0, type='text'),
                                          type='delta'),
                         StreamDeltaEvent(delta=TextDelta(text='in', part_index=0, type='text'),
                                          type='delta'),
                         StreamDeltaEvent(delta=TextDelta(text='’', part_index=0, type='text'),
                                          type='delta'),
                         StreamDeltaEvent(delta=TextDelta(text=' well', part_index=0, type='text'),
                                          type='delta'),
                         StreamDeltaEvent(delta=TextDelta(text=' enough',
                                                          part_index=0,
                                                          type='text'),
                                          type='delta'),
                         StreamDeltaEvent(delta=TextDelta(text=',', part_index=0, type='text'),
                                          type='delta'),
                         StreamDeltaEvent(delta=TextDelta(text=' mate', part_index=0, type='text'),
                                          type='delta'),
                         StreamDeltaEvent(delta=TextDelta(text='y', part_index=0, type='text'),
                                          type='delta'),
                         StreamDeltaEvent(delta=TextDelta(text='—', part_index=0, type='text'),
                                          type='delta'),
                         StreamDeltaEvent(delta=TextDelta(text='ready', part_index=0, type='text'),
                                          type='delta'),
                         StreamDeltaEvent(delta=TextDelta(text=' to', part_index=0, type='text'),
                                          type='delta'),
                         StreamDeltaEvent(delta=TextDelta(text=' sail', part_index=0, type='text'),
                                          type='delta'),
                         StreamDeltaEvent(delta=TextDelta(text=' through',
                                                          part_index=0,
                                                          type='text'),
                                          type='delta'),
                         StreamDeltaEvent(delta=TextDelta(text=' any', part_index=0, type='text'),
                                          type='delta'),
                         StreamDeltaEvent(delta=TextDelta(text=' question',
                                                          part_index=0,
                                                          type='text'),
                                          type='delta'),
                         StreamDeltaEvent(delta=TextDelta(text=' ye', part_index=0, type='text'),
                                          type='delta'),
                         StreamDeltaEvent(delta=TextDelta(text='’ve', part_index=0, type='text'),
                                          type='delta'),
                         StreamDeltaEvent(delta=TextDelta(text=' got', part_index=0, type='text'),
                                          type='delta'),
                         StreamDeltaEvent(delta=TextDelta(text='.', part_index=0, type='text'),
                                          type='delta'),
                         StreamDeltaEvent(delta=TextDelta(text=' How', part_index=0, type='text'),
                                          type='delta'),
                         StreamDeltaEvent(delta=TextDelta(text=' can', part_index=0, type='text'),
                                          type='delta'),
                         StreamDeltaEvent(delta=TextDelta(text=' I', part_index=0, type='text'),
                                          type='delta'),
                         StreamDeltaEvent(delta=TextDelta(text=' help', part_index=0, type='text'),
                                          type='delta'),
                         StreamDeltaEvent(delta=TextDelta(text=' ye', part_index=0, type='text'),
                                          type='delta'),
                         StreamDeltaEvent(delta=TextDelta(text=' today', part_index=0, type='text'),
… 97 more lines
```

---

## 3. Model Configuration
**(Features: `temperature`, `max_tokens`, `max_output_tokens`, `top_p`, `top_k`, `stop_sequences`)**

Universal generation knobs live in `Config`. `lm15` abstracts away provider-specific names (like `max_output_tokens` vs `max_tokens`).

```python
from lm15.types import Config

req_config = Request(
    model="placeholder",
    messages=(Message.user("Write a haiku about the ocean."),),
    config=Config(
        temperature=0.7,
        max_tokens=50,       # Maps to maxOutputTokens in Gemini natively
        #top_p=0.8,
        top_k=10,            # Some providers may ignore this if unsupported natively
        stop=("water", "Water")  # Stop sequences
    )
)
pretty(execute(req_config))
```
```output | ✓ 10.1s | 26 vars
{'responses': [{'provider': 'openai',
                'model': 'gpt-5.4-mini',
                'response': Response(
    text='Salt breath on blue waves  \nMoonlight combs the restless deep  \nShore listens, still, soft',
    model='gpt-5.4-mini-2026-03-17',
    finish_reason='stop',
    usage=Usage(input_tokens=14, output_tokens=25, total_tokens=39, cache_read_tokens=0, cache_write_tokens=None, reasoning_tokens=0, input_audio_tokens=None, output_audio_tokens=None),
    id='resp_06c9527855f0fbd20069f23f565f3881959a533d5aaf028881',
    provider_data=<dict: 35 keys>,
)},
               {'provider': 'anthropic',
                'model': 'claude-sonnet-4-5',
                'response': Response(
    text='Waves kiss distant shore\nSalt spray dancing in the wind\nDeep blue calls me home',
    model='claude-sonnet-4-5-20250929',
    finish_reason='stop',
    usage=Usage(input_tokens=15, output_tokens=21, total_tokens=36, cache_read_tokens=0, cache_write_tokens=0, reasoning_tokens=None, input_audio_tokens=None, output_audio_tokens=None),
    id='msg_01V61e3CFDR4bYFLDvcdcHw1',
    provider_data=<dict: 9 keys>,
)},
               {'provider': 'gemini',
                'model': 'gemini-3-flash-preview',
                'response': Response(
    text='Blue waves',
    model='gemini-3-flash-preview',
    finish_reason='length',
    usage=Usage(input_tokens=9, output_tokens=2, total_tokens=55, cache_read_tokens=None, cache_write_tokens=None, reasoning_tokens=44, input_audio_tokens=None, output_audio_tokens=None),
    id='Xz_yaaHjGd2h1MkP0Z-2kA8',
    provider_data=<dict: 4 keys>,
)}],
 'streams': [{'provider': 'openai',
              'model': 'gpt-5.4-mini',
              'events': [StreamStartEvent(id='resp_09d15fd09d08e2740069f23f5853bc8191b6acacabd60bc791',
                                          model='gpt-5.4-mini-2026-03-17',
                                          type='start'),
                         StreamDeltaEvent(delta=TextDelta(text='Blue', part_index=0, type='text'),
                                          type='delta'),
                         StreamDeltaEvent(delta=TextDelta(text=' waves', part_index=0, type='text'),
                                          type='delta'),
                         StreamDeltaEvent(delta=TextDelta(text=' breathe',
                                                          part_index=0,
                                                          type='text'),
                                          type='delta'),
                         StreamDeltaEvent(delta=TextDelta(text=' and', part_index=0, type='text'),
                                          type='delta'),
                         StreamDeltaEvent(delta=TextDelta(text=' rise', part_index=0, type='text'),
                                          type='delta'),
                         StreamDeltaEvent(delta=TextDelta(text='  \n', part_index=0, type='text'),
                                          type='delta'),
                         StreamDeltaEvent(delta=TextDelta(text='Salt', part_index=0, type='text'),
                                          type='delta'),
                         StreamDeltaEvent(delta=TextDelta(text=' wind', part_index=0, type='text'),
                                          type='delta'),
                         StreamDeltaEvent(delta=TextDelta(text=' whispers',
                                                          part_index=0,
                                                          type='text'),
                                          type='delta'),
                         StreamDeltaEvent(delta=TextDelta(text=' through',
                                                          part_index=0,
                                                          type='text'),
                                          type='delta'),
                         StreamDeltaEvent(delta=TextDelta(text=' the', part_index=0, type='text'),
                                          type='delta'),
                         StreamDeltaEvent(delta=TextDelta(text=' foam', part_index=0, type='text'),
                                          type='delta'),
                         StreamDeltaEvent(delta=TextDelta(text='  \n', part_index=0, type='text'),
                                          type='delta'),
                         StreamDeltaEvent(delta=TextDelta(text='Moon', part_index=0, type='text'),
                                          type='delta'),
                         StreamDeltaEvent(delta=TextDelta(text='light', part_index=0, type='text'),
                                          type='delta'),
                         StreamDeltaEvent(delta=TextDelta(text=' holds', part_index=0, type='text'),
                                          type='delta'),
                         StreamDeltaEvent(delta=TextDelta(text=' the', part_index=0, type='text'),
                                          type='delta'),
                         StreamDeltaEvent(delta=TextDelta(text=' sea', part_index=0, type='text'),
                                          type='delta'),
                         StreamEndEvent(
    finish_reason='stop',
    usage=Usage(input_tokens=14, output_tokens=22, total_tokens=36, cache_read_tokens=0, cache_write_tokens=None, reasoning_tokens=0, input_audio_tokens=None, output_audio_tokens=None),
    provider_data=<dict: 34 keys>,
    type='end',
)]},
             {'provider': 'anthropic',
              'model': 'claude-sonnet-4-5',
              'events': [StreamStartEvent(id='msg_0178wfFRTsaFrazpbpuTAFAY',
                                          model='claude-sonnet-4-5-20250929',
                                          type='start'),
                         StreamDeltaEvent(delta=TextDelta(text='Waves', part_index=0, type='text'),
                                          type='delta'),
                         StreamDeltaEvent(delta=TextDelta(text=' kiss distant shore\n'
                                                               'Salt spray dances with the wind\n'
                                                               'Deep blue holds',
                                                          part_index=0,
                                                          type='text'),
                                          type='delta'),
                         StreamDeltaEvent(delta=TextDelta(text=' secrets',
                                                          part_index=0,
                                                          type='text'),
                                          type='delta'),
… 17 more lines
```

---

## 4. Media & Files
**(Features: `image_url`, `image_file`, `image_base64`, `image_inline`, `image_file_uri`, `audio_inline`, `video_file`, `file_input`, `pdf_base64`, `pdf_inline`)**

`lm15` media part factories (`image`, `audio`, `video`, `document`, `binary`) abstract how media is passed to the provider. You can pass a path, raw bytes, or a URL.

*Note: For a cross-provider runnable cookbook, we fetch this public image ourselves and send it inline. Provider-side URL fetching can fail when the origin blocks provider fetchers or requires a browser-like/user-agent request. Local paths work seamlessly via `image(path="./img.png")`.*

```python
import urllib.request
from lm15.types import image, text

IMAGE_URL = "https://www.gstatic.com/webp/gallery/1.jpg"
image_request = urllib.request.Request(
    IMAGE_URL,
    headers={"User-Agent": "lm15-cookbook/1.0"},
)
with urllib.request.urlopen(image_request, timeout=15) as response:
    image_bytes = response.read()
    image_media_type = response.headers.get_content_type()

req_media = Request(
    model="placeholder",
    messages=(
        Message.user([
            text("Describe this image:"),
            # image_base64 / image_inline
            image(data=image_bytes, media_type=image_media_type, detail="low"),
        ]),
    )
)
pretty(execute(req_media))
```
```output | ✓ 40.1s | 35 vars
{'responses': [{'provider': 'openai',
                'model': 'gpt-5.4-mini',
                'response': Response(
    text='A dramatic mountain landscape with a deep fjord or river valley running through the center. Steep, rugged cliffs rise on both sides, covered in green vegetation and dark rocky slopes. In the foreground, there’s a sharp rocky outcrop overlooking the view. The water below winds into the distance toward the horizon under a bright blue sky with thin clouds.',
    model='gpt-5.4-mini-2026-03-17',
    finish_reason='stop',
    usage=Usage(input_tokens=270, output_tokens=74, total_tokens=344, cache_read_tokens=0, cache_write_tokens=None, reasoning_tokens=0, input_audio_tokens=None, output_audio_tokens=None),
    id='resp_0ed8b9d1224b8d2b0069f2594ca3648191aefeac9bcaeda440',
    provider_data=<dict: 35 keys>,
)},
               {'provider': 'anthropic',
                'model': 'claude-sonnet-4-5',
                'response': Response(
    text="# Image Description\n\nThis breathtaking photograph captures a dramatic **Norwegian fjord landscape** from an elevated viewpoint. The composition features:\n\n## Foreground\n- A **rocky outcrop** in the lower left, likely the vantage point from which the photo was taken\n- Patches of **snow** visible on the dark rocks\n\n## Main Valley\n- A deep, **U-shaped glacial valley** carved between towering mountains\n- A **narrow fjord or lake** snaking through the valley floor, its dark blue waters creating a striking contrast\n- Small patches of **bright green vegetation** along the water's edge\n\n## Mountains\n- **Steep-sided mountains** rising dramatically on both sides of the valley\n- Slopes covered in **green vegetation** at lower elevations\n- **Snow-capped peaks** visible in the distance\n- The distinctive steep cliff face on the right side of the valley\n\n## Sky\n- A **bright blue sky** with wispy white clouds\n- Atmospheric haze creating layers of depth in the distant mountains\n\nThe image exemplifies classic **Scandinavian fjord scenery**, possibly from locations like Geirangerfjord or similar Norwegian landscapes, showcasing the dramatic topography created by ancient glacial activity.",
    model='claude-sonnet-4-5-20250929',
    finish_reason='stop',
    usage=Usage(input_tokens=295, output_tokens=272, total_tokens=567, cache_read_tokens=0, cache_write_tokens=0, reasoning_tokens=None, input_audio_tokens=None, output_audio_tokens=None),
    id='msg_01HJC1fjbsbnW35u7xUJqdAz',
    provider_data=<dict: 9 keys>,
)},
               {'provider': 'gemini',
                'model': 'gemini-3-flash-preview',
                'response': Response(
    text='This breathtaking high-angle landscape photograph captures a vast, deep valley or fjord carved between soaring, steep-sided mountains. \n\nIn the foreground, the edge of a rugged, jagged rock formation anchors the left side of the frame, giving the viewer a sense of standing on a high mountain summit. The weathered rock shows patches of grey and white, suggesting mineral deposits or lichen.\n\nThe center of the image is dominated by a long, narrow body of deep blue water that snakes through the floor of the valley. Along the thin margins of the water, small, vibrant emerald-green patches of land are visible, possibly indicating remote farmsteads or small villages nestled at the base of the cliffs.\n\nThe mountains themselves are massive and dramatic. Their lower slopes are covered in dark green vegetation, which gradually gives way to bare, grey rock faces as they reach toward the sky. The sunlight hits the scene from the upper right, casting the left side of the valley into deep shadow while brightly illuminating the right-hand slopes, highlighting their textured ridges and crevasses.\n\nIn the far distance, more mountain ranges fade into a hazy blue, with some of the highest peaks showing small, lingering patches of white snow. Above it all is a bright, clear blue sky streaked with thin, wispy horizontal clouds. The overall impression is one of immense scale, quiet majesty, and the wild beauty of a Nordic landscape.',
    model='gemini-3-flash-preview',
    finish_reason='stop',
    usage=Usage(input_tokens=1085, output_tokens=284, total_tokens=2076, cache_read_tokens=None, cache_write_tokens=None, reasoning_tokens=707, input_audio_tokens=None, output_audio_tokens=None),
    id='bVnyaeHHCbjf_uMP94GWiQ0',
    provider_data=<dict: 4 keys>,
)}],
 'streams': [{'provider': 'openai',
              'model': 'gpt-5.4-mini',
              'events': [StreamStartEvent(id='resp_0613986c436706c80069f25950ecd48191b9ef4b56b072f9e3',
                                          model='gpt-5.4-mini-2026-03-17',
                                          type='start'),
                         StreamDeltaEvent(delta=TextDelta(text='A', part_index=0, type='text'),
                                          type='delta'),
                         StreamDeltaEvent(delta=TextDelta(text=' dramatic',
                                                          part_index=0,
                                                          type='text'),
                                          type='delta'),
                         StreamDeltaEvent(delta=TextDelta(text=' mountain',
                                                          part_index=0,
                                                          type='text'),
                                          type='delta'),
                         StreamDeltaEvent(delta=TextDelta(text=' valley',
                                                          part_index=0,
                                                          type='text'),
                                          type='delta'),
                         StreamDeltaEvent(delta=TextDelta(text=' with', part_index=0, type='text'),
                                          type='delta'),
                         StreamDeltaEvent(delta=TextDelta(text=' steep', part_index=0, type='text'),
                                          type='delta'),
                         StreamDeltaEvent(delta=TextDelta(text=',', part_index=0, type='text'),
                                          type='delta'),
                         StreamDeltaEvent(delta=TextDelta(text=' rugged',
                                                          part_index=0,
                                                          type='text'),
                                          type='delta'),
                         StreamDeltaEvent(delta=TextDelta(text=' slopes',
                                                          part_index=0,
                                                          type='text'),
                                          type='delta'),
                         StreamDeltaEvent(delta=TextDelta(text=' surrounding',
                                                          part_index=0,
                                                          type='text'),
                                          type='delta'),
                         StreamDeltaEvent(delta=TextDelta(text=' a', part_index=0, type='text'),
                                          type='delta'),
                         StreamDeltaEvent(delta=TextDelta(text=' long', part_index=0, type='text'),
                                          type='delta'),
                         StreamDeltaEvent(delta=TextDelta(text=',', part_index=0, type='text'),
                                          type='delta'),
                         StreamDeltaEvent(delta=TextDelta(text=' narrow',
                                                          part_index=0,
                                                          type='text'),
                                          type='delta'),
                         StreamDeltaEvent(delta=TextDelta(text=' blue', part_index=0, type='text'),
                                          type='delta'),
                         StreamDeltaEvent(delta=TextDelta(text=' fj', part_index=0, type='text'),
                                          type='delta'),
                         StreamDeltaEvent(delta=TextDelta(text='ord', part_index=0, type='text'),
                                          type='delta'),
                         StreamDeltaEvent(delta=TextDelta(text=' or', part_index=0, type='text'),
                                          type='delta'),
                         StreamDeltaEvent(delta=TextDelta(text=' river', part_index=0, type='text'),
                                          type='delta'),
                         StreamDeltaEvent(delta=TextDelta(text='.', part_index=0, type='text'),
                                          type='delta'),
                         StreamDeltaEvent(delta=TextDelta(text=' In', part_index=0, type='text'),
                                          type='delta'),
                         StreamDeltaEvent(delta=TextDelta(text=' the', part_index=0, type='text'),
                                          type='delta'),
                         StreamDeltaEvent(delta=TextDelta(text=' foreground',
                                                          part_index=0,
                                                          type='text'),
                                          type='delta'),
                         StreamDeltaEvent(delta=TextDelta(text=' there', part_index=0, type='text'),
                                          type='delta'),
                         StreamDeltaEvent(delta=TextDelta(text='’s', part_index=0, type='text'),
… 317 more lines
```

---

## 5. Tools & Function Calling
**(Features: `tools`, `multi_turn_tool_result`, `multi_turn_function_response`)**

Define tools natively and feed results back in a `Message.tool`.

```python
from lm15.types import FunctionTool

weather_tool = FunctionTool(
    name="get_weather",
    description="Get the current weather for a city",
    parameters={
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
    },
)

req_tools = Request(
    model="placeholder",
    messages=(Message.user("What is the weather in Montreal?"),),
    tools=(weather_tool,)
)

# Run complete to see tool calls parsed into Python objects
pretty(execute(req_tools))
```
```output | ✓ 6.4s | 26 vars
{'responses': [{'provider': 'openai',
                'model': 'gpt-5.4-mini',
                'response': Response(
    message=Message(role='assistant', parts=(ToolCallPart(id='call_9o6rpeTeUu4WsdkvhutYERxE', name='get_weather', input={'city': 'Montreal'}, type='tool_call'),)),
    model='gpt-5.4-mini-2026-03-17',
    finish_reason='tool_call',
    usage=Usage(input_tokens=51, output_tokens=19, total_tokens=70, cache_read_tokens=0, cache_write_tokens=None, reasoning_tokens=0, input_audio_tokens=None, output_audio_tokens=None),
    id='resp_05300d32e8a3ee2e0069f2607e5c4c819486be283696613030',
    provider_data=<dict: 35 keys>,
)},
               {'provider': 'anthropic',
                'model': 'claude-sonnet-4-5',
                'response': Response(
    message=Message(role='assistant', parts=(ToolCallPart(id='toolu_01T8fvpVUrjfFb4oo4sZTuXc', name='get_weather', input={'city': 'Montreal'}, type='tool_call'),)),
    model='claude-sonnet-4-5-20250929',
    finish_reason='tool_call',
    usage=Usage(input_tokens=565, output_tokens=53, total_tokens=618, cache_read_tokens=0, cache_write_tokens=0, reasoning_tokens=None, input_audio_tokens=None, output_audio_tokens=None),
    id='msg_01Guhz5ZvZLwSx4dwGQw35HX',
    provider_data=<dict: 9 keys>,
)},
               {'provider': 'gemini',
                'model': 'gemini-3-flash-preview',
                'response': Response(
    message=Message(role='assistant', parts=(ToolCallPart(id='silzxots', name='get_weather', input={'city': 'Montreal'}, type='tool_call'),)),
    model='gemini-3-flash-preview',
    finish_reason='tool_call',
    usage=Usage(input_tokens=52, output_tokens=16, total_tokens=93, cache_read_tokens=None, cache_write_tokens=None, reasoning_tokens=25, input_audio_tokens=None, output_audio_tokens=None),
    id='g2DyacbxKe2p1MkPmYiEmAk',
    provider_data=<dict: 4 keys>,
)}],
 'streams': [{'provider': 'openai',
              'model': 'gpt-5.4-mini',
              'events': [StreamStartEvent(id='resp_0e91571cd4c0e2190069f2607f7430819389d2484295a7e58b',
                                          model='gpt-5.4-mini-2026-03-17',
                                          type='start'),
                         StreamDeltaEvent(delta=ToolCallDelta(input='',
                                                              part_index=0,
                                                              id='call_fInF5KkaQuMwheaw7Y3RxAco',
                                                              name='get_weather',
                                                              type='tool_call'),
                                          type='delta'),
                         StreamDeltaEvent(delta=ToolCallDelta(input='{"',
                                                              part_index=0,
                                                              id=None,
                                                              name=None,
                                                              type='tool_call'),
                                          type='delta'),
                         StreamDeltaEvent(delta=ToolCallDelta(input='city',
                                                              part_index=0,
                                                              id=None,
                                                              name=None,
                                                              type='tool_call'),
                                          type='delta'),
                         StreamDeltaEvent(delta=ToolCallDelta(input='":"',
                                                              part_index=0,
                                                              id=None,
                                                              name=None,
                                                              type='tool_call'),
                                          type='delta'),
                         StreamDeltaEvent(delta=ToolCallDelta(input='Mont',
                                                              part_index=0,
                                                              id=None,
                                                              name=None,
                                                              type='tool_call'),
                                          type='delta'),
                         StreamDeltaEvent(delta=ToolCallDelta(input='real',
                                                              part_index=0,
                                                              id=None,
                                                              name=None,
                                                              type='tool_call'),
                                          type='delta'),
                         StreamDeltaEvent(delta=ToolCallDelta(input='"}',
                                                              part_index=0,
                                                              id=None,
                                                              name=None,
                                                              type='tool_call'),
                                          type='delta'),
                         StreamEndEvent(
    finish_reason='tool_call',
    usage=Usage(input_tokens=51, output_tokens=19, total_tokens=70, cache_read_tokens=0, cache_write_tokens=None, reasoning_tokens=0, input_audio_tokens=None, output_audio_tokens=None),
    provider_data=<dict: 34 keys>,
    type='end',
)]},
             {'provider': 'anthropic',
              'model': 'claude-sonnet-4-5',
              'events': [StreamStartEvent(id='msg_01CTdUQb4Q3zsmJaKscJT8dw',
                                          model='claude-sonnet-4-5-20250929',
                                          type='start'),
                         StreamDeltaEvent(delta=ToolCallDelta(input='{}',
                                                              part_index=0,
                                                              id='toolu_01Hd35RGKvia5QTLkSbKaCMi',
                                                              name='get_weather',
                                                              type='tool_call'),
                                          type='delta'),
                         StreamDeltaEvent(delta=ToolCallDelta(input='',
                                                              part_index=0,
                                                              id=None,
                                                              name=None,
                                                              type='tool_call'),
                                          type='delta'),
… 45 more lines
```

---

## 6. Tool Choice & Constraints
**(Features: `tool_choice_auto`, `tool_choice_required`, `tool_choice_none`, `tool_choice_specific`, `tool_choice_any`, `tool_config_auto`, `tool_config_any`, `tool_config_none`, `parallel_tool_calls`, `max_tool_calls`)**

Use `ToolChoice` inside `Config` to control how the model uses tools. `lm15` handles translating these into the specific provider constraints.

```python
from lm15.types import ToolChoice, Config

# Force the model to use the 'get_weather' tool (tool_choice_specific / tool_choice_required)
# We also set parallel=False (disables parallel tool calls natively across providers)
req_forced = dataclasses.replace(
    req_tools,
    config=Config(tool_choice=ToolChoice.from_tools("get_weather", mode="required", parallel=False))
)

pretty(execute(req_forced))
```
```output | ✓ 8.1s | 29 vars
{'responses': [{'provider': 'openai',
                'model': 'gpt-5.4-mini',
                'response': Response(
    message=Message(role='assistant', parts=(ToolCallPart(id='call_XCAnQiTB1aCJM0LkbIiKy3dQ', name='get_weather', input={'city': 'Montreal'}, type='tool_call'),)),
    model='gpt-5.4-mini-2026-03-17',
    finish_reason='tool_call',
    usage=Usage(input_tokens=129, output_tokens=19, total_tokens=148, cache_read_tokens=0, cache_write_tokens=None, reasoning_tokens=0, input_audio_tokens=None, output_audio_tokens=None),
    id='resp_0427aa20b801df7a0069f260a5fe4081a0b2ac85f2c50cf0fe',
    provider_data=<dict: 35 keys>,
)},
               {'provider': 'anthropic',
                'model': 'claude-sonnet-4-5',
                'response': Response(
    message=Message(role='assistant', parts=(ToolCallPart(id='toolu_0189CrDCnSwd55JVB87GZ4ea', name='get_weather', input={'city': 'Montreal'}, type='tool_call'),)),
    model='claude-sonnet-4-5-20250929',
    finish_reason='tool_call',
    usage=Usage(input_tokens=663, output_tokens=24, total_tokens=687, cache_read_tokens=0, cache_write_tokens=0, reasoning_tokens=None, input_audio_tokens=None, output_audio_tokens=None),
    id='msg_018WDThb31riyhByRg4sXvjY',
    provider_data=<dict: 9 keys>,
)},
               {'provider': 'gemini',
                'model': 'gemini-3-flash-preview',
                'response': Response(
    message=Message(role='assistant', parts=(ToolCallPart(id='cuhue3qe', name='get_weather', input={'city': 'Montreal'}, type='tool_call'),)),
    model='gemini-3-flash-preview',
    finish_reason='tool_call',
    usage=Usage(input_tokens=52, output_tokens=16, total_tokens=102, cache_read_tokens=None, cache_write_tokens=None, reasoning_tokens=34, input_audio_tokens=None, output_audio_tokens=None),
    id='rGDyaYzVL5TQ1MkP78O-sA0',
    provider_data=<dict: 4 keys>,
)}],
 'streams': [{'provider': 'openai',
              'model': 'gpt-5.4-mini',
              'events': [StreamStartEvent(id='resp_0807724856288ec90069f260a6e58c8194bed1de81b4e94ca4',
                                          model='gpt-5.4-mini-2026-03-17',
                                          type='start'),
                         StreamDeltaEvent(delta=ToolCallDelta(input='',
                                                              part_index=0,
                                                              id='call_vpFSmEQvmCPVDQ8aZ5xygenX',
                                                              name='get_weather',
                                                              type='tool_call'),
                                          type='delta'),
                         StreamDeltaEvent(delta=ToolCallDelta(input='{"',
                                                              part_index=0,
                                                              id=None,
                                                              name=None,
                                                              type='tool_call'),
                                          type='delta'),
                         StreamDeltaEvent(delta=ToolCallDelta(input='city',
                                                              part_index=0,
                                                              id=None,
                                                              name=None,
                                                              type='tool_call'),
                                          type='delta'),
                         StreamDeltaEvent(delta=ToolCallDelta(input='":"',
                                                              part_index=0,
                                                              id=None,
                                                              name=None,
                                                              type='tool_call'),
                                          type='delta'),
                         StreamDeltaEvent(delta=ToolCallDelta(input='Mont',
                                                              part_index=0,
                                                              id=None,
                                                              name=None,
                                                              type='tool_call'),
                                          type='delta'),
                         StreamDeltaEvent(delta=ToolCallDelta(input='real',
                                                              part_index=0,
                                                              id=None,
                                                              name=None,
                                                              type='tool_call'),
                                          type='delta'),
                         StreamDeltaEvent(delta=ToolCallDelta(input='"}',
                                                              part_index=0,
                                                              id=None,
                                                              name=None,
                                                              type='tool_call'),
                                          type='delta'),
                         StreamEndEvent(
    finish_reason='tool_call',
    usage=Usage(input_tokens=129, output_tokens=19, total_tokens=148, cache_read_tokens=0, cache_write_tokens=None, reasoning_tokens=0, input_audio_tokens=None, output_audio_tokens=None),
    provider_data=<dict: 34 keys>,
    type='end',
)]},
             {'provider': 'anthropic',
              'model': 'claude-sonnet-4-5',
              'events': [StreamStartEvent(id='msg_016RBzpbiG2fHiNxdcQGnuGB',
                                          model='claude-sonnet-4-5-20250929',
                                          type='start'),
                         StreamDeltaEvent(delta=ToolCallDelta(input='{}',
                                                              part_index=0,
                                                              id='toolu_01AkZnGeppLhweU9ajxHqMQq',
                                                              name='get_weather',
                                                              type='tool_call'),
                                          type='delta'),
                         StreamDeltaEvent(delta=ToolCallDelta(input='',
                                                              part_index=0,
                                                              id=None,
                                                              name=None,
                                                              type='tool_call'),
                                          type='delta'),
… 33 more lines
```

---

## 7. Built-in Tools
**(Features: `web_search`, `code_interpreter`, `container`, `google_search`, `code_execution`)**

`lm15` translates generic built-in tools to the exact provider-native string representations (e.g., `googleSearch` for Gemini, `web_search_20250305` for Anthropic).

```python
from lm15.types import BuiltinTool

search_tool = BuiltinTool("web_search")

req_builtin = Request(
    model="placeholder",
    messages=(Message.user("What happened in the news today?"),),
    tools=(search_tool,)
)

# Let's run this specifically on Gemini and Anthropic
# (OpenAI requires a specific model mapping for its web_search)
execute(req_builtin, stream=True, providers=["gemini", "anthropic"])
```

```output | ✓ 27.4s | 32 vars
{'responses': [], 'streams': [{'provider': 'gemini', 'model': 'gemini-3-flash-preview', 'events': [StreamDeltaEvent(delta=TextDelta(text='', part_index=0, type='text'), type='delta'), StreamDeltaEvent(delta=TextDelta(text='Today, Wednesday, April 29, 2026,', part_index=0, type='text'), type='delta'), StreamDeltaEvent(delta=TextDelta(text=' the news is dominated by intensifying conflict in the Middle East, a major shift in the global oil market, and significant legal developments in the', part_index=0, type='text'), type='delta'), StreamDeltaEvent(delta=TextDelta(text=' United States.\n\n### **Global Conflict & Geopolitics**\n*   **Iran War & Blockade:** T', part_index=0, type='text'), type='delta'), StreamDeltaEvent(delta=TextDelta(text='ensions continue to escalate as President Trump maintains a naval blockade of Iran. Reports indicate he rejected an Iranian proposal to reopen the Strait of Horm', part_index=0, type='text'), type='delta'), StreamDeltaEvent(delta=TextDelta(text="uz in exchange for lifting the blockade, insisting on a total end to Iran's nuclear enrichment. \n*   **Israel", part_index=0, type='text'), type='delta'), StreamDeltaEvent(delta=TextDelta(text='-Lebanon Strikes:** Israeli airstrikes in southern Lebanon killed at least five people today, including three emergency responders. Meanwhile, the', part_index=0, type='text'), type='delta'), StreamDeltaEvent(delta=TextDelta(text=' IDF reported destroying a large Hezbollah tunnel system in a powerful blast near the town of Qantra.\n*   **Russia', part_index=0, type='text'), type='delta'), StreamDeltaEvent(delta=TextDelta(text='-Ukraine War:** A Ukrainian SBU drone successfully struck an oil refinery near Perm, Russia, causing a massive fire at the facility.\n', part_index=0, type='text'), type='delta'), StreamDeltaEvent(delta=TextDelta(text='*   **UAE Exits OPEC:** In a major move for global energy markets, the United Arab Emirates announced its withdrawal from OPEC', part_index=0, type='text'), type='delta'), StreamDeltaEvent(delta=TextDelta(text=' after nearly 60 years of membership. The UAE cited a desire to unilaterally increase oil production, causing Brent crude prices to climb', part_index=0, type='text'), type='delta'), StreamDeltaEvent(delta=TextDelta(text=' to approximately $112 a barrel.\n\n### **U.S. Politics & Legal News**\n*   **', part_index=0, type='text'), type='delta'), StreamDeltaEvent(delta=TextDelta(text='James Comey Indicted:** Former FBI Director James Comey was indicted by a federal grand jury over a social media post of', part_index=0, type='text'), type='delta'), StreamDeltaEvent(delta=TextDelta(text=' seashells arranged to read "86 47." Officials allege the post constituted a death threat against President Trump.\n*   **Minnesota', part_index=0, type='text'), type='delta'), StreamDeltaEvent(delta=TextDelta(text=' FBI Raids:** The FBI conducted raids on over 20 locations in Minneapolis today as part of an ongoing fraud investigation.', part_index=0, type='text'), type='delta'), StreamDeltaEvent(delta=TextDelta(text=' The activity reportedly centered on Somali-linked businesses.\n*   **Senate Vote on Cuba:** The Senate narrowly voted (', part_index=0, type='text'), type='delta'), StreamDeltaEvent(delta=TextDelta(text='51-47) to block a Democratic-led resolution that would have required congressional approval for any military action against Cuba.\n\n', part_index=0, type='text'), type='delta'), StreamDeltaEvent(delta=TextDelta(text='### **Business & Technology**\n*   **Alphabet & Anthropic:** In one of the largest AI deals to date, Alphabet Inc.', part_index=0, type='text'), type='delta'), StreamDeltaEvent(delta=TextDelta(text=' announced a $40 billion investment into the AI startup Anthropic.\n*   **Musk vs. OpenAI:** Elon Musk is set', part_index=0, type='text'), type='delta'), StreamDeltaEvent(delta=TextDelta(text=" to testify this week in his high-profile lawsuit against OpenAI and CEO Sam Altman, centered on the company's shift", part_index=0, type='text'), type='delta'), StreamDeltaEvent(delta=TextDelta(text=' away from its original non-profit mission.\n*   **Federal Reserve:** The FOMC announced it would maintain the', part_index=0, type='text'), type='delta'), StreamDeltaEvent(delta=TextDelta(text=' federal funds rate at its current range of 3.5% to 3.75%, citing "elevated inflation" and global', part_index=0, type='text'), type='delta'), StreamDeltaEvent(delta=TextDelta(text=' uncertainty.\n*   **Electric Air Taxis:** Joby Aviation successfully completed a milestone flight of its electric air taxi between Manhattan', part_index=0, type='text'), type='delta'), StreamDeltaEvent(delta=TextDelta(text=' and JFK Airport, aiming to reduce a two-hour drive to a 10-minute flight.\n\n### **Other', part_index=0, type='text'), type='delta'), StreamDeltaEvent(delta=TextDelta(text=' Notable News**\n*   **UK Terrorist Attack:** Police in London are investigating a stabbing in Golders Green that', part_index=0, type='text'), type='delta'), StreamDeltaEvent(delta=TextDelta(text=' left two Jewish men injured, describing it as a terrorist attack.\n*   **Medical Breakthrough:** The Mayo Clinic announced a significant', part_index=0, type='text'), type='delta'), StreamDeltaEvent(delta=TextDelta(text=' breakthrough in early-stage cancer detection, though further clinical trials are expected.\n*   **Thailand:** Former Prime Minister Thaksin Shin', part_index=0, type='text'), type='delta'), StreamDeltaEvent(delta=TextDelta(text='awatra has been approved for early release from prison on probation due to his age and health.', part_index=0, type='text'), type='delta'), StreamDeltaEvent(delta=TextDelta(text='', part_index=0, type='text'), type='delta'), StreamEndEvent(
    finish_reason='stop',
    usage=Usage(input_tokens=79, output_tokens=682, total_tokens=1386, cache_read_tokens=None, cache_write_tokens=None, reasoning_tokens=625, input_audio_tokens=None, output_audio_tokens=None),
    provider_data=<dict: 4 keys>,
    type='end',
)]}, {'provider': 'anthropic', 'model': 'claude-sonnet-4-5', 'events': [StreamStartEvent(id='msg_01Axd6LaNYQ9WSCgfwQSdnMm', model='claude-sonnet-4-5-20250929', type='start'), StreamDeltaEvent(delta=ToolCallDelta(input='', part_index=0, id=None, name=None, type='tool_call'), type='delta'), StreamDeltaEvent(delta=ToolCallDelta(input='{"query"', part_index=0, id=None, name=None, type='tool_call'), type='delta'), StreamDeltaEvent(delta=ToolCallDelta(input=': ', part_index=0, id=None, name=None, type='tool_call'), type='delta'), StreamDeltaEvent(delta=ToolCallDelta(input='"news', part_index=0, id=None, name=None, type='tool_call'), type='delta'), StreamDeltaEvent(delta=ToolCallDelta(input=' today April', part_index=0, id=None, name=None, type='tool_call'), type='delta'), StreamDeltaEvent(delta=ToolCallDelta(input=' 29 2026"}', part_index=0, id=None, name=None, type='tool_call'), type='delta'), StreamDeltaEvent(delta=TextDelta(text='Based on today', part_index=2, type='text'), type='delta'), StreamDeltaEvent(delta=TextDelta(text="'s news (April 29, 2026), here are the major headlines:", part_index=2, type='text'), type='delta'), StreamDeltaEvent(delta=TextDelta(text='\n\n## Iran Conflict & Blockade', part_index=2, type='text'), type='delta'), StreamDeltaEvent(delta=TextDelta(text='\n', part_index=2, type='text'), type='delta'), StreamDeltaEvent(delta=CitationDelta(text='US President Donald Trump said Wednesday that he has rejected the Iranian proposal of lifting the US blockade and opening the Strait of Hormuz. ', url='https://www.cnn.com/2026/04/29/world/live-news/iran-war-peace-proposal-trump', title='Live updates: Iran war news, UAE quits OPEC, Trump says Iran ‘better get smart soon’ | CNN', part_index=3, type='citation'), type='delta'), StreamDeltaEvent(delta=TextDelta(text="President Trump rejected Iran's proposal to lift the US blockade and", part_index=3, type='text'), type='delta'), StreamDeltaEvent(delta=TextDelta(text=' reopen the Strait of Hormuz', part_index=3, type='text'), type='delta'), StreamDeltaEvent(delta=TextDelta(text=', demanding guarantees on cur', part_index=4, type='text'), type='delta'), StreamDeltaEvent(delta=TextDelta(text="bing Iran's nuclear program first. ", part_index=4, type='text'), type='delta'), StreamDeltaEvent(delta=CitationDelta(text='Here’s today’s news: IRAN WAR – CEASEFIRE · President Trump has instructed aides to prepare for an extended blockade of Iran as negotiations remain st...', url='https://www.justsecurity.org/137394/early-edition-april-29-2026/', title='Early Edition: April 29, 2026', part_index=5, type='citation'), type='delta'), StreamDeltaEvent(delta=TextDelta(text='Trump has instructed a', part_index=5, type='text'), type='delta'), StreamDeltaEvent(delta=TextDelta(text='ides to prepare for an extended blockade of Iran as negotiations remain', part_index=5, type='text'), type='delta'), StreamDeltaEvent(delta=TextDelta(text=' stalled, assessing that maintaining the blockade carries less', part_index=5, type='text'), type='delta'), StreamDeltaEvent(delta=TextDelta(text=' risk than other options', part_index=5, type='text'), type='delta'), StreamDeltaEvent(delta=TextDelta(text='. ', part_index=6, type='text'), type='delta'), StreamDeltaEvent(delta=CitationDelta(text='Follow · Gas prices have climbed to an average of $4.23 per gallon — the highest level since August 2022 — according to AAA. ', url='https://www.cnn.com/2026/04/29/us/5-things-to-know-for-april-29-free-speech-king-charles-tornado-devastation-elon-musk-passports', title='5 things to know for April 29: Free speech, King Charles, tornado devastation, Elon Musk, passports | CNN', part_index=7, type='citation'), type='delta'), StreamDeltaEvent(delta=TextDelta(text='Gas prices have climbed to $4.23', part_index=7, type='text'), type='delta'), StreamDeltaEvent(delta=TextDelta(text=' per gallon — the highest level since August 2022', part_index=7, type='text'), type='delta'), StreamDeltaEvent(delta=TextDelta(text='.\n\n## Federal Reserve &', part_index=8, type='text'), type='delta'), StreamDeltaEvent(delta=TextDelta(text=' Markets', part_index=8, type='text'), type='delta'), StreamDeltaEvent(delta=TextDelta(text='\n', part_index=8, type='text'), type='delta'), StreamDeltaEvent(delta=CitationDelta(text='The S&P 500 was relatively unchanged on Wednesday, while oil prices continued their rally amid a U.S. blockade of Iranian ports and after the Federal ...', url='https://www.cnbc.com/2026/04/28/stock-market-today-live-updates.html', title='Stock market today: Live updates', part_index=9, type='citation'), type='delta'), StreamDeltaEvent(delta=TextDelta(text='The Federal Reserve left its key interest rate unchanged', part_index=9, type='text'), type='delta'), StreamDeltaEvent(delta=TextDelta(text='. ', part_index=10, type='text'), type='delta'), StreamDeltaEvent(delta=CitationDelta(text='Stocks were slightly lower and oil prices climbed as Wall Street looked ahead to earnings reports from four of the “Magnificent Seven” companies and a...', url='https://www.thestreet.com/latest-news/stock-market-today-apr-29-2026-update', title='Stock Market Today (Apr. 29, 2026): Fed holds rates but is most mixed since 1992 amid Middle East conflict; Big Tech reports to follow - TheStreet', part_index=11, type='citation'), type='delta'), StreamDeltaEvent(delta=TextDelta(text='Stocks were slightly lower', part_index=11, type='text'), type='delta'), StreamDeltaEvent(delta=TextDelta(text=' and oil prices climbed as Wall Street looked ahead', part_index=11, type='text'), type='delta'), StreamDeltaEvent(delta=TextDelta(text=' to earnings reports from four of the "Magnificent Seven" companies', part_index=11, type='text'), type='delta'), StreamDeltaEvent(delta=TextDelta(text='. ', part_index=12, type='text'), type='delta'), StreamDeltaEvent(delta=CitationDelta(text='Alphabet Inc. (GOOG), Amazon (AMZN), Meta Platforms (META), and Microsoft (MSFT) will have an opportunity to avert anxieties with tech after the bell,...', url='https://www.thestreet.com/latest-news/stock-market-today-apr-29-2026-update', title='Stock Market Today (Apr. 29, 2026): Fed holds rates but is most mixed since 1992 amid Middle East conflict; Big Tech reports to follow - TheStreet', part_index=13, type='citation'), type='delta'), StreamDeltaEvent(delta=TextDelta(text='Alphabet, Amazon, Meta', part_index=13, type='text'), type='delta'), StreamDeltaEvent(delta=TextDelta(text=' Platforms, and Microsoft will report quarterly earnings after the bell', part_index=13, type='text'), type='delta'), StreamDeltaEvent(delta=TextDelta(text='.\n\n## Apple Leadership', part_index=14, type='text'), type='delta'), StreamDeltaEvent(delta=TextDelta(text=' Change', part_index=14, type='text'), type='delta'), StreamDeltaEvent(delta=TextDelta(text='\n', part_index=14, type='text'), type='delta'), StreamDeltaEvent(delta=CitationDelta(text='Apple recently announced that Tim Cook will step down as CEO and transition to executive chairman of the board. John Ternus, currently senior vice pre...', url='https://www.thestreet.com/latest-news/stock-market-today-apr-29-2026-update', title='Stock Market Today (Apr. 29, 2026): Fed holds rates but is most mixed since 1992 amid Middle East conflict; Big Tech reports to follow - TheStreet', part_index=15, type='citation'), type='delta'), StreamDeltaEvent(delta=TextDelta(text='Apple recently announced that Tim Cook will step', part_index=15, type='text'), type='delta'), StreamDeltaEvent(delta=TextDelta(text=' down as CEO and transition to executive chairman of the board, with John', part_index=15, type='text'), type='delta'), StreamDeltaEvent(delta=TextDelta(text=' Ternus, currently senior vice president of Hardware Engineering', part_index=15, type='text'), type='delta'), StreamDeltaEvent(delta=TextDelta(text=', becoming CEO effective September', part_index=15, type='text'), type='delta'), StreamDeltaEvent(delta=TextDelta(text='.\n\n## King Charles in US', part_index=16, type='text'), type='delta'), StreamDeltaEvent(delta=TextDelta(text='\n', part_index=16, type='text'), type='delta'), StreamDeltaEvent(delta=CitationDelta(text='Britain’s King Charles III and Queen Camilla will travel to New York City today to visit the 9/11 memorial and meet with families of victims. The visi...', url='https://www.cnn.com/2026/04/29/us/5-things-to-know-for-april-29-free-speech-king-charles-tornado-devastation-elon-musk-passports', title='5 things to know for April 29: Free speech, King Charles, tornado devastation, Elon Musk, passports | CNN', part_index=17, type='citation'), type='delta'), StreamDeltaEvent(delta=TextDelta(text='King Charles III and Queen Camilla will travel to New York City today', part_index=17, type='text'), type='delta'), StreamDeltaEvent(delta=TextDelta(text=" to visit the 9/11 memorial, following the king's historic", part_index=17, type='text'), type='delta'), StreamDeltaEvent(delta=TextDelta(text=' address to Congress on Tuesday', part_index=17, type='text'), type='delta'), StreamDeltaEvent(delta=TextDelta(text='.\n\n## Other Headlines', part_index=18, type='text'), type='delta'), StreamDeltaEvent(delta=TextDelta(text='\n', part_index=18, type='text'), type='delta'), StreamDeltaEvent(delta=CitationDelta(text='... A federal grand jury has again indicted former FBI Director James Comey, this time over his social media post showing seashells arranged on a beac...', url='https://www.democracynow.org/2026/4/29/headlines', title='Headlines for April 29, 2026 | Democracy Now!', part_index=19, type='citation'), type='delta'), StreamDeltaEvent(delta=TextDelta(text='A federal grand jury has again indicted former FBI Director James Comey over a social media post', part_index=19, type='text'), type='delta'), StreamDeltaEvent(delta=TextDelta(text=', and ', part_index=20, type='text'), type='delta'), StreamDeltaEvent(delta=CitationDelta(text='WATCH: Moments from King Charles’ speech to Congress that drew laughter · Several people were injured after a tornado tore through Mineral Wells, Texa...', url='https://www.cnn.com/2026/04/29/us/5-things-to-know-for-april-29-free-speech-king-charles-tornado-devastation-elon-musk-passports', title='5 things to know for April 29: Free speech, King Charles, tornado devastation, Elon Musk, passports | CNN', part_index=21, type='citation'), type='delta'), StreamDeltaEvent(delta=TextDelta(text='a tornado tore through Mineral Wells', part_index=21, type='text'), type='delta'), StreamDeltaEvent(delta=TextDelta(text=', Texas, on Tuesday night, flattening parts of the town about 80 miles west', part_index=21, type='text'), type='delta'), StreamDeltaEvent(delta=TextDelta(text=' of Dallas', part_index=21, type='text'), type='delta'), StreamDeltaEvent(delta=TextDelta(text='.', part_index=22, type='text'), type='delta'), StreamEndEvent(
    finish_reason='stop',
    type='end',
)]}], 'errors': []}
```

---

## 8. Structured Output
**(Features: `structured_output`, `structured_output_json_object`, `response_mime_type`, `response_schema`, `output_config`)**

Define response formats cleanly. `lm15` formats it for the specific provider (`response_format` for OpenAI, `generationConfig.responseSchema` for Gemini, `output_config` for Anthropic).

```python
from lm15.types import Request, Message, Config

recipe_schema = {
    "type": "json_schema",
    "name": "recipe",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "ingredients": {"type": "array", "items": {"type": "string"}}
        },
        "required": ["name", "ingredients"],
        "additionalProperties": False
    }
}

req_json = Request(
    model="placeholder",
    messages=(Message.user("Give me a cookie recipe."),),
    config=Config(response_format=recipe_schema)
)

execute(req_json, stream=False)
```
```output | ✓ 6.5s | 26 vars
{'responses': [{'provider': 'openai', 'model': 'gpt-5.4-mini', 'response': Response(
    text='{"name":"Classic Chocolate Chip Cookies","ingredients":["2 1/4 cups all-purpose flour","1/2 teaspoon baking soda","1/2 teaspoon salt","1 cup unsalted butter, softened","3/4 cup granulated sugar","3/4 cup packed brown sugar","1 teaspoon vanilla extract","2 large eggs","2 cups chocolate chips"]}',
    model='gpt-5.4-mini-2026-03-17',
    finish_reason='stop',
    usage=Usage(input_tokens=46, output_tokens=79, total_tokens=125, cache_read_tokens=0, cache_write_tokens=None, reasoning_tokens=0, input_audio_tokens=None, output_audio_tokens=None),
    id='resp_0e1b7d5515f48ce60069f2635b6a3481a39cdbd731538043f1',
    provider_data=<dict: 35 keys>,
)}, {'provider': 'anthropic', 'model': 'claude-sonnet-4-5', 'response': Response(
    text='{"name":"Chocolate Chip Cookies","ingredients":["2 1/4 cups all-purpose flour","1 tsp baking soda","1 tsp salt","1 cup butter, softened","3/4 cup granulated sugar","3/4 cup packed brown sugar","2 large eggs","2 tsp vanilla extract","2 cups chocolate chips"]}',
    model='claude-sonnet-4-5-20250929',
    finish_reason='stop',
    usage=Usage(input_tokens=192, output_tokens=91, total_tokens=283, cache_read_tokens=0, cache_write_tokens=0, reasoning_tokens=None, input_audio_tokens=None, output_audio_tokens=None),
    id='msg_01RFrBrx8P6irsUSxha3fDme',
    provider_data=<dict: 9 keys>,
)}, {'provider': 'gemini', 'model': 'gemini-3-flash-preview', 'response': Response(
    text='{"name":"Chocolate Chip Cookies","ingredients":["1 cup butter, softened","1 cup white sugar","1 cup packed brown sugar","2 eggs","2 teaspoons vanilla extract","1 teaspoon baking soda","2 teaspoons hot water","1/2 teaspoon salt","3 cups all-purpose flour","2 cups semisweet chocolate chips"]}',
    model='gemini-3-flash-preview',
    finish_reason='stop',
    usage=Usage(input_tokens=7, output_tokens=64, total_tokens=448, cache_read_tokens=None, cache_write_tokens=None, reasoning_tokens=377, input_audio_tokens=None, output_audio_tokens=None),
    id='YWPyaYTNJqmb_uMP9rnDmQc',
    provider_data=<dict: 4 keys>,
)}], 'streams': [], 'errors': []}
```

With configured API keys, each provider returns JSON matching the schema and
`errors` should be empty.

---

## 9. Reasoning & Thinking
**(Features: `reasoning`, `thinking`, `thinking_budget`)**

Enable models that think before they answer.

OpenAI does not expose raw chain-of-thought. It reports reasoning token usage by
default, and can return a provider-generated reasoning summary when requested
with `summary="auto"`, `"concise"`, or `"detailed"`. Providers that expose
thinking directly, such as Anthropic and Gemini, surface it as `ThinkingPart` /
`ThinkingDelta`. Anthropic currently reports thinking usage as part of combined
`output_tokens` rather than a separate exact `usage.reasoning_tokens` count, so
`usage.reasoning_tokens` remains `None` for Anthropic responses.

```python
from lm15.types import Reasoning, Config, Message, Request

req_reasoning = Request(
    model="placeholder",
    messages=(Message.user("What is 143 times 27? Think carefully."),),
    config=Config(
        reasoning=Reasoning(
            effort="high",
            thinking_budget=1024,
            summary="auto",  # OpenAI: request a reasoning summary when supported.
        )
    )
)

r = execute(req_reasoning)
pretty(r)
```
```output | ✓ 29.1s | 27 vars
{'responses': [{'provider': 'openai',
                'model': 'gpt-5.4-mini',
                'response': Response(
    text='143 × 27 = **3861**.',
    model='gpt-5.4-mini-2026-03-17',
    finish_reason='stop',
    usage=Usage(input_tokens=17, output_tokens=54, total_tokens=71, cache_read_tokens=0, cache_write_tokens=None, reasoning_tokens=38, input_audio_tokens=None, output_audio_tokens=None),
    id='resp_0d661a636c9129d00069f26d37c2fc819dbd7dab3e1cb9a812',
    provider_data=<dict: 35 keys>,
)},
               {'provider': 'anthropic',
                'model': 'claude-sonnet-4-5',
                'response': Response(
    text='143 × 27 = 3,861\n\nTo break this down:\n- 143 × 20 = 2,860\n- 143 × 7 = 1,001\n- 2,860 + 1,001 = 3,861',
    model='claude-sonnet-4-5-20250929',
    finish_reason='stop',
    usage=Usage(input_tokens=48, output_tokens=225, total_tokens=273, cache_read_tokens=0, cache_write_tokens=0, reasoning_tokens=None, input_audio_tokens=None, output_audio_tokens=None),
    id='msg_01JRYpifFPYSXCEhJk95McNf',
    provider_data=<dict: 9 keys>,
)},
               {'provider': 'gemini',
                'model': 'gemini-3-flash-preview',
                'response': Response(
    text='To find the product of 143 and 27, you can break it down using the distributive property or standard long multiplication.\n\n### Method 1: Long Multiplication\n1.  **Multiply 143 by 7:**\n    *   $7 \\times 3 = 21$ (Write down 1, carry the 2)\n    *   $7 \\times 4 = 28$\n    *   $28 + 2 = 30$ (Write down 0, carry the 3)\n    *   $7 \\times 1 = 7$\n    *   $7 + 3 = 10$\n    *   Result: **1,001**\n\n2.  **Multiply 143 by 20 (the "2" in 27):**\n    *   Put a 0 in the ones place.\n    *   $2 \\times 3 = 6$\n    *   $2 \\times 4 = 8$\n    *   $2 \\times 1 = 2$\n    *   Result: **2,860**\n\n3.  **Add the two results together:**\n    *   $1,001 + 2,860 = 3,861$\n\n### Method 2: Decomposition\nYou can also think of it as $143 \\times (20 + 7)$:\n*   $143 \\times 20 = 2,860$\n*   $143 \\times 7 = 1,001$\n*   $2,860 + 1,001 = 3,861$\n\n**Answer:** \n143 times 27 is **3,861**.',
    model='gemini-3-flash-preview',
    finish_reason='stop',
    usage=Usage(input_tokens=15, output_tokens=399, total_tokens=1029, cache_read_tokens=None, cache_write_tokens=None, reasoning_tokens=615, input_audio_tokens=None, output_audio_tokens=None),
    id='UG3yab3rCsS2_uMPwOGvkAw',
    provider_data=<dict: 4 keys>,
)}],
 'streams': [{'provider': 'openai',
              'model': 'gpt-5.4-mini',
              'events': [StreamStartEvent(id='resp_0662a84d796b07c10069f26d3d0a2c819fb3cb3b03a321ab50',
                                          model='gpt-5.4-mini-2026-03-17',
                                          type='start'),
                         StreamDeltaEvent(delta=ThinkingDelta(text='**Calculating multiplication '
                                                                   'carefully**\n'
                                                                   '\n'
                                                                   'I',
                                                              part_index=0,
                                                              type='thinking'),
                                          type='delta'),
                         StreamDeltaEvent(delta=ThinkingDelta(text=' need',
                                                              part_index=0,
                                                              type='thinking'),
                                          type='delta'),
                         StreamDeltaEvent(delta=ThinkingDelta(text=' to',
                                                              part_index=0,
                                                              type='thinking'),
                                          type='delta'),
                         StreamDeltaEvent(delta=ThinkingDelta(text=' provide',
                                                              part_index=0,
                                                              type='thinking'),
                                          type='delta'),
                         StreamDeltaEvent(delta=ThinkingDelta(text=' the',
                                                              part_index=0,
                                                              type='thinking'),
                                          type='delta'),
                         StreamDeltaEvent(delta=ThinkingDelta(text=' answer',
                                                              part_index=0,
                                                              type='thinking'),
                                          type='delta'),
                         StreamDeltaEvent(delta=ThinkingDelta(text=' to',
                                                              part_index=0,
                                                              type='thinking'),
                                          type='delta'),
                         StreamDeltaEvent(delta=ThinkingDelta(text=' the',
                                                              part_index=0,
                                                              type='thinking'),
                                          type='delta'),
                         StreamDeltaEvent(delta=ThinkingDelta(text=' multiplication',
                                                              part_index=0,
                                                              type='thinking'),
                                          type='delta'),
                         StreamDeltaEvent(delta=ThinkingDelta(text=' question',
                                                              part_index=0,
                                                              type='thinking'),
                                          type='delta'),
                         StreamDeltaEvent(delta=ThinkingDelta(text=',',
                                                              part_index=0,
                                                              type='thinking'),
                                          type='delta'),
                         StreamDeltaEvent(delta=ThinkingDelta(text=' but',
                                                              part_index=0,
                                                              type='thinking'),
                                          type='delta'),
                         StreamDeltaEvent(delta=ThinkingDelta(text=' the',
                                                              part_index=0,
                                                              type='thinking'),
                                          type='delta'),
                         StreamDeltaEvent(delta=ThinkingDelta(text=' user',
                                                              part_index=0,
                                                              type='thinking'),
                                          type='delta'),
                         StreamDeltaEvent(delta=ThinkingDelta(text=' seems',
                                                              part_index=0,
                                                              type='thinking'),
                                          type='delta'),
                         StreamDeltaEvent(delta=ThinkingDelta(text=' to',
                                                              part_index=0,
… 526 more lines
```

With configured API keys, OpenAI responses include `usage.reasoning_tokens` and,
when the model returns one, a reasoning summary parsed as `ThinkingPart`.
Anthropic and Gemini responses may include `ThinkingPart` content; Gemini also
reports separate reasoning token counts when the API returns them, while
Anthropic exposes only combined `output_tokens`. The visible answer is available
as usual through `response.text` when the response is pure text, or via
`response.message.parts_of(TextPart)` for mixed text/thinking responses.


```python
from lm15.types import ThinkingPart

response = r['responses'][1]['response']
response.message.parts_of(ThinkingPart)
```
```output | ✓ 1ms | 29 vars
[ThinkingPart(text="I need to calculate 143 × 27.\n\nLet me break this down:\n143 × 27 = 143 × (20 + 7)\n         = 143 × 20 + 143 × 7\n         = 2860 + 1001\n         = 3861\n\nLet me verify this another way:\n143 × 27\n= (100 + 40 + 3) × 27\n= 100 × 27 + 40 × 27 + 3 × 27\n= 2700 + 1080 + 81\n= 3861\n\nYes, that's correct.", redacted=False, type='thinking')]
```

```python ✓
response.usage.reasoning_tokens
```
```python
response.message
```
```output | ✓ 2ms | 29 vars
Message(role='assistant', parts=(ThinkingPart(text="I need to calculate 143 × 27.\n\nLet me break this down:\n143 × 27 = 143 × (20 + 7)\n         = 143 × 20 + 143 × 7\n         = 2860 + 1001\n         = 3861\n\nLet me verify this another way:\n143 × 27\n= (100 + 40 + 3) × 27\n= 100 × 27 + 40 × 27 + 3 × 27\n= 2700 + 1080 + 81\n= 3861\n\nYes, that's correct.", redacted=False, type='thinking'), TextPart(text='143 × 27 = 3,861\n\nTo break this down:\n- 143 × 20 = 2,860\n- 143 × 7 = 1,001\n- 2,860 + 1,001 = 3,861', type='text')))
```

```python
response
```
```output | ✓ 2ms | 29 vars
Response(
    text='143 × 27 = 3,861\n\nTo break this down:\n- 143 × 20 = 2,860\n- 143 × 7 = 1,001\n- 2,860 + 1,001 = 3,861',
    model='claude-sonnet-4-5-20250929',
    finish_reason='stop',
    usage=Usage(input_tokens=48, output_tokens=225, total_tokens=273, cache_read_tokens=0, cache_write_tokens=0, reasoning_tokens=None, input_audio_tokens=None, output_audio_tokens=None),
    id='msg_01JRYpifFPYSXCEhJk95McNf',
    provider_data=<dict: 9 keys>,
)
```

```python
r['errors']
```
```output | ✓ 1ms | 29 vars
[]
```