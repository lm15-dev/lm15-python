from lm15 import LMRouter, Message, Request, RouterConfig
from lm15.doctor import explain_auth

request = Request(
    model="anthropic:claude-haiku-4-5",
    system=(
        "You are the field assistant for a wildlife research station. "
        "Answer in two sentences."
    ),
    messages=[Message.user(
        "What might be eating the acorns under our oak trees at "
        "night?"
    )],
)

print(explain_auth("anthropic"))

import os

router = LMRouter(RouterConfig(
    api_keys={"anthropic": os.environ["STATION_API_KEY"]},
))
print(explain_auth("anthropic", config=router.config))

from pathlib import Path

def station_key():
    print("(reading the key)")
    return Path("secrets/anthropic.key").read_text().strip()

router = LMRouter(RouterConfig(api_keys={"anthropic": station_key}))
first = router.complete(request)
second = router.complete(request)
print(second.text)

from dataclasses import replace

print(explain_auth("openai-codex"))

router = LMRouter()
codex = replace(request, model="openai-codex:gpt-5.6-sol")
print(router.complete(codex).text)

from lm15.login import Auth

auth = Auth.local()
router = LMRouter(RouterConfig(auth=auth))
print(explain_auth("anthropic", config=router.config))

auth.configure(
    "anthropic", method="env", answers={"name": "ANTHROPIC_API_KEY"},
)
print(router.complete(request).text)

for connection in auth.connections():
    status = auth.status(connection.provider)
    print(connection.label)
    print("   ", status.usability, "until", status.expires_at)

for method in auth.methods("openai-codex"):
    print(f"{method.availability:<11} {method.label}")

auth.logout("anthropic")
print(auth.status("anthropic").detail)
router.complete(request)
