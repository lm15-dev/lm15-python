import json

from lm15 import LMRouter, Message, Request
from lm15.serde import request_from_dict, request_to_dict

router = LMRouter()
model = "anthropic:claude-haiku-4-5"
instructions = (
    "You are the field assistant for a wildlife research station. "
    "Answer in two sentences."
)
messages = []
for question in [
    "What might be eating the acorns under our oak trees at night?",
    "Would the same animals eat hazelnuts?",
    "How could we find out which one it is?",
]:
    messages.append(Message.user(question))
    response = router.complete(Request(
        model=model, system=instructions, messages=messages,
    ))
    messages.append(response.message)
    print(">", question)
    print(response.text)
    print(response.usage.input_tokens, "tokens in")

saved = Request(model=model, system=instructions, messages=messages)
with open("conversation.json", "w") as f:
    json.dump(request_to_dict(saved), f)

with open("conversation.json") as f:
    saved = request_from_dict(json.load(f))
followup = Request(
    model=saved.model,
    system=saved.system,
    messages=[*saved.messages, Message.user(
        "Which of those should we look for first?"
    )],
)
router = LMRouter()
print(router.complete(followup).text)
