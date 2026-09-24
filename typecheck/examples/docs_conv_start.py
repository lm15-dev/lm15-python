from lm15 import LMRouter, Message, Request

router = LMRouter()
messages = [Message.user(
    "What might be eating the acorns under our oak trees at night?"
)]
response = router.complete(Request(
    model="anthropic:claude-haiku-4-5",
    system=(
        "You are the field assistant for a wildlife research "
        "station. Answer in two sentences."
    ),
    messages=messages,
))
messages.append(response.message)
print(response.text)

for part in response.message.parts:
    print(part.type)
