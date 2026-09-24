from lm15 import LMRouter, Message, Request

request = Request(
    model="anthropic:claude-haiku-4-5",
    system=(
        "You are the field assistant for a wildlife research "
        "station. Answer in two sentences."
    ),
    messages=[Message.user(
        "What might be eating the acorns under our oak trees at "
        "night?"
    )],
)

router = LMRouter()
response = router.complete(request)
print(response.text)
print(response.finish_reason)  # "stop", "length", "tool_call"…
print(response.usage.input_tokens, response.usage.output_tokens)

followup = Request(
    model="anthropic:claude-haiku-4-5",
    system=(
        "You are the field assistant for a wildlife research "
        "station. Answer in two sentences."
    ),
    messages=[Message.user("Would the same animals eat hazelnuts?")],
)
print(router.complete(followup).text)
