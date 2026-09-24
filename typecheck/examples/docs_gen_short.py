from lm15 import Config, LMRouter, Message, Request

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
    config=Config(max_tokens=20),
)

router = LMRouter()
response = router.complete(request)
print(response.text)
print(response.finish_reason)
