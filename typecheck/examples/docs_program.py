from lm15 import Config, LMRouter, Message, Request, ResponseStream

router = LMRouter()
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
    config=Config(max_tokens=1000),
)

stream = ResponseStream(router.stream(request), request)
for text in stream:
    print(text, end="", flush=True)
print()
print(stream.usage.output_tokens, "tokens")
