from lm15 import Config, LMRouter, Message, Request

request = Request(
    model="anthropic:claude-haiku-4-5",
    system=(
        "You are the field assistant for a wildlife research "
        "station."
    ),
    messages=[Message.user(
        "List five animals that might eat acorns at night, one per "
        "line, numbered."
    )],
    config=Config(stop=["4."]),
)

router = LMRouter()
response = router.complete(request)
print(response.text)
print(response.finish_reason)
