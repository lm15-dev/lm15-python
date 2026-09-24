from lm15 import LMRouter, Message, Request

request = Request(
    model="anthropic:claude-haiku-4-5",
    messages=[Message.user(
        "What might be eating the acorns under our oak trees at "
        "night?"
    )],
)

router = LMRouter()
response = router.complete(request)
print(response.text)
