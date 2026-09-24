from lm15 import LMRouter, Message, Request, UnsupportedModelError

request = Request(
    model="anthropic:no-such-model",
    messages=[Message.user(
        "What might be eating the acorns under our oak trees at "
        "night?"
    )],
)

router = LMRouter()
try:
    response = router.complete(request)
    print(response.text)
except UnsupportedModelError as error:
    print("No such model at", error.provider)
    print(error)
