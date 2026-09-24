import os

from lm15 import LMRouter, Message, Request, RouterConfig

router = LMRouter(RouterConfig(
    api_keys={"anthropic": os.environ["STATION_API_KEY"]},
))

request = Request(
    model="anthropic:claude-haiku-4-5",
    messages=[Message.user(
        "What might be eating the acorns under our oak trees at "
        "night?"
    )],
)
response = router.complete(request)
print(response.text)
