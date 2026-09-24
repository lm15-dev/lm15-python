from lm15 import (
    Config, LMRouter, Message, Request, RouterConfig,
    UnsupportedFeatureError,
)

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
    config=Config(temperature=1.5, seed=42),
)

router = LMRouter()
response = router.complete(request)
print(response.text)
for a in response.adaptations:
    print(a.field, a.action)

for a in router.plan(request):
    print(a.field, a.action)

strict = LMRouter(RouterConfig(adaptations="refuse"))
try:
    print(strict.complete(request).text)
except UnsupportedFeatureError as error:
    print("Refused:", error.feature)
