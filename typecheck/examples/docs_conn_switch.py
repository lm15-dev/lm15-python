from lm15 import LMRouter, Message, Request

question = (
    "What might be eating the acorns under our oak trees at night?"
)

router = LMRouter()
for model in [
    "anthropic:claude-haiku-4-5",
    "anthropic:claude-haiku-4-5",
]:
    request = Request(
        model=model,
        system=(
            "You are the field assistant for a wildlife research "
            "station. Answer in two sentences."
        ),
        messages=[Message.user(question)],
    )
    print(model, "->", router.complete(request).text)
