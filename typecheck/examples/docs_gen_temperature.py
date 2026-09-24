from lm15 import Config, LMRouter, Message, Request

router = LMRouter()
for temperature in [0.0, 1.0]:
    for run in range(2):
        response = router.complete(Request(
            model="anthropic:claude-haiku-4-5",
            system=(
                "You are the field assistant for a wildlife research "
                "station. Answer in two sentences."
            ),
            messages=[Message.user(
                "What might be eating the acorns under our oak trees "
                "at night?"
            )],
            config=Config(temperature=temperature),
        ))
        print(temperature, response.text)
