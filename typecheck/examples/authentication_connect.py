from lm15 import Message
from lm15.interactive import connect

with connect("openai-codex") as lm:
    answer = lm.complete(
        system=(
            "You are the field assistant for a wildlife research "
            "station. Answer in two sentences."
        ),
        messages=[Message.user(
            "What might be eating the acorns under our oak trees at "
            "night?"
        )],
    )
    print(answer.text)
