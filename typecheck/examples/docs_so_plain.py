from lm15 import LMRouter, Message, Request

note = """Checked the stream camera this morning. Three
badgers came through overnight, one of them limping.
A fox passed later, just before dawn."""

request = Request(
    model="anthropic:claude-haiku-4-5",
    system=(
        "You are the field assistant for a wildlife research "
        "station. Turn each field note into sighting records."
    ),
    messages=[Message.user(note)],
)
router = LMRouter()
response = router.complete(request)
print(response.text)
