from lm15 import LMRouter, Message, Request
from lm15.types import document

logbook = document(
    path="oak-grove-log.pdf", media_type="application/pdf",
)
request = Request(
    model="anthropic:claude-haiku-4-5",
    system=(
        "You are the field assistant for a wildlife research "
        "station. Answer in two sentences."
    ),
    messages=[Message.user([
        (
            "Which animals did this camera record, and on how many "
            "nights did it see each one?"
        ),
        logbook,
    ])],
)
router = LMRouter()
response = router.complete(request)
print(response.text)
print(response.usage.input_tokens, "tokens in")
