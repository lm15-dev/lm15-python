from lm15 import LMRouter, Message, Request
from lm15.types import image

photo = image(path="badger.jpg", media_type="image/jpeg")
request = Request(
    model="anthropic:claude-haiku-4-5",
    system=(
        "You are the field assistant for a wildlife research "
        "station. Answer in two sentences."
    ),
    messages=[Message.user([
        "What animal is this, and what is it doing?",
        photo,
    ])],
)
router = LMRouter()
response = router.complete(request)
print(response.text)
print(response.usage.input_tokens, "tokens in")
