import time

from lm15 import LMRouter, Message, RETRYABLE_ERRORS, Request

def complete_with_retries(router, request, attempts=4):
    for attempt in range(1, attempts + 1):
        try:
            return router.complete(request)
        except RETRYABLE_ERRORS as error:
            if attempt == attempts:
                raise
            wait = error.retry_after
            if wait is None:
                wait = 2 ** attempt
            print(f"{type(error).__name__}, waiting {wait} s")
            time.sleep(wait)

request = Request(
    model="anthropic:busy-model",
    system=(
        "You are the field assistant for a wildlife research "
        "station. Answer in two sentences."
    ),
    messages=[Message.user(
        "What might be eating the acorns under our oak trees at "
        "night?"
    )],
)

router = LMRouter()
response = complete_with_retries(router, request)
print(response.text)
