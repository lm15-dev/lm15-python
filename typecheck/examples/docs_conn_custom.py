from lm15 import LMRouter, Message, Request, RouterConfig

router = LMRouter(RouterConfig(
    base_urls={"vllm": "http://127.0.0.1:11434/v1"},
))

request = Request(
    model="vllm:Qwen/Qwen3-8B",
    messages=[Message.user(
        "What might be eating the acorns under our oak trees at "
        "night?"
    )],
)
response = router.complete(request)
print(response.text)
