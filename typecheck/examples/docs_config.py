from lm15 import Config, FunctionTool, LMRouter, Message, Request

sightings_tool = FunctionTool(
    name="search_sightings",
    description=(
        "Find the station's recorded sightings by species, place or "
        "date."
    ),
    parameters={
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": (
                "One species, place or date, such as 'oak grove' or "
                "'2026-09-18'."
            )},
        },
        "required": ["query"],
    },
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
    tools=[sightings_tool],
    config=Config(max_tokens=1000),
)

router = LMRouter()
response = router.complete(request)
print(response.text)
print(response.finish_reason)  # "stop", "length", "tool_call"…
print(response.usage.input_tokens, response.usage.output_tokens)
