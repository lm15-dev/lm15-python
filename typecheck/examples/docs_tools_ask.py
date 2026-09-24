from lm15 import FunctionTool, LMRouter, Message, Request

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
        "Which animals has the station recorded at the oak grove?"
    )],
    tools=[sightings_tool],
)
router = LMRouter()
response = router.complete(request)
print(response.finish_reason)  # "tool_call"
for call in response.tool_calls:
    print(call.name, call.input)
