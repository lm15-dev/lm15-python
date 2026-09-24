import json

from lm15 import FunctionTool, LMRouter, Message, Request

SIGHTINGS = [
    {"date": "2026-09-18", "place": "oak grove",
     "species": "wood mouse", "count": 4},
    {"date": "2026-09-19", "place": "oak grove",
     "species": "roe deer", "count": 2},
    {"date": "2026-09-20", "place": "stream",
     "species": "red fox", "count": 1},
    {"date": "2026-09-21", "place": "oak grove",
     "species": "wild boar", "count": 3},
]


def search_sightings(query):
    query = query.lower()
    return [s for s in SIGHTINGS
            if query in (s["species"], s["place"], s["date"])]

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

results = {
    call.id: json.dumps(search_sightings(**call.input))
    for call in response.tool_calls
}
followup = Request(
    model=request.model,
    system=request.system,
    messages=[
        *request.messages,
        response.message,
        Message.tool(results),
    ],
    tools=request.tools,
)
print(router.complete(followup).text)
