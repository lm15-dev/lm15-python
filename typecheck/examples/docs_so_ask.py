from lm15 import Config, LMRouter, Message, Request

places = ["oak grove", "stream", "meadow"]
sighting_schema = {
    "type": "object",
    "properties": {
        "sightings": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "species": {
                        "type": "string",
                        "description": "Common name, singular.",
                    },
                    "count": {"type": "integer"},
                    "place": {
                        "type": "string",
                        "enum": places,
                    },
                },
                "required": ["species", "count", "place"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["sightings"],
    "additionalProperties": False,
}

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
    config=Config(response_format={
        "type": "json_schema",
        "name": "sightings",
        "schema": sighting_schema,
        "strict": True,
    }),
)
router = LMRouter()
response = router.complete(request)
print(response.data)

for s in response.data["sightings"]:
    print(s["count"], s["species"], "at", s["place"])
