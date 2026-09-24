from lm15 import (Config, LMRouter, Message, Request,
                  choice, judgments, score, yes_no)

answers = judgments(
    animal=choice("Which animal is the note mainly about?",
                  ["badger", "fox", "owl", "hare", "deer"]),
    certainty=score("How sure is the observer of the species?",
                    ["guess", "probable", "confident"]),
    hurt=yes_no("Does the note report a hurt animal?"),
)

note = ("Checked the stream camera this morning. Three\n"
        "badgers came through overnight, one of them limping.\n"
        "A fox passed later, just before dawn.")

router = LMRouter()
reply = router.complete(Request(
    model="jev-latest",
    messages=[Message.user(note)],
    config=Config(response_format=answers,
                  probabilities="if_available"),
))
print(reply.data)
print(reply.probabilities)
print(reply.method)

print(reply.expected("certainty"))

reply = router.complete(Request(
    model="anthropic:claude-haiku-4-5",
    messages=[Message.user(note)],
    config=Config(response_format=answers,
                  probabilities="if_available"),
))
print(reply.data)
print(reply.probabilities)
for change in reply.adaptations:
    print(change.field, change.action)

reply = router.complete(Request(
    model="anthropic:claude-haiku-4-5",
    messages=[Message.user(note)],
    config=Config(response_format=answers,
                  probabilities="required"),
))

notes = {
    "stream": note,
    "barn": ("Around midnight an owl was calling from the old\n"
             "barn roof, probably a tawny. Two hares in the barn\n"
             "field at first light."),
    "deer": ("Dusk, edge of the oak grove. Two deer browsing on\n"
             "fallen acorns, one small with spots still showing.\n"
             "Too far to be sure of the species: roe or fallow."),
}
for name, text in notes.items():
    reply = router.complete(Request(
        model="jev-latest",
        messages=[Message.user(text)],
        config=Config(response_format=answers,
                      probabilities="if_available"),
    ))
    sure = reply.expected("certainty")
    print(f"{name:7}{reply.data['animal']:8}{sure:.2f}")
