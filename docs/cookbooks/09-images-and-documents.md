# Images, PDFs & documents

**Problem** — You want a model to look at a picture or read a PDF. Each
provider wants media in a different shape — data URIs, base64 source
blocks, `inlineData` — and each has its own opinion about URLs and
upload handles. lm15 gives you two factories, `image()` and
`document()`, that take the media however you hold it and let the
provider adapter worry about the wire.

Keys loaded as in [recipe 01](01-first-request.md).

## Recipe

A media part is addressed by **exactly one** of `data` (bytes or
base64 string), `url`, `file_id` (a provider upload handle), or `path`
(a local file). Start with `path` — the factory guesses the media type
from the extension:

```python
import urllib.request
from pathlib import Path

from lm15 import LMRouter, Message, Request
from lm15.types import document, image, text

fetch = urllib.request.Request(
    "https://www.gstatic.com/webp/gallery/1.jpg",
    headers={"User-Agent": "lm15-cookbook/1.0"},
)
Path("fjord.jpg").write_bytes(urllib.request.urlopen(fetch, timeout=15).read())

part = image(path="fjord.jpg")
print(part)
```
```output
ImagePart(media_type='image/jpeg', data=None, url=None, file_id=None, path=PosixPath('fjord.jpg'))
```

The file is not read yet — `path` stays a `Path` until send time. Mix
the part with text in one user message:

```python
router = LMRouter()
response = router.complete(Request(
    model="claude-sonnet-4-5",
    messages=(Message.user([text("One sentence: what is this?"), part]),),
))
print(response.text)
```
```output
This is a dramatic Norwegian fjord landscape viewed from a mountain peak, showing steep valley walls, a winding river or lake below, and distant snow-capped mountains under a bright sky.
```

A `url` ships the URL itself; the provider fetches it. All three
providers accept public image URLs, but the fetch happens on their
side — an origin that blocks bot traffic fails there, not here:

```python
response = router.complete(Request(
    model="gpt-4.1-mini",
    messages=(Message.user([
        text("One sentence: what is this?"),
        image(url="https://www.gstatic.com/webp/gallery/1.jpg"),
    ]),),
))
print(response.text)
```
```output
This is a scenic view of a deep river valley or fjord surrounded by steep, rocky mountains covered with vegetation.
```

`data` takes raw `bytes` (encoded to base64 for you) or an
already-encoded base64 string. Images also take `detail` — a hint to
OpenAI's vision stack to downsample (`"low"`) or tile at full
resolution (`"high"`). On tile-priced models the cost difference is
an order of magnitude:

```python
fetch = urllib.request.Request(
    "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/"
    "Gfp-wisconsin-madison-the-nature-boardwalk.jpg/1280px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg",
    headers={"User-Agent": "lm15-cookbook/1.0"},
)
photo = urllib.request.urlopen(fetch, timeout=20).read()

for level in ("low", "high"):
    response = router.complete(Request(
        model="gpt-4o-mini",
        messages=(Message.user([
            text("One word: the structure."),
            image(data=photo, media_type="image/jpeg", detail=level),
        ]),),
    ))
    print(level, "->", response.text, "| input_tokens:", response.usage.input_tokens)
```
```output
low -> Pathway. | input_tokens: 2846
high -> Pathway | input_tokens: 36848
```

`document()` works the same way and defaults to `application/pdf`.
Fabricate a one-page PDF so this page is self-contained (skip this
block if you have a real one):

```python
content = b"BT /F1 18 Tf 72 720 Td (INVOICE #4521) Tj 0 -28 Td (Total due: $1,337.00 by 2026-07-01) Tj ET"
objects = [
    b"<< /Type /Catalog /Pages 2 0 R >>",
    b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
    b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R"
    b" /Resources << /Font << /F1 5 0 R >> >> >>",
    b"<< /Length %d >>\nstream\n%s\nendstream" % (len(content), content),
    b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
]
pdf, offsets = b"%PDF-1.4\n", []
for n, obj in enumerate(objects, 1):
    offsets.append(len(pdf))
    pdf += b"%d 0 obj\n%s\nendobj\n" % (n, obj)
xref = len(pdf)
pdf += b"xref\n0 6\n0000000000 65535 f \n" + b"".join(b"%010d 00000 n \n" % o for o in offsets)
pdf += b"trailer\n<< /Size 6 /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF" % xref
Path("invoice.pdf").write_bytes(pdf)

response = router.complete(Request(
    model="claude-sonnet-4-5",
    messages=(Message.user([
        text("Invoice number and amount due?"),
        document(path="invoice.pdf"),
    ]),),
))
print(response.text)
```
```output
Based on the invoice document:

**Invoice Number:** #4521

**Amount Due:** $1,337.00

The payment is due by July 1, 2026.
```

The same `DocumentPart` shape works on OpenAI — pass `data` there
(see Variations for why):

```python
response = router.complete(Request(
    model="gpt-4.1-mini",
    messages=(Message.user([
        text("Invoice number and amount due?"),
        document(data=Path("invoice.pdf").read_bytes()),
    ]),),
))
print(response.text)
```
```output
The invoice number is 4521 and the amount due is $1,337.00.
```

The exactly-one-of rule is enforced at construction, not at send time.
Two sources is ambiguous; zero is empty:

```python
try:
    image(path="fjord.jpg", url="https://example.com/fjord.jpg")
except ValueError as error:
    print(error)
try:
    image()
except ValueError as error:
    print(error)
```
```output
ImagePart requires exactly one of data, url, file_id, or path
ImagePart requires exactly one of data, url, file_id, or path
```

## How it works

`image()` and `document()` build frozen `ImagePart` / `DocumentPart`
dataclasses — two of the variants in lm15's `Part` union, alongside
`audio()`, `video()`, and `binary()`, which take the same four
sources. The factory counts the sources you passed and raises unless
exactly one is set; `bytes` data is base64-encoded immediately; a
`path` gets its `media_type` guessed by `mimetypes.guess_type`, with
an explicit `media_type=` overriding the guess. Defaults are
`image/png` and `application/pdf` when nothing else is known.

What goes on the wire is the provider adapter's problem. OpenAI gets a
`data:` URI (`input_image`) or `input_file` with inline `file_data`;
Anthropic gets a base64 / `url` / `file` source block; Gemini gets
`inlineData` or `fileData`. `path`-addressed media is read lazily —
the Anthropic and Gemini adapters call `path.read_bytes()` while
building the payload, so the part itself stays a cheap reference.

lm15 does **not** upload files, resize images, or fetch URLs for you.
`file_id` is passed through verbatim — you create the upload with the
provider's Files API (the LM `file_upload` surface) and hand lm15 the
handle. The router only picks the provider; the parts are identical
across all three. See [../using-the-router.md](../using-the-router.md).

## Variations

- **Async mirror.** `AsyncLMRouter().complete(...)` takes the same
  `Request`; parts are plain data, nothing to await.
- **OpenAI does not read `path`.** As of this writing the OpenAI
  adapter inlines `data`, `url`, and `file_id` only; a path-only image
  is silently dropped from the prompt (the model answers as if no
  image were attached). Pass `data=part.bytes` — the `.bytes` property
  reads `path` (or decodes inline `data`) for you. Anthropic and
  Gemini read `path` at send time.
- **`detail` is OpenAI-only.** Anthropic and Gemini ignore the field.
  And on patch-priced OpenAI models (`gpt-4.1-mini`) we measured
  identical input tokens at every level; the low/high gap above is the
  tile-priced `gpt-4o` family.
- **PDFs by URL.** `document(url=...)` maps to OpenAI `file_url` and
  Anthropic's `url` source. Gemini maps any `url` to
  `fileData.fileUri`; public HTTP works on current Gemini models, but
  Files-API URIs are the documented path for repeated use.
- **Responses contain media too.** Image generation returns the same
  `ImagePart`s — `.images` on an `ImageGenerationResponse`, `.images` /
  `.image_bytes` on a chat `Result` (recipe 12).

## See also

- [02 — Multi-turn conversations](02-conversations.md)
- [10 — Audio, video & reasoning models](10-audio-video-reasoning.md)
- [12 — Embeddings, batch & media generation](12-embeddings-batch-generation.md)
- [15 — Errors, retries & testing](15-errors-and-testing.md)
- [../using-the-router.md](../using-the-router.md)
