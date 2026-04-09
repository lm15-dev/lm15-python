# API versions explained

> [!IMPORTANT]
> We have updated our [Terms of Service](https://ai.google.dev/gemini-api/terms).

This document provides a high-level overview of the differences between the `v1`
and `v1beta` versions of the Gemini API.

- **v1**: Stable version of the API. Features in the stable version are fully-supported over the lifetime of the major version. If there are any breaking changes, then the next major version of the API will be created and the existing version will be deprecated after a reasonable period of time. Non-breaking changes may be introduced to the API without changing the major version.
- **v1beta** : This version includes early features that may be under development and are subject to breaking changes. There is also no guarantee that the features in the Beta version will move to the stable version. **If you require stability in your production environment and cannot risk breaking changes, you should not use this version in production.**

| Feature | v1 | v1beta |
|---|---|---|
| Generate Content - Text-only input | Yes | Yes |
| Generate Content - Text-and-image input | Yes | Yes |
| Generate Content - Text output | Yes | Yes |
| Generate Content - Multi-turn conversations (chat) | Yes | Yes |
| Generate Content - Function calls |   | Yes |
| Generate Content - Streaming | Yes | Yes |
| Embed Content - Text-only input | Yes | Yes |
| Generate Answer |   | Yes |
| Semantic retriever |   | Yes |
| Interactions API |   | Yes |

- Yes - Supported
- No - Will never be supported

## Configure API version in an SDK

The Gemini API SDK's default to `v1beta`, but you can opt to use other versions
by setting the API version as shown in the following code sample:

### Python

    from google import genai

    client = genai.Client(http_options={'api_version': 'v1'})

    response = client.models.generate_content(
        model='gemini-3-flash-preview',
        contents="Explain how AI works",
    )

    print(response.text)

### JavaScript

    import { GoogleGenAI } from "@google/genai";

    const ai = new GoogleGenAI({
      httpOptions: { apiVersion: "v1" },
    });

    async function main() {
      const response = await ai.models.generateContent({
        model: "gemini-3-flash-preview",
        contents: "Explain how AI works",
      });
      console.log(response.text);
    }

    await main();

### REST

    curl "https://generativelanguage.googleapis.com/v1/models/gemini-3-flash-preview:generateContent" \
    -H "x-goog-api-key: $GEMINI_API_KEY" \
    -H 'Content-Type: application/json' \
    -X POST \
    -d '{
      "contents": [{
        "parts":[{"text": "Explain how AI works."}]
        }]
       }'