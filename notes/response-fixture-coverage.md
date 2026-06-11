# Provider response fixture coverage

This note maps provider response shapes to lm15 response types and the offline
fixture tests that exercise them.

| Provider | Response shape | lm15 type | Test fixture | Status |
|---|---|---|---|---|
| OpenAI | `output[].type == "message"` + `content[].type == "output_text"` | `TextPart` | `openai.basic_text` | tested |
| OpenAI | `output_text.annotations[].type == "url_citation"` | `CitationPart` | `openai.web_search/2026-04-13T14-44-39Z.txt` | tested |
| OpenAI | `output[].type == "function_call"` | `ToolCallPart` | `openai.tools` | tested |
| OpenAI | `output[].type == "reasoning"` with non-empty summary/text | `ThinkingPart` | `openai.reasoning` | parser covered; saved fixture has empty summary |
| OpenAI | `output[].type == "web_search_call"` | provider data only | `openai.web_search` | intentionally ignored |
| OpenAI | `output[].type == "code_interpreter_call"` | provider data only | `openai.code_interpreter` | intentionally ignored |
| Anthropic | `content[].type == "text"` | `TextPart` | `anthropic.basic_text` | tested |
| Anthropic | `content[].type == "tool_use"` | `ToolCallPart` | `anthropic.tools` | tested |
| Anthropic | `content[].type == "thinking"` | `ThinkingPart` | `anthropic.thinking` | tested |
| Anthropic | `content[].type == "text"` + `citations[]` | `CitationPart` | `anthropic.web_search` | parser covered; no saved body yet |
| Gemini | `candidates[].content.parts[].text` | `TextPart` | `gemini.basic_text` | tested |
| Gemini | `candidates[].content.parts[].functionCall` | `ToolCallPart` | `gemini.tools` | tested |
| Gemini | `candidates[].content.parts[].thought == true` | `ThinkingPart` | `gemini.thinking` | parser covered by existing unit path; fixture expectation pending |
| Gemini | `candidates[].groundingMetadata.groundingSupports[]` + `groundingChunks[]` | `CitationPart` | `gemini.google_search` | tested |
| Gemini | `candidates[].content.parts[].executableCode` / `codeExecutionResult` | provider data only | `gemini.code_execution` | intentionally ignored |
| OpenAI | SSE `response.output_text.delta` + `response.completed` | `StreamDeltaEvent` / `StreamEndEvent` | `openai.streaming` | tested |
| Anthropic | SSE `content_block_delta.text_delta` + `message_stop` | `StreamDeltaEvent` / `StreamEndEvent` | `anthropic.streaming` | tested |
| Gemini | SSE `data: { candidates: ... finishReason }` | `StreamDeltaEvent` / `StreamEndEvent` | `gemini.streaming` | tested |

The corresponding structural expectations live in `curl-fixtures/cases/*/*.json`
under `expect_lm15`. The Python offline runner is
`lm15-python/tests/test_response_fixtures.py`.

Complete-response parsers attach unexpected provider shapes to
`response.provider_data["_lm15_unmapped"]`; the fixture runner fails if that
list is present. Provider-executed tool records such as OpenAI
`web_search_call` and Gemini `executableCode` are intentionally ignored.
