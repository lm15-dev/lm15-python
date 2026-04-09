## Create session

**post** `/realtime/sessions`

Create an ephemeral API token for use in client-side applications with the
Realtime API. Can be configured with the same session parameters as the
`session.update` client event.

It responds with a session object, plus a `client_secret` key which contains
a usable ephemeral API token that can be used to authenticate browser clients
for the Realtime API.

Returns the created Realtime session object, plus an ephemeral key.

### Body Parameters

- `client_secret: object { expires_at, value }`

  Ephemeral key returned by the API.

  - `expires_at: number`

    Timestamp for when the token expires. Currently, all tokens expire
    after one minute.

  - `value: string`

    Ephemeral key usable in client environments to authenticate connections
    to the Realtime API. Use this in client-side environments rather than
    a standard API token, which should only be used server-side.

- `input_audio_format: optional string`

  The format of input audio. Options are `pcm16`, `g711_ulaw`, or `g711_alaw`.

- `input_audio_transcription: optional object { model }`

  Configuration for input audio transcription, defaults to off and can be
  set to `null` to turn off once on. Input audio transcription is not native
  to the model, since the model consumes audio directly. Transcription runs
  asynchronously and should be treated as rough guidance
  rather than the representation understood by the model.

  - `model: optional string`

    The model to use for transcription.

- `instructions: optional string`

  The default system instructions (i.e. system message) prepended to model calls. This field allows the client to guide the model on desired responses. The model can be instructed on response content and format, (e.g. "be extremely succinct", "act friendly", "here are examples of good responses") and on audio behavior (e.g. "talk quickly", "inject emotion into your voice", "laugh frequently"). The instructions are not guaranteed to be followed by the model, but they provide guidance to the model on the desired behavior.
  Note that the server sets default instructions which will be used if this field is not set and are visible in the `session.created` event at the start of the session.

- `max_response_output_tokens: optional number or "inf"`

  Maximum number of output tokens for a single assistant response,
  inclusive of tool calls. Provide an integer between 1 and 4096 to
  limit output tokens, or `inf` for the maximum available tokens for a
  given model. Defaults to `inf`.

  - `number`

  - `"inf"`

    - `"inf"`

- `modalities: optional array of "text" or "audio"`

  The set of modalities the model can respond with. To disable audio,
  set this to ["text"].

  - `"text"`

  - `"audio"`

- `output_audio_format: optional string`

  The format of output audio. Options are `pcm16`, `g711_ulaw`, or `g711_alaw`.

- `prompt: optional ResponsePrompt`

  Reference to a prompt template and its variables.
  [Learn more](/docs/guides/text?api-mode=responses#reusable-prompts).

  - `id: string`

    The unique identifier of the prompt template to use.

  - `variables: optional map[string or ResponseInputText or ResponseInputImage or ResponseInputFile]`

    Optional map of values to substitute in for variables in your
    prompt. The substitution values can either be strings, or other
    Response input types like images or files.

    - `string`

    - `ResponseInputText = object { text, type }`

      A text input to the model.

      - `text: string`

        The text input to the model.

      - `type: "input_text"`

        The type of the input item. Always `input_text`.

        - `"input_text"`

    - `ResponseInputImage = object { detail, type, file_id, image_url }`

      An image input to the model. Learn about [image inputs](/docs/guides/vision).

      - `detail: "low" or "high" or "auto" or "original"`

        The detail level of the image to be sent to the model. One of `high`, `low`, `auto`, or `original`. Defaults to `auto`.

        - `"low"`

        - `"high"`

        - `"auto"`

        - `"original"`

      - `type: "input_image"`

        The type of the input item. Always `input_image`.

        - `"input_image"`

      - `file_id: optional string`

        The ID of the file to be sent to the model.

      - `image_url: optional string`

        The URL of the image to be sent to the model. A fully qualified URL or base64 encoded image in a data URL.

    - `ResponseInputFile = object { type, file_data, file_id, 2 more }`

      A file input to the model.

      - `type: "input_file"`

        The type of the input item. Always `input_file`.

        - `"input_file"`

      - `file_data: optional string`

        The content of the file to be sent to the model.

      - `file_id: optional string`

        The ID of the file to be sent to the model.

      - `file_url: optional string`

        The URL of the file to be sent to the model.

      - `filename: optional string`

        The name of the file to be sent to the model.

  - `version: optional string`

    Optional version of the prompt template.

- `speed: optional number`

  The speed of the model's spoken response. 1.0 is the default speed. 0.25 is
  the minimum speed. 1.5 is the maximum speed. This value can only be changed
  in between model turns, not while a response is in progress.

- `temperature: optional number`

  Sampling temperature for the model, limited to [0.6, 1.2]. Defaults to 0.8.

- `tool_choice: optional string`

  How the model chooses tools. Options are `auto`, `none`, `required`, or
  specify a function.

- `tools: optional array of object { description, name, parameters, type }`

  Tools (functions) available to the model.

  - `description: optional string`

    The description of the function, including guidance on when and how
    to call it, and guidance about what to tell the user when calling
    (if anything).

  - `name: optional string`

    The name of the function.

  - `parameters: optional unknown`

    Parameters of the function in JSON Schema.

  - `type: optional "function"`

    The type of the tool, i.e. `function`.

    - `"function"`

- `tracing: optional "auto" or object { group_id, metadata, workflow_name }`

  Configuration options for tracing. Set to null to disable tracing. Once
  tracing is enabled for a session, the configuration cannot be modified.

  `auto` will create a trace for the session with default values for the
  workflow name, group id, and metadata.

  - `"auto"`

    Default tracing mode for the session.

    - `"auto"`

  - `TracingConfiguration = object { group_id, metadata, workflow_name }`

    Granular configuration for tracing.

    - `group_id: optional string`

      The group id to attach to this trace to enable filtering and
      grouping in the traces dashboard.

    - `metadata: optional unknown`

      The arbitrary metadata to attach to this trace to enable
      filtering in the traces dashboard.

    - `workflow_name: optional string`

      The name of the workflow to attach to this trace. This is used to
      name the trace in the traces dashboard.

- `truncation: optional RealtimeTruncation`

  When the number of tokens in a conversation exceeds the model's input token limit, the conversation be truncated, meaning messages (starting from the oldest) will not be included in the model's context. A 32k context model with 4,096 max output tokens can only include 28,224 tokens in the context before truncation occurs.

  Clients can configure truncation behavior to truncate with a lower max token limit, which is an effective way to control token usage and cost.

  Truncation will reduce the number of cached tokens on the next turn (busting the cache), since messages are dropped from the beginning of the context. However, clients can also configure truncation to retain messages up to a fraction of the maximum context size, which will reduce the need for future truncations and thus improve the cache rate.

  Truncation can be disabled entirely, which means the server will never truncate but would instead return an error if the conversation exceeds the model's input token limit.

  - `"auto" or "disabled"`

    The truncation strategy to use for the session. `auto` is the default truncation strategy. `disabled` will disable truncation and emit errors when the conversation exceeds the input token limit.

    - `"auto"`

    - `"disabled"`

  - `RetentionRatioTruncation = object { retention_ratio, type, token_limits }`

    Retain a fraction of the conversation tokens when the conversation exceeds the input token limit. This allows you to amortize truncations across multiple turns, which can help improve cached token usage.

    - `retention_ratio: number`

      Fraction of post-instruction conversation tokens to retain (`0.0` - `1.0`) when the conversation exceeds the input token limit. Setting this to `0.8` means that messages will be dropped until 80% of the maximum allowed tokens are used. This helps reduce the frequency of truncations and improve cache rates.

    - `type: "retention_ratio"`

      Use retention ratio truncation.

      - `"retention_ratio"`

    - `token_limits: optional object { post_instructions }`

      Optional custom token limits for this truncation strategy. If not provided, the model's default token limits will be used.

      - `post_instructions: optional number`

        Maximum tokens allowed in the conversation after instructions (which including tool definitions). For example, setting this to 5,000 would mean that truncation would occur when the conversation exceeds 5,000 tokens after instructions. This cannot be higher than the model's context window size minus the maximum output tokens.

- `turn_detection: optional object { prefix_padding_ms, silence_duration_ms, threshold, type }`

  Configuration for turn detection. Can be set to `null` to turn off. Server
  VAD means that the model will detect the start and end of speech based on
  audio volume and respond at the end of user speech.

  - `prefix_padding_ms: optional number`

    Amount of audio to include before the VAD detected speech (in
    milliseconds). Defaults to 300ms.

  - `silence_duration_ms: optional number`

    Duration of silence to detect speech stop (in milliseconds). Defaults
    to 500ms. With shorter values the model will respond more quickly,
    but may jump in on short pauses from the user.

  - `threshold: optional number`

    Activation threshold for VAD (0.0 to 1.0), this defaults to 0.5. A
    higher threshold will require louder audio to activate the model, and
    thus might perform better in noisy environments.

  - `type: optional string`

    Type of turn detection, only `server_vad` is currently supported.

- `voice: optional string or "alloy" or "ash" or "ballad" or 7 more or object { id }`

  The voice the model uses to respond. Supported built-in voices are
  `alloy`, `ash`, `ballad`, `coral`, `echo`, `sage`, `shimmer`, `verse`,
  `marin`, and `cedar`. You may also provide a custom voice object with an
  `id`, for example `{ "id": "voice_1234" }`. Voice cannot be changed during
  the session once the model has responded with audio at least once.

  - `string`

  - `"alloy" or "ash" or "ballad" or 7 more`

    - `"alloy"`

    - `"ash"`

    - `"ballad"`

    - `"coral"`

    - `"echo"`

    - `"sage"`

    - `"shimmer"`

    - `"verse"`

    - `"marin"`

    - `"cedar"`

  - `ID = object { id }`

    Custom voice reference.

    - `id: string`

      The custom voice ID, e.g. `voice_1234`.

### Returns

- `id: optional string`

  Unique identifier for the session that looks like `sess_1234567890abcdef`.

- `audio: optional object { input, output }`

  Configuration for input and output audio for the session.

  - `input: optional object { format, noise_reduction, transcription, turn_detection }`

    - `format: optional RealtimeAudioFormats`

      The PCM audio format. Only a 24kHz sample rate is supported.

      - `PCMAudioFormat = object { rate, type }`

        The PCM audio format. Only a 24kHz sample rate is supported.

        - `rate: optional 24000`

          The sample rate of the audio. Always `24000`.

          - `24000`

        - `type: optional "audio/pcm"`

          The audio format. Always `audio/pcm`.

          - `"audio/pcm"`

      - `PCMUAudioFormat = object { type }`

        The G.711 μ-law format.

        - `type: optional "audio/pcmu"`

          The audio format. Always `audio/pcmu`.

          - `"audio/pcmu"`

      - `PCMAAudioFormat = object { type }`

        The G.711 A-law format.

        - `type: optional "audio/pcma"`

          The audio format. Always `audio/pcma`.

          - `"audio/pcma"`

    - `noise_reduction: optional object { type }`

      Configuration for input audio noise reduction.

      - `type: optional NoiseReductionType`

        Type of noise reduction. `near_field` is for close-talking microphones such as headphones, `far_field` is for far-field microphones such as laptop or conference room microphones.

        - `"near_field"`

        - `"far_field"`

    - `transcription: optional AudioTranscription`

      Configuration for input audio transcription.

      - `language: optional string`

        The language of the input audio. Supplying the input language in
        [ISO-639-1](https://en.wikipedia.org/wiki/List_of_ISO_639-1_codes) (e.g. `en`) format
        will improve accuracy and latency.

      - `model: optional string or "whisper-1" or "gpt-4o-mini-transcribe" or "gpt-4o-mini-transcribe-2025-12-15" or 2 more`

        The model to use for transcription. Current options are `whisper-1`, `gpt-4o-mini-transcribe`, `gpt-4o-mini-transcribe-2025-12-15`, `gpt-4o-transcribe`, and `gpt-4o-transcribe-diarize`. Use `gpt-4o-transcribe-diarize` when you need diarization with speaker labels.

        - `string`

        - `"whisper-1" or "gpt-4o-mini-transcribe" or "gpt-4o-mini-transcribe-2025-12-15" or 2 more`

          The model to use for transcription. Current options are `whisper-1`, `gpt-4o-mini-transcribe`, `gpt-4o-mini-transcribe-2025-12-15`, `gpt-4o-transcribe`, and `gpt-4o-transcribe-diarize`. Use `gpt-4o-transcribe-diarize` when you need diarization with speaker labels.

          - `"whisper-1"`

          - `"gpt-4o-mini-transcribe"`

          - `"gpt-4o-mini-transcribe-2025-12-15"`

          - `"gpt-4o-transcribe"`

          - `"gpt-4o-transcribe-diarize"`

      - `prompt: optional string`

        An optional text to guide the model's style or continue a previous audio
        segment.
        For `whisper-1`, the [prompt is a list of keywords](/docs/guides/speech-to-text#prompting).
        For `gpt-4o-transcribe` models (excluding `gpt-4o-transcribe-diarize`), the prompt is a free text string, for example "expect words related to technology".

    - `turn_detection: optional object { prefix_padding_ms, silence_duration_ms, threshold, type }`

      Configuration for turn detection.

      - `prefix_padding_ms: optional number`

      - `silence_duration_ms: optional number`

      - `threshold: optional number`

      - `type: optional string`

        Type of turn detection, only `server_vad` is currently supported.

  - `output: optional object { format, speed, voice }`

    - `format: optional RealtimeAudioFormats`

      The PCM audio format. Only a 24kHz sample rate is supported.

      - `PCMAudioFormat = object { rate, type }`

        The PCM audio format. Only a 24kHz sample rate is supported.

        - `rate: optional 24000`

          The sample rate of the audio. Always `24000`.

          - `24000`

        - `type: optional "audio/pcm"`

          The audio format. Always `audio/pcm`.

          - `"audio/pcm"`

      - `PCMUAudioFormat = object { type }`

        The G.711 μ-law format.

        - `type: optional "audio/pcmu"`

          The audio format. Always `audio/pcmu`.

          - `"audio/pcmu"`

      - `PCMAAudioFormat = object { type }`

        The G.711 A-law format.

        - `type: optional "audio/pcma"`

          The audio format. Always `audio/pcma`.

          - `"audio/pcma"`

    - `speed: optional number`

    - `voice: optional string or "alloy" or "ash" or "ballad" or 7 more`

      - `string`

      - `"alloy" or "ash" or "ballad" or 7 more`

        - `"alloy"`

        - `"ash"`

        - `"ballad"`

        - `"coral"`

        - `"echo"`

        - `"sage"`

        - `"shimmer"`

        - `"verse"`

        - `"marin"`

        - `"cedar"`

- `expires_at: optional number`

  Expiration timestamp for the session, in seconds since epoch.

- `include: optional array of "item.input_audio_transcription.logprobs"`

  Additional fields to include in server outputs.

  - `item.input_audio_transcription.logprobs`: Include logprobs for input audio transcription.

  - `"item.input_audio_transcription.logprobs"`

- `instructions: optional string`

  The default system instructions (i.e. system message) prepended to model
  calls. This field allows the client to guide the model on desired
  responses. The model can be instructed on response content and format,
  (e.g. "be extremely succinct", "act friendly", "here are examples of good
  responses") and on audio behavior (e.g. "talk quickly", "inject emotion
  into your voice", "laugh frequently"). The instructions are not guaranteed
  to be followed by the model, but they provide guidance to the model on the
  desired behavior.

  Note that the server sets default instructions which will be used if this
  field is not set and are visible in the `session.created` event at the
  start of the session.

- `max_output_tokens: optional number or "inf"`

  Maximum number of output tokens for a single assistant response,
  inclusive of tool calls. Provide an integer between 1 and 4096 to
  limit output tokens, or `inf` for the maximum available tokens for a
  given model. Defaults to `inf`.

  - `number`

  - `"inf"`

    - `"inf"`

- `model: optional string`

  The Realtime model used for this session.

- `object: optional string`

  The object type. Always `realtime.session`.

- `output_modalities: optional array of "text" or "audio"`

  The set of modalities the model can respond with. To disable audio,
  set this to ["text"].

  - `"text"`

  - `"audio"`

- `tool_choice: optional string`

  How the model chooses tools. Options are `auto`, `none`, `required`, or
  specify a function.

- `tools: optional array of RealtimeFunctionTool`

  Tools (functions) available to the model.

  - `description: optional string`

    The description of the function, including guidance on when and how
    to call it, and guidance about what to tell the user when calling
    (if anything).

  - `name: optional string`

    The name of the function.

  - `parameters: optional unknown`

    Parameters of the function in JSON Schema.

  - `type: optional "function"`

    The type of the tool, i.e. `function`.

    - `"function"`

- `tracing: optional "auto" or object { group_id, metadata, workflow_name }`

  Configuration options for tracing. Set to null to disable tracing. Once
  tracing is enabled for a session, the configuration cannot be modified.

  `auto` will create a trace for the session with default values for the
  workflow name, group id, and metadata.

  - `"auto"`

    Default tracing mode for the session.

    - `"auto"`

  - `TracingConfiguration = object { group_id, metadata, workflow_name }`

    Granular configuration for tracing.

    - `group_id: optional string`

      The group id to attach to this trace to enable filtering and
      grouping in the traces dashboard.

    - `metadata: optional unknown`

      The arbitrary metadata to attach to this trace to enable
      filtering in the traces dashboard.

    - `workflow_name: optional string`

      The name of the workflow to attach to this trace. This is used to
      name the trace in the traces dashboard.

- `turn_detection: optional object { prefix_padding_ms, silence_duration_ms, threshold, type }`

  Configuration for turn detection. Can be set to `null` to turn off. Server
  VAD means that the model will detect the start and end of speech based on
  audio volume and respond at the end of user speech.

  - `prefix_padding_ms: optional number`

    Amount of audio to include before the VAD detected speech (in
    milliseconds). Defaults to 300ms.

  - `silence_duration_ms: optional number`

    Duration of silence to detect speech stop (in milliseconds). Defaults
    to 500ms. With shorter values the model will respond more quickly,
    but may jump in on short pauses from the user.

  - `threshold: optional number`

    Activation threshold for VAD (0.0 to 1.0), this defaults to 0.5. A
    higher threshold will require louder audio to activate the model, and
    thus might perform better in noisy environments.

  - `type: optional string`

    Type of turn detection, only `server_vad` is currently supported.

### Example

```http
curl https://api.openai.com/v1/realtime/sessions \
    -H 'Content-Type: application/json' \
    -H "Authorization: Bearer $OPENAI_API_KEY" \
    -d '{
          "client_secret": {
            "expires_at": 0,
            "value": "value"
          }
        }'
```

#### Response

```json
{
  "id": "id",
  "audio": {
    "input": {
      "format": {
        "rate": 24000,
        "type": "audio/pcm"
      },
      "noise_reduction": {
        "type": "near_field"
      },
      "transcription": {
        "language": "language",
        "model": "string",
        "prompt": "prompt"
      },
      "turn_detection": {
        "prefix_padding_ms": 0,
        "silence_duration_ms": 0,
        "threshold": 0,
        "type": "type"
      }
    },
    "output": {
      "format": {
        "rate": 24000,
        "type": "audio/pcm"
      },
      "speed": 0,
      "voice": "ash"
    }
  },
  "expires_at": 0,
  "include": [
    "item.input_audio_transcription.logprobs"
  ],
  "instructions": "instructions",
  "max_output_tokens": 0,
  "model": "model",
  "object": "object",
  "output_modalities": [
    "text"
  ],
  "tool_choice": "tool_choice",
  "tools": [
    {
      "description": "description",
      "name": "name",
      "parameters": {},
      "type": "function"
    }
  ],
  "tracing": "auto",
  "turn_detection": {
    "prefix_padding_ms": 0,
    "silence_duration_ms": 0,
    "threshold": 0,
    "type": "type"
  }
}
```

### Example

```http
curl -X POST https://api.openai.com/v1/realtime/sessions \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-realtime",
    "modalities": ["audio", "text"],
    "instructions": "You are a friendly assistant."
  }'
```

#### Response

```json
{
  "id": "sess_001",
  "object": "realtime.session",
  "model": "gpt-realtime-2025-08-25",
  "modalities": ["audio", "text"],
  "instructions": "You are a friendly assistant.",
  "voice": "alloy",
  "input_audio_format": "pcm16",
  "output_audio_format": "pcm16",
  "input_audio_transcription": {
      "model": "whisper-1"
  },
  "turn_detection": null,
  "tools": [],
  "tool_choice": "none",
  "temperature": 0.7,
  "max_response_output_tokens": 200,
  "speed": 1.1,
  "tracing": "auto",
  "client_secret": {
    "value": "ek_abc123", 
    "expires_at": 1234567890
  }
}
```
