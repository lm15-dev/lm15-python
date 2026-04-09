# Counting tokens

For a detailed guide on counting tokens using the Gemini API, including how images, audio and video are counted, see the [Token counting guide](https://ai.google.dev/gemini-api/docs/tokens) and accompanying [Cookbook recipe](https://colab.research.google.com/github/google-gemini/cookbook/blob/main/quickstarts/Counting_Tokens.ipynb).

## Method: models.countTokens

- [Endpoint](https://ai.google.dev/api/tokens#body.HTTP_TEMPLATE)
- [Path parameters](https://ai.google.dev/api/tokens#body.PATH_PARAMETERS)
- [Request body](https://ai.google.dev/api/tokens#body.request_body)
  - [JSON representation](https://ai.google.dev/api/tokens#body.request_body.SCHEMA_REPRESENTATION)
- [Response body](https://ai.google.dev/api/tokens#body.response_body)
  - [JSON representation](https://ai.google.dev/api/tokens#body.CountTokensResponse.SCHEMA_REPRESENTATION)
- [Authorization scopes](https://ai.google.dev/api/tokens#body.aspect)
- [Example request](https://ai.google.dev/api/tokens#body.codeSnippets)
  - [Text](https://ai.google.dev/api/tokens#body.codeSnippets.group)
  - [Chat](https://ai.google.dev/api/tokens#body.codeSnippets.group_1)
  - [Inline media](https://ai.google.dev/api/tokens#body.codeSnippets.group_2)
  - [Video](https://ai.google.dev/api/tokens#body.codeSnippets.group_3)
  - [PDF](https://ai.google.dev/api/tokens#body.codeSnippets.group_4)
  - [Cache](https://ai.google.dev/api/tokens#body.codeSnippets.group_5)
  - [System Instruction](https://ai.google.dev/api/tokens#body.codeSnippets.group_6)
  - [Tools](https://ai.google.dev/api/tokens#body.codeSnippets.group_7)

Runs a model's tokenizer on input `Content` and returns the token count. Refer to the [tokens guide](https://ai.google.dev/gemini-api/docs/tokens) to learn more about tokens.

### Endpoint

post `https://generativelanguage.googleapis.com/v1beta/{model=models/*}:countTokens`   

### Path parameters

`model` `string` Required. The model's resource name. This serves as an ID for the Model to use.

This name should match a model name returned by the `models.list` method.

Format: `models/{model}` It takes the form `models/{model}`.

### Request body

The request body contains data with the following structure:
Fields `contents[]` ``object (`https://ai.google.dev/api/caching#Content`)`` Optional. The input given to the model as a prompt. This field is ignored when `generateContentRequest` is set.
`generateContentRequest` ``object (`https://ai.google.dev/api/batch-api#GenerateContentRequest`)`` Optional. The overall input given to the `Model`. This includes the prompt as well as other model steering information like [system instructions](https://ai.google.dev/gemini-api/docs/system-instructions), and/or function declarations for [function calling](https://ai.google.dev/gemini-api/docs/function-calling). `Model`s/`Content`s and `generateContentRequest`s are mutually exclusive. You can either send `Model` + `Content`s or a `generateContentRequest`, but never both.

### Example request

### Text

### Python

    from google import genai

    client = genai.Client()
    prompt = "The quick brown fox jumps over the lazy dog."

    # Count tokens using the new client method.
    total_tokens = client.models.count_tokens(
        model="gemini-2.0-flash", contents=prompt
    )
    print("total_tokens: ", total_tokens)
    # ( e.g., total_tokens: 10 )

    response = client.models.generate_content(
        model="gemini-2.0-flash", contents=prompt
    )

    # The usage_metadata provides detailed token counts.
    print(response.usage_metadata)
    # ( e.g., prompt_token_count: 11, candidates_token_count: 73, total_token_count: 84 )https://github.com/google-gemini/api-examples/blob/4ce9033e1d2f857db3f728d78399e3d7ded8ef05/python/count_tokens.py#L36-L54

### Node.js

    // Make sure to include the following import:
    // import {GoogleGenAI} from '@google/genai';
    const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY });
    const prompt = "The quick brown fox jumps over the lazy dog.";
    const countTokensResponse = await ai.models.countTokens({
      model: "gemini-2.0-flash",
      contents: prompt,
    });
    console.log(countTokensResponse.totalTokens);

    const generateResponse = await ai.models.generateContent({
      model: "gemini-2.0-flash",
      contents: prompt,
    });
    console.log(generateResponse.usageMetadata);https://github.com/google-gemini/api-examples/blob/4ce9033e1d2f857db3f728d78399e3d7ded8ef05/javascript/count_tokens.js#L38-L52

### Go

    ctx := context.Background()
    client, err := genai.NewClient(ctx, &genai.ClientConfig{
    	APIKey:  os.Getenv("GEMINI_API_KEY"),
    	Backend: genai.BackendGeminiAPI,
    })
    if err != nil {
    	log.Fatal(err)
    }
    prompt := "The quick brown fox jumps over the lazy dog."

    // Convert prompt to a slice of *genai.Content using the helper.
    contents := []*genai.Content{
    	genai.NewContentFromText(prompt, genai.RoleUser),
    }
    countResp, err := client.Models.CountTokens(ctx, "gemini-2.0-flash", contents, nil)
    if err != nil {
    	return err
    }
    fmt.Println("total_tokens:", countResp.TotalTokens)

    response, err := client.Models.GenerateContent(ctx, "gemini-2.0-flash", contents, nil)
    if err != nil {
    	log.Fatal(err)
    }
    usageMetadata, err := json.MarshalIndent(response.UsageMetadata, "", "  ")
    if err != nil {
    	log.Fatal(err)
    }
    fmt.Println(string(usageMetadata))https://github.com/google-gemini/api-examples/blob/4ce9033e1d2f857db3f728d78399e3d7ded8ef05/go/count_tokens.go#L38-L66

### Shell

    curl https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:countTokens?key=$GEMINI_API_KEY \
        -H 'Content-Type: application/json' \
        -X POST \
        -d '{
          "contents": [{
            "parts":[{
              "text": "The quick brown fox jumps over the lazy dog."
              }],
            }],
          }'https://github.com/google-gemini/deprecated-generative-ai-python/blob/7a7cc5474ddaa0255a4410e05361028a24400abd/samples/rest/count_tokens.sh#L28-L38

### Chat

### Python

    from google import genai
    from google.genai import types

    client = genai.Client()

    chat = client.chats.create(
        model="gemini-2.0-flash",
        history=[
            types.Content(
                role="user", parts=[types.Part(text="Hi my name is Bob")]
            ),
            types.Content(role="model", parts=[types.Part(text="Hi Bob!")]),
        ],
    )
    # Count tokens for the chat history.
    print(
        client.models.count_tokens(
            model="gemini-2.0-flash", contents=chat.get_history()
        )
    )
    # ( e.g., total_tokens: 10 )

    response = chat.send_message(
        message="In one sentence, explain how a computer works to a young child."
    )
    print(response.usage_metadata)
    # ( e.g., prompt_token_count: 25, candidates_token_count: 21, total_token_count: 46 )

    # You can count tokens for the combined history and a new message.
    extra = types.UserContent(
        parts=[
            types.Part(
                text="What is the meaning of life?",
            )
        ]
    )
    history = chat.get_history()
    history.append(extra)
    print(client.models.count_tokens(model="gemini-2.0-flash", contents=history))
    # ( e.g., total_tokens: 56 )https://github.com/google-gemini/api-examples/blob/4ce9033e1d2f857db3f728d78399e3d7ded8ef05/python/count_tokens.py#L59-L98

### Node.js

    // Make sure to include the following import:
    // import {GoogleGenAI} from '@google/genai';
    const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY });
    // Initial chat history.
    const history = [
      { role: "user", parts: [{ text: "Hi my name is Bob" }] },
      { role: "model", parts: [{ text: "Hi Bob!" }] },
    ];
    const chat = ai.chats.create({
      model: "gemini-2.0-flash",
      history: history,
    });

    // Count tokens for the current chat history.
    const countTokensResponse = await ai.models.countTokens({
      model: "gemini-2.0-flash",
      contents: chat.getHistory(),
    });
    console.log(countTokensResponse.totalTokens);

    const chatResponse = await chat.sendMessage({
      message: "In one sentence, explain how a computer works to a young child.",
    });
    console.log(chatResponse.usageMetadata);

    // Add an extra user message to the history.
    const extraMessage = {
      role: "user",
      parts: [{ text: "What is the meaning of life?" }],
    };
    const combinedHistory = chat.getHistory();
    combinedHistory.push(extraMessage);
    const combinedCountTokensResponse = await ai.models.countTokens({
      model: "gemini-2.0-flash",
      contents: combinedHistory,
    });
    console.log(
      "Combined history token count:",
      combinedCountTokensResponse.totalTokens,
    );https://github.com/google-gemini/api-examples/blob/4ce9033e1d2f857db3f728d78399e3d7ded8ef05/javascript/count_tokens.js#L62-L101

### Go

    ctx := context.Background()
    client, err := genai.NewClient(ctx, &genai.ClientConfig{
    	APIKey:  os.Getenv("GEMINI_API_KEY"),
    	Backend: genai.BackendGeminiAPI,
    })
    if err != nil {
    	log.Fatal(err)
    }

    // Initialize chat with some history.
    history := []*genai.Content{
    	{Role: genai.RoleUser, Parts: []*genai.Part{{Text: "Hi my name is Bob"}}},
    	{Role: genai.RoleModel, Parts: []*genai.Part{{Text: "Hi Bob!"}}},
    }
    chat, err := client.Chats.Create(ctx, "gemini-2.0-flash", nil, history)
    if err != nil {
    	log.Fatal(err)
    }

    firstTokenResp, err := client.Models.CountTokens(ctx, "gemini-2.0-flash", chat.History(false), nil)
    if err != nil {
    	log.Fatal(err)
    }
    fmt.Println(firstTokenResp.TotalTokens)

    resp, err := chat.SendMessage(ctx, genai.Part{
    	Text: "In one sentence, explain how a computer works to a young child."},
    )
    if err != nil {
    	log.Fatal(err)
    }
    fmt.Printf("%#v\n", resp.UsageMetadata)

    // Append an extra user message and recount.
    extra := genai.NewContentFromText("What is the meaning of life?", genai.RoleUser)
    hist := chat.History(false)
    hist = append(hist, extra)

    secondTokenResp, err := client.Models.CountTokens(ctx, "gemini-2.0-flash", hist, nil)
    if err != nil {
    	log.Fatal(err)
    }
    fmt.Println(secondTokenResp.TotalTokens)https://github.com/google-gemini/api-examples/blob/4ce9033e1d2f857db3f728d78399e3d7ded8ef05/go/count_tokens.go#L73-L115

### Shell

    curl https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:countTokens?key=$GEMINI_API_KEY \
        -H 'Content-Type: application/json' \
        -X POST \
        -d '{
          "contents": [
            {"role": "user",
            "parts": [{"text": "Hi, my name is Bob."}],
            },
            {"role": "model",
             "parts":[{"text": "Hi Bob"}],
            },
          ],
          }'https://github.com/google-gemini/deprecated-generative-ai-python/blob/7a7cc5474ddaa0255a4410e05361028a24400abd/samples/rest/count_tokens.sh#L42-L55

### Inline media

### Python

    from google import genai
    import PIL.Image

    client = genai.Client()
    prompt = "Tell me about this image"
    your_image_file = PIL.Image.open(media / "organ.jpg")

    # Count tokens for combined text and inline image.
    print(
        client.models.count_tokens(
            model="gemini-2.0-flash", contents=[prompt, your_image_file]
        )
    )
    # ( e.g., total_tokens: 263 )

    response = client.models.generate_content(
        model="gemini-2.0-flash", contents=[prompt, your_image_file]
    )
    print(response.usage_metadata)
    # ( e.g., prompt_token_count: 264, candidates_token_count: 80, total_token_count: 345 )https://github.com/google-gemini/api-examples/blob/4ce9033e1d2f857db3f728d78399e3d7ded8ef05/python/count_tokens.py#L103-L122

### Node.js

    // Make sure to include the following import:
    // import {GoogleGenAI} from '@google/genai';
    const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY });
    const prompt = "Tell me about this image";
    const imageBuffer = fs.readFileSync(path.join(media, "organ.jpg"));

    // Convert buffer to base64 string.
    const imageBase64 = imageBuffer.toString("base64");

    // Build contents using createUserContent and createPartFromBase64.
    const contents = createUserContent([
      prompt,
      createPartFromBase64(imageBase64, "image/jpeg"),
    ]);

    const countTokensResponse = await ai.models.countTokens({
      model: "gemini-2.0-flash",
      contents: contents,
    });
    console.log(countTokensResponse.totalTokens);

    const generateResponse = await ai.models.generateContent({
      model: "gemini-2.0-flash",
      contents: contents,
    });
    console.log(generateResponse.usageMetadata);https://github.com/google-gemini/api-examples/blob/4ce9033e1d2f857db3f728d78399e3d7ded8ef05/javascript/count_tokens.js#L112-L137

### Go

    model := client.GenerativeModel("gemini-1.5-flash")
    prompt := "Tell me about this image"
    imageFile, err := os.ReadFile(filepath.Join(testDataDir, "personWorkingOnComputer.jpg"))
    if err != nil {
    	log.Fatal(err)
    }
    // Call `CountTokens` to get the input token count
    // of the combined text and file (`total_tokens`).
    // An image's display or file size does not affect its token count.
    // Optionally, you can call `count_tokens` for the text and file separately.
    tokResp, err := model.CountTokens(ctx, genai.Text(prompt), genai.ImageData("jpeg", imageFile))
    if err != nil {
    	log.Fatal(err)
    }
    fmt.Println("total_tokens:", tokResp.TotalTokens)
    // ( total_tokens: 264 )

    resp, err := model.GenerateContent(ctx, genai.Text(prompt), genai.ImageData("jpeg", imageFile))
    if err != nil {
    	log.Fatal(err)
    }

    fmt.Println("prompt_token_count:", resp.UsageMetadata.PromptTokenCount)
    fmt.Println("candidates_token_count:", resp.UsageMetadata.CandidatesTokenCount)
    fmt.Println("total_token_count:", resp.UsageMetadata.TotalTokenCount)
    // ( prompt_token_count: 264, candidates_token_count: 100, total_token_count: 364 )https://github.com/google/generative-ai-go/blob/b1a1f5eba2c10785895c91f4189f1ef7940c4764/genai/internal/samples/docs-snippets_test.go#L682-L707

### Shell

    curl "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:countTokens?key=$GEMINI_API_KEY" \
        -H 'Content-Type: application/json' \
        -X POST \
        -d '{
          "contents": [{
            "parts":[
                {"text": "Tell me about this instrument"},
                {
                  "inline_data": {
                    "mime_type":"image/jpeg",
                    "data": "'$(base64 $B64FLAGS $IMG_PATH)'"
                  }
                }
            ]
            }]
           }' 2> /dev/nullhttps://github.com/google-gemini/deprecated-generative-ai-python/blob/7a7cc5474ddaa0255a4410e05361028a24400abd/samples/rest/count_tokens.sh#L59-L75

### Video

### Python

    from google import genai
    import time

    client = genai.Client()
    prompt = "Tell me about this video"
    your_file = client.files.upload(file=media / "Big_Buck_Bunny.mp4")

    # Poll until the video file is completely processed (state becomes ACTIVE).
    while not your_file.state or your_file.state.name != "ACTIVE":
        print("Processing video...")
        print("File state:", your_file.state)
        time.sleep(5)
        your_file = client.files.get(name=your_file.name)

    print(
        client.models.count_tokens(
            model="gemini-2.0-flash", contents=[prompt, your_file]
        )
    )
    # ( e.g., total_tokens: 300 )

    response = client.models.generate_content(
        model="gemini-2.0-flash", contents=[prompt, your_file]
    )
    print(response.usage_metadata)
    # ( e.g., prompt_token_count: 301, candidates_token_count: 60, total_token_count: 361 )https://github.com/google-gemini/api-examples/blob/4ce9033e1d2f857db3f728d78399e3d7ded8ef05/python/count_tokens.py#L149-L174

### Node.js

    // Make sure to include the following import:
    // import {GoogleGenAI} from '@google/genai';
    const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY });
    const prompt = "Tell me about this video";
    let videoFile = await ai.files.upload({
      file: path.join(media, "Big_Buck_Bunny.mp4"),
      config: { mimeType: "video/mp4" },
    });

    // Poll until the video file is completely processed (state becomes ACTIVE).
    while (!videoFile.state || videoFile.state.toString() !== "ACTIVE") {
      console.log("Processing video...");
      console.log("File state: ", videoFile.state);
      await sleep(5000);
      videoFile = await ai.files.get({ name: videoFile.name });
    }

    const countTokensResponse = await ai.models.countTokens({
      model: "gemini-2.0-flash",
      contents: createUserContent([
        prompt,
        createPartFromUri(videoFile.uri, videoFile.mimeType),
      ]),
    });
    console.log(countTokensResponse.totalTokens);

    const generateResponse = await ai.models.generateContent({
      model: "gemini-2.0-flash",
      contents: createUserContent([
        prompt,
        createPartFromUri(videoFile.uri, videoFile.mimeType),
      ]),
    });
    console.log(generateResponse.usageMetadata);https://github.com/google-gemini/api-examples/blob/4ce9033e1d2f857db3f728d78399e3d7ded8ef05/javascript/count_tokens.js#L183-L216

### Go

    ctx := context.Background()
    client, err := genai.NewClient(ctx, &genai.ClientConfig{
    	APIKey:  os.Getenv("GEMINI_API_KEY"),
    	Backend: genai.BackendGeminiAPI,
    })
    if err != nil {
    	log.Fatal(err)
    }

    file, err := client.Files.UploadFromPath(
    	ctx, 
    	filepath.Join(getMedia(), "Big_Buck_Bunny.mp4"), 
    	&genai.UploadFileConfig{
    		MIMEType : "video/mp4",
    	},
    )
    if err != nil {
    	log.Fatal(err)
    }

    // Poll until the video file is completely processed (state becomes ACTIVE).
    for file.State == genai.FileStateUnspecified || file.State != genai.FileStateActive {
    	fmt.Println("Processing video...")
    	fmt.Println("File state:", file.State)
    	time.Sleep(5 * time.Second)

    	file, err = client.Files.Get(ctx, file.Name, nil)
    	if err != nil {
    		log.Fatal(err)
    	}
    }

    parts := []*genai.Part{
    	genai.NewPartFromText("Tell me about this video"),
    	genai.NewPartFromURI(file.URI, file.MIMEType),
    }
    contents := []*genai.Content{
    	genai.NewContentFromParts(parts, genai.RoleUser),
    }

    tokenResp, err := client.Models.CountTokens(ctx, "gemini-2.0-flash", contents, nil)
    if err != nil {
    	log.Fatal(err)
    }
    fmt.Println("Multimodal video/audio token count:", tokenResp.TotalTokens)
    response, err := client.Models.GenerateContent(ctx, "gemini-2.0-flash", contents, nil)
    if err != nil {
    	log.Fatal(err)
    }
    usageMetadata, err := json.MarshalIndent(response.UsageMetadata, "", "  ")
    if err != nil {
    	log.Fatal(err)
    }
    fmt.Println(string(usageMetadata))https://github.com/google-gemini/api-examples/blob/4ce9033e1d2f857db3f728d78399e3d7ded8ef05/go/count_tokens.go#L171-L224

### Shell


    MIME_TYPE=$(file -b --mime-type "${VIDEO_PATH}")
    NUM_BYTES=$(wc -c < "${VIDEO_PATH}")
    DISPLAY_NAME=VIDEO_PATH

    # Initial resumable request defining metadata.
    # The upload url is in the response headers dump them to a file.
    curl "${BASE_URL}/upload/v1beta/files?key=${GOOGLE_API_KEY}" \
      -D upload-header.tmp \
      -H "X-Goog-Upload-Protocol: resumable" \
      -H "X-Goog-Upload-Command: start" \
      -H "X-Goog-Upload-Header-Content-Length: ${NUM_BYTES}" \
      -H "X-Goog-Upload-Header-Content-Type: ${MIME_TYPE}" \
      -H "Content-Type: application/json" \
      -d "{'file': {'display_name': '${DISPLAY_NAME}'}}" 2> /dev/null

    upload_url=$(grep -i "x-goog-upload-url: " "${tmp_header_file}" | cut -d" " -f2 | tr -d "\r")
    rm "${tmp_header_file}"

    # Upload the actual bytes.
    curl "${upload_url}" \
      -H "Content-Length: ${NUM_BYTES}" \
      -H "X-Goog-Upload-Offset: 0" \
      -H "X-Goog-Upload-Command: upload, finalize" \
      --data-binary "@${VIDEO_PATH}" 2> /dev/null > file_info.json

    file_uri=$(jq ".file.uri" file_info.json)

    state=$(jq ".file.state" file_info.json)

    name=$(jq ".file.name" file_info.json)

    while [[ "($state)" = *"PROCESSING"* ]];
    do
      echo "Processing video..."
      sleep 5
      # Get the file of interest to check state
      curl https://generativelanguage.googleapis.com/v1beta/files/$name > file_info.json
      state=$(jq ".file.state" file_info.json)
    done

    curl "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:countTokens?key=$GOOGLE_API_KEY" \
        -H 'Content-Type: application/json' \
        -X POST \
        -d '{
          "contents": [{
            "parts":[
              {"text": "Describe this video clip"},
              {"file_data":{"mime_type": "video/mp4", "file_uri": '$file_uri'}}]
            }]
           }'https://github.com/google-gemini/deprecated-generative-ai-python/blob/7a7cc5474ddaa0255a4410e05361028a24400abd/samples/rest/count_tokens.sh#L125-L176

### PDF

### Python

    from google import genai

    client = genai.Client()
    sample_pdf = client.files.upload(file=media / "test.pdf")
    token_count = client.models.count_tokens(
        model="gemini-2.0-flash",
        contents=["Give me a summary of this document.", sample_pdf],
    )
    print(f"{token_count=}")

    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=["Give me a summary of this document.", sample_pdf],
    )
    print(response.usage_metadata)https://github.com/google-gemini/api-examples/blob/4ce9033e1d2f857db3f728d78399e3d7ded8ef05/python/count_tokens.py#L179-L193

### Cache

### Python

    from google import genai
    from google.genai import types
    import time

    client = genai.Client()
    your_file = client.files.upload(file=media / "a11.txt")

    cache = client.caches.create(
        model="gemini-1.5-flash-001",
        config={
            "contents": ["Here the Apollo 11 transcript:", your_file],
            "system_instruction": None,
            "tools": None,
        },
    )

    # Create a prompt.
    prompt = "Please give a short summary of this file."

    # Count tokens for the prompt (the cached content is not passed here).
    print(client.models.count_tokens(model="gemini-2.0-flash", contents=prompt))
    # ( e.g., total_tokens: 9 )

    response = client.models.generate_content(
        model="gemini-1.5-flash-001",
        contents=prompt,
        config=types.GenerateContentConfig(
            cached_content=cache.name,
        ),
    )
    print(response.usage_metadata)
    # ( e.g., prompt_token_count: ..., cached_content_token_count: ..., candidates_token_count: ... )
    client.caches.delete(name=cache.name)https://github.com/google-gemini/api-examples/blob/4ce9033e1d2f857db3f728d78399e3d7ded8ef05/python/count_tokens.py#L198-L230

### Node.js

      // Make sure to include the following import:
      // import {GoogleGenAI} from '@google/genai';
      const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY });
      const textFile = await ai.files.upload({
        file: path.join(media, "a11.txt"),
        config: { mimeType: "text/plain" },
      });

      const cache = await ai.caches.create({
        model: "gemini-1.5-flash-001",
        config: {
          contents: createUserContent([
            "Here the Apollo 11 transcript:",
            createPartFromUri(textFile.uri, textFile.mimeType),
          ]),
          system_instruction: null,
          tools: null,
        },
      });

      const prompt = "Please give a short summary of this file.";
      const countTokensResponse = await ai.models.countTokens({
        model: "gemini-2.0-flash",
        contents: prompt,
      });
      console.log(countTokensResponse.totalTokens);

      const generateResponse = await ai.models.generateContent({
        model: "gemini-1.5-flash-001",
        contents: prompt,
        config: { cachedContent: cache.name },
      });
      console.log(generateResponse.usageMetadata);

      await ai.caches.delete({ name: cache.name });
      return {
        totalTokens: countTokensResponse.totalTokens,
        usage: generateResponse.usageMetadata,
      };
    }
    https://github.com/google-gemini/api-examples/blob/4ce9033e1d2f857db3f728d78399e3d7ded8ef05/javascript/count_tokens.js#L261-L-1

### Go

    ctx := context.Background()
    client, err := genai.NewClient(ctx, &genai.ClientConfig{
    	APIKey:  os.Getenv("GEMINI_API_KEY"),
    	Backend: genai.BackendGeminiAPI,
    })
    if err != nil {
    	log.Fatal(err)
    }

    file, err := client.Files.UploadFromPath(
    	ctx, 
    	filepath.Join(getMedia(), "a11.txt"), 
    	&genai.UploadFileConfig{
    		MIMEType : "text/plain",
    	},
    )
    if err != nil {
    	log.Fatal(err)
    }
    parts := []*genai.Part{
    	genai.NewPartFromText("Here the Apollo 11 transcript:"),
    	genai.NewPartFromURI(file.URI, file.MIMEType),
    }
    contents := []*genai.Content{
    	genai.NewContentFromParts(parts, genai.RoleUser),
    }

    // Create cached content using a simple slice with text and a file.
    cache, err := client.Caches.Create(ctx, "gemini-1.5-flash-001", &genai.CreateCachedContentConfig{
    	Contents: contents,
    })
    if err != nil {
    	log.Fatal(err)
    }

    prompt := "Please give a short summary of this file."
    countResp, err := client.Models.CountTokens(ctx, "gemini-2.0-flash", []*genai.Content{
    	genai.NewContentFromText(prompt, genai.RoleUser),
    }, nil)
    if err != nil {
    	log.Fatal(err)
    }
    fmt.Printf("%d", countResp.TotalTokens)
    response, err := client.Models.GenerateContent(ctx, "gemini-1.5-flash-001", []*genai.Content{
    	genai.NewContentFromText(prompt, genai.RoleUser),
    }, &genai.GenerateContentConfig{
    	CachedContent: cache.Name,
    })
    if err != nil {
    	log.Fatal(err)
    }

    usageMetadata, err := json.MarshalIndent(response.UsageMetadata, "", "  ")
    if err != nil {
    	log.Fatal(err)
    }
    // Returns `nil` for some reason
    fmt.Println(string(usageMetadata))
    _, err = client.Caches.Delete(ctx, cache.Name, &genai.DeleteCachedContentConfig{})https://github.com/google-gemini/api-examples/blob/4ce9033e1d2f857db3f728d78399e3d7ded8ef05/go/count_tokens.go#L278-L336

### System Instruction

### Go

    ctx := context.Background()
    client, err := genai.NewClient(ctx, &genai.ClientConfig{
    	APIKey:  os.Getenv("GEMINI_API_KEY"),
    	Backend: genai.BackendGeminiAPI,
    })
    if err != nil {
    	log.Fatal(err)
    }

    // Construct the user message contents.
    contents := []*genai.Content{
    	genai.NewContentFromText("Good morning! How are you?", genai.RoleUser),
    }

    // Set the system instruction as a *genai.Content.
    config := &genai.GenerateContentConfig{
    	SystemInstruction: genai.NewContentFromText("You are a cat. Your name is Neko.", genai.RoleUser),
    }

    response, err := client.Models.GenerateContent(ctx, "gemini-2.0-flash", contents, config)
    if err != nil {
    	log.Fatal(err)
    }
    printResponse(response)https://github.com/google-gemini/api-examples/blob/4ce9033e1d2f857db3f728d78399e3d7ded8ef05/go/system_instruction.go#L13-L36

### Tools

### Java

    FunctionDeclaration multiplyDefinition =
        defineFunction(
            /* name  */ "multiply",
            /* description */ "returns a * b.",
            /* parameters */ Arrays.asList(
                Schema.numDouble("a", "First parameter"),
                Schema.numDouble("b", "Second parameter")),
            /* required */ Arrays.asList("a", "b"));

    Tool tool = new Tool(Arrays.asList(multiplyDefinition), null);
    ;

    // Specify a Gemini model appropriate for your use case
    GenerativeModel gm =
        new GenerativeModel(
            /* modelName */ "gemini-1.5-flash",
            // Access your API key as a Build Configuration variable (see "Set up your API key"
            // above)
            /* apiKey */ BuildConfig.apiKey,
            /* generationConfig (optional) */ null,
            /* safetySettings (optional) */ null,
            /* requestOptions (optional) */ new RequestOptions(),
            /* tools (optional) */ Arrays.asList(tool));
    GenerativeModelFutures model = GenerativeModelFutures.from(gm);

    Content inputContent = new Content.Builder().addText("What's your name?.").build();

    // For illustrative purposes only. You should use an executor that fits your needs.
    Executor executor = Executors.newSingleThreadExecutor();

    // For text-only input
    ListenableFuture<CountTokensResponse> countTokensResponse = model.countTokens(inputContent);

    Futures.addCallback(
        countTokensResponse,
        new FutureCallback<CountTokensResponse>() {
          @Override
          public void onSuccess(CountTokensResponse result) {
            int totalTokens = result.getTotalTokens();
            System.out.println("TotalTokens = " + totalTokens);
          }

          @Override
          public void onFailure(Throwable t) {
            t.printStackTrace();
          }
        },
        executor);https://github.com/google-gemini/generative-ai-android/blob/a77dc5e4dc07d7aa710f59fde1908fbfff2e0e70/samples/src/main/java/com/google/ai/client/generative/samples/java/count_tokens.java#L239-L286

### Response body

A response from `models.countTokens`.

It returns the model's `tokenCount` for the `prompt`.

If successful, the response body contains data with the following structure:
Fields `totalTokens` `integer` The number of tokens that the `Model` tokenizes the `prompt` into. Always non-negative.
`cachedContentTokenCount` `integer` Number of tokens in the cached part of the prompt (the cached content).
`promptTokensDetails[]` ``object (`https://ai.google.dev/api/generate-content#v1beta.ModalityTokenCount`)`` Output only. List of modalities that were processed in the request input.
`cacheTokensDetails[]` ``object (`https://ai.google.dev/api/generate-content#v1beta.ModalityTokenCount`)`` Output only. List of modalities that were processed in the cached content.

| JSON representation |
|---|
| ``` { "totalTokens": integer, "cachedContentTokenCount": integer, "promptTokensDetails": [ { object (`https://ai.google.dev/api/generate-content#v1beta.ModalityTokenCount`) } ], "cacheTokensDetails": [ { object (`https://ai.google.dev/api/generate-content#v1beta.ModalityTokenCount`) } ] } ``` |