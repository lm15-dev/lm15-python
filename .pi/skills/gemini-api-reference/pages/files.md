# Using files

The Gemini API supports uploading media files separately from the prompt input, allowing your media to be reused across multiple requests and multiple prompts. For more details, check out the [Prompting with media](https://ai.google.dev/gemini-api/docs/prompting_with_media) guide.

## Method: media.upload

- [Endpoint](https://ai.google.dev/api/files#body.HTTP_TEMPLATE)
- [Request body](https://ai.google.dev/api/files#body.request_body)
  - [JSON representation](https://ai.google.dev/api/files#body.request_body.SCHEMA_REPRESENTATION)
- [Response body](https://ai.google.dev/api/files#body.response_body)
  - [JSON representation](https://ai.google.dev/api/files#body.CreateFileResponse.SCHEMA_REPRESENTATION)
- [Example request](https://ai.google.dev/api/files#body.codeSnippets)
  - [Image](https://ai.google.dev/api/files#body.codeSnippets.group)
  - [Audio](https://ai.google.dev/api/files#body.codeSnippets.group_1)
  - [Text](https://ai.google.dev/api/files#body.codeSnippets.group_2)
  - [Video](https://ai.google.dev/api/files#body.codeSnippets.group_3)
  - [PDF](https://ai.google.dev/api/files#body.codeSnippets.group_4)

Creates a `File`.

### Endpoint

- Upload URI, for media upload requests:  
post `https://generativelanguage.googleapis.com/upload/v1beta/files`
- Metadata URI, for metadata-only requests:  
post `https://generativelanguage.googleapis.com/v1beta/files`

### Request body

The request body contains data with the following structure:
Fields `file` ``object (`https://ai.google.dev/api/files#File`)`` Optional. Metadata for the file to create.

### Example request

### Image

### Python

    from google import genai

    client = genai.Client()
    myfile = client.files.upload(file=media / "Cajun_instruments.jpg")
    print(f"{myfile=}")

    result = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=[
            myfile,
            "\n\n",
            "Can you tell me about the instruments in this photo?",
        ],
    )
    print(f"{result.text=}")https://github.com/google-gemini/api-examples/blob/4ce9033e1d2f857db3f728d78399e3d7ded8ef05/python/files.py#L40-L54

### Node.js

    // Make sure to include the following import:
    // import {GoogleGenAI} from '@google/genai';
    const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY });
    const myfile = await ai.files.upload({
      file: path.join(media, "Cajun_instruments.jpg"),
      config: { mimeType: "image/jpeg" },
    });
    console.log("Uploaded file:", myfile);

    const result = await ai.models.generateContent({
      model: "gemini-2.0-flash",
      contents: createUserContent([
        createPartFromUri(myfile.uri, myfile.mimeType),
        "\n\n",
        "Can you tell me about the instruments in this photo?",
      ]),
    });
    console.log("result.text=", result.text);https://github.com/google-gemini/api-examples/blob/4ce9033e1d2f857db3f728d78399e3d7ded8ef05/javascript/files.js#L59-L76

### Go

    ctx := context.Background()
    client, err := genai.NewClient(ctx, &genai.ClientConfig{
    	APIKey:  os.Getenv("GEMINI_API_KEY"),
    	Backend: genai.BackendGeminiAPI,
    })
    if err != nil {
    	log.Fatal(err)
    }
    myfile, err := client.Files.UploadFromPath(
    	ctx, 
    	filepath.Join(getMedia(), "Cajun_instruments.jpg"), 
    	&genai.UploadFileConfig{
    		MIMEType : "image/jpeg",
    	},
    )
    if err != nil {
    	log.Fatal(err)
    }
    fmt.Printf("myfile=%+v\n", myfile)

    parts := []*genai.Part{
    	genai.NewPartFromURI(myfile.URI, myfile.MIMEType),
    	genai.NewPartFromText("\n\n"),
    	genai.NewPartFromText("Can you tell me about the instruments in this photo?"),
    }

    contents := []*genai.Content{
    	genai.NewContentFromParts(parts, genai.RoleUser),
    }

    response, err := client.Models.GenerateContent(ctx, "gemini-2.0-flash", contents, nil)
    if err != nil {
    	log.Fatal(err)
    }
    text := response.Text()
    fmt.Printf("result.text=%s\n", text)https://github.com/google-gemini/api-examples/blob/4ce9033e1d2f857db3f728d78399e3d7ded8ef05/go/files.go#L59-L94

### Shell

    MIME_TYPE=$(file -b --mime-type "${IMG_PATH_2}")
    NUM_BYTES=$(wc -c < "${IMG_PATH_2}")
    DISPLAY_NAME=TEXT

    tmp_header_file=upload-header.tmp

    # Initial resumable request defining metadata.
    # The upload url is in the response headers dump them to a file.
    curl "${BASE_URL}/upload/v1beta/files?key=${GEMINI_API_KEY}" \
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
      --data-binary "@${IMG_PATH_2}" 2> /dev/null > file_info.json

    file_uri=$(jq ".file.uri" file_info.json)
    echo file_uri=$file_uri

    # Now generate content using that file
    curl "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key=$GEMINI_API_KEY" \
        -H 'Content-Type: application/json' \
        -X POST \
        -d '{
          "contents": [{
            "parts":[
              {"text": "Can you tell me about the instruments in this photo?"},
              {"file_data":
                {"mime_type": "image/jpeg", 
                "file_uri": '$file_uri'}
            }]
            }]
           }' 2> /dev/null > response.json

    cat response.json
    echo

    jq ".candidates[].content.parts[].text" response.jsonhttps://github.com/google-gemini/deprecated-generative-ai-python/blob/7a7cc5474ddaa0255a4410e05361028a24400abd/samples/rest/files.sh#L84-L133

### Audio

### Python

    from google import genai

    client = genai.Client()
    myfile = client.files.upload(file=media / "sample.mp3")
    print(f"{myfile=}")

    result = client.models.generate_content(
        model="gemini-2.0-flash", contents=[myfile, "Describe this audio clip"]
    )
    print(f"{result.text=}")https://github.com/google-gemini/api-examples/blob/4ce9033e1d2f857db3f728d78399e3d7ded8ef05/python/files.py#L59-L68

### Node.js

    // Make sure to include the following import:
    // import {GoogleGenAI} from '@google/genai';
    const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY });
    const myfile = await ai.files.upload({
      file: path.join(media, "sample.mp3"),
      config: { mimeType: "audio/mpeg" },
    });
    console.log("Uploaded file:", myfile);

    const result = await ai.models.generateContent({
      model: "gemini-2.0-flash",
      contents: createUserContent([
        createPartFromUri(myfile.uri, myfile.mimeType),
        "Describe this audio clip",
      ]),
    });
    console.log("result.text=", result.text);https://github.com/google-gemini/api-examples/blob/4ce9033e1d2f857db3f728d78399e3d7ded8ef05/javascript/files.js#L83-L99

### Go

    ctx := context.Background()
    client, err := genai.NewClient(ctx, &genai.ClientConfig{
    	APIKey:  os.Getenv("GEMINI_API_KEY"),
    	Backend: genai.BackendGeminiAPI,
    })
    if err != nil {
    	log.Fatal(err)
    }
    myfile, err := client.Files.UploadFromPath(
    	ctx, 
    	filepath.Join(getMedia(), "sample.mp3"), 
    	&genai.UploadFileConfig{
    		MIMEType : "audio/mpeg",
    	},
    )
    if err != nil {
    	log.Fatal(err)
    }
    fmt.Printf("myfile=%+v\n", myfile)

    parts := []*genai.Part{
    	genai.NewPartFromURI(myfile.URI, myfile.MIMEType),
    	genai.NewPartFromText("Describe this audio clip"),
    }

    contents := []*genai.Content{
    	genai.NewContentFromParts(parts, genai.RoleUser),
    }

    response, err := client.Models.GenerateContent(ctx, "gemini-2.0-flash", contents, nil)
    if err != nil {
    	log.Fatal(err)
    }
    text := response.Text()
    fmt.Printf("result.text=%s\n", text)https://github.com/google-gemini/api-examples/blob/4ce9033e1d2f857db3f728d78399e3d7ded8ef05/go/files.go#L101-L135

### Shell

    MIME_TYPE=$(file -b --mime-type "${AUDIO_PATH}")
    NUM_BYTES=$(wc -c < "${AUDIO_PATH}")
    DISPLAY_NAME=AUDIO

    tmp_header_file=upload-header.tmp

    # Initial resumable request defining metadata.
    # The upload url is in the response headers dump them to a file.
    curl "${BASE_URL}/upload/v1beta/files?key=${GEMINI_API_KEY}" \
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
      --data-binary "@${AUDIO_PATH}" 2> /dev/null > file_info.json

    file_uri=$(jq ".file.uri" file_info.json)
    echo file_uri=$file_uri

    # Now generate content using that file
    curl "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key=$GEMINI_API_KEY" \
        -H 'Content-Type: application/json' \
        -X POST \
        -d '{
          "contents": [{
            "parts":[
              {"text": "Describe this audio clip"},
              {"file_data":{"mime_type": "audio/mp3", "file_uri": '$file_uri'}}]
            }]
           }' 2> /dev/null > response.json

    cat response.json
    echo

    jq ".candidates[].content.parts[].text" response.jsonhttps://github.com/google-gemini/deprecated-generative-ai-python/blob/7a7cc5474ddaa0255a4410e05361028a24400abd/samples/rest/files.sh#L137-L183

### Text

### Python

    from google import genai

    client = genai.Client()
    myfile = client.files.upload(file=media / "poem.txt")
    print(f"{myfile=}")

    result = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=[myfile, "\n\n", "Can you add a few more lines to this poem?"],
    )
    print(f"{result.text=}")https://github.com/google-gemini/api-examples/blob/4ce9033e1d2f857db3f728d78399e3d7ded8ef05/python/files.py#L25-L35

### Node.js

    // Make sure to include the following import:
    // import {GoogleGenAI} from '@google/genai';
    const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY });
    const myfile = await ai.files.upload({
      file: path.join(media, "poem.txt"),
    });
    console.log("Uploaded file:", myfile);

    const result = await ai.models.generateContent({
      model: "gemini-2.0-flash",
      contents: createUserContent([
        createPartFromUri(myfile.uri, myfile.mimeType),
        "\n\n",
        "Can you add a few more lines to this poem?",
      ]),
    });
    console.log("result.text=", result.text);https://github.com/google-gemini/api-examples/blob/4ce9033e1d2f857db3f728d78399e3d7ded8ef05/javascript/files.js#L36-L52

### Go

    ctx := context.Background()
    client, err := genai.NewClient(ctx, &genai.ClientConfig{
    	APIKey:  os.Getenv("GEMINI_API_KEY"),
    	Backend: genai.BackendGeminiAPI,
    })
    if err != nil {
    	log.Fatal(err)
    }

    myfile, err := client.Files.UploadFromPath(
    	ctx, 
    	filepath.Join(getMedia(), "poem.txt"), 
    	&genai.UploadFileConfig{
    		MIMEType : "text/plain",
    	},
    )
    if err != nil {
    	log.Fatal(err)
    }
    fmt.Printf("myfile=%+v\n", myfile)

    parts := []*genai.Part{
    	genai.NewPartFromURI(myfile.URI, myfile.MIMEType),
    	genai.NewPartFromText("\n\n"),
    	genai.NewPartFromText("Can you add a few more lines to this poem?"),
    }

    contents := []*genai.Content{
    	genai.NewContentFromParts(parts, genai.RoleUser),
    }

    response, err := client.Models.GenerateContent(ctx, "gemini-2.0-flash", contents, nil)
    if err != nil {
    	log.Fatal(err)
    }
    text := response.Text()
    fmt.Printf("result.text=%s\n", text)https://github.com/google-gemini/api-examples/blob/4ce9033e1d2f857db3f728d78399e3d7ded8ef05/go/files.go#L16-L52

### Shell

    MIME_TYPE=$(file -b --mime-type "${TEXT_PATH}")
    NUM_BYTES=$(wc -c < "${TEXT_PATH}")
    DISPLAY_NAME=TEXT

    tmp_header_file=upload-header.tmp

    # Initial resumable request defining metadata.
    # The upload url is in the response headers dump them to a file.
    curl "${BASE_URL}/upload/v1beta/files?key=${GEMINI_API_KEY}" \
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
      --data-binary "@${TEXT_PATH}" 2> /dev/null > file_info.json

    file_uri=$(jq ".file.uri" file_info.json)
    echo file_uri=$file_uri

    # Now generate content using that file
    curl "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key=$GEMINI_API_KEY" \
        -H 'Content-Type: application/json' \
        -X POST \
        -d '{
          "contents": [{
            "parts":[
              {"text": "Can you add a few more lines to this poem?"},
              {"file_data":{"mime_type": "text/plain", "file_uri": '$file_uri'}}]
            }]
           }' 2> /dev/null > response.json

    cat response.json
    echo

    jq ".candidates[].content.parts[].text" response.json

    name=$(jq ".file.name" file_info.json)
    # Get the file of interest to check state
    curl https://generativelanguage.googleapis.com/v1beta/files/$name > file_info.json
    # Print some information about the file you got
    name=$(jq ".file.name" file_info.json)
    echo name=$name
    file_uri=$(jq ".file.uri" file_info.json)
    echo file_uri=$file_uri

    curl --request "DELETE" https://generativelanguage.googleapis.com/v1beta/files/$name?key=$GEMINI_API_KEY
    https://github.com/google-gemini/deprecated-generative-ai-python/blob/7a7cc5474ddaa0255a4410e05361028a24400abd/samples/rest/files.sh#L16-L80

### Video

### Python

    from google import genai
    import time

    client = genai.Client()
    # Video clip (CC BY 3.0) from https://peach.blender.org/download/
    myfile = client.files.upload(file=media / "Big_Buck_Bunny.mp4")
    print(f"{myfile=}")

    # Poll until the video file is completely processed (state becomes ACTIVE).
    while not myfile.state or myfile.state.name != "ACTIVE":
        print("Processing video...")
        print("File state:", myfile.state)
        time.sleep(5)
        myfile = client.files.get(name=myfile.name)

    result = client.models.generate_content(
        model="gemini-2.0-flash", contents=[myfile, "Describe this video clip"]
    )
    print(f"{result.text=}")https://github.com/google-gemini/api-examples/blob/4ce9033e1d2f857db3f728d78399e3d7ded8ef05/python/files.py#L73-L91

### Node.js

    // Make sure to include the following import:
    // import {GoogleGenAI} from '@google/genai';
    const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY });
    let myfile = await ai.files.upload({
      file: path.join(media, "Big_Buck_Bunny.mp4"),
      config: { mimeType: "video/mp4" },
    });
    console.log("Uploaded video file:", myfile);

    // Poll until the video file is completely processed (state becomes ACTIVE).
    while (!myfile.state || myfile.state.toString() !== "ACTIVE") {
      console.log("Processing video...");
      console.log("File state: ", myfile.state);
      await sleep(5000);
      myfile = await ai.files.get({ name: myfile.name });
    }

    const result = await ai.models.generateContent({
      model: "gemini-2.0-flash",
      contents: createUserContent([
        createPartFromUri(myfile.uri, myfile.mimeType),
        "Describe this video clip",
      ]),
    });
    console.log("result.text=", result.text);https://github.com/google-gemini/api-examples/blob/4ce9033e1d2f857db3f728d78399e3d7ded8ef05/javascript/files.js#L106-L130

### Go

    ctx := context.Background()
    client, err := genai.NewClient(ctx, &genai.ClientConfig{
    	APIKey:  os.Getenv("GEMINI_API_KEY"),
    	Backend: genai.BackendGeminiAPI,
    })
    if err != nil {
    	log.Fatal(err)
    }
    myfile, err := client.Files.UploadFromPath(
    	ctx, 
    	filepath.Join(getMedia(), "Big_Buck_Bunny.mp4"), 
    	&genai.UploadFileConfig{
    		MIMEType : "video/mp4",
    	},
    )
    if err != nil {
    	log.Fatal(err)
    }
    fmt.Printf("myfile=%+v\n", myfile)

    // Poll until the video file is completely processed (state becomes ACTIVE).
    for myfile.State == genai.FileStateUnspecified || myfile.State != genai.FileStateActive {
    	fmt.Println("Processing video...")
    	fmt.Println("File state:", myfile.State)
    	time.Sleep(5 * time.Second)

    	myfile, err = client.Files.Get(ctx, myfile.Name, nil)
    	if err != nil {
    		log.Fatal(err)
    	}
    }

    parts := []*genai.Part{
    	genai.NewPartFromURI(myfile.URI, myfile.MIMEType),
    	genai.NewPartFromText("Describe this video clip"),
    }

    contents := []*genai.Content{
    	genai.NewContentFromParts(parts, genai.RoleUser),
    }

    response, err := client.Models.GenerateContent(ctx, "gemini-2.0-flash", contents, nil)
    if err != nil {
    	log.Fatal(err)
    }
    text := response.Text()
    fmt.Printf("result.text=%s\n", text)https://github.com/google-gemini/api-examples/blob/4ce9033e1d2f857db3f728d78399e3d7ded8ef05/go/files.go#L142-L188

### Shell

    MIME_TYPE=$(file -b --mime-type "${VIDEO_PATH}")
    NUM_BYTES=$(wc -c < "${VIDEO_PATH}")
    DISPLAY_NAME=VIDEO_PATH

    # Initial resumable request defining metadata.
    # The upload url is in the response headers dump them to a file.
    curl "${BASE_URL}/upload/v1beta/files?key=${GEMINI_API_KEY}" \
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
    echo file_uri=$file_uri

    state=$(jq ".file.state" file_info.json)
    echo state=$state

    # Ensure the state of the video is 'ACTIVE'
    while [[ "($state)" = *"PROCESSING"* ]];
    do
      echo "Processing video..."
      sleep 5
      # Get the file of interest to check state
      curl https://generativelanguage.googleapis.com/v1beta/files/$name > file_info.json
      state=$(jq ".file.state" file_info.json)
    done

    # Now generate content using that file
    curl "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key=$GEMINI_API_KEY" \
        -H 'Content-Type: application/json' \
        -X POST \
        -d '{
          "contents": [{
            "parts":[
              {"text": "Describe this video clip"},
              {"file_data":{"mime_type": "video/mp4", "file_uri": '$file_uri'}}]
            }]
           }' 2> /dev/null > response.json

    cat response.json
    echo

    jq ".candidates[].content.parts[].text" response.jsonhttps://github.com/google-gemini/deprecated-generative-ai-python/blob/7a7cc5474ddaa0255a4410e05361028a24400abd/samples/rest/files.sh#L187-L244

### PDF

### Python

    from google import genai

    client = genai.Client()
    sample_pdf = client.files.upload(file=media / "test.pdf")
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=["Give me a summary of this pdf file.", sample_pdf],
    )
    print(response.text)https://github.com/google-gemini/api-examples/blob/4ce9033e1d2f857db3f728d78399e3d7ded8ef05/python/files.py#L96-L104

### Go

    ctx := context.Background()
    client, err := genai.NewClient(ctx, &genai.ClientConfig{
    	APIKey:  os.Getenv("GEMINI_API_KEY"),
    	Backend: genai.BackendGeminiAPI,
    })
    if err != nil {
    	log.Fatal(err)
    }
    samplePdf, err := client.Files.UploadFromPath(
    	ctx,
    	filepath.Join(getMedia(), "test.pdf"),
    	&genai.UploadFileConfig{
    		MIMEType: "application/pdf",
    	},
    )
    if err != nil {
    	log.Fatal(err)
    }

    parts := []*genai.Part{
    	genai.NewPartFromText("Give me a summary of this pdf file."),
    	genai.NewPartFromURI(samplePdf.URI, samplePdf.MIMEType),
    }

    contents := []*genai.Content{
    	genai.NewContentFromParts(parts, genai.RoleUser),
    }

    response, err := client.Models.GenerateContent(ctx, "gemini-2.0-flash", contents, nil)
    if err != nil {
    	log.Fatal(err)
    }
    text := response.Text()
    fmt.Println(text)https://github.com/google-gemini/api-examples/blob/4ce9033e1d2f857db3f728d78399e3d7ded8ef05/go/files.go#L195-L228

### Response body

Response for `media.upload`.

If successful, the response body contains data with the following structure:
Fields `file` ``object (`https://ai.google.dev/api/files#File`)`` Metadata for the created file.

| JSON representation |
|---|
| ``` { "file": { object (`https://ai.google.dev/api/files#File`) } } ``` |

## Method: files.get

- [Endpoint](https://ai.google.dev/api/files#body.HTTP_TEMPLATE)
- [Path parameters](https://ai.google.dev/api/files#body.PATH_PARAMETERS)
- [Request body](https://ai.google.dev/api/files#body.request_body)
- [Response body](https://ai.google.dev/api/files#body.response_body)
- [Example request](https://ai.google.dev/api/files#body.codeSnippets)
  - [Basic](https://ai.google.dev/api/files#body.codeSnippets.group)

Gets the metadata for the given `File`.

### Endpoint

get `https://generativelanguage.googleapis.com/v1beta/{name=files/*}`   

### Path parameters

`name` `string` Required. The name of the `File` to get. Example: `files/abc-123` It takes the form `files/{file}`.

### Request body

The request body must be empty.

### Example request

### Python

    from google import genai

    client = genai.Client()
    myfile = client.files.upload(file=media / "poem.txt")
    file_name = myfile.name
    print(file_name)  # "files/*"

    myfile = client.files.get(name=file_name)
    print(myfile)https://github.com/google-gemini/api-examples/blob/4ce9033e1d2f857db3f728d78399e3d7ded8ef05/python/files.py#L137-L145

### Node.js

    // Make sure to include the following import:
    // import {GoogleGenAI} from '@google/genai';
    const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY });
    const myfile = await ai.files.upload({
      file: path.join(media, "poem.txt"),
    });
    const fileName = myfile.name;
    console.log(fileName);

    const fetchedFile = await ai.files.get({ name: fileName });
    console.log(fetchedFile);https://github.com/google-gemini/api-examples/blob/4ce9033e1d2f857db3f728d78399e3d7ded8ef05/javascript/files.js#L180-L190

### Go

    ctx := context.Background()
    client, err := genai.NewClient(ctx, &genai.ClientConfig{
    	APIKey:  os.Getenv("GEMINI_API_KEY"),
    	Backend: genai.BackendGeminiAPI,
    })
    if err != nil {
    	log.Fatal(err)
    }
    myfile, err := client.Files.UploadFromPath(
    	ctx,
    	filepath.Join(getMedia(), "poem.txt"), 
    	&genai.UploadFileConfig{
    		MIMEType: "text/plain",
    	},
    )
    if err != nil {
    	log.Fatal(err)
    }
    fileName := myfile.Name
    fmt.Println(fileName)
    file, err := client.Files.Get(ctx, fileName, nil)
    if err != nil {
    	log.Fatal(err)
    }
    fmt.Println(file)https://github.com/google-gemini/api-examples/blob/4ce9033e1d2f857db3f728d78399e3d7ded8ef05/go/files.go#L298-L322

### Shell

    name=$(jq ".file.name" file_info.json)
    # Get the file of interest to check state
    curl https://generativelanguage.googleapis.com/v1beta/files/$name > file_info.json
    # Print some information about the file you got
    name=$(jq ".file.name" file_info.json)
    echo name=$name
    file_uri=$(jq ".file.uri" file_info.json)
    echo file_uri=$file_urihttps://github.com/google-gemini/deprecated-generative-ai-python/blob/7a7cc5474ddaa0255a4410e05361028a24400abd/samples/rest/files.sh#L65-L73

### Response body

If successful, the response body contains an instance of `https://ai.google.dev/api/files#File`.

## Method: files.list

- [Endpoint](https://ai.google.dev/api/files#body.HTTP_TEMPLATE)
- [Query parameters](https://ai.google.dev/api/files#body.QUERY_PARAMETERS)
- [Request body](https://ai.google.dev/api/files#body.request_body)
- [Response body](https://ai.google.dev/api/files#body.response_body)
  - [JSON representation](https://ai.google.dev/api/files#body.ListFilesResponse.SCHEMA_REPRESENTATION)
- [Example request](https://ai.google.dev/api/files#body.codeSnippets)
  - [Basic](https://ai.google.dev/api/files#body.codeSnippets.group)

Lists the metadata for `File`s owned by the requesting project.

### Endpoint

get `https://generativelanguage.googleapis.com/v1beta/files`   

### Query parameters

`pageSize` `integer` Optional. Maximum number of `File`s to return per page. If unspecified, defaults to 10. Maximum `pageSize` is 100.
`pageToken` `string` Optional. A page token from a previous `files.list` call.

### Request body

The request body must be empty.

### Example request

### Python

    from google import genai

    client = genai.Client()
    print("My files:")
    for f in client.files.list():
        print("  ", f.name)https://github.com/google-gemini/api-examples/blob/4ce9033e1d2f857db3f728d78399e3d7ded8ef05/python/files.py#L127-L132

### Node.js

    // Make sure to include the following import:
    // import {GoogleGenAI} from '@google/genai';
    const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY });
    console.log("My files:");
    // Using the pager style to list files
    const pager = await ai.files.list({ config: { pageSize: 10 } });
    let page = pager.page;
    const names = [];
    while (true) {
      for (const f of page) {
        console.log("  ", f.name);
        names.push(f.name);
      }
      if (!pager.hasNextPage()) break;
      page = await pager.nextPage();
    }https://github.com/google-gemini/api-examples/blob/4ce9033e1d2f857db3f728d78399e3d7ded8ef05/javascript/files.js#L158-L173

### Go

    ctx := context.Background()
    client, err := genai.NewClient(ctx, &genai.ClientConfig{
    	APIKey:  os.Getenv("GEMINI_API_KEY"),
    	Backend: genai.BackendGeminiAPI,
    })
    if err != nil {
    	log.Fatal(err)
    }
    fmt.Println("My files:")
    page, err := client.Files.List(ctx, nil)
    if err != nil {
    	log.Fatal(err)
    }
    for _, f := range page.Items {
    	fmt.Println("  ", f.Name)
    }https://github.com/google-gemini/api-examples/blob/4ce9033e1d2f857db3f728d78399e3d7ded8ef05/go/files.go#L276-L291

### Shell

    echo "My files: "

    curl "https://generativelanguage.googleapis.com/v1beta/files?key=$GEMINI_API_KEY"https://github.com/google-gemini/deprecated-generative-ai-python/blob/7a7cc5474ddaa0255a4410e05361028a24400abd/samples/rest/files.sh#L296-L299

### Response body

Response for `files.list`.

If successful, the response body contains data with the following structure:
Fields `files[]` ``object (`https://ai.google.dev/api/files#File`)`` The list of `File`s.
`nextPageToken` `string` A token that can be sent as a `pageToken` into a subsequent `files.list` call.

| JSON representation |
|---|
| ``` { "files": [ { object (`https://ai.google.dev/api/files#File`) } ], "nextPageToken": string } ``` |

## Method: files.delete

- [Endpoint](https://ai.google.dev/api/files#body.HTTP_TEMPLATE)
- [Path parameters](https://ai.google.dev/api/files#body.PATH_PARAMETERS)
- [Request body](https://ai.google.dev/api/files#body.request_body)
- [Response body](https://ai.google.dev/api/files#body.response_body)
- [Example request](https://ai.google.dev/api/files#body.codeSnippets)
  - [Basic](https://ai.google.dev/api/files#body.codeSnippets.group)

Deletes the `File`.

### Endpoint

delete `https://generativelanguage.googleapis.com/v1beta/{name=files/*}`   

### Path parameters

`name` `string` Required. The name of the `File` to delete. Example: `files/abc-123` It takes the form `files/{file}`.

### Request body

The request body must be empty.

### Example request

### Python

    from google import genai

    client = genai.Client()
    myfile = client.files.upload(file=media / "poem.txt")

    client.files.delete(name=myfile.name)

    try:
        result = client.models.generate_content(
            model="gemini-2.0-flash", contents=[myfile, "Describe this file."]
        )
        print(result)
    except genai.errors.ClientError:
        passhttps://github.com/google-gemini/api-examples/blob/4ce9033e1d2f857db3f728d78399e3d7ded8ef05/python/files.py#L150-L163

### Node.js

    // The Gen AI SDK for TypeScript and JavaScript is in preview.
    // Some features have not been implemented.https://github.com/google-gemini/api-examples/blob/4ce9033e1d2f857db3f728d78399e3d7ded8ef05/javascript/README.md#L20-L21

### Go

    ctx := context.Background()
    client, err := genai.NewClient(ctx, &genai.ClientConfig{
    	APIKey:  os.Getenv("GEMINI_API_KEY"),
    	Backend: genai.BackendGeminiAPI,
    })
    if err != nil {
    	log.Fatal(err)
    }
    myfile, err := client.Files.UploadFromPath(
    	ctx, 
    	filepath.Join(getMedia(), "poem.txt"), 
    	&genai.UploadFileConfig{
    		MIMEType: "text/plain",
    	},
    )
    if err != nil {
    	log.Fatal(err)
    }
    // Delete the file.
    _, err = client.Files.Delete(ctx, myfile.Name, nil)
    if err != nil {
    	log.Fatal(err)
    }
    // Attempt to use the deleted file.
    parts := []*genai.Part{
    	genai.NewPartFromURI(myfile.URI, myfile.MIMEType,),
    	genai.NewPartFromText("Describe this file."),
    }

    contents := []*genai.Content{
    	genai.NewContentFromParts(parts, genai.RoleUser),
    }

    _, err = client.Models.GenerateContent(ctx, "gemini-2.0-flash", contents, nil)
    // Expect an error when using a deleted file.
    if err != nil {
    	return nil
    }
    return fmt.Errorf("expected an error when using deleted file")https://github.com/google-gemini/api-examples/blob/4ce9033e1d2f857db3f728d78399e3d7ded8ef05/go/files.go#L329-L367

### Shell

    curl --request "DELETE" https://generativelanguage.googleapis.com/v1beta/files/$name?key=$GEMINI_API_KEYhttps://github.com/google-gemini/deprecated-generative-ai-python/blob/7a7cc5474ddaa0255a4410e05361028a24400abd/samples/rest/files.sh#L77-L78

### Response body

If successful, the response body is an empty JSON object.

## Method: files.register

- [Endpoint](https://ai.google.dev/api/files#body.HTTP_TEMPLATE)
- [Request body](https://ai.google.dev/api/files#body.request_body)
  - [JSON representation](https://ai.google.dev/api/files#body.request_body.SCHEMA_REPRESENTATION)
- [Response body](https://ai.google.dev/api/files#body.response_body)
  - [JSON representation](https://ai.google.dev/api/files#body.RegisterFilesResponse.SCHEMA_REPRESENTATION)
- [Authorization scopes](https://ai.google.dev/api/files#body.aspect)

Registers a Google Cloud Storage files with FileService. The user is expected to provide Google Cloud Storage URIs and will receive a File resource for each URI in return. Note that the files are not copied, just registered with File API. If one file fails to register, the whole request fails.

### Endpoint

post `https://generativelanguage.googleapis.com/v1beta/files:register`   

### Request body

The request body contains data with the following structure:
Fields `uris[]` `string` Required. The Google Cloud Storage URIs to register. Example: `gs://bucket/object`.

### Response body

Response for `files.register`.

If successful, the response body contains data with the following structure:
Fields `files[]` ``object (`https://ai.google.dev/api/files#File`)`` The registered files to be used when calling GenerateContent.

| JSON representation |
|---|
| ``` { "files": [ { object (`https://ai.google.dev/api/files#File`) } ] } ``` |

## REST Resource: files

- [Resource: File](https://ai.google.dev/api/files#File)
  - [JSON representation](https://ai.google.dev/api/files#File.SCHEMA_REPRESENTATION)
- [VideoFileMetadata](https://ai.google.dev/api/files#VideoFileMetadata)
  - [JSON representation](https://ai.google.dev/api/files#VideoFileMetadata.SCHEMA_REPRESENTATION)
- [State](https://ai.google.dev/api/files#State)
- [Source](https://ai.google.dev/api/files#Source)
- [Methods](https://ai.google.dev/api/files#METHODS_SUMMARY)

## Resource: File

A file uploaded to the API. Next ID: 15
Fields `name` `string` Immutable. Identifier. The `File` resource name. The ID (name excluding the "files/" prefix) can contain up to 40 characters that are lowercase alphanumeric or dashes (-). The ID cannot start or end with a dash. If the name is empty on create, a unique name will be generated. Example: `files/123-456`
`displayName` `string` Optional. The human-readable display name for the `File`. The display name must be no more than 512 characters in length, including spaces. Example: "Welcome Image"
`mimeType` `string` Output only. MIME type of the file.
`sizeBytes` `string (https://developers.google.com/discovery/v1/type-format format)` Output only. Size of the file in bytes.
`createTime` ``string (`https://protobuf.dev/reference/protobuf/google.protobuf#timestamp` format)`` Output only. The timestamp of when the `File` was created.

Uses RFC 3339, where generated output will always be Z-normalized and use 0, 3, 6 or 9 fractional digits. Offsets other than "Z" are also accepted. Examples: `"2014-10-02T15:01:23Z"`, `"2014-10-02T15:01:23.045123456Z"` or `"2014-10-02T15:01:23+05:30"`.
`updateTime` ``string (`https://protobuf.dev/reference/protobuf/google.protobuf#timestamp` format)`` Output only. The timestamp of when the `File` was last updated.

Uses RFC 3339, where generated output will always be Z-normalized and use 0, 3, 6 or 9 fractional digits. Offsets other than "Z" are also accepted. Examples: `"2014-10-02T15:01:23Z"`, `"2014-10-02T15:01:23.045123456Z"` or `"2014-10-02T15:01:23+05:30"`.
`expirationTime` ``string (`https://protobuf.dev/reference/protobuf/google.protobuf#timestamp` format)`` Output only. The timestamp of when the `File` will be deleted. Only set if the `File` is scheduled to expire.

Uses RFC 3339, where generated output will always be Z-normalized and use 0, 3, 6 or 9 fractional digits. Offsets other than "Z" are also accepted. Examples: `"2014-10-02T15:01:23Z"`, `"2014-10-02T15:01:23.045123456Z"` or `"2014-10-02T15:01:23+05:30"`.
`sha256Hash` `string (https://developers.google.com/discovery/v1/type-format format)` Output only. SHA-256 hash of the uploaded bytes.

A base64-encoded string.
`uri` `string` Output only. The uri of the `File`.
`downloadUri` `string` Output only. The download uri of the `File`.
`state` ``enum (`https://ai.google.dev/api/files#State`)`` Output only. Processing state of the File.
`source` ``enum (`https://ai.google.dev/api/files#Source`)`` Source of the File.
`error` ``object (`https://ai.google.dev/api/files#v1beta.Status`)`` Output only. Error status if File processing failed.
`metadata` `Union type` Metadata for the File. `metadata` can be only one of the following: `videoMetadata` ``object (`https://ai.google.dev/api/files#VideoFileMetadata`)`` Output only. Metadata for a video.

| JSON representation |
|---|
| ``` { "name": string, "displayName": string, "mimeType": string, "sizeBytes": string, "createTime": string, "updateTime": string, "expirationTime": string, "sha256Hash": string, "uri": string, "downloadUri": string, "state": enum (`https://ai.google.dev/api/files#State`), "source": enum (`https://ai.google.dev/api/files#Source`), "error": { object (`https://ai.google.dev/api/files#v1beta.Status`) }, // metadata "videoMetadata": { object (`https://ai.google.dev/api/files#VideoFileMetadata`) } // Union type } ``` |

## VideoFileMetadata

Metadata for a video `File`.
Fields `videoDuration` ``string (`https://protobuf.dev/reference/protobuf/google.protobuf#duration` format)`` Duration of the video.

A duration in seconds with up to nine fractional digits, ending with '`s`'. Example: `"3.5s"`.

| JSON representation |
|---|
| ``` { "videoDuration": string } ``` |

## State

States for the lifecycle of a File.

| Enums ||
|---|---|
| `STATE_UNSPECIFIED` | The default value. This value is used if the state is omitted. |
| `PROCESSING` | File is being processed and cannot be used for inference yet. |
| `ACTIVE` | File is processed and available for inference. |
| `FAILED` | File failed processing. |

## Source

| Enums ||
|---|---|
| `SOURCE_UNSPECIFIED` | Used if source is not specified. |
| `UPLOADED` | Indicates the file is uploaded by the user. |
| `GENERATED` | Indicates the file is generated by Google. |
| `REGISTERED` | Indicates the file is a registered, i.e. a Google Cloud Storage file. |

## Status

- [JSON representation](https://ai.google.dev/api/files#SCHEMA_REPRESENTATION)

The `Status` type defines a logical error model that is suitable for different programming environments, including REST APIs and RPC APIs. It is used by [gRPC](https://github.com/grpc). Each `Status` message contains three pieces of data: error code, error message, and error details.

You can find out more about this error model and how to work with it in the [API Design Guide](https://cloud.google.com/apis/design/errors).
Fields `code` `integer` The status code, which should be an enum value of `google.rpc.Code`.
`message` `string` A developer-facing error message, which should be in English. Any user-facing error message should be localized and sent in the `https://ai.google.dev/api/files#FIELDS.details` field, or localized by the client.
`details[]` `object` A list of messages that carry the error details. There is a common set of message types for APIs to use.

An object containing fields of an arbitrary type. An additional field `"@type"` contains a URI identifying the type. Example: `{ "id": 1234, "@type": "types.example.com/standard/id" }`.

| JSON representation |
|---|
| ``` { "code": integer, "message": string, "details": [ { "@type": string, field1: ..., ... } ] } ``` |