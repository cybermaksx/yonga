# OpenAI-compatible server

`yonga serve` puts a prepared model behind a local HTTP API that speaks OpenAI's chat completions format, so editors, scripts and chat front ends that work with OpenAI can use your NPU, GPU or CPU instead. It runs the same session as `yonga chat`, with the same chat profile, generation defaults and hidden reasoning. This page shows how to start it, call it with `curl` and the official `openai` Python client, and what it accepts and refuses.

## Start the server {#start}

To serve a prepared model, run:

```bash
yonga serve Qwen/Qwen3-0.6B
```

yonga loads the model before it starts listening, so a model that cannot load on the device makes `serve` itself fail instead of every request. When it is ready, it prints the address:

```text
Loading the model on NPU...
Serving Qwen/Qwen3-0.6B on NPU at http://127.0.0.1:8000/v1
Press Ctrl+C to stop.
```

Press Ctrl+C to stop it. A port that is already in use is reported before the model is loaded.

!!! note
    `serve` needs the `serve` extra (FastAPI, uvicorn and Pillow). The AppImage includes it; with pip, install `yonga[serve]`. See [Installation](../installation.md).

### Options {#options}

| Option | Default | What it does |
|---|---|---|
| `--host ADDRESS` | `127.0.0.1` | Address to bind. Loopback only unless you say otherwise. |
| `--port N` | `8000` | TCP port to listen on. |
| `--device auto\|npu\|gpu\|cpu` | as in `chat` | Device to run the model on. Without it, your configured default device, then the device the artifact was prepared for. |
| `--max-new-tokens N` | the chat profile's | Output token limit when a request sets none. |
| `--show-reasoning` | off | Return reasoning blocks instead of hiding them. |
| `--max-body-mb N` | `8` | Largest request body in MiB, 1 to 64. Embedded images count towards it. |
| `--carry-images-forward` | off | Move images sent with earlier messages onto the last one instead of refusing them (see [Images](#images)). |

`serve` does not accept `--json`; `GET /health` reports the server state as JSON instead. There is no option for an API key: it is read from the `YONGA_API_KEY` environment variable (see [Access from other machines](#api-key)).

## Endpoints {#endpoints}

| Method and path | What it returns |
|---|---|
| `GET /v1/models` | The one served model, in OpenAI's model-object shape. |
| `GET /v1/models/{id}` | The same model, if `{id}` names it. |
| `POST /v1/chat/completions` | OpenAI chat completions, streamed or not. |
| `GET /health` | `status`, `model`, `device`, `pipeline` and `uptime_seconds`. Never requires a key. |

The model answers to three names in the `model` field: the name you gave `yonga serve`, its artifact id, and its source repository. With `yonga serve Qwen/Qwen3-0.6B`, a request can name `Qwen/Qwen3-0.6B` or the artifact id; `GET /v1/models` lists the name you gave `serve` as the model `id`. Any other name gets a 404 `model_not_found`.

## Call it with curl {#curl}

```bash
curl -s http://127.0.0.1:8000/health
curl -s http://127.0.0.1:8000/v1/models
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model": "Qwen/Qwen3-0.6B", "messages": [{"role": "user", "content": "Say hello in one short sentence."}]}'
```

A real answer from the reference machine (NPU, Qwen3-0.6B):

```text
{"id":"chatcmpl-…","object":"chat.completion","created":1791738528,"model":"…",
 "choices":[{"index":0,"message":{"role":"assistant","content":"Hello! How can I assist you today?"},
 "finish_reason":"stop"}],"usage":{"prompt_tokens":19,"completion_tokens":10,"total_tokens":29}}
```

## Use the openai Python client {#python}

Any OpenAI client works when you point its base URL at the server. The client requires an API key value; when the server has no key set, any string will do.

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="unused")

reply = client.chat.completions.create(
    model="Qwen/Qwen3-0.6B",
    messages=[{"role": "user", "content": "What is an NPU? One sentence."}],
)
print(reply.choices[0].message.content)
```

The client owns the conversation: send all of it with every request. Nothing is kept on the server between requests.

## Streaming {#streaming}

Set `stream` to get the answer as server-sent events, ending in `data: [DONE]`:

=== "Python"

    ```python
    stream = client.chat.completions.create(
        model="Qwen/Qwen3-0.6B",
        messages=[{"role": "user", "content": "Name three rivers in Turkey."}],
        stream=True,
    )
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="", flush=True)
    print()
    ```

=== "curl"

    ```bash
    curl -sN http://127.0.0.1:8000/v1/chat/completions \
      -H 'Content-Type: application/json' \
      -d '{"model": "Qwen/Qwen3-0.6B", "messages": [{"role": "user", "content": "Count from 1 to 5."}], "stream": true}'
    ```

`stream_options: {"include_usage": true}` adds a final chunk with token usage. Hidden reasoning stays hidden while streaming too. A client that disconnects stops its generation.

## Generation settings {#generation}

A request may set `max_tokens` (or `max_completion_tokens`), `temperature` (`0` is greedy), `top_p`, `top_k`, `seed` and `repetition_penalty`. They are range-checked like the matching slash commands in [`yonga chat`](run.md#chat-commands) (`temperature` 0 to 2, `top_p` above 0 and at most 1, `top_k` 0 or more); a value outside its range is a 400 naming the field. Fields yonga does not know are ignored.

### Stop sequences {#stop}

`stop` takes a string or a list of up to 4 strings, each at most 256 characters, as in OpenAI's API:

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model": "Qwen/Qwen3-0.6B", "messages": [{"role": "user", "content": "List three colours, one per line."}], "stop": ["\n"]}'
```

On the reference machine this answered `"content":"1. Red"` with `"finish_reason":"stop"`. The answer ends before the first stop sequence, without it, and generation stops there, streamed or not. yonga matches stop sequences on the visible answer only, so a stop such as `"\n\n"` cannot fire inside a hidden reasoning block and leave you with an empty answer. With `--show-reasoning` the reasoning is part of the visible answer, so a stop sequence can end it there.

## Tool calling {#tools}

Tools work on the model families the registry has seen writing usable tool calls on the reference machine: **Qwen2.5** (the Instruct models), **Qwen3** (not its MoE models), **SmolLM3** and **Granite**. `yonga registry explain MODEL` shows "Tool calling" under the chat profile when a model has it. Any other model refuses `tools` with a 400 `serve.request.tools_unsupported` that says why, rather than guessing.

Send OpenAI `tools`; the answer comes back as `message.tool_calls` with `finish_reason` `tool_calls`. Send the result back in a `tool` message, exactly as with OpenAI. This example ran against Qwen3-0.6B on the NPU:

```python
import json

from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="unused")
tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Current weather for a city",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}]
messages = [{"role": "user", "content": "What is the weather in Ankara? Use the tool."}]

reply = client.chat.completions.create(model="Qwen/Qwen3-0.6B", messages=messages, tools=tools)
message = reply.choices[0].message
if not message.tool_calls:
    print(message.content)  # the model answered without calling a tool
else:
    messages.append(message)
    for call in message.tool_calls:
        args = json.loads(call.function.arguments)
        result = {"city": args.get("city"), "celsius": 21}  # run your real function here
        messages.append({"role": "tool", "tool_call_id": call.id, "content": json.dumps(result)})
    final = client.chat.completions.create(
        model="Qwen/Qwen3-0.6B", messages=messages, tools=tools
    )
    print(final.choices[0].message.content)
```

- `tool_choice` may be `auto` (the default) or `none`. `required` or a named function is refused with 400 `unsupported_parameter`: forcing a call would need constrained decoding.
- `parallel_tool_calls: false` keeps only the first call of an answer that makes several.
- Streamed, text arrives until a call could begin; each call then arrives whole, so the stream pauses while the model writes it.
- An answer that looks like a call but is not a valid one (broken JSON, a function you did not offer) comes back as plain text, never as a guessed call. A call the model only drafted while reasoning is never made.
- Up to 128 tools per request.

!!! tip
    Small models call tools unreliably, so always handle an answer without `tool_calls`, as the example does. The tool schemas also make every prompt longer, which matters on the NPU (see [NPU prompt length](#npu-prompt-length)).

## Images {#images}

A VLM artifact (Qwen2.5-VL, Qwen3.5) also takes images, as OpenAI `image_url` content parts holding a `data:` URL. An LLM artifact refuses them with `serve.request.images_unsupported`.

```bash
IMG=$(base64 -w0 photo.jpg)
curl -s http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' -d @- <<EOF
{"model": "ARTIFACT", "messages": [{"role": "user", "content": [
  {"type": "text", "text": "What is in this picture?"},
  {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,$IMG"}}]}]}
EOF
```

The rules:

- **Only `data:` URLs**, base64-encoded PNG, JPEG, WebP or GIF. yonga never fetches an image, so `http(s)://` and `file://` URLs are refused with a 400, and there is no option to allow them: a server that fetches what its clients name could be made to reach anything on its host or network.
- **At most 4 images per request**, counted over all messages; more is a 400 `serve.request.too_many_images`.
- **At most 4096 × 4096 pixels (about 16.8 million, width times height) per image.** A larger image is refused, not scaled down; scale it before sending.
- **Body size.** Images count towards the 8 MiB request limit, and base64 adds a third to their size. For larger photos, raise the limit with `--max-body-mb` (up to 64).
- **Images go with the last user message only.** OpenVINO GenAI cannot attach an image to an earlier turn, so an image in an earlier message is refused with `serve.request.images_earlier_turn`.

Chat front ends such as Open WebUI resend the earlier images of a conversation with every follow-up question. For them, start the server with `--carry-images-forward`: earlier images then move onto the newest message, with a line telling the model they were attached earlier. The model no longer knows which words each image went with, so a question about "the first picture I sent" may be answered less well. The 4-image limit still counts all of them.

## Access from other machines and API keys {#api-key}

By default the server listens on `127.0.0.1` only, so nothing outside your machine can reach it. Binding anything else (`--host 0.0.0.0`, for example) prints a warning, which also says so when no key is set. With a key set, the startup line ends in `(API key required)`.

Whenever you bind beyond loopback, set `YONGA_API_KEY` in the environment:

```bash
export YONGA_API_KEY='choose-a-long-random-string'
yonga serve Qwen/Qwen3-0.6B --host 0.0.0.0 --port 8000
```

Every endpoint except `/health` then requires `Authorization: Bearer <key>`; a request without it gets 401 `invalid_api_key`. With the `openai` client, pass the key as `api_key`. The key is read from the environment only, never from an option, so it does not show up in the process list, and it is never logged.

!!! warning
    yonga serves plain HTTP. If you expose it beyond your own machine, keep it on a network you trust, or put it behind a reverse proxy that adds TLS.

## Use it from OpenCode {#opencode}

```bash
yonga connect opencode 073d5d7c8ea75283 --port 8765
yonga serve 073d5d7c8ea75283 --port 8765
```

The first command adds the artifact to OpenCode's config (`~/.config/opencode/opencode.json`, or the file you give with `--config`) as a provider, with the context and output limits read from the artifact. It shows the change and asks first; `--yes` skips the question. Then start the server with the command it prints and choose `yonga-8765/073d5d7c8ea75283` in `opencode`. Running it again changes nothing when the entry is already right.

yonga refuses, and writes nothing, when OpenCode could not use the artifact. OpenCode sends its 12 tools with every request, and its first request is about 5300 tokens (system prompt plus the 12 tool definitions), more with skills or an `AGENTS.md`, so:

- An artifact without tool calling is refused (`connect.opencode.tools_unsupported`).
- An artifact whose prompt length is fixed under 8192 tokens is refused (`connect.opencode.prompt_too_short`). Every built-in NPU profile is fixed at 1024. The message names a prepared GPU or CPU artifact of the same model to use instead, or gives the `yonga prepare MODEL --device gpu` command.
- A config file with comments or trailing commas (JSONC) is not rewritten (`connect.opencode.config_not_json`). Add the provider by hand from the JSON it prints.

When `YONGA_API_KEY` is set, the entry refers to it as `{env:YONGA_API_KEY}`; the key itself is never written. Into a provider you already have, yonga writes only `npm`, `options.baseURL` and `options.apiKey`; the provider's name and your other options stay.

```bash
yonga connect opencode --remove 073d5d7c8ea75283
```

`--remove` deletes only the entries yonga wrote, also after `yonga remove` deleted the artifact. An entry you edited since is kept and listed, and the command exits 70. The config file is never deleted. A connection made with `--config FILE` is removed with the same `--config FILE`; without it, `--remove` names the file the connection is in, and `yonga remove` prints the full command.

## Limits and refusals {#limits}

yonga refuses what it cannot honour instead of silently answering a different request. Errors use OpenAI's shape, `{"error": {"message", "type", "param", "code"}}`, and `param` names the offending field.

| Request | Answer |
|---|---|
| `n` other than 1 | 400 `unsupported_parameter` |
| non-zero `presence_penalty` or `frequency_penalty` | 400 `unsupported_parameter` (use `repetition_penalty`) |
| `response_format` other than text (JSON mode, JSON schema) | 400 `unsupported_parameter` |
| `logprobs` | 400 `unsupported_parameter` |
| legacy `functions` / `function_call` | 400 `unsupported_parameter` (use `tools`) |
| `tool_choice` `required` or a named function | 400 `unsupported_parameter` |
| last message neither the user's nor a `tool` result | 400 `unsupported_parameter` |
| an image URL that is not an accepted `data:` URL | 400 `unsupported_parameter` |
| more than 4096 messages | 400 `invalid_request` |
| a `model` the server does not serve | 404 `model_not_found` |
| body larger than `--max-body-mb` (8 MiB by default) | 413 `request_too_large` |
| missing or wrong key, when `YONGA_API_KEY` is set | 401 `invalid_api_key` |

A runtime failure during generation is a 500 whose `code` is yonga's diagnostic code. With `stream: true`, a request refused before generation starts gets the same status and body as without it; only a failure in the middle of a stream arrives as a final `data: {"error": …}` event. The full list of codes is in the [serve spec](https://github.com/ooguz/yonga/blob/main/docs/CLI_SPEC.md#14-serve).

### NPU prompt length {#npu-prompt-length}

On the NPU, the prompt length is fixed when the model is compiled: 1024 tokens with the built-in NPU profiles. A longer prompt, counting the whole conversation, the offered tools and any images, is refused with 400 `context_length_exceeded`, streamed or not. This is the real message from the reference machine:

```text
The prompt is too long for this model: It has 3013 tokens, and this model on NPU takes at most 1024:
its prompt length (MAX_PROMPT_LEN) is fixed when it is compiled. Shorten the conversation, or use a
GPU or CPU artifact of this model, whose prompt length is not fixed.
```

Shorten the conversation, offer fewer tools, or serve a GPU or CPU artifact of the same model.

### One request at a time {#one-at-a-time}

The server holds one model on one device and answers requests one at a time. Further requests wait their turn. It is meant for one user and their tools on a local machine, not for many concurrent users.

## Next steps {#next-steps}

- Use the same model interactively: [Chat, probe, bench and eval](run.md).
- When `serve` fails to start or a request fails: [Troubleshooting](../troubleshooting.md).
- Every option: [Command reference](../reference/cli.md).
