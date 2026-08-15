# Example Agent

The example agent for [Welt](https://github.com/iwamot/welt): the smallest complete agent that exercises the wire in both directions through welt-io-openai-agents.

## Stack

| Package | Role |
|---------|------|
| [Bedrock AgentCore SDK](https://github.com/aws/bedrock-agentcore-sdk-python) | Serves the endpoint |
| [OpenAI Agents SDK](https://openai.github.io/openai-agents-python/) | Runs the model and the tools (`Runner.run_streamed`) |
| [OpenAI Python SDK](https://github.com/openai/openai-python) | Talks to Bedrock's OpenAI-compatible endpoint |
| welt-io-openai-agents | Adapts the wire to Welt |

The model runs on Amazon Bedrock through the OpenAI-compatible [`bedrock-mantle` endpoint](https://docs.aws.amazon.com/bedrock/latest/userguide/bedrock-mantle.html) — the OpenAI client gets a different base URL and a Bedrock API key, and no OpenAI account is involved. To run against another OpenAI-compatible service instead, change the `base_url` in `main.py` and the key it is paired with.

## Run Locally

The agent runs on your machine as-is — the AgentCore SDK serves the same HTTP surface locally, on port 8080, that AgentCore Runtime serves in the cloud, and [Welt's local mode](https://github.com/iwamot/welt#quick-start) invokes it there.

Generate a [Bedrock API key](https://docs.aws.amazon.com/bedrock/latest/userguide/api-keys.html) in the Amazon Bedrock console, then fetch the agent and run it with [uv](https://docs.astral.sh/uv/):

```sh
curl -O https://raw.githubusercontent.com/iwamot/welt-io-openai-agents/main/examples/agent/main.py
AWS_BEARER_TOKEN_BEDROCK="<your Bedrock API key>" \
  uv run --with bedrock-agentcore --with openai-agents \
  --with welt-io-openai-agents main.py
```

The endpoint's region comes from your AWS configuration the way boto3 resolves it (`AWS_DEFAULT_REGION` before `AWS_REGION`, then a profile), falling back to `us-east-1`. `MODEL_ID` takes any model the account may invoke on the endpoint's `/v1/models` listing; unset, the agent uses `openai.gpt-oss-120b`. The agent talks to the endpoint through the Responses API, and the endpoint serves some models through the Chat Completions API alone — such a model answers with a 400 ("does not support the '/v1/responses' API") rather than a reply.

One difference from the cloud: AgentCore Runtime gives every session its own microVM, while the local server is a single process for all sessions — the agent stashes an interrupted run in one slot, so keep approval experiments to one thread at a time.

## Deploy

Deploy with the [AgentCore CLI](https://github.com/aws/agentcore-cli), replacing the generated agent with this one:

```sh
agentcore create --name WeltExample --framework OpenAIAgents
cd WeltExample

curl -o app/WeltExample/main.py https://raw.githubusercontent.com/iwamot/welt-io-openai-agents/main/examples/agent/main.py
uv add --project app/WeltExample welt-io-openai-agents boto3

agentcore deploy
```

The CLI's OpenAIAgents template assumes the OpenAI platform; this agent points the client at Bedrock instead, so what the deployed runtime needs in its environment is `AWS_BEARER_TOKEN_BEDROCK` — a [Bedrock API key](https://docs.aws.amazon.com/bedrock/latest/userguide/api-keys.html) — rather than an OpenAI key, plus `MODEL_ID` for a model other than the default `openai.gpt-oss-120b`. Note the agent runtime ARN from the deploy output: Welt's `AGENT_ARN` points at it.

## Tools

- `current_time` — the minimal tool: plain text streaming, nothing else. Ask "what time is it?" to see tool use in the thread.
- `create_sample_file` — writes a small CSV that Welt uploads to the thread. The Bedrock endpoint takes a tool's output only as a string — the file content parts the OpenAI platform accepts are rejected — so the tool queues the file and the entrypoint yields it as a `file` event itself, beside the tool's result. The result string carries the file's exact content, that being the one channel this endpoint gives the model: without it, the model describes the upload by making one up. Ask it for a sample file.
- `sample_draft_report` — the model drafts a report and passes it as an argument, so the approval question shows the draft itself, and an approved call publishes exactly what was shown (the SDK resumes it with the arguments it was approved with). The published draft reaches the thread as a markdown file. Ask for two reports on different topics to see several questions pend and resolve in one round trip.
- `sample_dangerous_action` — a pretend dangerous action (no side effects, no extra AWS permissions) gated by `needs_approval=True`: the tool itself carries no approval code, and the run pauses before its body starts. Welt renders **Approve** / **Reject** buttons in the Slack thread, and the pressed one decides whether the tool runs. Ask "deploy to prod", then press a button. See [Welt's Interrupts doc](https://github.com/iwamot/welt/blob/main/docs/interrupts.md) for the round trip.

The adapter's `files_from` is not used here: it takes files from tool outputs, which is a shape this stack's endpoint refuses — see `create_sample_file` above. Against an endpoint that accepts file content in tool outputs, naming the tool in `files_from` replaces the queue-and-yield pattern.

## Optional: file input

The agent can also read files uploaded to Slack — disabled by default, and it needs a model with vision / file input (the default `openai.gpt-oss-120b` is text-only). To try it, point `MODEL_ID` at a model that reads images or documents and set in Welt's `.env`:

```sh
FILE_INPUT_MODALITIES=image,document
```

`video` is not supported: the Responses API has no video input, so the adapter refuses video blocks outright — see [Welt's Files doc](https://github.com/iwamot/welt/blob/main/docs/files.md) for the Welt side.
