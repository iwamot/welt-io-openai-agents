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

`MODEL_ID` takes any model the account may invoke on the endpoint's `/openai/v1` path; unset, the agent uses `google.gemma-4-31b`. The agent talks to the endpoint through the Responses API, and the endpoint serves some models through the Chat Completions API alone — such a model answers with a 400 ("does not support the '/openai/v1/responses' API") rather than a reply.

`BEDROCK_REGION` names the region the endpoint is reached in — useful locally, when the model access you want is not where your credentials point. Unset, boto3 resolves one: `AWS_DEFAULT_REGION`, then the profile's own `region` (`AWS_REGION` is not read at all), falling back to `us-east-1` when nothing names one.

One difference from the cloud: AgentCore Runtime gives every session its own microVM, while the local server is a single process for all sessions — the interrupted states this example keeps all share that one process, outlive the session that raised them, and accumulate while unanswered until the process exits.

## Deploy

Deploy with the [AgentCore CLI](https://github.com/aws/agentcore-cli):

```sh
agentcore create --name WeltExample
cd WeltExample
agentcore add runtime --name WeltExample --language python \
  --environment-variables '{"AWS_BEARER_TOKEN_BEDROCK": "<your Bedrock API key>"}'

curl -o app/WeltExample/main.py https://raw.githubusercontent.com/iwamot/welt-io-openai-agents/main/examples/agent/main.py

# welt-io-openai-agents' floor; left out, uv takes the local Python's, which may be newer than AgentCore runs
uv init --bare --python 3.12 app/WeltExample
uv add --project app/WeltExample bedrock-agentcore aws-opentelemetry-distro boto3 welt-io-openai-agents

agentcore deploy
```

The agent reaches the model through Bedrock's OpenAI-compatible endpoint, so the deployed runtime needs `AWS_BEARER_TOKEN_BEDROCK` — a [Bedrock API key](https://docs.aws.amazon.com/bedrock/latest/userguide/api-keys.html) — in its environment, plus `MODEL_ID` for a model other than the default `google.gemma-4-31b` and `BEDROCK_REGION` for an endpoint region other than the one the runtime resolves. All three go in the `--environment-variables` object.

`agentcore status` reports the agent runtime ARN: Welt's `AGENT_ARN` points at it.

To remove the deployment, take the runtime out of the project and deploy the empty project, which deletes the CloudFormation stack, `AgentCore-WeltExample-default`:

```sh
agentcore remove runtime --name WeltExample
agentcore deploy --yes
```

## Tools

- `current_time` — the minimal tool: plain text streaming, nothing else. Ask "what time is it?" to see tool use in the thread.
- `create_sample_file` — returns a small CSV as a file beside its text, which reaches the model and, because the tool is named in `files_from`, the Slack thread. Ask it for a sample file.
- `sample_draft_report` — the model drafts a report and passes it as an argument, so the approval question shows the draft itself, and an approved call publishes exactly what was shown (the SDK resumes it with the arguments it was approved with). The published draft reaches the thread as a markdown file. Ask for two reports on different topics to see several questions pend and resolve in one round trip.
- `sample_dangerous_action` — a pretend dangerous action (no side effects, no extra AWS permissions) gated by `needs_approval=True`: the tool itself carries no approval code, and the run pauses before its body starts. Welt renders **Approve** / **Reject** buttons in the Slack thread, and the pressed one decides whether the tool runs. Ask "deploy to prod", then press a button. See [Welt's Interrupts doc](https://github.com/iwamot/welt/blob/main/docs/interrupts.md) for the round trip.

## Optional: file input

The agent can also read files uploaded to Slack — disabled by default, and it needs a model with vision / file input, which the default is. To try it, set in Welt's `.env`:

```sh
FILE_INPUT_MODALITIES=image,document
```

`video` can go in that list too: a video rides in the file slot, which an endpoint that reads video types by the filename's extension. Whether one is read at all is the endpoint's and the model's answer, and a refusal arrives as the error it is — see [Welt's Files doc](https://github.com/iwamot/welt/blob/main/docs/files.md) for the Welt side.
