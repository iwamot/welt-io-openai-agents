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

The agent keeps a run that stops for approval in an S3 bucket until its buttons are answered, so that the answer can resume it in whichever process receives it. [Create the session bucket](#create-the-session-bucket) first, and generate a [Bedrock API key](https://docs.aws.amazon.com/bedrock/latest/userguide/api-keys.html) in the Amazon Bedrock console. Then fetch the agent and run it with [uv](https://docs.astral.sh/uv/), naming the bucket in `SESSION_BUCKET`:

```sh
curl -O https://raw.githubusercontent.com/iwamot/welt-io-openai-agents/main/examples/agent/main.py
AWS_BEARER_TOKEN_BEDROCK="<your Bedrock API key>" SESSION_BUCKET='<session-bucket>' \
  uv run --with bedrock-agentcore --with openai-agents \
  --with welt-io-openai-agents --with "botocore[crt]" main.py
```

The process also needs AWS credentials the standard SDK way — environment variables, `AWS_PROFILE`, an SSO session, `aws login` (which is why `botocore[crt]` is included) — for the bucket, which is in S3.

`MODEL_ID` takes any model the account may invoke on the endpoint's `/openai/v1` path; unset, the agent uses `google.gemma-4-31b`. The agent talks to the endpoint through the Responses API, and the endpoint serves some models through the Chat Completions API alone — such a model answers with a 400 ("does not support the '/openai/v1/responses' API") rather than a reply.

`BEDROCK_REGION` names the region the endpoint is reached in — useful locally, when the model access you want is not where your credentials point. Unset, boto3 resolves one: `AWS_DEFAULT_REGION`, then the profile's own `region` (`AWS_REGION` is not read at all), falling back to `us-east-1` when nothing names one.

## Create the Session Bucket

Create a bucket for the runs that wait for approval, and let S3 remove the ones nobody answers after a week:

```sh
aws s3 mb 's3://<session-bucket>' --region '<region>'
aws s3api put-bucket-lifecycle-configuration --bucket '<session-bucket>' \
  --lifecycle-configuration '{"Rules":[{"ID":"expire-interrupted","Filter":{"Prefix":"interrupted/"},"Status":"Enabled","Expiration":{"Days":7}}]}'
```

The agent reads and writes objects under `interrupted/` only. It may also look for a run that is no longer there, after a second press or a week's silence, and S3 reports that as missing rather than forbidden only to a role that may list the bucket. Save this policy as `session-policy.json`; the deploy steps below attach it to the agent's role:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": ["s3:GetObject", "s3:PutObject", "s3:DeleteObject"],
      "Resource": "arn:aws:s3:::<session-bucket>/interrupted/*"
    },
    {
      "Effect": "Allow",
      "Action": "s3:ListBucket",
      "Resource": "arn:aws:s3:::<session-bucket>"
    }
  ]
}
```

## Deploy to AgentCore Runtime

Deploy with the [AgentCore CLI](https://github.com/aws/agentcore-cli):

```sh
agentcore create --name WeltExample
cd WeltExample
agentcore add runtime --name WeltExample --language python \
  --environment-variables '{"AWS_BEARER_TOKEN_BEDROCK": "<your Bedrock API key>", "SESSION_BUCKET": "<session-bucket>"}'

curl -o app/WeltExample/main.py https://raw.githubusercontent.com/iwamot/welt-io-openai-agents/main/examples/agent/main.py

# welt-io-openai-agents' floor; left out, uv takes the local Python's, which may be newer than AgentCore runs
uv init --bare --python 3.12 app/WeltExample
uv add --project app/WeltExample bedrock-agentcore aws-opentelemetry-distro boto3 welt-io-openai-agents

agentcore deploy
```

Then let the runtime's role use the bucket. `agentcore status --json` reports the agent runtime ARN; use the ID after `runtime/` in it as `<runtime-id>`, and its region as `<runtime-region>`:

```sh
role=$(aws bedrock-agentcore-control get-agent-runtime --region '<runtime-region>' \
  --agent-runtime-id '<runtime-id>' --query roleArn --output text)
aws iam put-role-policy --role-name "${role##*/}" \
  --policy-name welt-session-bucket --policy-document file://session-policy.json
```

The agent reaches the model through Bedrock's OpenAI-compatible endpoint, so the deployed runtime needs `AWS_BEARER_TOKEN_BEDROCK` — a [Bedrock API key](https://docs.aws.amazon.com/bedrock/latest/userguide/api-keys.html) — in its environment, plus `MODEL_ID` for a model other than the default `google.gemma-4-31b` and `BEDROCK_REGION` for an endpoint region other than the one the runtime resolves. All three go in the `--environment-variables` object, next to `SESSION_BUCKET`.

Generate that key with an expiration: it stays in the runtime's environment for as long as the runtime exists, and AWS [recommends restricting Bedrock API keys to exploration](https://docs.aws.amazon.com/bedrock/latest/userguide/api-keys-generate.html).

`agentcore status` reports the agent runtime ARN: Welt's `AGENT_ARN` points at it.

### Verify the Deployment

Display the deployed Runtime ARN:

```sh
agentcore status --json
```

Use the ID after `runtime/` in that ARN as `<runtime-id>`, and its region as `<runtime-region>`. Invoke the agent directly with AgentCore CLI v1:

```sh
agentcore runtime invoke \
  --region '<runtime-region>' \
  --id '<runtime-id>' \
  --session-id manual-check-0000000000000000000000 \
  --content-type application/json \
  --accept text/event-stream \
  --payload '{"messages":[{"role":"user","content":[{"text":"Say hello in one sentence."}]}]}' \
  --json
```

This sends the same `messages` envelope Welt uses, exercising the deployed agent and adapter before connecting Slack. Look for the reply text in the response body's SSE `data:` lines, where each JSON event's `data` field carries a text chunk. The invoke command also works outside the AgentCore project directory.

### Remove the Deployment

Take the bucket policy off the runtime's role, then take the runtime out of the project and deploy the empty project, which deletes the CloudFormation stack, `AgentCore-WeltExample-default`:

```sh
aws iam delete-role-policy --role-name "${role##*/}" --policy-name welt-session-bucket
agentcore remove runtime --name WeltExample
agentcore deploy --yes
```

Then delete the Bedrock API key in the Amazon Bedrock console.

## Deploy to AWS Lambda Web Functions

The same `main.py` also runs on [Lambda Web Functions](https://docs.aws.amazon.com/lambda/latest/dg/web-functions.html), which load the ASGI app named `app` in `main.py`. Python on Web Functions is `python3.15`, which is in public preview there: not for production workloads. Web Functions run on arm64, and `aws lambda-web deploy` needs AWS CLI 2.42 or later.

Collect the agent and its dependencies, built for that platform, in one directory:

```sh
mkdir build
curl -o build/main.py https://raw.githubusercontent.com/iwamot/welt-io-openai-agents/main/examples/agent/main.py
uv pip install --target build --python-platform aarch64-manylinux_2_34 \
  --python-version 3.15 --only-binary :all: bedrock-agentcore boto3 welt-io-openai-agents
```

Create the function's role, allowed to write logs and use the session bucket. The model is reached with the Bedrock API key, so the role needs no Bedrock permission of its own:

```sh
aws iam create-role --role-name welt-example-web --assume-role-policy-document \
  '{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"lambda.amazonaws.com"},"Action":"sts:AssumeRole"}]}'
aws iam attach-role-policy --role-name welt-example-web \
  --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole
aws iam put-role-policy --role-name welt-example-web --policy-name welt-session-bucket \
  --policy-document file://session-policy.json
```

Deploy it behind an IAM-authenticated endpoint:

```sh
aws lambda-web deploy --name welt-example --code build --runtime python \
  --execution-role-arn "arn:aws:iam::<account-id>:role/welt-example-web" \
  --env SESSION_BUCKET='<session-bucket>' \
  --env AWS_BEARER_TOKEN_BEDROCK='<your Bedrock API key>' --timeout-seconds 900 \
  --auth-type IamAuth --force-include certifi/cacert.pem \
  --force-include botocore/cacert.pem --region '<region>' --create
```

- `--force-include` keeps the CA certificate bundles that `deploy` otherwise leaves out with every `*.pem`; without them the agent cannot reach the model or S3 over TLS.
- `--timeout-seconds 900` lets a reply stream for up to 15 minutes, the most a Web Function request takes; the default is 30 seconds, and a reply that outlasts it is cut off.
- `MODEL_ID` and `BEDROCK_REGION` go in further `--env` options, as they do in the AgentCore environment.

The command prints the endpoint URL. Point Welt at it with `AGENT_URL`, and have Welt sign its requests for Lambda:

```sh
AGENT_URL=https://<endpoint-id>.<shard>.lambda-web.<region>.on.aws
AGENT_SIGV4_SERVICE=lambda
AGENT_SIGV4_REGION=<region>
```

Welt's AWS identity needs `lambda:InvokeWebFunctionEndpoint` on `arn:aws:lambda:<region>:<account-id>:web-function/welt-example/endpoint/*`. See [Invoking an Agent by URL](https://github.com/iwamot/welt/blob/main/docs/agent-url.md).

### Remove the Web Function

```sh
aws lambda-web delete-web-function --function-name welt-example --region '<region>'
aws iam delete-role-policy --role-name welt-example-web --policy-name welt-session-bucket
aws iam detach-role-policy --role-name welt-example-web \
  --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole
aws iam delete-role --role-name welt-example-web
```

Then delete the Bedrock API key in the Amazon Bedrock console.

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
