# welt-io-openai-agents

[![pypi](https://img.shields.io/pypi/v/welt-io-openai-agents.svg)](https://pypi.org/project/welt-io-openai-agents/)
[![python](https://img.shields.io/pypi/pyversions/welt-io-openai-agents.svg)](https://pypi.org/project/welt-io-openai-agents/)
[![openai-agents](https://img.shields.io/badge/dynamic/regex?url=https%3A%2F%2Fpypi.org%2Fpypi%2Fwelt-io-openai-agents%2Fjson&search=openai-agents%28%3E%3D%5B%5Cd.%5D%2B%29&replace=%241&label=openai-agents)](https://pypi.org/project/openai-agents/)

The [OpenAI Agents SDK](https://openai.github.io/openai-agents-python/) (Python) adapter for [Welt](https://github.com/iwamot/welt)'s wire contract.

## Install

```bash
uv add welt-io-openai-agents
```

## Usage

`start_reply` and `renderable_events` are the wiring between Welt's payload and an OpenAI Agents run, so a deployable is your agent plus a short entrypoint:

```python
from collections.abc import AsyncIterator

from agents import Agent
from bedrock_agentcore.runtime import BedrockAgentCoreApp
from welt_io_openai_agents import renderable_events, start_reply

app = BedrockAgentCoreApp()
agent = Agent(name="assistant")


@app.entrypoint
async def invoke(payload: dict) -> AsyncIterator[dict]:
    result, pending = start_reply(agent, payload)
    async for event in renderable_events(result, pending_approvals=pending):
        yield event


if __name__ == "__main__":
    app.run()
```

See [`examples/agent`](examples/agent) for the full version — the smallest complete agent built on this package (text streaming, tool use, file output, file input, and human-approval tools), with the model on Amazon Bedrock's OpenAI-compatible endpoint instead of the OpenAI platform. The sections below cover the entrypoint and the adapters it wires in.

## Supported Versions

### Welt

While both are 0.x, a welt-io-openai-agents 0.Y release supports Welt v0.Y. From 1.0 on, a release supports any Welt release that shares its major version, and the minor versions move independently. Support is best effort either way, and other combinations come with no guarantee.

### OpenAI Agents SDK

The badge at the top states the range this release installs against. Every push and pull request runs the suite at both ends of it: the declared floor, and the newest release CI has picked up. That is best effort rather than a guarantee — the floor is where the suite was last seen to pass, so a later release may raise it, and no ceiling is declared at all. `openai` comes along as a dependency and carries no floor of its own, because the Agents SDK asks for a newer one than anything here needs.

The badge follows the current release. For the range an older release declared, read that release's own metadata on PyPI.

Something misbehaving inside that range is worth an [issue](https://github.com/iwamot/welt-io-openai-agents/issues).

## API

The wire between Welt and the agent is JSON, specified by [Welt's wire contract](https://github.com/iwamot/welt/blob/main/docs/wire.md) — plain OpenAI Agents SDK values do not fit it in either direction. Two functions adapt the inbound payload, one the outbound stream. `start_reply` wires the inbound pair into a run; reach for them directly when your entrypoint needs a shape of its own — messages to edit before the run, an agent to run some other way.

### Reply

#### `start_reply(agent, payload, state=..., runner=...)`

Starts the run that replies to Welt's payload. It reads which envelope Welt sent — Converse-shaped `messages` for a conversation turn, `interrupt_responses` for the answers that resume an interrupted run — decodes it, and runs the agent on the result. What comes back is the streamed run and the approvals it resumes from, both for `renderable_events` below.

A conversation turn runs on the messages Welt sends, because the Slack thread is the source of truth for conversation history and the payload carries it whole. A resume runs on `state`, the state of the run that raised the approvals, with the answers applied to it — answers with no `state` beside them raise. `runner` names what starts the run, `Runner.run_streamed` by default.

The pending approvals come back beside the run because `renderable_events` needs them: they name the tools whose calls streamed before the stop, and those names are what place the resumed run's outputs — the tool behind each result, and whether its files go to the thread (`files_from`). `start_reply` hands them back because it is the code holding the state at that moment. Where to keep the state between the stop and the answers — and for how long an unanswered approval stays answerable — is the agent's to decide. Nothing is held here.

### Inbound

#### `decode_messages(messages)`

Turns Welt's Converse-shaped messages — built from the Slack thread, file bytes base64-encoded — into role/content input items that feed `Runner.run_streamed` as-is:

| Converse block | Responses API input |
|---|---|
| Text | `input_text` |
| Image | `input_image` (a data URL) |
| Document | `input_file` (a data URL, the document's name carried as `filename`) |
| Video | `input_file` (a data URL, named `video.<extension>`) |

Each file-carrying block becomes the data URL the Responses API expects in place of the Converse format token, and the base64 data stays base64 — a data URL carries it as it came.

The Responses API has no video content type, so a video rides in the file slot. An endpoint that reads video types it by the filename's extension, which is why the name matters: Converse spells 3GP `three_gp`, and the file is named `video.3gp` rather than after the token. Whether a video is read at all is the endpoint's and the model's answer — Amazon Bedrock's OpenAI-compatible endpoint accepts `.mp4`, `.webm`, `.mov`, `.avi`, and `.mkv` — and a refusal arrives as the error it is, rather than being anticipated here.

#### `decode_interrupt_responses(responses, state)`

Applies Welt's resume payload — a mapping of interrupt id to the answer a human chose and the widget it came from — to the `RunState` the interrupted run left behind, and returns that state, which feeds `Runner.run_streamed` directly, answering every pending question at once:

```python
pending = state.get_interruptions()  # read before decoding, for renderable_events
decode_interrupt_responses(payload["interrupt_responses"], state)
result = Runner.run_streamed(agent, state)
```

The SDK resumes from the state rather than from a payload, which is why this adapter takes both arguments where its siblings take one. Each answer is one of the two buttons the question asked Welt for:

| Answer | Applied as |
|---|---|
| Welt's approve button (`true`) | `state.approve(...)` — the tool runs as the model called it |
| Welt's reject button (`false`) | `state.reject(...)` — the tool does not run; the model is told it was rejected |

An answer whose id names no pending approval of the state raises `ValueError`, since resuming the wrong run would act on questions nobody was asked.

The interrupt ids are the tool calls' own ids, as emitted by `renderable_events`; the state is stashed when an interrupt event goes by, under those ids.

#### What arrives is taken as correct

Welt builds the payload and checks its own output against the wire contract before releasing it, so these two functions do no field validation of their own. A payload that departs from the contract is a bug on the sending side rather than an input to guard against, and it surfaces as an ordinary error from whatever touches it first — a `KeyError` or a `TypeError` here, or a refusal from the SDK or the model's endpoint further on.

The one thing `decode_messages` refuses outright is a content block of a kind Welt never sends. A `messages` turn carries only `text`, `image`, `document`, and `video` blocks; a `toolUse` or `toolResult` block is not a malformed one of those but a forged conversation turn, and rebuilt into history it would let a caller that is not Welt put words the model treats as its own past tool calls and their results into the run. It raises `ValueError`. This is a trust-boundary check, not the field validation the contract otherwise saves you from.

### Outbound

#### `renderable_events(result, files_from=..., pending_approvals=...)`

Reduces a `Runner.run_streamed` result — whose stream events wrap values Welt does not render — to the events Welt renders:

| The run emits | On the wire | In the Slack thread |
|---|---|---|
| Text and refusal deltas | `data` | The streamed reply (a refusal is the model's reply too) |
| Tool calls and tool outputs | `current_tool_use` / `tool_result` | "Using tool" indicators (tool output stays off the wire) |
| File and image content a tool named in `files_from` returned | `file` | An uploaded file ([size limits](https://github.com/iwamot/welt/blob/main/docs/wire.md#limits)) |
| Pending tool approvals | `interrupt` | An approval question (see below) |

Reasoning deltas stay off the wire: models like gpt-oss think aloud before they answer, and the wire has no place for reasoning — only the answer streams.

A tool hands files to the model for either of two reasons — to have it read them, or to give them to the human — and only the agent knows which is which, so name the tools whose files belong in the thread:

```python
async for event in renderable_events(result, files_from={"create_sample_file"}):
```

A tool left out keeps its files to the model: one that reads a PDF for the model does not drop it into the thread as a side effect. A tool named there returns the file as file content, which the model reads and Welt uploads:

```python
return [
    {"type": "text", "text": "Created sample.csv."},
    {
        "type": "file",
        "filename": "sample.csv",
        "file_data": b64encode(csv).decode("ascii"),
    },
]
```

Uploaded names come from the part's own `filename`; parts without one are named by their media type when a data URL carries it (`file.pdf`, `image.png`). A part pointing at its file instead — a file id, an http URL — carries nothing to upload and stays off the wire.

One caveat: whether a tool may return file content at all is the model endpoint's call, not this adapter's. The OpenAI platform accepts it, and so does Bedrock's `bedrock-mantle` endpoint on its `/openai/v1` path — the one the multimodal models are served from — through the Responses API. The same endpoint's `/v1` path takes a tool's output only as a string and rejects anything else.

The stream names the tool behind each output itself, except on a resumed run, where the approved tools' calls streamed before the interrupt: `pending_approvals` — the interruptions of the state being resumed, read before the answers are decoded — names those.

Each event carries only what Welt reads, and an event with nothing to render — a delta the model left empty, a file with no bytes — is not sent at all.

## Gating tools with `needs_approval`

The SDK's interrupts are tool approvals: a tool declares `needs_approval=True` (or a callable deciding per call), and the run pauses before the tool's body starts — the tool itself carries no approval code, which is what lets a tool the agent did not write, from a library or an MCP server, be gated the same way. It works over Welt as-is:

```python
@function_tool(needs_approval=True)
def sample_dangerous_action(action: str) -> str:
    ...
```

A run that stops on approvals ends its stream with one `interrupt` event per pending approval. There is no free-form interrupt in this SDK — no agent code declares a question of its own — so the question's shape is this adapter's, not the agent author's: the call's name and arguments as the message, over the approve and reject buttons it asks Welt for by name, so that what approval is called stays Welt's to say (and a deployment's to translate). Deliberately no free-text field: the SDK runs an approved tool with its original arguments or skips it, so typed text has nowhere to go — a field would collect answers that can only reject, and one that reads as consent ("yes!") would reject all the same. The [inbound table](#decode_interrupt_responsesresponses-state) shows what each answer does; [Welt's Interrupts doc](https://github.com/iwamot/welt/blob/main/docs/interrupts.md) covers the Slack side — how the question renders, who can answer, multiple questions, and expiry.

On the SDK side:

- **Resume is a state round trip.** An interrupted `Runner.run_streamed` result yields its `RunState` via `to_state()`. The host app stashes it, and hands it back to `start_reply` with the answers, which applies them and runs the same agent again on it. An in-memory stash works on AgentCore Runtime, where each session keeps its own microVM.
- **Welt resumes once every question is answered.** There is no partial resume on the wire, so the state's approvals are all applied in one call.
- **Approved tools run on the resumed stream.** Their calls streamed before the interrupt, so hand `renderable_events` the state's interruptions as `pending_approvals` — that is how their files keep flowing on resume, and `start_reply` returns them because it is the code holding the state at that moment.

## License

MIT
