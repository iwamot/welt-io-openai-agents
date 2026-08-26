"""A small AgentCore agent that Welt can drive.

Receives Welt's payload, feeds it to an OpenAI Agents SDK run, and
streams back the renderable subset of its stream — the AgentCore Runtime
SDK emits each event as SSE, which Welt (https://github.com/iwamot/welt)
renders into Slack. `start_reply` reads which envelope Welt sent (a
conversation turn, or the answers that resume an interrupted run),
decodes it, and runs the agent on the result; `renderable_events` reduces
what it streams.

The model runs on Amazon Bedrock through the OpenAI-compatible
`bedrock-mantle` endpoint, so the OpenAI client needs nothing beyond a
different base URL and a Bedrock API key — no OpenAI account is involved.

This example is a standalone deployable; Welt drives it only through the
JSON wire contract, which welt-io-openai-agents adapts in both directions.
"""

import base64
import os
from collections.abc import AsyncIterator, Mapping
from datetime import UTC, datetime
from uuid import uuid4

import boto3
from agents import (
    Agent,
    OpenAIResponsesModel,
    function_tool,
    set_tracing_disabled,
)
from bedrock_agentcore.runtime import BedrockAgentCoreApp
from openai import AsyncOpenAI

from welt_io_openai_agents import (
    InterruptedState,
    renderable_events,
    start_reply,
)

# The SDK traces to the OpenAI platform by default and asks for an OpenAI
# API key to do it; this agent runs on AWS credentials alone.
set_tracing_disabled(True)

app = BedrockAgentCoreApp()


@function_tool
def current_time() -> str:
    """
    Get the current date and time.

    Returns:
        str: The current UTC time in ISO 8601 format.
    """
    return datetime.now(UTC).isoformat()


@function_tool
def create_sample_file() -> list[dict]:
    """
    Create a small sample CSV file.

    The file rides the tool's result as a content part, which reaches the
    model — and the Slack thread, because this tool is named in
    `files_from` below.

    Returns:
        list: The outcome, the file beside it.
    """
    csv = b"fruit,count\napple,3\nbanana,5\n"
    return [
        {"type": "text", "text": "Created sample.csv."},
        {
            "type": "file",
            "filename": "sample.csv",
            "file_data": "data:text/csv;base64," + base64.b64encode(csv).decode(),
        },
    ]


def _document_name(stem: str) -> str:
    """
    Name a report apart from every other report of the run.

    One turn can publish several reports ("apple and banana, separately"),
    and the thread tells the uploads apart by name alone.

    Args:
        stem (str): The name's fixed part.

    Returns:
        str: The stem with a random tail.
    """
    return f"{stem}-{uuid4().hex[:8]}"


@function_tool(needs_approval=True)
def sample_draft_report(topic: str, draft: str) -> list[dict]:
    """
    Publish a small report on a topic.

    Draft the full report body and pass it as `draft`; a human reviews the
    draft before it is published.

    The sibling examples draft inside the tool and pause to show the
    draft. This SDK pauses before the tool starts, and the question shows
    the call's arguments — so here the model drafts, and the draft rides
    the arguments into the question. What the human approved is what
    publishes: an approved call resumes with the arguments it was shown
    with, so no memoization guards the draft the way the siblings need.

    Args:
        topic (str): The report topic.
        draft (str): The full report body, ready to publish.

    Returns:
        list: The outcome, the published draft beside it.
    """
    name = _document_name("report")
    return [
        {
            "type": "text",
            "text": f"Published the approved draft as {name}.md."
            " The publish flow is complete; nothing is left to approve.",
        },
        {
            "type": "file",
            "filename": f"{name}.md",
            "file_data": "data:text/markdown;base64,"
            + base64.b64encode(draft.encode()).decode(),
        },
    ]


@function_tool(needs_approval=True)
def sample_dangerous_action(action: str) -> str:
    """
    Pretend to run a dangerous or irreversible action the user asked for.

    Approval by declaration: `needs_approval` pauses the run before this
    body starts, and welt-io-openai-agents renders the pending approval as
    a question in the Slack thread. Nothing here knows about the approval —
    which is what lets a tool the agent did not write, from a library or
    an MCP server, be gated the same way. Nothing is actually executed.

    Args:
        action (str): The action to pretend to run.

    Returns:
        str: The outcome of the action.
    """
    return f"Ran: {action}. Completed successfully (simulated by this demo tool)."


# Bedrock's OpenAI-compatible endpoint, in the region the AWS SDK resolves
# (us-east-1 is mantle's home region, for environments that set none). To
# run against another OpenAI-compatible service instead, change base_url
# and the key it is paired with.
_REGION = boto3.Session().region_name or "us-east-1"

agent = Agent(
    name="welt-example",
    # A rejected approval reaches the model as the tool's result ("Tool
    # execution was not approved."), and models of several families read
    # right past it, reporting the action as completed. The rule exists
    # because nothing else in the conversation marks the call as unrun.
    instructions=(
        'When a tool call\'s result says its execution "was not approved",'
        " that tool did not run. Say plainly that the action was not"
        " performed — never describe it as completed, in progress, or"
        " pending."
    ),
    model=OpenAIResponsesModel(
        # Any model the account may invoke that serves
        # `/openai/v1/responses` and reads files; an empty MODEL_ID means
        # unset, like Welt's own variables.
        model=os.environ.get("MODEL_ID") or "google.gemma-4-31b",
        openai_client=AsyncOpenAI(
            base_url=f"https://bedrock-mantle.{_REGION}.api.aws/openai/v1",
            api_key=os.environ["AWS_BEARER_TOKEN_BEDROCK"],
        ),
    ),
    tools=[
        current_time,
        create_sample_file,
        sample_draft_report,
        sample_dangerous_action,
    ],
)


# The tools whose files belong in the Slack thread. A tool left out keeps
# its files to the model.
_FILES_FROM = {"create_sample_file", "sample_draft_report"}

# The states of the runs that stopped for approval, under the ids of the
# approvals they stopped on — Welt sends those ids back when the buttons
# are answered. An entry lives as long as this process: AgentCore Runtime
# gives each session its own microVM, so a resume that arrives after it
# was recycled finds nothing and raises, which Welt renders as its
# resume-failure notice.
_interrupted: dict[str, InterruptedState] = {}


def _resumed(answers: Mapping[str, object]) -> InterruptedState:
    """
    Take the state the answered approvals belong to out of the map.

    A stop's questions are answered together, so every id in one payload
    names the same run, and any answered id finds it. The whole stop
    leaves the map with it: the ids the stop raised all hold that one
    state, and identity is what tells them from the ids of another stop.
    The values the answers carry are not read here — the adapter reads
    them when it applies them to the state.

    Args:
        answers (Mapping): Welt's `interrupt_responses`, keyed by the
            approvals' ids.

    Returns:
        InterruptedState: The state that resumes the answers.

    Raises:
        RuntimeError: If no answered id is held — this process no longer
            has the run, so there is nothing to resume.
    """
    state = next((_interrupted[i] for i in answers if i in _interrupted), None)
    if state is None:
        raise RuntimeError("No interrupted run to resume in this session.")
    for approval_id in [i for i, held in _interrupted.items() if held is state]:
        del _interrupted[approval_id]
    return state


@app.entrypoint
async def invoke(payload: dict) -> AsyncIterator[dict]:
    """
    Stream a reply to the conversation or approval answers Welt sent.

    Args:
        payload (dict): Welt's invocation payload.

    Yields:
        dict: The events Welt renders.
    """
    state: InterruptedState | None = None
    if "interrupt_responses" in payload:
        # The envelope key says which of the two arrived; a turn carries
        # no answers and needs no state.
        state = _resumed(payload["interrupt_responses"])

    result, pending = start_reply(agent, payload, state=state)
    state_of_stop: InterruptedState | None = None
    async for event in renderable_events(
        result, files_from=_FILES_FROM, pending_approvals=pending
    ):
        interrupt = event.get("interrupt")
        if interrupt is not None:
            # The run stopped here, and its state is what answers these
            # questions when the buttons come back. One snapshot serves
            # the whole stop: every id points at the same object, which
            # is what lets `_resumed` release the stop as one.
            if state_of_stop is None:
                state_of_stop = result.to_state()
            _interrupted[interrupt["id"]] = state_of_stop
        yield event


if __name__ == "__main__":
    app.run()
