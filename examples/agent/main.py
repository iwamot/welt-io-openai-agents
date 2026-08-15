"""A small AgentCore agent that Welt can drive.

Receives Welt's payload, feeds it to an OpenAI Agents SDK run, and yields
the renderable subset of its stream — the AgentCore Runtime SDK emits
each event as SSE, which Welt (https://github.com/iwamot/welt) renders
into Slack. The payload carries one of two envelopes: Converse-shaped
`messages` for a conversation turn, or `interrupt_responses` when a human
answered the approval buttons of an interrupted run.

The model runs on Amazon Bedrock through the OpenAI-compatible
`bedrock-mantle` endpoint, so the OpenAI client needs nothing beyond a
different base URL and a Bedrock API key — no OpenAI account is involved.

This example is a standalone deployable; Welt drives it only through the
JSON wire contract, which welt-io-openai-agents adapts in both directions.
"""

import os
from base64 import b64encode
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from uuid import uuid4

import boto3
from agents import (
    Agent,
    OpenAIResponsesModel,
    Runner,
    RunState,
    function_tool,
    set_tracing_disabled,
)
from bedrock_agentcore.runtime import BedrockAgentCoreApp
from openai import AsyncOpenAI

from welt_io_openai_agents import (
    decode_interrupt_responses,
    decode_messages,
    renderable_events,
)

# The SDK traces to the OpenAI platform by default and asks for an OpenAI
# API key to do it; this agent runs on AWS credentials alone.
set_tracing_disabled(True)

app = BedrockAgentCoreApp()

# Where an interrupted run waits for its answers. One slot is enough:
# AgentCore Runtime runs each session in its own microVM, so this process
# never serves two sessions. Resume only: a normal turn always runs on the
# messages Welt sends (the Slack thread is the source of truth for
# conversation history, so the state must not stand in for it). No
# persistence either — the slot lives and dies with the session's microVM
# (recycled on idle timeout, 8 hours at most).
_interrupted_state: RunState | None = None


@function_tool
def current_time() -> str:
    """
    Get the current date and time.

    Returns:
        str: The current UTC time in ISO 8601 format.
    """
    return datetime.now(UTC).isoformat()


# Files the tools made this turn, on their way to the thread. Bedrock's
# OpenAI-compatible endpoint takes a tool's output only as a string — the
# file content parts the OpenAI platform accepts there are rejected as
# malformed — so a tool on this stack cannot hand its file to the model.
# It hands the thread the file directly instead: the tool queues it here,
# and the entrypoint puts it on the wire beside the tool's own result.
_pending_files: list[dict] = []


@function_tool
def create_sample_file() -> str:
    """
    Create a small sample CSV file.

    Returns:
        str: The outcome, the file's exact content included — the result
            string is the one channel this endpoint gives the model, and
            a model that never saw the content would describe the upload
            by making one up. The file itself goes to the Slack thread.
    """
    csv = b"fruit,count\napple,3\nbanana,5\n"
    _pending_files.append(
        {"name": "sample.csv", "bytes": b64encode(csv).decode("ascii")}
    )
    return (
        "Created sample.csv and sent it to the Slack thread."
        " Its exact content is:\n" + csv.decode("ascii")
    )


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
def sample_draft_report(topic: str, draft: str) -> str:
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
        str: The outcome of the publish.
    """
    name = _document_name("report")
    _pending_files.append(
        {"name": f"{name}.md", "bytes": b64encode(draft.encode()).decode("ascii")}
    )
    return (
        f"Published the approved draft to the Slack thread as {name}.md."
        " The publish flow is complete; nothing is left to approve."
    )


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
    return f"Ran: {action}. (This example doesn't actually run anything.)"


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
        # Any model on the endpoint's /v1/models listing the account may
        # invoke; an empty MODEL_ID means unset, like Welt's own variables.
        model=os.environ.get("MODEL_ID") or "openai.gpt-oss-120b",
        openai_client=AsyncOpenAI(
            base_url=f"https://bedrock-mantle.{_REGION}.api.aws/v1",
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


@app.entrypoint
async def invoke(payload: dict) -> AsyncIterator[dict]:
    """
    Stream a reply to the conversation or approval answers Welt sent.

    Args:
        payload (dict): The invocation payload: Converse-shaped `messages`
            built by Welt from the Slack thread (file blocks
            base64-encoded), or `interrupt_responses` carrying the button
            answers that resume an interrupted run.

    Yields:
        dict: The renderable subset of the run's stream.
    """
    global _interrupted_state

    pending = []
    if "interrupt_responses" in payload:
        state = _interrupted_state
        _interrupted_state = None
        if state is None:  # The microVM was recycled while the buttons waited.
            # The SDK reports the raise as an `error` event, and Welt renders
            # its resume-failure notice.
            raise RuntimeError("No interrupted run to resume in this session.")
        # Read before the answers are applied: these name the tools whose
        # outputs stream without their calls on the resumed run.
        pending = list(state.get_interruptions())
        decode_interrupt_responses(payload["interrupt_responses"], state)
        result = Runner.run_streamed(agent, state)
    else:
        # The envelope key is the discriminator, so a payload without
        # either one is Welt's bug, and the KeyError it raises is reported
        # as an `error` event by the SDK.
        result = Runner.run_streamed(agent, decode_messages(payload["messages"]))

    interrupted = False
    # Reduce the stream to the JSON-serializable events Welt renders
    async for event in renderable_events(result, pending_approvals=pending):
        if "interrupt" in event:
            interrupted = True
        yield event
        # The files the tools queued ride the wire beside their results —
        # see _pending_files for why they do not ride the tool outputs.
        while _pending_files:
            yield {"file": _pending_files.pop(0)}

    if interrupted:
        # Re-stashed on every interrupted stop, so a resume that interrupts
        # again keeps working.
        _interrupted_state = result.to_state()


if __name__ == "__main__":
    app.run()
