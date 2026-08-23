"""A small AgentCore agent that Welt can drive.

Receives Welt's payload, feeds it to an OpenAI Agents SDK run, and
streams back the renderable subset of its stream — the AgentCore Runtime
SDK emits each event as SSE, which Welt (https://github.com/iwamot/welt)
renders into Slack. `welt_agent` is the whole connection: it reads which
envelope Welt sent (a conversation turn, or the answers that resume an
interrupted run), runs the agent, and keeps an interrupted run until its
answers arrive.

The model runs on Amazon Bedrock through the OpenAI-compatible
`bedrock-mantle` endpoint, so the OpenAI client needs nothing beyond a
different base URL and a Bedrock API key — no OpenAI account is involved.

This example is a standalone deployable; Welt drives it only through the
JSON wire contract, which welt-io-openai-agents adapts in both directions.
"""

import os
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

from welt_io_openai_agents.agentcore import send_file, welt_agent

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
def create_sample_file() -> str:
    """
    Create a small sample CSV file.

    Bedrock's OpenAI-compatible endpoint takes a tool's output only as a
    string — the file content parts the OpenAI platform accepts there are
    rejected as malformed — so a tool on this stack cannot hand its file
    to the model. `send_file` hands the thread the file directly instead.

    Returns:
        str: The outcome, the file's exact content included — the result
            string is the one channel this endpoint gives the model, and
            a model that never saw the content would describe the upload
            by making one up. The file itself goes to the Slack thread.
    """
    csv = b"fruit,count\napple,3\nbanana,5\n"
    send_file("sample.csv", csv)
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
    send_file(f"{name}.md", draft.encode())
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


app.entrypoint(welt_agent(agent))


if __name__ == "__main__":
    app.run()
