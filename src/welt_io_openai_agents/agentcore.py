"""The AgentCore Runtime entrypoint for an OpenAI Agents run Welt drives.

`welt_agent` builds the entrypoint that `BedrockAgentCoreApp` serves, so
an agent connects to Welt without rewriting the wiring every deployable
needs: telling a conversation turn from the answers that resume an
interrupted run, decoding each envelope, keeping the interrupted run's
state until its answers arrive, and reducing the stream to the events
Welt renders. The example agent of this repository once wrote this wiring
out by hand; this module is the same wiring as a function.

The interrupted run's state waits inside the returned entrypoint, under
the runtime's own lifecycle: AgentCore Runtime serves each session from
its own microVM, so one slot is enough, and the slot lives and dies with
that microVM — resuming after it was recycled (idle timeout, 8 hours at
most) raises an error the AgentCore Runtime SDK reports as an `error`
event, which Welt renders as its resume-failure notice. The slot is
resume-only: a normal turn always runs on the messages Welt sends,
because the Slack thread is the source of truth for conversation history
and the payload already carries it whole.

`send_file` hands the Slack thread a file without handing it to the
model: a tool queues the file, and the entrypoint puts it on the wire
beside the events of the reply being streamed. On Bedrock's
OpenAI-compatible endpoint this is the one road a tool's file has — the
endpoint takes tool output only as a string — and the model never sees
what was sent either way, so a tool whose file matters to the
conversation says what it holds in its result string.
"""

import base64
from collections.abc import AsyncIterator, Callable, Collection, Sequence
from functools import partial
from typing import Protocol

from agents import Agent, Runner, StreamEvent, ToolApprovalItem

from welt_io_openai_agents import (
    decode_interrupt_responses,
    decode_messages,
    renderable_events,
)

__all__ = ["send_file", "welt_agent"]


class _ResumeState(Protocol):
    """What the entrypoint holds of an interrupted run's RunState.

    Importing the SDK's RunState to hold and hand back one value would
    bind this signature to its generics for nothing. This names the
    methods `decode_interrupt_responses` uses instead, and a RunState
    satisfies it.
    """

    def get_interruptions(self) -> Sequence[ToolApprovalItem]: ...

    def approve(self, approval_item: ToolApprovalItem) -> None: ...

    def reject(self, approval_item: ToolApprovalItem) -> None: ...


class _StreamedLike(Protocol):
    """What the entrypoint reads from the streamed run.

    A RunResultStreaming satisfies it: the members `renderable_events`
    reads, plus `to_state()`, which is where an interrupted run waits for
    its answers.
    """

    interruptions: list[ToolApprovalItem]

    def stream_events(self) -> AsyncIterator[StreamEvent]: ...

    def to_state(self) -> _ResumeState: ...


# The files queued by `send_file`, on their way to the Slack thread. One
# queue for the process, like the interrupt slot is one per entrypoint:
# AgentCore Runtime serves each session from its own microVM, so no other
# reply's files can interleave with the running one's.
_pending_files: list[dict] = []


def send_file(name: str, data: bytes) -> None:
    """
    Queue one file for the Slack thread, beside the reply being streamed.

    The file rides the wire between the events of the running reply, and
    never reaches the model. A tool that wants the model to know what the
    file holds says so in its result string — on Bedrock's
    OpenAI-compatible endpoint, which takes tool output only as a string,
    a model that never saw the content would describe the upload by
    making one up.

    A file queued by a turn that failed before draining does not ride a
    later reply: the entrypoint starts every turn with the queue empty.

    Args:
        name (str): The upload filename, extension included.
        data (bytes): The raw file bytes.

    Raises:
        TypeError: If the name or the data is of the wrong type.
        ValueError: If either is empty. Slack refuses a zero-byte upload,
            and the whole reply fails with it, so an empty file is refused
            here, where the tool that queued it is still on the stack.
    """
    name = _checked_name(name)
    data = _checked_data(data)
    _pending_files.append(
        {"file": {"name": name, "bytes": base64.b64encode(data).decode("ascii")}}
    )


def _checked_name(name: object) -> str:
    """
    Check an upload filename.

    Args:
        name (object): The name the caller passed.

    Returns:
        str: The name.

    Raises:
        TypeError: If it is not a string.
        ValueError: If it is empty.
    """
    if not isinstance(name, str):
        raise TypeError(f"name must be a str, not {type(name).__name__}")
    if not name:
        raise ValueError("name must not be empty")
    return name


def _checked_data(data: object) -> bytes:
    """
    Check a file's bytes.

    Args:
        data (object): The data the caller passed.

    Returns:
        bytes: The data.

    Raises:
        TypeError: If it is not bytes.
        ValueError: If it is empty.
    """
    if not isinstance(data, bytes):
        raise TypeError(f"data must be bytes, not {type(data).__name__}")
    if not data:
        raise ValueError("data must not be empty; Slack refuses an empty upload")
    return data


def _drained() -> list[dict]:
    """
    Take every queued file event off the queue, in order.

    Returns:
        list[dict]: The `file` events queued since the last drain.
    """
    events = _pending_files[:]
    _pending_files.clear()
    return events


def welt_agent(
    agent: Agent,
    *,
    files_from: Collection[str] | None = None,
) -> Callable[[dict], AsyncIterator[dict]]:
    """
    Build the AgentCore Runtime entrypoint for an agent Welt drives.

    The returned function is what `BedrockAgentCoreApp` takes::

        app = BedrockAgentCoreApp()
        app.entrypoint(welt_agent(agent))

    It reads which envelope Welt sent — Converse-shaped `messages` for a
    conversation turn, `interrupt_responses` for the answers that resume
    an interrupted run — runs the agent through `Runner.run_streamed`,
    and yields the events Welt renders, the files tools queued with
    `send_file` among them.

    Args:
        agent (Agent): The agent to run.
        files_from (Collection[str] | None): The names of the tools whose
            file content becomes `file` events, as `renderable_events`
            takes it. None takes files from none of them. On Bedrock's
            OpenAI-compatible endpoint a tool cannot hand the model a
            file at all, and `send_file` is the road instead.

    Returns:
        Callable[[dict], AsyncIterator[dict]]: The entrypoint. It raises
            `RuntimeError` when asked to resume a run its microVM no
            longer holds — the session was recycled while the buttons
            waited — which the AgentCore Runtime SDK reports as an
            `error` event and Welt renders as its resume-failure notice.
    """
    return _agent_entrypoint(partial(Runner.run_streamed, agent), files_from=files_from)


def _agent_entrypoint(
    run: Callable[..., _StreamedLike],
    *,
    files_from: Collection[str] | None = None,
) -> Callable[[dict], AsyncIterator[dict]]:
    """
    Build the entrypoint around one run function.

    The seam `welt_agent` closes over `Runner.run_streamed`: the run
    function takes a turn's decoded messages or an interrupted run's
    answered state — the two inputs `Runner.run_streamed` resumes from.

    Args:
        run (Callable[..., _StreamedLike]): Starts one streamed run on a
            turn's input.
        files_from (Collection[str] | None): The names of the tools whose
            file content becomes `file` events.

    Returns:
        Callable[[dict], AsyncIterator[dict]]: The entrypoint.
    """
    interrupted_state: _ResumeState | None = None

    async def entrypoint(payload: dict) -> AsyncIterator[dict]:
        """
        Stream a reply to the conversation or approval answers Welt sent.

        Args:
            payload (dict): The invocation payload, carrying one of the
                two envelopes. What Welt sends is taken as correct, so a
                payload carrying neither is Welt's bug, and the KeyError
                it raises is reported as an `error` event by the SDK.

        Yields:
            dict: The renderable subset of the run's stream, and the
                `file` events tools queued with `send_file`.

        Raises:
            RuntimeError: If there is no interrupted run to resume.
        """
        nonlocal interrupted_state
        # A failed turn's leftovers stay off this reply.
        _pending_files.clear()

        pending: list[ToolApprovalItem] = []
        if "interrupt_responses" in payload:
            state = interrupted_state
            interrupted_state = None
            if state is None:  # The microVM was recycled while the buttons waited.
                raise RuntimeError("No interrupted run to resume in this session.")
            # Read before the answers are applied: these name the tools
            # whose outputs stream without their calls on the resumed run.
            pending = list(state.get_interruptions())
            result = run(
                decode_interrupt_responses(payload["interrupt_responses"], state)
            )
        else:
            result = run(decode_messages(payload["messages"]))

        interrupted = False
        async for event in renderable_events(
            result, files_from=files_from, pending_approvals=pending
        ):
            if "interrupt" in event:
                interrupted = True
            yield event
            for file_event in _drained():
                yield file_event
        # Files a tool queued after its result's events had already
        # drained — the stream's tail — still belong to this reply.
        for file_event in _drained():
            yield file_event

        if interrupted:
            # Re-stashed on every interrupted stop, so a resume that
            # interrupts again keeps working.
            interrupted_state = result.to_state()

    return entrypoint
