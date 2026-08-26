import asyncio
import base64
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

import pytest
from agents import (
    Agent,
    RawResponsesStreamEvent,
    StreamEvent,
    ToolApprovalItem,
)
from openai.types.responses import ResponseFunctionToolCall, ResponseTextDeltaEvent

from welt_io_openai_agents import decode_messages, renderable_events, start_reply

PNG_BYTES = b"\x89PNG\r\n\x1a\n"
PNG_BASE64 = base64.b64encode(PNG_BYTES).decode("ascii")

agent = Agent(name="test-agent")


def text_delta(delta: str) -> RawResponsesStreamEvent:
    return RawResponsesStreamEvent(
        data=ResponseTextDeltaEvent(
            content_index=0,
            delta=delta,
            item_id="msg_1",
            logprobs=[],
            output_index=0,
            sequence_number=0,
            type="response.output_text.delta",
        )
    )


def approval(call_id: str = "call_1") -> ToolApprovalItem:
    raw = ResponseFunctionToolCall(
        arguments='{"action": "wipe"}',
        call_id=call_id,
        name="risky",
        type="function_call",
    )
    return ToolApprovalItem(agent=agent, raw_item=raw)


@dataclass
class FakeState:
    """The three members the entrypoint and the decode use of a RunState."""

    pending: list[ToolApprovalItem] = field(default_factory=list)
    approved: list[ToolApprovalItem] = field(default_factory=list)
    rejected: list[ToolApprovalItem] = field(default_factory=list)

    def get_interruptions(self) -> list[ToolApprovalItem]:
        return self.pending

    def approve(self, approval_item: ToolApprovalItem) -> None:
        self.approved.append(approval_item)

    def reject(self, approval_item: ToolApprovalItem) -> None:
        self.rejected.append(approval_item)


@dataclass
class FakeRun:
    """The members the entrypoint reads off a streamed run."""

    events: list
    interruptions: list[ToolApprovalItem] = field(default_factory=list)
    state: FakeState = field(default_factory=FakeState)

    async def stream_events(self) -> AsyncIterator[StreamEvent]:
        for event in self.events:
            yield event

    def to_state(self) -> FakeState:
        return self.state


class ReplayRunner:
    """A run function that replays scripted runs, one per call.

    Constructed input data, not a mock: it holds the runs to hand out and
    the inputs it was started on, and verifies nothing itself.
    """

    def __init__(self, *runs: FakeRun) -> None:
        self.runs = list(runs)
        self.inputs: list = []

    def __call__(self, agent: object, run_input: object) -> FakeRun:
        self.inputs.append(run_input)
        return self.runs.pop(0)


def replies(
    runner: ReplayRunner,
    payload: dict,
    *,
    state: FakeState | None = None,
    files_from: set[str] | None = None,
) -> list[dict]:
    """Stream one reply and gather the events it renders."""
    result, pending = start_reply(agent, payload, state=state, runner=runner)

    async def gather() -> list[dict]:
        return [
            event
            async for event in renderable_events(
                result, files_from=files_from, pending_approvals=pending
            )
        ]

    return asyncio.run(gather())


def test_a_turn_streams_the_renderable_events() -> None:
    runner = ReplayRunner(FakeRun([text_delta("hi")]))

    assert replies(runner, {"messages": []}) == [{"data": "hi"}]


def test_a_turn_runs_on_the_decoded_messages() -> None:
    runner = ReplayRunner(FakeRun([]))
    messages = [{"role": "user", "content": [{"text": "hello"}]}]

    replies(runner, {"messages": messages})

    assert runner.inputs == [decode_messages(messages)]


def test_a_resume_runs_on_the_state_it_was_given() -> None:
    item = approval("call_1")
    state = FakeState(pending=[item])
    runner = ReplayRunner(FakeRun([text_delta("resumed")]))

    resumed = replies(
        runner,
        {"interrupt_responses": {"call_1": {"value": True}}},
        state=state,
    )

    assert resumed == [{"data": "resumed"}]
    assert state.approved == [item]
    assert runner.inputs[0] is state


def test_a_rejected_approval_is_recorded_as_rejected() -> None:
    item = approval("call_1")
    state = FakeState(pending=[item])
    runner = ReplayRunner(FakeRun([]))

    replies(
        runner,
        {"interrupt_responses": {"call_1": {"value": False}}},
        state=state,
    )

    assert state.rejected == [item]


def test_answers_without_a_state_to_resume_are_refused() -> None:
    runner = ReplayRunner()

    with pytest.raises(RuntimeError, match="no state to resume"):
        replies(runner, {"interrupt_responses": {"call_1": {"value": True}}})


def test_a_resume_returns_the_approvals_it_resumes_from() -> None:
    item = approval("call_1")
    state = FakeState(pending=[item])
    runner = ReplayRunner(FakeRun([]))

    _, pending = start_reply(
        agent,
        {"interrupt_responses": {"call_1": {"value": True}}},
        state=state,
        runner=runner,
    )

    # These are what renderable_events takes as pending_approvals.
    assert pending == [item]


def test_a_run_that_stops_for_approval_ends_with_its_interrupts() -> None:
    item = approval("call_1")
    runner = ReplayRunner(
        FakeRun([], interruptions=[item], state=FakeState(pending=[item]))
    )

    streamed = replies(runner, {"messages": []})

    assert [list(event) for event in streamed] == [["interrupt"]]
