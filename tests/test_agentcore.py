import asyncio
import base64
from collections.abc import AsyncIterator, Callable, Iterator
from dataclasses import dataclass, field

import pytest
from agents import (
    Agent,
    RawResponsesStreamEvent,
    StreamEvent,
    ToolApprovalItem,
)
from openai.types.responses import ResponseFunctionToolCall, ResponseTextDeltaEvent

from welt_io_openai_agents import decode_messages
from welt_io_openai_agents.agentcore import (
    _agent_entrypoint,
    _checked_data,
    _checked_name,
    _drained,
    send_file,
    welt_agent,
)

PNG_BYTES = b"\x89PNG\r\n\x1a\n"
PNG_BASE64 = base64.b64encode(PNG_BYTES).decode("ascii")

agent = Agent(name="test-agent")


@pytest.fixture(autouse=True)
def empty_queue() -> Iterator[None]:
    """Start and leave every test with no files queued."""
    _drained()
    yield
    _drained()


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

    def __call__(self, run_input: object) -> FakeRun:
        self.inputs.append(run_input)
        return self.runs.pop(0)


def replies(
    entrypoint: Callable[[dict], AsyncIterator[dict]], payload: dict
) -> list[dict]:
    """Run the entrypoint on one payload and gather what it streams."""

    async def gather() -> list[dict]:
        return [event async for event in entrypoint(payload)]

    return asyncio.run(gather())


def test_welt_agent_builds_an_entrypoint() -> None:
    entrypoint = welt_agent(agent)

    assert callable(entrypoint)


def test_a_turn_streams_the_renderable_events() -> None:
    runner = ReplayRunner(FakeRun([text_delta("hi")]))

    entrypoint = _agent_entrypoint(runner)

    assert replies(entrypoint, {"messages": []}) == [{"data": "hi"}]


def test_a_turn_runs_on_the_decoded_messages() -> None:
    runner = ReplayRunner(FakeRun([]))
    messages = [{"role": "user", "content": [{"text": "hello"}]}]

    replies(_agent_entrypoint(runner), {"messages": messages})

    assert runner.inputs == [decode_messages(messages)]


class SendingRun(FakeRun):
    """A run whose stream queues a file the way a tool would."""

    def __init__(self, *, after_last_event: bool = False) -> None:
        super().__init__(events=[])
        self.after_last_event = after_last_event

    async def stream_events(self) -> AsyncIterator[StreamEvent]:
        yield text_delta("before")
        if not self.after_last_event:
            send_file("chart.png", PNG_BYTES)
            yield text_delta("after")
        else:
            send_file("chart.png", PNG_BYTES)


def test_a_file_a_tool_queued_rides_beside_the_reply() -> None:
    entrypoint = _agent_entrypoint(ReplayRunner(SendingRun()))

    assert replies(entrypoint, {"messages": []}) == [
        {"data": "before"},
        {"data": "after"},
        {"file": {"name": "chart.png", "bytes": PNG_BASE64}},
    ]


def test_a_file_queued_after_the_last_event_still_rides_the_reply() -> None:
    entrypoint = _agent_entrypoint(ReplayRunner(SendingRun(after_last_event=True)))

    assert replies(entrypoint, {"messages": []}) == [
        {"data": "before"},
        {"file": {"name": "chart.png", "bytes": PNG_BASE64}},
    ]


def test_a_failed_turns_leftover_files_stay_off_the_next_reply() -> None:
    send_file("stale.txt", b"left behind")

    entrypoint = _agent_entrypoint(ReplayRunner(FakeRun([text_delta("fresh")])))

    assert replies(entrypoint, {"messages": []}) == [{"data": "fresh"}]


def test_resume_without_an_interrupted_run_is_refused() -> None:
    entrypoint = _agent_entrypoint(ReplayRunner())

    with pytest.raises(RuntimeError, match="No interrupted run"):
        replies(entrypoint, {"interrupt_responses": {}})


def test_an_interrupted_run_resumes_on_its_answered_state() -> None:
    item = approval("call_1")
    state = FakeState(pending=[item])
    runner = ReplayRunner(
        FakeRun([], interruptions=[item], state=state),
        FakeRun([text_delta("resumed")]),
    )

    entrypoint = _agent_entrypoint(runner)
    first = replies(entrypoint, {"messages": []})
    second = replies(entrypoint, {"interrupt_responses": {"call_1": {"value": True}}})

    assert [list(event) for event in first] == [["interrupt"]]
    assert second == [{"data": "resumed"}]
    # The answer was applied to the stashed state, which the resume ran on.
    assert state.approved == [item]
    assert runner.inputs[1] is state


def test_a_rejected_approval_is_recorded_as_rejected() -> None:
    item = approval("call_1")
    state = FakeState(pending=[item])
    runner = ReplayRunner(
        FakeRun([], interruptions=[item], state=state),
        FakeRun([]),
    )

    entrypoint = _agent_entrypoint(runner)
    replies(entrypoint, {"messages": []})
    replies(entrypoint, {"interrupt_responses": {"call_1": {"value": False}}})

    assert state.rejected == [item]


def test_the_slot_empties_once_resumed() -> None:
    item = approval("call_1")
    runner = ReplayRunner(
        FakeRun([], interruptions=[item], state=FakeState(pending=[item])),
        FakeRun([text_delta("resumed")]),
    )

    entrypoint = _agent_entrypoint(runner)
    replies(entrypoint, {"messages": []})
    replies(entrypoint, {"interrupt_responses": {"call_1": {"value": True}}})

    with pytest.raises(RuntimeError, match="No interrupted run"):
        replies(entrypoint, {"interrupt_responses": {"call_1": {"value": True}}})


def test_a_resume_that_interrupts_again_can_resume_again() -> None:
    first_item = approval("call_1")
    second_item = approval("call_2")
    runner = ReplayRunner(
        FakeRun(
            [],
            interruptions=[first_item],
            state=FakeState(pending=[first_item]),
        ),
        FakeRun(
            [],
            interruptions=[second_item],
            state=FakeState(pending=[second_item]),
        ),
        FakeRun([text_delta("done")]),
    )

    entrypoint = _agent_entrypoint(runner)
    replies(entrypoint, {"messages": []})
    replies(entrypoint, {"interrupt_responses": {"call_1": {"value": True}}})
    third = replies(entrypoint, {"interrupt_responses": {"call_2": {"value": True}}})

    assert third == [{"data": "done"}]


def test_sent_file_becomes_a_file_wire_event() -> None:
    send_file("chart.png", PNG_BYTES)
    assert _drained() == [{"file": {"name": "chart.png", "bytes": PNG_BASE64}}]


# The checks below go through the private helpers, which take `object`: a
# deliberately wrong value handed to the typed public function would be a
# type error in this file, and the helpers are where the checks live.


def test_a_name_that_is_not_a_string_is_refused() -> None:
    with pytest.raises(TypeError, match="name must be a str, not int"):
        _checked_name(1)


def test_an_empty_name_is_refused() -> None:
    with pytest.raises(ValueError, match="name must not be empty"):
        _checked_name("")


def test_data_that_is_not_bytes_is_refused() -> None:
    with pytest.raises(TypeError, match="data must be bytes, not str"):
        _checked_data("not bytes")


def test_empty_data_is_refused() -> None:
    with pytest.raises(ValueError, match="data must not be empty"):
        _checked_data(b"")


def test_a_refused_file_is_not_queued() -> None:
    with pytest.raises(ValueError):
        send_file("chart.png", b"")
    assert _drained() == []
