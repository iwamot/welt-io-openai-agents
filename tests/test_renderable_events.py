import asyncio
import base64
import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

import pytest
from agents import (
    Agent,
    AgentUpdatedStreamEvent,
    MessageOutputItem,
    RawResponsesStreamEvent,
    RunItemStreamEvent,
    StreamEvent,
    ToolApprovalItem,
    ToolCallItem,
    ToolCallOutputItem,
)
from openai.types.responses import (
    ResponseFunctionShellToolCallOutput,
    ResponseFunctionToolCall,
    ResponseOutputMessage,
    ResponseOutputText,
    ResponseReasoningTextDeltaEvent,
    ResponseRefusalDeltaEvent,
    ResponseTextDeltaEvent,
)

from welt_io_openai_agents import renderable_events

agent = Agent(name="test-agent")


@dataclass
class _Run:
    """The two members `renderable_events` reads off a streamed run."""

    events: list
    interruptions: list[ToolApprovalItem] = field(default_factory=list)

    async def stream_events(self) -> AsyncIterator[StreamEvent]:
        for event in self.events:
            yield event


def rendered(
    run: _Run,
    *,
    files_from: set[str] | None = None,
    pending_approvals: list[ToolApprovalItem] | None = None,
) -> list[dict]:
    async def gather() -> list[dict]:
        return [
            event
            async for event in renderable_events(
                run, files_from=files_from, pending_approvals=pending_approvals
            )
        ]

    return asyncio.run(gather())


def encoded(raw: bytes) -> str:
    return base64.b64encode(raw).decode()


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


def tool_call(name: str = "make_file", call_id: str = "call_1") -> RunItemStreamEvent:
    raw = ResponseFunctionToolCall(
        arguments='{"kind": "csv"}',
        call_id=call_id,
        name=name,
        type="function_call",
    )
    return RunItemStreamEvent(
        item=ToolCallItem(agent=agent, raw_item=raw), name="tool_called"
    )


def tool_output(output: object = "done", call_id: str = "call_1") -> RunItemStreamEvent:
    raw = {"type": "function_call_output", "call_id": call_id, "output": output}
    return RunItemStreamEvent(
        item=ToolCallOutputItem(agent=agent, raw_item=raw, output=output),
        name="tool_output",
    )


def approval(
    name: str = "risky",
    call_id: str = "call_1",
    arguments: str = '{"action": "wipe"}',
) -> ToolApprovalItem:
    raw = ResponseFunctionToolCall(
        arguments=arguments,
        call_id=call_id,
        name=name,
        type="function_call",
    )
    return ToolApprovalItem(agent=agent, raw_item=raw)


def test_text_delta_becomes_a_data_event() -> None:
    assert rendered(_Run([text_delta("hel")])) == [{"data": "hel"}]


def test_empty_text_delta_yields_nothing() -> None:
    assert rendered(_Run([text_delta("")])) == []


def test_reasoning_delta_stays_off_the_wire() -> None:
    # gpt-oss thinks aloud before it answers; the wire has no place for
    # reasoning, so only the answer's deltas travel.
    reasoning = RawResponsesStreamEvent(
        data=ResponseReasoningTextDeltaEvent(
            content_index=0,
            delta="The user asks...",
            item_id="rs_1",
            output_index=0,
            sequence_number=0,
            type="response.reasoning_text.delta",
        )
    )

    assert rendered(_Run([reasoning])) == []


def test_refusal_delta_becomes_a_data_event() -> None:
    # A refusal is the model's reply; dropping it would end the turn with
    # nothing in the thread.
    refusal = RawResponsesStreamEvent(
        data=ResponseRefusalDeltaEvent(
            content_index=0,
            delta="I can't help with that.",
            item_id="msg_1",
            output_index=0,
            sequence_number=0,
            type="response.refusal.delta",
        )
    )

    assert rendered(_Run([refusal])) == [{"data": "I can't help with that."}]


def test_tool_call_becomes_a_current_tool_use_event() -> None:
    assert rendered(_Run([tool_call("make_file", "call_1")])) == [
        {"current_tool_use": {"name": "make_file", "toolUseId": "call_1"}}
    ]


def test_tool_call_without_id_stays_off_the_wire() -> None:
    nameless = RunItemStreamEvent(
        item=ToolCallItem(agent=agent, raw_item={"name": "make_file"}),
        name="tool_called",
    )

    assert rendered(_Run([nameless])) == []


def test_tool_output_is_slimmed_to_id_and_status() -> None:
    events = rendered(_Run([tool_call(), tool_output("a long tool log")]))

    assert events[1] == {"tool_result": {"toolUseId": "call_1", "status": "success"}}


def test_files_of_a_tool_named_in_files_from_follow_the_result() -> None:
    output = [
        {"type": "input_text", "text": "Created sample.csv."},
        {
            "type": "input_file",
            "file_data": encoded(b"a,b\n1,2\n"),
            "filename": "sample.csv",
        },
    ]

    events = rendered(
        _Run([tool_call(), tool_output(output)]), files_from={"make_file"}
    )

    assert events[2] == {
        "file": {"name": "sample.csv", "bytes": encoded(b"a,b\n1,2\n")}
    }


def test_files_of_a_tool_left_out_of_files_from_stay_off_the_wire() -> None:
    output = [{"type": "input_file", "file_data": encoded(b"x"), "filename": "x.csv"}]

    events = rendered(_Run([tool_call(), tool_output(output)]), files_from={"other"})

    assert [list(event) for event in events] == [["current_tool_use"], ["tool_result"]]


def test_files_of_an_unnamed_output_stay_off_the_wire() -> None:
    # No tool_called in this stream and no pending approvals: nothing
    # names the tool, so its files cannot be claimed by files_from.
    output = [{"type": "input_file", "file_data": encoded(b"x"), "filename": "x.csv"}]

    events = rendered(_Run([tool_output(output)]), files_from={"make_file"})

    assert [list(event) for event in events] == [["tool_result"]]


def test_pending_approvals_name_the_tools_of_a_resumed_run() -> None:
    # A resumed run streams the approved tool's output without its call —
    # that streamed before the interrupt — so the pending approvals of the
    # state being resumed carry the name.
    output = [{"type": "input_file", "file_data": encoded(b"x"), "filename": "x.csv"}]

    events = rendered(
        _Run([tool_output(output)]),
        files_from={"risky"},
        pending_approvals=[approval("risky", "call_1")],
    )

    assert events[1] == {"file": {"name": "x.csv", "bytes": encoded(b"x")}}


def test_file_data_url_names_the_file_by_its_media_type() -> None:
    output = [
        {
            "type": "input_file",
            "file_data": f"data:application/pdf;base64,{encoded(b'doc')}",
        }
    ]

    events = rendered(
        _Run([tool_call(), tool_output(output)]), files_from={"make_file"}
    )

    assert events[2] == {"file": {"name": "file.pdf", "bytes": encoded(b"doc")}}


def test_bare_file_data_without_a_name_falls_back_to_bin() -> None:
    output = [{"type": "input_file", "file_data": encoded(b"blob")}]

    events = rendered(
        _Run([tool_call(), tool_output(output)]), files_from={"make_file"}
    )

    assert events[2] == {"file": {"name": "file.bin", "bytes": encoded(b"blob")}}


def test_image_data_url_becomes_a_file_event() -> None:
    output = [
        {"type": "input_image", "image_url": f"data:image/png;base64,{encoded(b'img')}"}
    ]

    events = rendered(
        _Run([tool_call(), tool_output(output)]), files_from={"make_file"}
    )

    assert events[2] == {"file": {"name": "image.png", "bytes": encoded(b"img")}}


def test_media_subtypes_that_are_not_extensions_are_mapped() -> None:
    output = [
        {
            "type": "input_file",
            "file_data": f"data:text/markdown;base64,{encoded(b'# hi')}",
        },
        {
            "type": "input_file",
            "file_data": f"data:application/x-unknown+zip;base64,{encoded(b'z')}",
        },
    ]

    events = rendered(
        _Run([tool_call(), tool_output(output)]), files_from={"make_file"}
    )

    assert events[2]["file"]["name"] == "file.md"
    assert events[3]["file"]["name"] == "file.bin"


def test_pointer_parts_stay_off_the_wire() -> None:
    # A part pointing at its file — an http URL, a file id — carries
    # nothing to upload.
    output = [
        {"type": "input_image", "image_url": "https://example.com/x.png"},
        {"type": "input_image", "file_id": "file-123"},
        {"type": "input_file", "file_id": "file-456"},
        "not a mapping",
    ]

    events = rendered(
        _Run([tool_call(), tool_output(output)]), files_from={"make_file"}
    )

    assert [list(event) for event in events] == [["current_tool_use"], ["tool_result"]]


def test_a_file_with_no_bytes_stays_off_the_wire(
    caplog: pytest.LogCaptureFixture,
) -> None:
    output = [
        {"type": "input_file", "file_data": "", "filename": "empty.csv"},
        {"type": "input_image", "image_url": "data:image/png;base64,"},
    ]

    with caplog.at_level(logging.WARNING, logger="welt_io_openai_agents"):
        events = rendered(
            _Run([tool_call(), tool_output(output)]), files_from={"make_file"}
        )

    assert [list(event) for event in events] == [["current_tool_use"], ["tool_result"]]
    assert "empty.csv" in caplog.text
    assert "image.png" in caplog.text
    assert "make_file" in caplog.text


def test_string_tool_output_carries_no_file(
    caplog: pytest.LogCaptureFixture,
) -> None:
    events = rendered(
        _Run([tool_call(), tool_output("plain text result")]),
        files_from={"make_file"},
    )

    assert [list(event) for event in events] == [["current_tool_use"], ["tool_result"]]


def test_a_pending_approval_ends_the_stream_as_an_interrupt() -> None:
    run = _Run(
        [tool_call("risky", "call_9")], interruptions=[approval("risky", "call_9")]
    )

    events = rendered(run)

    assert events[-1] == {
        "interrupt": {
            "id": "call_9",
            "name": "risky",
            "reason": {
                "message": 'May I run `risky`?\n```\n{\n  "action": "wipe"\n}\n```',
                "approve": {},
                "reject": {},
            },
        }
    }


def test_unparseable_arguments_are_shown_as_they_came() -> None:
    run = _Run([], interruptions=[approval(arguments='{"broken"')])

    events = rendered(run)

    assert '{"broken"' in events[0]["interrupt"]["reason"]["message"]


def test_non_ascii_arguments_stay_readable_in_the_question() -> None:
    # The arguments are what the human decides on, so a Japanese value
    # must reach the thread as itself, not as \uXXXX escapes.
    interruptions = [approval("risky", "call_9", '{"action": "観葉植物に水やり"}')]

    events = rendered(_Run([], interruptions=interruptions))

    assert "観葉植物に水やり" in events[0]["interrupt"]["reason"]["message"]


def test_empty_arguments_leave_the_question_bare() -> None:
    run = _Run([], interruptions=[approval(arguments="{}")])

    events = rendered(run)

    assert events[0]["interrupt"]["reason"]["message"] == "May I run `risky`?"


def test_an_approval_without_a_call_id_is_skipped(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # dict raw items carry no call id unless one is set, and a question
    # Welt cannot resume is worse than none.
    unanswerable = ToolApprovalItem(agent=agent, raw_item={"name": "risky"})
    run = _Run([], interruptions=[unanswerable])

    with caplog.at_level(logging.WARNING, logger="welt_io_openai_agents"):
        events = rendered(run)

    assert events == []
    assert "risky" in caplog.text


def test_other_stream_events_stay_off_the_wire() -> None:
    message = ResponseOutputMessage(
        id="msg_1",
        content=[ResponseOutputText(annotations=[], text="Hello!", type="output_text")],
        role="assistant",
        status="completed",
        type="message",
    )
    events = [
        AgentUpdatedStreamEvent(new_agent=agent),
        RunItemStreamEvent(
            item=MessageOutputItem(agent=agent, raw_item=message),
            name="message_output_created",
        ),
    ]

    assert rendered(_Run(events)) == []


def test_a_dict_approval_shows_its_arguments_too() -> None:
    # Hosted and MCP approvals arrive as plain dicts; the question reads
    # the same fields off them.
    raw = {"name": "hosted", "call_id": "call_2", "arguments": '{"a": 1}'}
    run = _Run([], interruptions=[ToolApprovalItem(agent=agent, raw_item=raw)])

    events = rendered(run)

    assert events[0]["interrupt"]["id"] == "call_2"
    assert '"a": 1' in events[0]["interrupt"]["reason"]["message"]


def test_a_dict_approval_without_arguments_asks_bare() -> None:
    raw = {"name": "hosted", "call_id": "call_2"}
    run = _Run([], interruptions=[ToolApprovalItem(agent=agent, raw_item=raw)])

    events = rendered(run)

    assert events[0]["interrupt"]["reason"]["message"] == "May I run `hosted`?"


def test_a_pending_approval_without_a_call_id_names_nothing() -> None:
    output = [{"type": "input_file", "file_data": encoded(b"x"), "filename": "x.csv"}]
    nameless = ToolApprovalItem(agent=agent, raw_item={"name": "risky"})

    events = rendered(
        _Run([tool_output(output)]),
        files_from={"risky"},
        pending_approvals=[nameless],
    )

    assert [list(event) for event in events] == [["tool_result"]]


def test_an_output_that_is_no_mapping_carries_no_file() -> None:
    # Shell and hosted tool outputs are model objects, not the function
    # call output dict whose content carries files.
    shell_output = ResponseFunctionShellToolCallOutput(
        id="so_1",
        call_id="call_1",
        output=[],
        status="completed",
        type="shell_call_output",
    )
    item = ToolCallOutputItem(agent=agent, raw_item=shell_output, output="x")

    events = rendered(
        _Run([tool_call(), RunItemStreamEvent(item=item, name="tool_output")]),
        files_from={"make_file"},
    )

    assert [list(event) for event in events] == [["current_tool_use"], ["tool_result"]]


def test_a_data_url_without_base64_is_a_pointer() -> None:
    output = [{"type": "input_image", "image_url": "data:image/png,notbase64"}]

    events = rendered(
        _Run([tool_call(), tool_output(output)]), files_from={"make_file"}
    )

    assert [list(event) for event in events] == [["current_tool_use"], ["tool_result"]]
