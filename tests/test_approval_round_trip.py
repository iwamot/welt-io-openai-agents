"""The adapters against the approval machinery they translate, over the
real SDK.

The unit tests either side of the wire work on items written by hand; this
drives `Runner` itself — over a scripted model, so no network is needed —
to pin what the two ends have to agree on: the approval a run stops on
becomes the question Welt renders, and the answers applied to the state
resume the run with the decision each stands for.
"""

import asyncio
import base64
from collections.abc import AsyncIterator

import pytest
from agents import (
    Agent,
    Model,
    ModelResponse,
    Runner,
    RunState,
    function_tool,
    set_tracing_disabled,
)
from openai.types.responses import (
    Response,
    ResponseCompletedEvent,
    ResponseFunctionToolCall,
    ResponseOutputMessage,
    ResponseOutputText,
    ResponseTextDeltaEvent,
)

from welt_io_openai_agents import decode_interrupt_responses, renderable_events

set_tracing_disabled(True)

CSV = base64.b64encode(b"fruit,count\napple,3\n").decode()

ran: list[str] = []


@function_tool(needs_approval=True)
def risky(action: str) -> list[dict]:
    """Do something risky.

    Args:
        action: What to do.
    """
    ran.append(action)
    return [
        {"type": "text", "text": f"did {action}"},
        {"type": "file", "file_data": CSV, "filename": "out.csv"},
    ]


def _response(output: list) -> Response:
    return Response(
        id="resp_1",
        created_at=0,
        model="scripted",
        object="response",
        output=output,
        parallel_tool_calls=False,
        tool_choice="auto",
        tools=[],
    )


class ScriptedModel(Model):
    """Calls the gated tool on its first turn, closes on the second.

    The input of every turn is kept, so a test can read what the resumed
    model was told about the tool call it never saw finish.
    """

    def __init__(self) -> None:
        self.seen_inputs: list = []

    async def get_response(self, *args: object, **kwargs: object) -> ModelResponse:
        raise NotImplementedError("the round trip streams")

    async def stream_response(
        self, *args: object, **kwargs: object
    ) -> AsyncIterator[ResponseCompletedEvent | ResponseTextDeltaEvent]:
        self.seen_inputs.append(args[1])
        if len(self.seen_inputs) == 1:
            output: list = [
                ResponseFunctionToolCall(
                    arguments='{"action": "wipe"}',
                    call_id="call_1",
                    name="risky",
                    type="function_call",
                )
            ]
        else:
            # A real backend streams the text ahead of the completed
            # response that repeats it.
            yield ResponseTextDeltaEvent(
                content_index=0,
                delta="Done.",
                item_id="msg_1",
                logprobs=[],
                output_index=0,
                sequence_number=0,
                type="response.output_text.delta",
            )
            output = [
                ResponseOutputMessage(
                    id="msg_1",
                    content=[
                        ResponseOutputText(
                            annotations=[], text="Done.", type="output_text"
                        )
                    ],
                    role="assistant",
                    status="completed",
                    type="message",
                )
            ]
        yield ResponseCompletedEvent(
            response=_response(output),
            sequence_number=0,
            type="response.completed",
        )


def interrupted() -> tuple[Agent, list[dict], RunState]:
    """Run one turn to its approval stop.

    Returns:
        The agent, the wire events of the stopped turn, and its RunState.
    """
    ran.clear()
    agent = Agent(name="round-trip", model=ScriptedModel(), tools=[risky])

    async def turn() -> tuple[list[dict], RunState]:
        result = Runner.run_streamed(agent, "please wipe")
        events = [event async for event in renderable_events(result)]
        return events, result.to_state()

    events, state = asyncio.run(turn())
    return agent, events, state


def resumed(agent: Agent, state: RunState, answers: dict) -> tuple[list, list[dict]]:
    """Answer the stop and stream the run to its end.

    Returns:
        The input items the resumed model saw, and the wire events.
    """

    async def turn() -> tuple[list, list[dict]]:
        pending = state.get_interruptions()
        result = Runner.run_streamed(agent, decode_interrupt_responses(answers, state))
        events = [
            event
            async for event in renderable_events(
                result, files_from={"risky"}, pending_approvals=pending
            )
        ]
        model = agent.model
        assert isinstance(model, ScriptedModel)
        return model.seen_inputs[-1], events

    return asyncio.run(turn())


def function_call_outputs(items: list) -> list:
    return [
        item
        for item in items
        if isinstance(item, dict) and item.get("type") == "function_call_output"
    ]


def test_the_stop_asks_one_question_per_approval() -> None:
    _, events, _ = interrupted()

    # The gated call streams as an ordinary tool-use indicator first; the
    # question that gates it ends the stream.
    assert events == [
        {"current_tool_use": {"name": "risky", "toolUseId": "call_1"}},
        {
            "interrupt": {
                "id": "call_1",
                "name": "risky",
                "reason": {
                    "message": (
                        'May I run `risky`?\n```\n{\n  "action": "wipe"\n}\n```'
                    ),
                    "options": [
                        {"value": "approve", "label": "Approve", "style": "primary"},
                        {"value": "reject", "label": "Reject", "style": "danger"},
                    ],
                    "input": {},
                },
            }
        },
    ]
    assert ran == []  # the tool waits on the answer


def test_approval_runs_the_tool_and_releases_its_files() -> None:
    agent, _, state = interrupted()

    _, events = resumed(
        agent, state, {"call_1": {"value": "approve", "source": "option"}}
    )

    assert ran == ["wipe"]
    assert {"tool_result": {"toolUseId": "call_1", "status": "success"}} in events
    assert {"file": {"name": "out.csv", "bytes": CSV}} in events
    assert {"data": "Done."} in events


def test_rejection_keeps_the_tool_unrun() -> None:
    agent, _, state = interrupted()

    seen, _ = resumed(agent, state, {"call_1": {"value": "reject", "source": "option"}})

    assert ran == []
    # The model is told the call was rejected, as the call's output.
    outputs = function_call_outputs(seen)
    assert len(outputs) == 1
    assert outputs[0]["call_id"] == "call_1"


def test_typed_text_reaches_the_model_as_the_tools_answer() -> None:
    agent, _, state = interrupted()

    seen, _ = resumed(
        agent, state, {"call_1": {"value": "not now, ask Bob", "source": "input"}}
    )

    assert ran == []
    outputs = function_call_outputs(seen)
    assert len(outputs) == 1
    assert outputs[0]["output"] == "not now, ask Bob"


def test_an_unasked_question_cannot_be_answered() -> None:
    _, _, state = interrupted()

    with pytest.raises(ValueError, match="call_404"):
        decode_interrupt_responses(
            {"call_404": {"value": "approve", "source": "option"}}, state
        )
