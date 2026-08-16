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

import pytest
from agents import (
    Agent,
    Runner,
    RunState,
    function_tool,
    set_tracing_disabled,
)
from agents.testing import (
    ModelStep,
    ScriptedModel,
    assistant_message,
    function_call,
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


def scripted() -> ScriptedModel:
    """Call the gated tool on the first turn, close on the second.

    The SDK builds the stream a real backend would send around each step's
    output, so the adapter reads the same event sequence here as in a run.
    """
    return ScriptedModel(
        [
            ModelStep(
                output=[function_call("risky", {"action": "wipe"}, call_id="call_1")]
            ),
            ModelStep(output=[assistant_message("Done.", item_id="msg_1")]),
        ]
    )


def interrupted() -> tuple[Agent, list[dict], RunState]:
    """Run one turn to its approval stop.

    Returns:
        The agent, the wire events of the stopped turn, and its RunState.
    """
    ran.clear()
    agent = Agent(name="round-trip", model=scripted(), tools=[risky])

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
        return list(model.calls[-1].input), events

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
                    "approve": {},
                    "reject": {},
                },
            }
        },
    ]
    assert ran == []  # the tool waits on the answer


def test_approval_runs_the_tool_and_releases_its_files() -> None:
    agent, _, state = interrupted()

    _, events = resumed(agent, state, {"call_1": {"value": True, "source": "option"}})

    assert ran == ["wipe"]
    assert {"tool_result": {"toolUseId": "call_1", "status": "success"}} in events
    assert {"file": {"name": "out.csv", "bytes": CSV}} in events
    assert {"data": "Done."} in events


def test_rejection_keeps_the_tool_unrun() -> None:
    agent, _, state = interrupted()

    seen, _ = resumed(agent, state, {"call_1": {"value": False, "source": "option"}})

    assert ran == []
    # The model is told the call was rejected, as the call's output.
    outputs = function_call_outputs(seen)
    assert len(outputs) == 1
    assert outputs[0]["call_id"] == "call_1"


def test_a_button_this_adapter_never_built_rejects_too() -> None:
    agent, _, state = interrupted()

    seen, _ = resumed(
        agent, state, {"call_1": {"value": "something else", "source": "option"}}
    )

    assert ran == []
    outputs = function_call_outputs(seen)
    assert len(outputs) == 1


def test_an_unasked_question_cannot_be_answered() -> None:
    _, _, state = interrupted()

    with pytest.raises(ValueError, match="call_404"):
        decode_interrupt_responses(
            {"call_404": {"value": True, "source": "option"}}, state
        )
