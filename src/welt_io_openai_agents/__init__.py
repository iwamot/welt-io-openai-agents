"""Adapters for the two directions of Welt's wire contract.

The wire between Welt and the agent is JSON, and plain OpenAI Agents SDK
values do not fit it in either direction:

- Inbound, Welt sends Bedrock Converse-shaped messages with base64-encoded
  file bytes, while an Agents SDK run consumes Responses API input items
  whose file parts carry data URLs. `decode_messages` rebuilds each
  message accordingly. Welt resumes an interrupted run with a plain
  mapping of interrupt id to the chosen answer; the SDK resumes from the
  interrupted `RunState` instead of from a payload, so
  `decode_interrupt_responses` applies each answer to the state as the
  approval decision it stands for and returns the state for
  `Runner.run_streamed`.
- Outbound, raw `stream_events` items carry values that are not
  JSON-serializable (SDK dataclasses wrapping OpenAI model objects), which
  the AgentCore Runtime SDK would degrade to a plain string on the SSE
  wire. `renderable_events` reduces the run to the events Welt renders,
  the files of the tools the agent names among them — carried as the
  base64 their data URLs already hold. A run that stops on tool approvals
  ends with one `interrupt` event per pending approval, its reason built
  here — the SDK's interrupts are tool approvals, not free-form questions,
  so the question's shape is this adapter's to decide, not the agent
  author's.

What Welt sends is taken as correct. Welt builds the payload and checks its
own output against the wire contract before releasing it, so a payload that
departs from the contract is a bug on the sending side, not an input to
validate against runtime errors — a malformed one surfaces as an ordinary
error from whatever touches it first. The one thing `decode_messages`
does refuse is a content block of a kind Welt never sends: a `toolUse` or
`toolResult` is not a shape error but a forged conversation turn, and
rebuilt as history it would let whoever reached the runtime put words the
model treats as its own past actions into the run.

The reply stream is read as the types that define it: the SDK's stream
event and run item dataclasses, and the OpenAI delta events they wrap,
rather than whatever carries the right attribute names. Only what Welt
reads goes on the wire — an event carrying more than that costs bandwidth
for something the renderer discards.
"""

import json
import logging
from collections.abc import AsyncIterator, Collection, Mapping, Sequence
from typing import Protocol

from agents import (
    RawResponsesStreamEvent,
    RunItemStreamEvent,
    StreamEvent,
    ToolApprovalItem,
    ToolCallItem,
    ToolCallOutputItem,
)
from openai.types.responses import ResponseRefusalDeltaEvent, ResponseTextDeltaEvent

try:
    from ._version import __version__
except ImportError:
    __version__ = "0.0.0+unknown"

__all__ = [
    "decode_interrupt_responses",
    "decode_messages",
    "renderable_events",
]

logger = logging.getLogger(__name__)


# The media types the Responses API expects in a data URL, by Converse
# format token.
_IMAGE_MIME_TYPES = {
    "gif": "image/gif",
    "jpeg": "image/jpeg",
    "png": "image/png",
    "webp": "image/webp",
}

_DOCUMENT_MIME_TYPES = {
    "csv": "text/csv",
    "doc": "application/msword",
    "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "html": "text/html",
    "md": "text/markdown",
    "pdf": "application/pdf",
    "txt": "text/plain",
    "xls": "application/vnd.ms-excel",
    "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
}


def decode_messages(messages: list) -> list:
    """
    Decode Welt's messages payload into the input an Agents SDK run takes.

    A run consumes Responses API input items, whose file parts are
    `input_image` / `input_file` content carrying a data URL instead of a
    Converse format token plus base64 slot. This rebuilds each message
    accordingly — text blocks become `input_text`, image blocks
    `input_image`, and document blocks `input_file`, the document's name
    (extension included) carried as `filename`. The result feeds
    `Runner.run_streamed` as-is.

    Video blocks are refused: the Responses API has no video input shape,
    so there is nothing to rebuild one into — a silent drop would leave
    the model answering a conversation with a piece missing.

    Args:
        messages (list): The `messages` value of Welt's payload.

    Returns:
        list: Role/content input items for `Runner.run_streamed`.

    Raises:
        ValueError: If a block is of a kind Welt does not send, or a video
            block arrives.
    """
    return [
        {
            "role": message["role"],
            "content": [_decoded_block(block) for block in message["content"]],
        }
        for message in messages
    ]


# The content block kinds Welt sends. A block of any other kind — a toolUse or
# toolResult in particular — is a forged conversation turn, not something Welt
# builds, and rebuilt as history it would let a caller put words the model
# treats as its own past actions into the run. It is refused, not rebuilt.
_ALLOWED_BLOCKS = frozenset({"text", "image", "document", "video"})


def _decoded_block(block: dict) -> dict:
    """
    Decode one Converse content block into its Responses API counterpart.

    Args:
        block (dict): A Converse content block.

    Returns:
        dict: The Responses API input content.

    Raises:
        ValueError: If the block is of a kind Welt does not send, or is a
            video block — the Responses API has no video input shape.
    """
    if not _ALLOWED_BLOCKS.issuperset(block):
        raise ValueError(f"unexpected content block: {sorted(block)}")
    if "text" in block:
        return {"type": "input_text", "text": block["text"]}
    if "image" in block:
        media = block["image"]
        mime_type = _IMAGE_MIME_TYPES[media["format"]]
        return {
            "type": "input_image",
            "detail": "auto",
            "image_url": f"data:{mime_type};base64,{media['source']['bytes']}",
        }
    if "document" in block:
        media = block["document"]
        mime_type = _DOCUMENT_MIME_TYPES[media["format"]]
        # Document format tokens double as filename extensions.
        return {
            "type": "input_file",
            "filename": f"{media['name']}.{media['format']}",
            "file_data": f"data:{mime_type};base64,{media['source']['bytes']}",
        }
    raise ValueError("the Responses API has no video input")


class _InterruptedState(Protocol):
    """What `decode_interrupt_responses` uses of the interrupted RunState.

    Importing the SDK's RunState to call three methods on it would bind
    this signature to its generics for nothing. This names the methods
    instead, and a RunState satisfies it.
    """

    def get_interruptions(self) -> Sequence[ToolApprovalItem]: ...

    def approve(self, approval_item: ToolApprovalItem) -> None: ...

    def reject(self, approval_item: ToolApprovalItem) -> None: ...


def decode_interrupt_responses[StateT: _InterruptedState](
    responses: dict, state: StateT
) -> StateT:
    """
    Apply Welt's interrupt answers to the interrupted run's state.

    Welt resumes an interrupted run with a payload mapping each interrupt
    id to the answer a human chose in the thread and the widget it came
    from. The SDK resumes from the `RunState` the interrupted run left
    behind, answers recorded on it — so this applies each answer to its
    pending approval and returns the state, which feeds
    `Runner.run_streamed` directly, answering every pending question at
    once.

    The question asks Welt for its own approve and reject buttons, which
    answer with `True` and `False`. A value carrying neither came from no
    question this adapter built, and rejecting is the direction that does
    not act on an answer nobody can read.

    Args:
        responses (dict): The `interrupt_responses` value of Welt's
            payload.
        state (StateT): The `RunState` the interrupted run left behind
            (`result.to_state()`).

    Returns:
        StateT: The state passed in, every answer applied.

    Raises:
        ValueError: If an answer names no pending approval of this state —
            resuming the wrong run acts on questions nobody was asked.
    """
    pending = {item.call_id: item for item in state.get_interruptions() if item.call_id}
    for interrupt_id, answer in responses.items():
        item = pending.get(interrupt_id)
        if item is None:
            raise ValueError(f"no pending approval for interrupt id: {interrupt_id}")
        if answer["value"] is True:
            state.approve(item)
        else:
            state.reject(item)
    return state


class _StreamedRun(Protocol):
    """What `renderable_events` reads from the streamed run.

    Importing the SDK's RunResultStreaming to read one attribute and one
    method off it would say what two lines of code already say. This names
    them instead, and a RunResultStreaming satisfies it.
    """

    interruptions: list[ToolApprovalItem]

    def stream_events(self) -> AsyncIterator[StreamEvent]: ...


async def renderable_events(
    run: _StreamedRun,
    *,
    files_from: Collection[str] | None = None,
    pending_approvals: Sequence[ToolApprovalItem] | None = None,
) -> AsyncIterator[dict]:
    """
    Reduce a streamed run to the events Welt renders.

    Iterates `Runner.run_streamed`'s `stream_events()` and yields the
    wire's renderable subset: text and refusal deltas (`data` — a refusal
    is the model's reply too), tool-use indicators (`current_tool_use` /
    `tool_result`, slimmed so tool output stays off the wire), and files
    (`file` — the file and image content a tool named in `files_from`
    returned). Reasoning deltas and everything else are dropped. A run
    that stops on tool approvals ends with one `interrupt` event per
    pending approval, read from the run after its stream closes — the
    reason renders in Slack as the call's name and arguments over the
    approve and reject buttons Welt words itself.

    Which of the agent's files belong in the reply is the agent's call, so
    a tool's files become `file` events only when the tool is named in
    `files_from` — a tool that hands the model a file to read stays off
    the wire unless it is listed. The stream names the tool behind each
    output itself, except for the approved tools of a resumed run, whose
    calls streamed before the interrupt: `pending_approvals` — the
    interruptions of the state being resumed — names those.

    Each event carries only what Welt reads, and an event with nothing to
    render — a delta the model left empty, a file with no bytes — is not
    sent at all.

    Args:
        run (_StreamedRun): The `RunResultStreaming` of
            `Runner.run_streamed`.
        files_from (Collection[str] | None): The names of the tools whose
            files become `file` events. None takes files from none of
            them.
        pending_approvals (Sequence[ToolApprovalItem] | None): On resume,
            the interruptions of the state being resumed
            (`state.get_interruptions()`, read before decoding the
            answers). None for a fresh conversation turn.

    Yields:
        dict: The renderable wire events, in stream order.
    """
    names_by_call: dict[str, str] = {}
    for approval in pending_approvals or ():
        if approval.call_id and approval.tool_name:
            names_by_call[approval.call_id] = approval.tool_name
    async for event in run.stream_events():
        if isinstance(event, RawResponsesStreamEvent):
            if (
                isinstance(
                    event.data, (ResponseTextDeltaEvent, ResponseRefusalDeltaEvent)
                )
                and event.data.delta
            ):
                yield {"data": event.data.delta}
        elif isinstance(event, RunItemStreamEvent):
            item = event.item
            if isinstance(item, ToolCallItem):
                if item.call_id and item.tool_name:
                    names_by_call[item.call_id] = item.tool_name
                    yield {
                        "current_tool_use": {
                            "name": item.tool_name,
                            "toolUseId": item.call_id,
                        }
                    }
            elif isinstance(item, ToolCallOutputItem) and item.call_id:
                # Always "success": the SDK folds a failed tool into the
                # text it sends the model, and an exception that escapes
                # that ends the run as an `error` event instead of
                # streaming a result.
                yield {"tool_result": {"toolUseId": item.call_id, "status": "success"}}
                name = names_by_call.get(item.call_id)
                if files_from and name is not None and name in files_from:
                    for file_event in _file_events(item.raw_item, name):
                        yield file_event
    for approval in run.interruptions:
        if not approval.call_id:
            # An approval that nothing can name cannot be answered, and a
            # question Welt cannot resume is worse than none.
            logger.warning(
                "Skipped an approval without a call id: %s", approval.tool_name
            )
            continue
        yield {
            "interrupt": {
                "id": approval.call_id,
                "name": approval.tool_name or "",
                "reason": _approval_reason(approval),
            }
        }


def _approval_reason(approval: ToolApprovalItem) -> dict:
    """
    Build the reason that asks a human to decide on one tool approval.

    The SDK's interrupts are tool approvals — no agent code declares a
    question of its own — so the question's shape is fixed here: the
    call's name and arguments as the message, and the two decisions the
    state resumes from asked of Welt by name, so that what approval is
    called stays Welt's to say. Deliberately no free-text field: the SDK
    runs an approved tool with its original arguments or skips it, so
    typed text has nowhere to go — a field would collect answers that can
    only reject, and one that reads as consent ("yes!") would reject all
    the same.

    Args:
        approval (ToolApprovalItem): The pending approval.

    Returns:
        dict: The structured reason Welt renders as those widgets.
    """
    name = approval.tool_name
    message = f"May I run `{name}`?" if name else "May I run this tool?"
    arguments = _formatted_arguments(approval.raw_item)
    if arguments:
        message = f"{message}\n```\n{arguments}\n```"
    return {"message": message, "approve": {}, "reject": {}}


def _formatted_arguments(raw_item: object) -> str:
    """
    Format a tool call's arguments for the approval question's body.

    Args:
        raw_item (object): The approval's raw tool call, whose
            `arguments` — when it has any — carry the model's JSON.

    Returns:
        str: The arguments, pretty-printed when they parse and as they
            came when they do not — the model wrote them, so a human
            deciding on the call sees them either way. Empty when there is
            nothing to show.
    """
    if isinstance(raw_item, Mapping):
        arguments = raw_item.get("arguments")
    else:
        arguments = getattr(raw_item, "arguments", None)
    if not isinstance(arguments, str) or not arguments:
        return ""
    try:
        parsed = json.loads(arguments)
    except json.JSONDecodeError:
        return arguments
    if not parsed:
        return ""
    return json.dumps(parsed, indent=2)


# Media subtypes double as filename extensions, except these.
_EXTENSION_BY_SUBTYPE = {
    "3gpp": "3gp",
    "markdown": "md",
    "plain": "txt",
    "quicktime": "mov",
    "x-matroska": "mkv",
}


def _file_events(raw_item: object, origin: str) -> list[dict]:
    """
    Build `file` events from a tool output's file-carrying content.

    The raw output holds the tool's return converted to Responses input
    content: `input_file` parts carry base64 in `file_data` (bare or as a
    data URL) and their name in `filename`; `input_image` parts carry a
    data URL in `image_url`. A part pointing at its file instead — a file
    id, an http URL — carries nothing to upload.

    Args:
        raw_item (object): The tool output's raw item.
        origin (str): The tool that produced it, for the log line an
            empty file leaves behind.

    Returns:
        list[dict]: One `file` event per file-carrying part, in content
            order.
    """
    if not isinstance(raw_item, Mapping):
        return []
    output = raw_item.get("output")
    if not isinstance(output, list):
        # A plain-text tool result carries no file.
        return []
    events = []
    for part in output:
        if not isinstance(part, Mapping):
            continue
        if part.get("type") == "input_file":
            event = _file_event_from_file(part, origin)
        elif part.get("type") == "input_image":
            event = _file_event_from_image(part, origin)
        else:
            event = None
        if event is not None:
            events.append(event)
    return events


def _file_event_from_file(part: Mapping, origin: str) -> dict | None:
    """
    Build a `file` event from an `input_file` part.

    Args:
        part (Mapping): The part, whose `file_data` carries the base64 —
            bare, or inside a data URL — and whose `filename` names the
            upload.
        origin (str): The tool that produced it.

    Returns:
        dict | None: The `file` event, or None for a part without bytes.
    """
    file_data = part.get("file_data")
    if not isinstance(file_data, str):
        return None
    mime_type = None
    data_url = _data_url_parts(file_data)
    if data_url is not None:
        mime_type, file_data = data_url
    name = part.get("filename")
    if not isinstance(name, str) or not name:
        name = f"file.{_extension(mime_type)}"
    if not file_data:
        # Slack refuses a zero-byte upload, and the whole reply fails
        # with it, so an empty file does not go on the wire.
        logger.warning("Skipped an empty file from %s: %s", origin, name)
        return None
    return {"file": {"name": name, "bytes": file_data}}


def _file_event_from_image(part: Mapping, origin: str) -> dict | None:
    """
    Build a `file` event from an `input_image` part.

    Args:
        part (Mapping): The part, whose `image_url` carries the image as
            a data URL — an http URL or a file id is a pointer, with
            nothing to upload.
        origin (str): The tool that produced it.

    Returns:
        dict | None: The `file` event, or None for a part without bytes.
    """
    image_url = part.get("image_url")
    if not isinstance(image_url, str):
        return None
    data_url = _data_url_parts(image_url)
    if data_url is None:
        return None
    mime_type, data = data_url
    name = f"image.{_extension(mime_type)}"
    if not data:
        logger.warning("Skipped an empty file from %s: %s", origin, name)
        return None
    return {"file": {"name": name, "bytes": data}}


def _data_url_parts(url: str) -> tuple[str, str] | None:
    """
    Split a data URL into its media type and base64 payload.

    Args:
        url (str): A string that may be a data URL.

    Returns:
        tuple[str, str] | None: The media type and the base64 payload, or
            None for anything but a base64 data URL.
    """
    if not url.startswith("data:"):
        return None
    head, separator, payload = url[5:].partition(",")
    if not separator or not head.endswith(";base64"):
        return None
    return head.removesuffix(";base64"), payload


def _extension(mime_type: str | None) -> str:
    """
    Pick a filename extension for a media type.

    Args:
        mime_type (str | None): The media type, when one is known.

    Returns:
        str: The subtype as extension, or `bin` for none worth writing.
    """
    subtype = mime_type.split("/", 1)[1] if mime_type and "/" in mime_type else ""
    extension = _EXTENSION_BY_SUBTYPE.get(subtype)
    if extension is None:
        extension = subtype if subtype.isalnum() and subtype.islower() else "bin"
    return extension
