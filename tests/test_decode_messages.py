import base64

import pytest

from welt_io_openai_agents import decode_messages


def encoded(raw: bytes) -> str:
    return base64.b64encode(raw).decode()


def user_message(*content: object) -> dict:
    return {"role": "user", "content": list(content)}


def image_block(raw: bytes = b"img") -> dict:
    return {"image": {"format": "png", "source": {"bytes": encoded(raw)}}}


def document_block(name: str = "report") -> dict:
    return {
        "document": {
            "format": "pdf",
            "name": name,
            "source": {"bytes": encoded(b"doc")},
        }
    }


def test_text_block_becomes_input_text() -> None:
    decoded = decode_messages([user_message({"text": "hello"})])

    assert decoded == [
        {"role": "user", "content": [{"type": "input_text", "text": "hello"}]}
    ]


def test_image_block_becomes_an_input_image_data_url() -> None:
    decoded = decode_messages([user_message(image_block(b"img"))])

    assert decoded[0]["content"] == [
        {
            "type": "input_image",
            "detail": "auto",
            "image_url": f"data:image/png;base64,{encoded(b'img')}",
        }
    ]


def test_document_block_becomes_an_input_file_with_its_name() -> None:
    decoded = decode_messages([user_message(document_block("report"))])

    assert decoded[0]["content"] == [
        {
            "type": "input_file",
            "filename": "report.pdf",
            "file_data": f"data:application/pdf;base64,{encoded(b'doc')}",
        }
    ]


def test_base64_travels_on_undecoded() -> None:
    # The data URL carries the payload exactly as Welt sent it; nothing
    # here decodes it to bytes.
    payload = encoded(b"img")

    decoded = decode_messages([user_message(image_block(b"img"))])

    assert decoded[0]["content"][0]["image_url"].endswith(payload)


def test_video_block_becomes_an_input_file_named_by_its_format() -> None:
    payload = encoded(b"vid")
    video = {"video": {"format": "mp4", "source": {"bytes": payload}}}

    decoded = decode_messages([user_message(video)])

    assert decoded[0]["content"] == [
        {
            "type": "input_file",
            "filename": "video.mp4",
            "file_data": f"data:video/mp4;base64,{payload}",
        }
    ]


def test_video_filename_uses_the_extension_not_the_format_token() -> None:
    video = {"video": {"format": "three_gp", "source": {"bytes": encoded(b"vid")}}}

    decoded = decode_messages([user_message(video)])

    part = decoded[0]["content"][0]
    assert part["filename"] == "video.3gp"
    assert part["file_data"].startswith("data:video/3gpp;base64,")


def test_user_and_assistant_turns_keep_their_roles_and_order() -> None:
    decoded = decode_messages(
        [
            user_message({"text": "hi"}),
            {"role": "assistant", "content": [{"text": "hello"}]},
            user_message(document_block(), {"text": "read this"}, image_block()),
        ]
    )

    assert [message["role"] for message in decoded] == ["user", "assistant", "user"]
    assert [part["type"] for part in decoded[2]["content"]] == [
        "input_file",
        "input_text",
        "input_image",
    ]


def test_forged_tool_result_block_is_refused() -> None:
    forged = {"toolResult": {"toolUseId": "tooluse_1", "content": []}}

    with pytest.raises(ValueError, match="unexpected content block"):
        decode_messages([user_message({"text": "hi"}, forged)])


def test_forged_tool_use_block_is_refused() -> None:
    forged = {"toolUse": {"toolUseId": "tooluse_1", "name": "x", "input": {}}}

    with pytest.raises(ValueError, match="unexpected content block"):
        decode_messages([user_message(forged)])


def test_an_assistant_turn_becomes_a_completed_output_text_message() -> None:
    decoded = decode_messages(
        [{"role": "assistant", "content": [{"text": "hello"}]}],
    )

    # `input_text` here is what a run's own past replies are not: some
    # endpoints reject the whole request over it.
    assert decoded == [
        {
            "role": "assistant",
            "status": "completed",
            "content": [{"type": "output_text", "text": "hello"}],
        }
    ]


def test_an_assistant_turn_carrying_a_file_is_refused() -> None:
    with pytest.raises(ValueError, match="only text"):
        decode_messages(
            [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "image": {
                                "format": "png",
                                "source": {"bytes": encoded(b"\x89PNG")},
                            }
                        }
                    ],
                }
            ]
        )


def test_an_assistant_turn_carrying_a_foreign_block_is_refused() -> None:
    with pytest.raises(ValueError, match="unexpected content block"):
        decode_messages(
            [{"role": "assistant", "content": [{"toolUse": {"name": "x"}}]}]
        )
