import copy
import re

import httpx
import pytest

from service import assembler, llm, main, parser, prompt, token_count


def conversation(id_type):
    return [
        {
            "role": "user",
            "global_id": id_type(1),
            "content": "Preserve parse(text) while fixing empty input. " * 15,
        },
        {
            "role": "assistant",
            "global_id": id_type(2),
            "content": "Obsolete diagnostic output. " * 600,
        },
        {
            "role": "user",
            "global_id": id_type(3),
            "content": "Keep all existing public signatures. " * 10,
        },
        {
            "role": "assistant",
            "global_id": id_type(4),
            "content": [
                {"type": "text", "text": "I inspected both results. " * 10},
                {
                    "type": "tool-call",
                    "toolCallId": "a",
                    "toolName": "read",
                    "input": {"path": "a.py", "global_id": "business-field"},
                },
                {
                    "type": "tool-call",
                    "toolCallId": "b",
                    "toolName": "read",
                    "input": {"path": "b.py"},
                },
            ],
        },
        {
            "role": "tool",
            "global_id": id_type(90),
            "content": [
                {
                    "type": "tool-result",
                    "toolCallId": "a",
                    "toolName": "read",
                    "output": {"type": "text", "value": "file A is inspected. " * 20},
                }
            ],
        },
        {
            "role": "tool",
            "global_id": id_type(91),
            "content": [
                {
                    "type": "tool-result",
                    "toolCallId": "b",
                    "toolName": "read",
                    "output": {"type": "text", "value": "file B is inspected. " * 20},
                }
            ],
        },
        {
            "role": "assistant",
            "global_id": id_type(6),
            "content": "Next: add the guard and rerun tests. " * 15,
        },
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("id_type", [int, str], ids=["integer-ids", "string-ids"])
@pytest.mark.parametrize(
    "fallback", [False, True], ids=["model-plan", "retention-fallback"]
)
async def test_compaction_output_can_be_submitted_again(monkeypatch, id_type, fallback):
    original = conversation(id_type)
    untouched = copy.deepcopy(original)
    calls = []

    async def backend(messages):
        ids = re.findall(r"<id>([^<>]+)</id>", messages[1]["content"])
        calls.append(ids)
        if len(calls) == 1:
            keep_ids = set(ids) - (set() if fallback else {"2"})
        else:
            keep_ids = {ids[-1]}
        decisions = "".join(
            f"<aix_decision_item><id>{gid}</id><item_analysis>Preserve useful context.</item_analysis>"
            f"<action>{'keep' if gid in keep_ids else 'remove'}</action></aix_decision_item>"
            for gid in ids
        )
        return (
            "<aix_trajectory_compress><aix_analysis>Continue the parser fix.</aix_analysis>"
            f"<aix_decisions>{decisions}</aix_decisions>"
            "<aix_summary>Preserve parse(text), add the guard, and rerun tests.</aix_summary>"
            "</aix_trajectory_compress>"
        )

    monkeypatch.setattr(prompt, "model_token_length", lambda text: (len(text) + 3) // 4)
    monkeypatch.setattr(main, "TRACE_ENABLED", False)
    monkeypatch.setattr(llm, "call", backend)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=main.app), base_url="http://test"
    ) as client:
        first = await client.post("/compact", json={"messages": original})
        assert first.status_code == 200, first.text
        data = first.json()["data"]
        assert data["stats"]["fallback_triggered"] is fallback
        retained = data["messages"]
        second = await client.post("/compact", json={"messages": retained})
        assert second.status_code == 200, second.text
        assert (
            await client.post("/validate", json={"messages": retained})
        ).status_code == 200

    assert calls[0] == ["1", "2", "3", "4", "5", "6"]
    assert len(calls) == 2
    assert all("global_id" not in message for message in retained)
    parts = [
        part
        for message in retained
        if isinstance(message["content"], list)
        for part in message["content"]
    ]
    tool_calls = {
        part["toolCallId"]: part for part in parts if part["type"] == "tool-call"
    }
    tool_results = {
        part["toolCallId"]: part for part in parts if part["type"] == "tool-result"
    }
    assert tool_calls.keys() == tool_results.keys() == {"a", "b"}
    assert tool_calls["a"]["input"]["global_id"] == "business-field"
    assert original == untouched


@pytest.mark.parametrize("mode", ["aix", "legacy", "fallback"])
def test_assembly_preserves_original_ids_and_unrelated_metadata(monkeypatch, mode):
    original = conversation(int)
    original[3]["client_tag"] = "preserve-this-metadata"
    untouched = copy.deepcopy(original)
    monkeypatch.setattr(prompt, "model_token_length", lambda text: (len(text) + 3) // 4)
    parsed = parser.ParseResult(
        schema="legacy" if mode == "legacy" else "aix",
        summary="Continue the parser fix.",
        decisions=[
            parser.PruneDecision(
                call_id=gid, name="read", action="keep_all", reason="Keep context."
            )
            for gid in (
                ["a", "b"] if mode == "legacy" else ["1", "2", "3", "4", "5", "6"]
            )
        ],
    )
    if mode == "fallback":
        result = assembler.build_retention_fallback(
            original, parsed, token_count.count_messages_tokens(original), 25
        )
    else:
        result = assembler.assemble(original, parsed, original)

    assert original == untouched
    assert all("global_id" not in message for message in result["messages"])
    assert any(
        message.get("client_tag") == "preserve-this-metadata"
        for message in result["messages"]
    )
