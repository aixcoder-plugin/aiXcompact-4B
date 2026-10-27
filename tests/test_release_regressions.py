import copy
import json
from pathlib import Path
from unittest.mock import patch

import httpx
import pytest

from service import assembler, llm, main, parser, prompt, token_count, validator


def protocol(actions, summary="Current task and constraints are preserved."):
    items = "".join(
        f"<aix_decision_item><id>{gid}</id><item_analysis>Relevant context.</item_analysis>"
        f"<action>{action}</action></aix_decision_item>"
        for gid, action in actions
    )
    return (
        "<aix_trajectory_compress><aix_analysis>Preserve the task.</aix_analysis>"
        f"<aix_decisions>{items}</aix_decisions><aix_summary>{summary}</aix_summary>"
        "</aix_trajectory_compress>"
    )


def separate_results():
    return [
        {"role": "user", "content": "Inspect both results."},
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool-call",
                    "toolCallId": "a",
                    "toolName": "read",
                    "input": {},
                },
                {
                    "type": "tool-call",
                    "toolCallId": "b",
                    "toolName": "read",
                    "input": {},
                },
            ],
        },
        {
            "role": "tool",
            "content": [
                {
                    "type": "tool-result",
                    "toolCallId": "a",
                    "toolName": "read",
                    "output": {"type": "text", "value": "result A"},
                },
            ],
        },
        {
            "role": "tool",
            "content": [
                {
                    "type": "tool-result",
                    "toolCallId": "b",
                    "toolName": "read",
                    "output": {"type": "text", "value": "result B"},
                },
            ],
        },
    ]


@pytest.fixture(autouse=True)
def isolate(monkeypatch):
    monkeypatch.setattr(prompt, "model_token_length", lambda text: (len(text) + 3) // 4)
    monkeypatch.setattr(main, "TRACE_ENABLED", False)


def test_quoted_action_in_analysis_does_not_override_keep():
    raw = protocol([(1, "keep")]).replace(
        "Relevant context.",
        "The quoted example <action>remove</action> does not apply.",
    )
    parsed = parser.parse(raw)
    assert parsed.decisions[0].action == "keep_all"
    original = [{"role": "user", "content": "Do not change the public API."}]
    output = assembler.assemble(original, parsed, original)["messages"]
    assert original[0] in output


def test_quoted_protocol_in_analysis_and_replacement_is_raw_text():
    replacement = "Example <aix_decision_item><id>9</id><action>remove</action></aix_decision_item> & x < y"
    raw = (
        protocol([(1, "keep_with_replaced_tool_result")])
        .replace(
            "<action>keep_with_replaced_tool_result</action>",
            "<action>keep_with_replaced_tool_result</action>"
            f"<replacement_tool_result_content>{replacement}</replacement_tool_result_content>",
        )
        .replace(
            "Preserve the task.", "Quote <aix_summary>wrong summary</aix_summary>."
        )
    )
    parsed = parser.parse(raw)
    assert parsed.summary == "Current task and constraints are preserved."
    assert len(parsed.decisions) == 1
    assert parsed.decisions[0].pruned_observation == replacement


def test_render_separate_tool_results_without_losing_payloads():
    messages = separate_results()
    validator.validate(messages)
    xml = prompt.build_coding_agent_trajectory_xml(messages)
    assert "result A" in xml and "result B" in xml
    assert xml.count("<assistant_tool_pair>") == 2
    assert xml.count("<user_message>") == 1


@pytest.mark.parametrize("action", ["keep", "remove"])
def test_assembly_keeps_or_removes_all_separate_results_with_calls(action):
    messages = separate_results()
    parsed = parser.parse(protocol([(1, "keep"), (2, action), (3, action)]))
    output = assembler.assemble(messages, parsed, messages)["messages"]
    calls = {
        i["toolCallId"]
        for m in output
        for i in m["content"]
        if isinstance(m["content"], list) and i.get("type") == "tool-call"
    }
    results = {
        i["toolCallId"]
        for m in output
        for i in m["content"]
        if isinstance(m["content"], list) and i.get("type") == "tool-result"
    }
    assert calls == results == ({"a", "b"} if action == "keep" else set())


def test_fallback_never_splits_separate_tool_results():
    messages = separate_results()
    parsed = parser.ParseResult(summary="s")
    with patch.object(token_count, "count_message_tokens", return_value=10):
        output = assembler.build_retention_fallback(messages, parsed, 100, 25)[
            "messages"
        ]
    assert not any(m["role"] == "tool" for m in output)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "corrupt",
    [
        lambda raw: raw.replace("<aix_summary>", "<missing_summary>").replace(
            "</aix_summary>", "</missing_summary>"
        ),
        lambda raw: raw.replace("<aix_trajectory_compress>", "").replace(
            "</aix_trajectory_compress>", ""
        ),
        lambda raw: raw.replace("<id>2</id>", "<id>1</id>"),
        lambda raw: raw.replace("<id>2</id>", "<id>99</id>"),
        lambda raw: raw.replace(
            "<action>remove</action>", "<action>invalid</action>", 1
        ),
        lambda raw: raw.replace("Current task and constraints are preserved.", ""),
    ],
)
async def test_invalid_protocol_fails_without_replacing_history(monkeypatch, corrupt):
    messages = [
        {"role": "user", "content": "Important request " * 400},
        {"role": "assistant", "content": "Pending fix " * 400},
    ]
    untouched = copy.deepcopy(messages)

    async def fake(_):
        return corrupt(protocol([(1, "remove"), (2, "remove")]))

    monkeypatch.setattr(llm, "call", fake)
    monkeypatch.setattr(
        token_count, "count_text_tokens", lambda text: (len(text) + 3) // 4
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=main.app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/compact", json={"messages": messages, "request_id": "invalid-output"}
        )
    assert response.status_code == 502
    assert response.json()["status"] == "error"
    assert "data" not in response.json()
    assert messages == untouched
    assert "invalid-output" not in main._inflight


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "field,value",
    [
        ("toolCallId", ["a"]),
        ("toolCallId", {"id": "a"}),
        ("toolName", ["read"]),
        ("type", ["tool-call"]),
    ],
)
async def test_invalid_nested_types_are_client_errors(field, value):
    messages = separate_results()
    messages[1]["content"][0][field] = value
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=main.app, raise_app_exceptions=False),
        base_url="http://test",
    ) as client:
        for endpoint in ["/validate", "/compact"]:
            response = await client.post(endpoint, json={"messages": messages})
            assert response.status_code == 422


def test_token_counts_use_local_model_tokenizer_without_network(monkeypatch):
    monkeypatch.setattr(prompt, "model_token_length", lambda text: len(text.split()))
    with patch(
        "urllib.request.urlopen", side_effect=AssertionError("No remote tokenize calls")
    ):
        assert [token_count.count_text_tokens("hello hello") for _ in range(3)] == [
            2,
            2,
            2,
        ]
    assert token_count.count_text_tokens("") == 0


@pytest.mark.asyncio
async def test_service_groups_separate_tool_results_before_inference(monkeypatch):
    messages = separate_results()
    captured = []

    async def fake(api_messages):
        captured.append(api_messages[1]["content"])
        return protocol([(1, "keep"), (2, "keep"), (3, "keep")])

    monkeypatch.setattr(llm, "call", fake)
    monkeypatch.setattr(
        token_count, "count_text_tokens", lambda text: (len(text) + 3) // 4
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=main.app), base_url="http://test"
    ) as client:
        response = await client.post("/compact", json={"messages": messages})
    assert response.status_code == 200
    assert "result B" in captured[0]


def test_cdata_protects_payloads_own_closing_delimiter():
    text = "Quoted </aix_summary> & x < y"
    raw = protocol([(1, "keep")], f"<![CDATA[{text}]]>")
    parsed = parser.parse(raw)
    parser.validate_output(parsed, {"1": False})
    assert parsed.summary == text


@pytest.mark.parametrize(
    "corrupt",
    [
        lambda raw: raw + "unexpected suffix",
        lambda raw: raw.replace("</aix_summary>", ""),
        lambda raw: raw.replace("Relevant context.", ""),
        lambda raw: raw.replace("<id>1</id>", "<id></id>"),
        lambda raw: raw.replace(
            "Current task and constraints are preserved.", "<![CDATA[unterminated"
        ),
    ],
)
def test_incomplete_and_ambiguous_protocol_is_not_validated(corrupt):
    parsed = parser.parse(corrupt(protocol([(1, "keep")])))
    with pytest.raises(parser.ProtocolError):
        parser.validate_output(parsed, {"1": False})


def test_replacement_cannot_target_a_plain_message():
    raw = protocol([(1, "keep_with_replaced_tool_result")]).replace(
        "</action>",
        "</action><replacement_tool_result_content>short</replacement_tool_result_content>",
    )
    with pytest.raises(parser.ProtocolError, match="only valid for tool pairs"):
        parser.validate_output(parser.parse(raw), {"1": False})


@pytest.mark.parametrize("replacement", ["", None])
def test_replacement_requires_nonempty_payload(replacement):
    raw = protocol([(1, "keep_with_replaced_tool_result")])
    if replacement is not None:
        raw = raw.replace(
            "</action>",
            "</action><replacement_tool_result_content></replacement_tool_result_content>",
        )
    with pytest.raises(parser.ProtocolError):
        parser.validate_output(parser.parse(raw), {"1": True})


def test_partial_decisions_remove_and_replace_only_their_own_pairs():
    messages = separate_results()
    raw = protocol(
        [(1, "keep"), (2, "remove"), (3, "keep_with_replaced_tool_result")]
    ).replace(
        "<action>keep_with_replaced_tool_result</action>",
        "<action>keep_with_replaced_tool_result</action>"
        "<replacement_tool_result_content>short B</replacement_tool_result_content>",
    )
    output = assembler.assemble(messages, parser.parse(raw), messages)["messages"]
    calls = [
        item["toolCallId"]
        for m in output
        if isinstance(m["content"], list)
        for item in m["content"]
        if item["type"] == "tool-call"
    ]
    results = [
        item
        for m in output
        if isinstance(m["content"], list)
        for item in m["content"]
        if item["type"] == "tool-result"
    ]
    assert calls == ["b"]
    assert results[0]["toolCallId"] == "b"
    assert results[0]["output"]["value"] == "short B"


def test_legacy_parser_and_assembly_keep_replace_and_remove_tool_pairs():
    messages = separate_results()
    messages += [
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Let me check the file:"},
                {
                    "type": "tool-call",
                    "toolCallId": "c",
                    "toolName": "read",
                    "input": {},
                },
            ],
        },
        {
            "role": "tool",
            "content": [
                {
                    "type": "tool-result",
                    "toolCallId": "c",
                    "toolName": "read",
                    "output": "obsolete",
                },
            ],
        },
    ]
    raw = """<analysis>Legacy format.</analysis><tool_pruning>
    <tool call_id="a" name="read"><reason>keep it</reason><action>keep_all</action></tool>
    <tool call_id="call_2" name="read"><action>pruner</action><pruned_observation>short B</pruned_observation></tool>
    <tool call_id="call_3" name="read"><action>remove_all</action></tool>
    <tool call_id="call_9999" name="read"><action>keep_all</action></tool>
    <tool call_id="unknown" name="read"><action>keep_all</action></tool>
    </tool_pruning><summary>Legacy summary.</summary>"""
    parsed = parser.parse(raw)
    output = assembler.assemble(messages, parsed, messages)["messages"]
    calls = {
        item["toolCallId"]
        for m in output
        for item in m["content"]
        if item["type"] == "tool-call"
    }
    assert calls == {"a", "b"}
    results = {
        item["toolCallId"]: item["output"]
        for m in output
        for item in m["content"]
        if item["type"] == "tool-result"
    }
    assert results["b"] == {"type": "text", "value": "short B"}
    assert parsed.summary == "Legacy summary."


@pytest.mark.parametrize(
    "messages",
    [
        None,
        [],
        ["not an object"],
        [{"role": [], "content": "text"}],
        [{"role": "system", "content": "text"}],
        [{"role": "user"}, {"role": "assistant", "content": "text"}],
        [{"role": "user", "content": 123}, {"role": "assistant", "content": "text"}],
        [{"role": "user", "content": [None]}, {"role": "assistant", "content": "text"}],
        [
            {"role": "user", "content": [{"type": "text", "text": []}]},
            {"role": "assistant", "content": "text"},
        ],
    ],
)
def test_invalid_message_shapes_raise_validation_error(messages):
    with pytest.raises(validator.ValidationError):
        validator.validate(messages)


@pytest.mark.parametrize(
    "change",
    [
        lambda m: m[1]["content"][1].update(toolCallId="a"),
        lambda m: m[3]["content"][0].update(toolCallId="a"),
        lambda m: m[3]["content"][0].update(toolName="wrong"),
        lambda m: m[0].update(global_id=2),
        lambda m: m[0].update(global_id=[]),
        lambda m: m[3].update(content="missing result ID"),
        lambda m: m.insert(3, {"role": "user", "content": "interruption"}),
    ],
)
def test_duplicate_ids_and_unsupported_result_layouts_are_rejected(change):
    messages = separate_results()
    change(messages)
    with pytest.raises(validator.ValidationError):
        validator.validate(messages)


@pytest.mark.asyncio
async def test_validate_success_and_compact_debug(monkeypatch):
    body = json.loads(Path("examples/request.json").read_text())

    async def fake(_):
        return Path("examples/response.xml").read_text()

    monkeypatch.setattr(llm, "call", fake)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=main.app), base_url="http://test"
    ) as client:
        response = await client.post("/validate", json={"messages": body["messages"]})
        assert response.status_code == 200
        response = await client.post("/compact", json={**body, "debug": True})
        assert response.status_code == 200
        assert response.json()["data"]["debug"]["summary"]


@pytest.mark.asyncio
@pytest.mark.parametrize("status,attempts", [(503, 2), (429, 2), (400, 1)])
async def test_model_retries_transient_failures_and_not_client_errors(
    monkeypatch, status, attempts
):
    requests = []

    def backend(request):
        requests.append(request)
        if len(requests) == 1:
            return httpx.Response(status, text="synthetic error")
        return httpx.Response(
            200,
            text='data: {"choices":[{"delta":{"content":"valid output"},"finish_reason":"stop"}]}\n\ndata: [DONE]\n\n',
        )

    transport = httpx.MockTransport(backend)
    monkeypatch.setattr(
        llm, "_client_modes", lambda: [("test", {"transport": transport})]
    )
    if status == 400:
        with pytest.raises(llm.LLMError):
            await llm.call([])
    else:
        assert await llm.call([]) == "valid output"
    assert len(requests) == attempts


@pytest.mark.asyncio
async def test_exhausted_model_retries_return_backend_error(monkeypatch):
    monkeypatch.setattr(llm, "LLM_MAX_RETRIES", 0)
    transport = httpx.MockTransport(
        lambda _: httpx.Response(503, text="temporarily unavailable")
    )
    monkeypatch.setattr(
        llm, "_client_modes", lambda: [("test", {"transport": transport})]
    )
    with pytest.raises(llm.LLMError, match="failed after 1 attempts"):
        await llm.call([])


def test_proxy_modes_are_explicit(monkeypatch):
    monkeypatch.delenv("LLM_TRUST_ENV", raising=False)
    monkeypatch.delenv("LLM_HTTP_PROXY", raising=False)
    assert [name for name, _ in llm._client_modes()] == ["direct"]
    monkeypatch.setenv("LLM_HTTP_PROXY", "http://127.0.0.1:9999")
    assert [name for name, _ in llm._client_modes()] == ["direct", "env-proxy-fallback"]
    monkeypatch.setenv("LLM_TRUST_ENV", "true")
    assert [name for name, _ in llm._client_modes()] == ["env-proxy"]
