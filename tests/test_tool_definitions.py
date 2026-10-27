import json
from pathlib import Path

import httpx
import pytest

from service import llm, main, prompt


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "empty_schema", [False, True], ids=["nested-schema", "empty-schema"]
)
async def test_complete_tool_schema_reaches_compaction_model(monkeypatch, empty_schema):
    body = json.loads(Path("examples/request.json").read_text())
    description = "Read the requested file. " * 20 + "Preserve the final warning."
    parameters = {
        "$defs": {"Mode": {"type": "string", "enum": ["text", "binary"]}},
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Project-relative file path"},
            "options": {
                "type": "object",
                "properties": {"mode": {"$ref": "#/$defs/Mode"}},
                "required": ["mode"],
                "additionalProperties": False,
            },
        },
        "required": ["path"],
        "additionalProperties": False,
    }
    if empty_schema:
        parameters = {}
    body["tool_defs"] = [
        {
            "name": "read_file",
            "description": description,
            "parameters": parameters,
            "args": [
                {"name": "legacy_hint", "description": "Existing argument metadata"}
            ],
        }
    ]
    captured = []

    async def backend(messages):
        captured.append(messages[1]["content"])
        return Path("examples/response.xml").read_text()

    monkeypatch.setattr(prompt, "model_token_length", lambda text: (len(text) + 3) // 4)
    monkeypatch.setattr(main, "TRACE_ENABLED", False)
    monkeypatch.setattr(llm, "call", backend)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=main.app), base_url="http://test"
    ) as client:
        response = await client.post("/compact", json=body)

    assert response.status_code == 200
    assert description in captured[0]
    encoded = captured[0].split("Input JSON Schema:\n", 1)[1]
    forwarded, _ = json.JSONDecoder().raw_decode(encoded)
    assert forwarded == parameters
    assert "<arg_name>legacy_hint</arg_name>" in captured[0]


@pytest.mark.asyncio
@pytest.mark.parametrize("parameters", [None, [], "not a schema", 1])
async def test_invalid_tool_parameter_schema_is_a_client_error(monkeypatch, parameters):
    body = json.loads(Path("examples/request.json").read_text())
    body["tool_defs"] = [{"name": "read_file", "parameters": parameters}]

    calls = []

    async def backend(messages):
        calls.append(messages)
        return Path("examples/response.xml").read_text()

    monkeypatch.setattr(prompt, "model_token_length", lambda text: (len(text) + 3) // 4)
    monkeypatch.setattr(main, "TRACE_ENABLED", False)
    monkeypatch.setattr(llm, "call", backend)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=main.app, raise_app_exceptions=False),
        base_url="http://test",
    ) as client:
        response = await client.post("/compact", json=body)

    assert response.status_code == 400
    assert "parameters" in response.json()["error"]
    assert not calls
