import asyncio
import json
from pathlib import Path
import pytest
import httpx
from service import main, llm, token_count, prompt, parser, assembler


@pytest.fixture(autouse=True)
def isolate(monkeypatch):
    monkeypatch.setattr(
        token_count,
        "count_text_tokens",
        lambda text: max(1, (len(text) + 3) // 4) if text else 0,
    )
    monkeypatch.setattr(prompt, "model_token_length", lambda text: (len(text) + 3) // 4)
    monkeypatch.setattr(main, "TRACE_ENABLED", False)


def body():
    return json.loads(Path("examples/request.json").read_text())


@pytest.mark.asyncio
async def test_service_success_and_no_default_traces(monkeypatch, tmp_path):
    monkeypatch.setattr(main, "_TRACE_DIR", tmp_path / "traces")

    async def fake(_):
        return Path("examples/response.xml").read_text()

    monkeypatch.setattr(llm, "call", fake)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=main.app), base_url="http://test"
    ) as client:
        response = await client.post("/compact", json=body())
        assert response.status_code == 200
        data = response.json()["data"]
        assert data["messages"] and "fallback_triggered" in data["stats"]
        assert (await client.get("/health")).status_code == 200
    assert not (tmp_path / "traces").exists()


@pytest.mark.asyncio
async def test_duplicate_and_cancel_before_model_dispatch(monkeypatch):
    started = asyncio.Event()

    async def slow(_):
        started.set()
        await asyncio.sleep(60)

    monkeypatch.setattr(llm, "call", slow)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=main.app), base_url="http://test"
    ) as client:
        data = {**body(), "request_id": "same-id"}
        task = asyncio.create_task(client.post("/compact", json=data))
        await asyncio.wait_for(started.wait(), 5)
        assert (await client.post("/compact", json=data)).status_code == 409
        assert (await client.post("/cancel/same-id")).status_code == 200
        assert (await task).status_code == 499
        assert "same-id" not in main._inflight
        assert (await client.post("/cancel/same-id")).status_code == 404


@pytest.mark.asyncio
@pytest.mark.parametrize("value", [0, 100, 25.5, True, "25", None])
async def test_invalid_target(value):
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=main.app), base_url="http://test"
    ) as client:
        assert (
            await client.post("/compact", json={**body(), "target_ratio_pct": value})
        ).status_code == 400


def test_retention_boundaries():
    low, high = main._retention_band(25)
    assert (low, high) == (0.05, 0.45)
    assert main._fallback_reason(high, low, high, 10000) == "under_compression"
    assert main._fallback_reason(0.049, low, high, 10000) == "over_compression"
    assert main._fallback_reason(low, low, high, 10000) is None
    assert main._fallback_reason(0.01, low, high, 100) is None


def test_fallback_preserves_pairs_and_reports_summary_exception():
    messages = body()["messages"]
    units = assembler._iter_units(messages)
    assert any(len(unit) == 2 for unit in units)
    parsed = parser.ParseResult(summary="long summary " * 1000)
    result = assembler.build_retention_fallback(messages, parsed, 1000, 25)
    assert result["stats"]["summary_over_budget"]
    assert not result["stats"]["summary_truncated"]
    assert len(result["messages"]) == 1


@pytest.mark.asyncio
async def test_invalid_json():
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=main.app), base_url="http://test"
    ) as client:
        for route in ["/compact", "/validate"]:
            assert (await client.post(route, content="{")).status_code == 400


@pytest.mark.asyncio
async def test_reservation_and_cancel_during_preparation(monkeypatch):
    preparing = asyncio.Event()
    called = []

    async def hold_trace(*_):
        preparing.set()
        await asyncio.sleep(60)

    async def unexpected_call(_):
        called.append(True)
        raise AssertionError("Inference must not begin")

    monkeypatch.setattr(main, "_save_trace_async", hold_trace)
    monkeypatch.setattr(llm, "call", unexpected_call)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=main.app), base_url="http://test"
    ) as client:
        data = {**body(), "request_id": "preparation-id"}
        pending = asyncio.create_task(client.post("/compact", json=data))
        await asyncio.wait_for(preparing.wait(), 5)
        assert (await client.post("/compact", json=data)).status_code == 409
        assert (await client.post("/cancel/preparation-id")).status_code == 200
        assert (await pending).status_code == 499
        assert not called


@pytest.mark.asyncio
@pytest.mark.parametrize("ending", ["length", None])
async def test_model_truncation_and_broken_stream_fail(ending):
    payload = (
        'data: {"choices":[{"delta":{"content":"<aix_trajectory_compress>"}}]}\n\n'
    )
    if ending:
        payload += 'data: {"choices":[{"delta":{},"finish_reason":"length"}]}\n\n'
    transport = httpx.MockTransport(lambda request: httpx.Response(200, text=payload))
    with pytest.raises(llm.LLMError):
        await llm._stream_once({"transport": transport}, {"model": "test"})


@pytest.mark.asyncio
async def test_stream_ignores_provider_reasoning_and_retains_protocol_analysis():
    xml = "<aix_trajectory_compress><aix_analysis>keep this</aix_analysis><aix_summary>summary</aix_summary></aix_trajectory_compress>"
    events = [
        {"choices": [{"delta": {"reasoning_content": "do not include"}}]},
        {"choices": [{"delta": {"content": xml}}]},
        {"choices": [{"delta": {}, "finish_reason": "stop"}]},
    ]
    payload = (
        "".join("data: " + json.dumps(event) + "\n\n" for event in events)
        + "data: [DONE]\n\n"
    )
    transport = httpx.MockTransport(lambda request: httpx.Response(200, text=payload))
    assert await llm._stream_once({"transport": transport}, {"model": "test"}) == xml
