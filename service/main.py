from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import time
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from . import assembler
from . import llm
from . import parser
from . import prompt
from . import token_count
from .messages import normalize_tool_results, trajectory_item_types
from .config import (
    FLOOR_SKIP_INPUT_TOKENS,
    LLM_ENDPOINT,
    LLM_MODEL,
    RETENTION_BAND_PCT,
    TARGET_RATIO_PCT,
    TRACE_MAX_DIRS,
    TRACE_ENABLED,
)
from logging.handlers import RotatingFileHandler
from pathlib import Path
from .validator import ValidationError, validate

_cfg_level = getattr(
    logging,
    os.environ.get("LOG_LEVEL", "INFO").upper(),
    logging.INFO,
)
_fmt = logging.Formatter("%(asctime)s [%(name)s] %(levelname)s %(message)s")
_root = logging.getLogger()
_root.setLevel(_cfg_level)
if not _root.handlers:
    _sh = logging.StreamHandler()
    _sh.setFormatter(_fmt)
    _root.addHandler(_sh)

_log_target = os.environ.get("LOG_FILE", "stderr")
if _log_target != "stderr":
    _log_path = Path(_log_target).expanduser()
    _log_path.parent.mkdir(parents=True, exist_ok=True)
    _fh = RotatingFileHandler(
        _log_path, maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8"
    )
    _fh.setFormatter(_fmt)
    _root.addHandler(_fh)

log = logging.getLogger("compaction.api")

_TRACE_DIR = Path(os.environ.get("TRACE_DIR", ".runtime/traces"))


def _save_trace(trace_dir: Path, filename: str, data: Any) -> None:
    """Save a trace file as pretty-printed JSON."""
    if not TRACE_ENABLED:
        return
    filepath = trace_dir / filename
    with open(filepath, "w", encoding="utf-8") as f:
        if isinstance(data, str):
            f.write(data)
        else:
            json.dump(data, f, ensure_ascii=False, indent=2)


async def _save_trace_async(trace_dir: Path, filename: str, data: Any) -> None:
    """Trace files can be tens of MB and land on NFS; keep the write off the
    event loop so it never stalls concurrent requests."""
    await asyncio.get_running_loop().run_in_executor(
        None, _save_trace, trace_dir, filename, data
    )


def _prune_traces() -> None:
    if not TRACE_ENABLED or TRACE_MAX_DIRS <= 0 or not _TRACE_DIR.exists():
        return
    dirs = sorted(d for d in _TRACE_DIR.iterdir() if d.is_dir())
    for d in dirs[: max(0, len(dirs) - TRACE_MAX_DIRS)]:
        shutil.rmtree(d, ignore_errors=True)


def _create_trace_dir() -> Path:
    """Create a timestamped trace directory for this request."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    trace_dir = _TRACE_DIR / ts
    if TRACE_ENABLED:
        trace_dir.mkdir(parents=True, exist_ok=True)
    return trace_dir


@asynccontextmanager
async def lifespan(_app: FastAPI):
    prompt._load_length_tokenizer()
    log.info("Compaction service ready log_file=%s", _log_target)
    yield
    log.info("Compaction service stopping")


app = FastAPI(
    title="aiXcompact",
    description="Standalone context compaction service for opencode context compression",
    version="2.2.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    ms = round((time.time() - start) * 1000, 2)
    line = "%s %s -> %s %sms"
    args = (request.method, request.url.path, response.status_code, ms)
    if request.url.path == "/health":
        log.debug(line, *args)
    else:
        log.info(line, *args)
    return response


def _ok(data: Any, **extra: Any) -> dict:
    return {"status": "ok", "data": data, **extra}


def _err(msg: str, **extra: Any) -> dict:
    return {"status": "error", "error": msg, **extra}


# Allowed top-level request-body fields per endpoint. Anything else is
# rejected with 400 so a mistyped parameter name fails loudly instead of
# being silently ignored in favour of the env-configured default.
_COMPACT_ALLOWED_FIELDS = {
    "messages",
    "debug",
    "request_id",
    "system_prompt",
    "tool_defs",
    "tool_names",
    "target_ratio_pct",
}
_VALIDATE_ALLOWED_FIELDS = {"messages"}


def _reject_unknown_fields(
    body: Any, allowed: set[str], endpoint: str
) -> JSONResponse | None:
    if not isinstance(body, dict):
        return JSONResponse(
            status_code=400,
            content=_err("request body must be a JSON object"),
        )
    unknown = sorted(set(body) - allowed)
    if unknown:
        log.warning("%s: unknown request fields %s", endpoint, unknown)
        return JSONResponse(
            status_code=400,
            content=_err(
                f"unknown request field(s): {', '.join(unknown)}",
                unknown_fields=unknown,
                allowed_fields=sorted(allowed),
            ),
        )
    return None


def _reject_bad_field_types(body: dict) -> JSONResponse | None:
    """Shape-check the non-numeric /compact fields so a wrong type fails with
    a 400 instead of a mid-pipeline 500 (e.g. tool_defs passed as an object)."""
    problems: list[str] = []
    if not isinstance(body.get("messages"), list):
        problems.append("'messages' must be an array")
    if not isinstance(body.get("debug", False), bool):
        problems.append("'debug' must be a boolean")
    if body.get("request_id") is not None and not isinstance(body["request_id"], str):
        problems.append("'request_id' must be a string")
    if not isinstance(body.get("system_prompt", ""), str):
        problems.append("'system_prompt' must be a string")
    tool_defs = body.get("tool_defs", [])
    if not isinstance(tool_defs, list) or any(
        not isinstance(td, dict) for td in tool_defs
    ):
        problems.append("'tool_defs' must be an array of objects")
    elif any(
        "parameters" in td and not isinstance(td["parameters"], dict)
        for td in tool_defs
    ):
        problems.append("'tool_defs[].parameters' must be a JSON Schema object")
    tool_names = body.get("tool_names", [])
    if not isinstance(tool_names, list) or any(
        not isinstance(n, str) for n in tool_names
    ):
        problems.append("'tool_names' must be an array of strings")
    if problems:
        log.warning("compact: bad field types: %s", "; ".join(problems))
        return JSONResponse(status_code=400, content=_err("; ".join(problems)))
    return None


def _retention_band(target_ratio_pct: int) -> tuple[float, float]:
    """Valid retention band derived from the target ratio: target +/-
    RETENTION_BAND_PCT percentage points, clamped to [0, 1]. The half-width
    is a fixed service invariant, deliberately not exposed as a knob."""
    lo = max(0.0, (target_ratio_pct - RETENTION_BAND_PCT) / 100)
    hi = min(1.0, (target_ratio_pct + RETENTION_BAND_PCT) / 100)
    return lo, hi


def _fallback_reason(
    retention: float,
    band_min: float,
    band_max: float,
    input_tokens: int,
) -> str | None:
    """Retention-band guardrail decision. 'under_compression' when the model
    barely compressed (retention >= band_max), 'over_compression' when it
    deleted almost everything (retention < band_min), else None. The
    lower-bound check is skipped for tiny inputs, which can legitimately
    compress to almost nothing."""
    if retention >= band_max:
        return "under_compression"
    if retention < band_min and input_tokens >= FLOOR_SKIP_INPUT_TOKENS:
        return "over_compression"
    return None


# Registry of in-flight model-inference tasks, keyed by client-supplied
# request_id, so an explicit POST /cancel/{request_id} can abort the running
# compression. Cancelling the task unwinds llm.call's streaming httpx request,
# which closes the connection to the model backend and makes vLLM abort.
_inflight: dict[str, asyncio.Task] = {}


@app.post("/cancel/{request_id}")
async def cancel(request_id: str):
    task = _inflight.get(request_id)
    if task is None or task.done():
        return JSONResponse(
            status_code=404,
            content=_err(
                "no in-flight request with that id (unknown or already finished)",
                request_id=request_id,
            ),
        )
    task.cancel()
    log.warning("cancel: aborting in-flight compaction request_id=%s", request_id)
    return _ok({"cancelled": True, "request_id": request_id})


@app.get("/")
@app.get("/health")
async def health():
    return _ok({"service": "compaction", "status": "healthy"})


@app.post("/validate")
async def validate_endpoint(request: Request):
    try:
        body = await request.json()
    except ValueError:
        return JSONResponse(status_code=400, content=_err("invalid JSON"))
    rejected = _reject_unknown_fields(body, _VALIDATE_ALLOWED_FIELDS, "validate")
    if rejected is not None:
        return rejected
    messages = body.get("messages")
    if messages is None:
        log.warning("validate: missing messages field")
        return JSONResponse(
            status_code=400,
            content=_err("request body must contain 'messages' field"),
        )
    try:
        stats = validate(messages)
    except ValidationError as e:
        log.warning("validate: failed errors=%s", e.errors)
        return JSONResponse(
            status_code=422,
            content=_err("validation failed", details=e.errors),
        )
    log.info(
        "validate: ok messages=%s tokens≈%s tools=%s/%s",
        stats["message_count"],
        stats["estimated_tokens"],
        stats["tool_calls"],
        stats["tool_results"],
    )
    return _ok(stats)


@app.post("/compact")
async def compact(request: Request):
    try:
        body = await request.json()
    except ValueError:
        return JSONResponse(status_code=400, content=_err("invalid JSON"))
    if not isinstance(body, dict):
        return JSONResponse(
            status_code=400, content=_err("request body must be a JSON object")
        )
    request_id = body.get("request_id")
    if request_id is not None and (
        not isinstance(request_id, str) or not request_id.strip()
    ):
        return JSONResponse(
            status_code=400, content=_err("request_id must be a non-empty string")
        )
    if request_id and request_id in _inflight:
        return JSONResponse(
            status_code=409, content=_err("request_id already in flight")
        )
    task = asyncio.create_task(_compact_impl(body))
    if request_id:
        _inflight[request_id] = task
    try:
        return await task
    except asyncio.CancelledError:
        return JSONResponse(
            status_code=499,
            content=_err("request cancelled; inference aborted", request_id=request_id),
        )
    finally:
        if request_id:
            _inflight.pop(request_id, None)


async def _compact_impl(body: dict):
    start = time.time()
    rejected = _reject_unknown_fields(body, _COMPACT_ALLOWED_FIELDS, "compact")
    if rejected is not None:
        return rejected
    if body.get("messages") is None:
        log.warning("compact: missing messages field")
        return JSONResponse(
            status_code=400,
            content=_err("request body must contain 'messages' field"),
        )
    bad_types = _reject_bad_field_types(body)
    if bad_types is not None:
        return bad_types

    messages = body["messages"]
    debug = body.get("debug", False)
    request_id = body.get("request_id")  # optional; enables POST /cancel/{request_id}
    system_prompt = body.get("system_prompt", "")
    tool_defs: list[dict] = body.get("tool_defs", [])
    tool_names: list[str] = body.get("tool_names", [])

    # Per-request target override; falls back to the env-configured default.
    # target_ratio_pct feeds the system prompt and the fallback budget; the
    # valid retention band [band_min, band_max) is derived from it (fixed
    # +/- RETENTION_BAND_PCT percentage points) — a result outside the band
    # triggers the deterministic fallback.
    target_ratio_pct = body.get("target_ratio_pct", TARGET_RATIO_PCT)
    if type(target_ratio_pct) is not int:
        return JSONResponse(
            status_code=400, content=_err("'target_ratio_pct' must be an integer")
        )
    if not 1 <= target_ratio_pct <= 99:
        return JSONResponse(
            status_code=400,
            content=_err("'target_ratio_pct' must be within [1, 99]"),
        )
    band_min, band_max = _retention_band(target_ratio_pct)

    loop = asyncio.get_running_loop()

    # Create trace directory for this request
    trace_dir = _create_trace_dir()
    loop.run_in_executor(None, _prune_traces)
    log.info("compact: trace_dir=%s", trace_dir)

    # [TRACE 1] 保存压缩服务接收到的原始内容（验证前保存以便调试）
    await _save_trace_async(
        trace_dir,
        "1_received_from_client.json",
        {
            "messages": messages,
            "system_prompt": system_prompt,
            "tool_defs": tool_defs,
            "tool_names": tool_names,
            "debug": debug,
            "target_ratio_pct": target_ratio_pct,
            "retention_band": [band_min, band_max],
        },
    )

    # 1. Validate
    try:
        stats = validate(messages)
    except ValidationError as e:
        log.warning("compact: validation failed errors=%s", e.errors)
        return JSONResponse(
            status_code=422,
            content=_err("validation failed", details=e.errors),
        )
    log.info(
        "compact: validated messages=%s tokens≈%s",
        stats["message_count"],
        stats["estimated_tokens"],
    )
    messages = normalize_tool_results(messages)

    # 2. Build tool definitions: prefer tool_defs (full definitions from
    # client), fall back to tool_names with default descriptions
    tool_definitions: list[dict] | None = None
    if tool_defs:
        tool_definitions = [
            {
                "name": td.get("name", ""),
                "description": td.get("description", ""),
                "args": td.get("args", []),
                **({"parameters": td["parameters"]} if "parameters" in td else {}),
            }
            for td in tool_defs
            if td.get("name")
        ]
    elif tool_names:
        tool_definitions = [prompt.get_tool_definition(name) for name in tool_names]

    # 3. Build the compressor input exactly once. Rendering the trajectory XML
    # and counting its tokens are CPU-heavy on large trajectories, so both run
    # off the event loop.
    trajectory_xml = await loop.run_in_executor(
        None,
        lambda: prompt.build_coding_agent_trajectory_xml(
            messages,
            coding_agent_system_prompt=system_prompt,
            tool_definitions=tool_definitions,
        ),
    )
    visible_message_tokens = await loop.run_in_executor(
        None, prompt.compute_visible_message_tokens, trajectory_xml
    )
    compressor_user_prompt = prompt.build_compressor_user_prompt(
        trajectory_xml,
        visible_message_tokens=visible_message_tokens,
        target_compression_ratio_pct=target_ratio_pct,
    )
    api_messages = llm.build_api_messages(compressor_user_prompt)
    log.info(
        "compact: prompt built tools=%s system_prompt_len=%s tool_defs=%s "
        "visible_message_tokens=%s target_ratio=%s%%%s band=[%.2f, %.2f)",
        len(prompt.collect_tools(messages)),
        len(system_prompt),
        len(tool_defs),
        visible_message_tokens,
        target_ratio_pct,
        " (overridden)" if target_ratio_pct != TARGET_RATIO_PCT else "",
        band_min,
        band_max,
    )

    # [TRACE 2] 保存转发给压缩模型的内容
    await _save_trace_async(
        trace_dir,
        "2_sent_to_llm.json",
        {
            "endpoint": LLM_ENDPOINT,
            "model": LLM_MODEL,
            "messages": api_messages,
        },
    )

    # 4. Call LLM. Run the model call as a registered task so POST
    # /cancel/{request_id} can abort it. Cancelling unwinds llm.call's
    # streaming request -> connection closes -> vLLM aborts the generation.
    llm_task = asyncio.ensure_future(llm.call(api_messages))
    try:
        raw = await llm_task
    except asyncio.CancelledError:
        log.warning(
            "compact: inference cancelled via /cancel request_id=%s trace=%s",
            request_id,
            trace_dir.name,
        )
        await _save_trace_async(
            trace_dir,
            "3_llm_response_CANCELLED.txt",
            f"cancelled via /cancel request_id={request_id}; model inference aborted",
        )
        return JSONResponse(
            status_code=499,
            content=_err("request cancelled; inference aborted", request_id=request_id),
        )
    except llm.LLMError as e:
        log.error("compact: LLM failed: %s", e)
        await _save_trace_async(trace_dir, "3_llm_response_ERROR.txt", str(e))
        return JSONResponse(
            status_code=502,
            content=_err(f"LLM call failed: {e}"),
        )
    log.info("compact: LLM response_chars=%s", len(raw))

    # [TRACE 3] 保存压缩模型返回的原始内容
    await _save_trace_async(trace_dir, "3_llm_response.txt", raw)

    # 5. Parse XML output
    parsed = parser.parse(raw)
    try:
        parser.validate_output(parsed, trajectory_item_types(messages))
    except parser.ProtocolError as error:
        log.warning("compact: invalid model protocol: %s", error)
        return JSONResponse(
            status_code=502,
            content=_err("invalid model compression protocol", details=str(error)),
        )
    has_structured = True
    log.info(
        "compact: parsed schema=%s analysis=%s summary=%s decisions=%s",
        parsed.schema,
        parsed.analysis is not None,
        parsed.summary is not None,
        len(parsed.decisions),
    )

    # 6. Assemble compressed output
    result = assembler.assemble(messages, parsed, messages)

    # 6b. Retention-band guardrails: under-compression (retention >= R1, the
    # model barely compressed) and over-compression (retention < R0, the model
    # deleted almost everything) both discard the per-item decisions and
    # deterministically rebuild as summary + most-recent items within the R2
    # token budget.
    # Count with the local model tokenizer off the event loop.
    input_tokens = await loop.run_in_executor(
        None, token_count.count_messages_tokens, messages
    )
    output_tokens = await loop.run_in_executor(
        None, token_count.count_messages_tokens, result["messages"]
    )
    retention = output_tokens / max(1, input_tokens)
    fallback_reason = _fallback_reason(retention, band_min, band_max, input_tokens)
    fallback_triggered = fallback_reason is not None
    if fallback_triggered:
        log.warning(
            "compact: %s retention=%.3f band=[%.2f, %.2f); "
            "applying deterministic fallback (target=%s%%)",
            fallback_reason,
            retention,
            band_min,
            band_max,
            target_ratio_pct,
        )
        result = await loop.run_in_executor(
            None,
            assembler.build_retention_fallback,
            messages,
            parsed,
            input_tokens,
            target_ratio_pct,
        )
        output_tokens = await loop.run_in_executor(
            None, token_count.count_messages_tokens, result["messages"]
        )
        retention = output_tokens / max(1, input_tokens)

    elapsed = round(time.time() - start, 2)
    response: dict[str, Any] = {
        "messages": result["messages"],
        "stats": {
            **result["stats"],
            "input_messages": stats["message_count"],
            "output_messages": len(result["messages"]),
            "estimated_input_tokens": stats["estimated_tokens"],
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "actual_retention": round(retention, 4),
            "target_ratio_pct": target_ratio_pct,
            "retention_band": [band_min, band_max],
            "fallback_triggered": fallback_triggered,
            "fallback_reason": fallback_reason,
            "fallback_kept_items": result["stats"].get("fallback_kept_items", 0),
            "has_structured_output": has_structured,
            "elapsed_seconds": elapsed,
        },
    }
    if stats.get("warnings"):
        response["warnings"] = stats["warnings"]

    if debug:
        response["debug"] = {
            "schema": parsed.schema,
            "analysis": parsed.analysis,
            "summary": parsed.summary,
            "decisions": [
                {
                    "call_id": d.call_id,
                    "name": d.name,
                    "action": d.action,
                    "reason": d.reason,
                    "has_pruned_observation": d.pruned_observation is not None,
                }
                for d in parsed.decisions
            ],
            "raw_llm_output": raw,
            "prompt": compressor_user_prompt,
        }

    # [TRACE 4] 保存返回给客户端的完整内容
    await _save_trace_async(trace_dir, "4_returned_to_client.json", _ok(response))

    log.info(
        "compact: done elapsed=%ss out_msgs=%s structured=%s schema=%s trace=%s",
        elapsed,
        len(result["messages"]),
        has_structured,
        parsed.schema,
        trace_dir.name,
    )
    return _ok(response)


def main() -> None:
    import uvicorn

    host = os.environ.get("COMPACTION_HOST", "0.0.0.0")
    port = int(os.environ.get("COMPACTION_PORT", "12003"))
    log.info("Uvicorn bind %s:%s", host, port)
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
