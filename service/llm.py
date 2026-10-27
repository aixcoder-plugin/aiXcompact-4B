from __future__ import annotations

import json
import logging
import os

import httpx

from .config import (
    LLM_CONNECT_TIMEOUT,
    LLM_ENDPOINT,
    LLM_MAX_RETRIES,
    LLM_MAX_TOKENS,
    LLM_MODEL,
    LLM_TEMPERATURE,
    LLM_TIMEOUT,
    LLM_TOKEN,
    LLM_TOP_P,
)
from .prompt import SYSTEM_PROMPT

log = logging.getLogger("compaction.llm")


class LLMError(Exception):
    """LLM call failure. ``retryable`` marks transient errors (5xx / 429 /
    empty content) worth another attempt; 4xx responses are not retried."""

    def __init__(self, message: str, *, retryable: bool = True):
        super().__init__(message)
        self.retryable = retryable


def _client_modes() -> list[tuple[str, dict]]:
    """Prefer direct requests, but fall back to env proxy when available."""
    t = httpx.Timeout(LLM_TIMEOUT, connect=LLM_CONNECT_TIMEOUT)
    proxy = os.environ.get("LLM_HTTP_PROXY")
    trust = os.environ.get("LLM_TRUST_ENV", "").lower() in ("1", "true", "yes")
    if trust:
        return [("env-proxy", {"timeout": t, "trust_env": True})]

    modes: list[tuple[str, dict]] = [
        ("direct", {"timeout": t, "proxy": None, "trust_env": False})
    ]
    if proxy:
        modes.append(
            ("env-proxy-fallback", {"timeout": t, "proxy": proxy, "trust_env": False})
        )
    return modes


def build_api_messages(compressor_user_prompt: str) -> list[dict[str, str]]:
    """Wrap the fully rendered compressor user prompt into the chat shape.

    The caller renders the trajectory XML / user prompt exactly once (it is
    expensive) and passes the same message list to trace logging and call().
    """
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": compressor_user_prompt},
    ]


async def _stream_once(client_kwargs: dict, payload: dict) -> str:
    """Make one streaming request and return the accumulated content.

    Closing the streaming connection on cancellation releases the vLLM request.
    """
    chunks: list[str] = []
    finished = False
    async with httpx.AsyncClient(**client_kwargs) as client:
        async with client.stream(
            "POST",
            LLM_ENDPOINT,
            json=payload,
            headers={
                "Content-Type": "application/json",
                **({"Authorization": f"Bearer {LLM_TOKEN}"} if LLM_TOKEN else {}),
            },
        ) as resp:
            if resp.status_code != 200:
                body = await resp.aread()
                raise LLMError(
                    f"LLM returned status {resp.status_code}: {body[:800]!r}",
                    retryable=resp.status_code >= 500 or resp.status_code == 429,
                )
            async for line in resp.aiter_lines():
                if not line or not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    obj = json.loads(data)
                except ValueError:
                    continue
                for ch in obj.get("choices", []):
                    if ch.get("finish_reason") == "length":
                        raise LLMError(
                            "Model output truncated at token limit", retryable=False
                        )
                    if ch.get("finish_reason") is not None:
                        finished = True
                    delta = ch.get("delta") or {}
                    piece = delta.get("content")
                    if piece:
                        chunks.append(piece)
    text = "".join(chunks)
    if not finished:
        raise LLMError("Model stream ended before completion")
    if not text:
        raise LLMError("LLM returned empty content")
    return text


async def call(api_messages: list[dict[str, str]]) -> str:
    """Call the LLM with prebuilt chat messages and return the raw text."""
    if not LLM_ENDPOINT:
        raise LLMError("LLM_ENDPOINT is not set", retryable=False)

    payload = {
        "model": LLM_MODEL,
        "messages": api_messages,
        "stream": True,
        "max_tokens": LLM_MAX_TOKENS,
        "temperature": LLM_TEMPERATURE,
        "top_p": LLM_TOP_P,
        "chat_template_kwargs": {"enable_thinking": False},
    }

    last_err: Exception | None = None
    modes = _client_modes()
    total = (LLM_MAX_RETRIES + 1) * len(modes)
    seq = 0
    for name, kwargs in modes:
        for attempt in range(1, LLM_MAX_RETRIES + 2):
            seq += 1
            try:
                text = await _stream_once(kwargs, payload)
                log.info(
                    "LLM call succeeded mode=%s attempt=%s/%s",
                    name,
                    seq,
                    total,
                )
                return text

            except (httpx.HTTPError, LLMError) as e:
                if isinstance(e, LLMError) and not e.retryable:
                    raise
                last_err = e
                log.warning(
                    "LLM transient error mode=%s attempt %s/%s (%s/%s in mode) %s: %s",
                    name,
                    seq,
                    total,
                    attempt,
                    LLM_MAX_RETRIES + 1,
                    type(e).__name__,
                    e,
                )

    raise LLMError(
        f"LLM call failed after {total} attempts: {type(last_err).__name__}: {last_err}"
    )
