"""Token counting for prompt budgets and retention using the local model tokenizer."""

from __future__ import annotations

import json
from typing import Any

from . import prompt


def count_text_tokens(text: str) -> int:
    """Use the cached model tokenizer; never mix remote counts and estimates."""
    return prompt.model_token_length(text)


def _message_text(msg: Any) -> str:
    """Extract the visible text a successor agent would pay tokens for."""
    if not isinstance(msg, dict):
        return ""
    content = msg.get("content")
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for item in content:
        if not isinstance(item, dict):
            parts.append(str(item))
            continue
        itype = item.get("type")
        if itype == "text":
            parts.append(str(item.get("text", "")))
        elif itype == "tool-call":
            parts.append(str(item.get("toolName", "")))
            inp = item.get("input")
            parts.append(
                inp if isinstance(inp, str) else json.dumps(inp, ensure_ascii=False)
            )
        elif itype == "tool-result":
            out = item.get("output")
            if isinstance(out, dict):
                parts.append(str(out.get("value", "")))
            else:
                parts.append(str(out))
        else:
            parts.append(json.dumps(item, ensure_ascii=False))
    return "\n".join(parts)


def count_message_tokens(msg: Any) -> int:
    return count_text_tokens(_message_text(msg))


def count_messages_tokens(messages: list) -> int:
    """Count joined visible text with the same local tokenizer as every budget."""
    if not isinstance(messages, list):
        return 0
    joined = "\n".join(_message_text(m) for m in messages)
    return count_text_tokens(joined)
