from __future__ import annotations

import json
from typing import Any

from .config import CHARS_PER_TOKEN, TOKEN_WARN_THRESHOLD
from .messages import trajectory_item_types

VALID_ROLES = {"user", "assistant", "tool"}
VALID_CONTENT_TYPES = {"text", "tool-call", "tool-result", "reasoning"}


class ValidationError(Exception):
    def __init__(self, errors: list[str]):
        self.errors = errors
        super().__init__("; ".join(errors))


def _estimate_tokens(messages: list[dict]) -> int:
    raw = json.dumps(messages, ensure_ascii=False)
    return int(len(raw) / CHARS_PER_TOKEN)


def _validate_item(item: Any, idx: int, role: str) -> list[str]:
    errors: list[str] = []
    if not isinstance(item, dict):
        errors.append(f"messages[{idx}].content item is not an object")
        return errors

    t = item.get("type")
    if not isinstance(t, str) or t not in VALID_CONTENT_TYPES:
        errors.append(
            f"messages[{idx}] content item has invalid type '{t}', "
            f"expected one of {VALID_CONTENT_TYPES}"
        )
        return errors

    if t in {"text", "reasoning"} and not isinstance(item.get("text"), str):
        errors.append(f"messages[{idx}] {t} content must have a string 'text'")

    if t == "tool-call":
        if role != "assistant":
            errors.append(f"messages[{idx}] tool-call must belong to an assistant")
        for field in ("toolCallId", "toolName"):
            if not isinstance(item.get(field), str) or not item[field].strip():
                errors.append(
                    f"messages[{idx}] tool-call '{field}' must be a non-empty string"
                )

    if t == "tool-result":
        if role != "tool":
            errors.append(f"messages[{idx}] tool-result must belong to a tool message")
        for field in ("toolCallId", "toolName"):
            if not isinstance(item.get(field), str) or not item[field].strip():
                errors.append(
                    f"messages[{idx}] tool-result '{field}' must be a non-empty string"
                )
        if "output" not in item:
            errors.append(f"messages[{idx}] tool-result missing 'output'")

    return errors


def validate(messages: list[dict]) -> dict:
    """Validate input messages. Returns dict with warnings and stats."""
    errors: list[str] = []
    warnings: list[str] = []

    if not isinstance(messages, list) or len(messages) == 0:
        raise ValidationError(["messages must be a non-empty array"])

    has_user = False
    has_assistant = False
    call_ids: set[str] = set()
    result_ids: set[str] = set()
    active_calls: dict[str, str] = {}

    for i, msg in enumerate(messages):
        if not isinstance(msg, dict):
            errors.append(f"messages[{i}] is not an object")
            continue

        role = msg.get("role")
        if not isinstance(role, str) or role not in VALID_ROLES:
            errors.append(
                f"messages[{i}] has invalid role '{role}', "
                f"expected one of {VALID_ROLES}"
            )
            continue

        if role != "tool":
            active_calls = {}
        if "global_id" in msg:
            gid = msg["global_id"]
            if not (
                (type(gid) is int and gid > 0) or (isinstance(gid, str) and gid.strip())
            ):
                errors.append(
                    f"messages[{i}].global_id must be a positive integer or non-empty string"
                )

        if role == "user":
            has_user = True
        if role == "assistant":
            has_assistant = True

        content = msg.get("content")
        if content is None:
            errors.append(f"messages[{i}] missing 'content'")
            continue

        if isinstance(content, str):
            if role == "tool":
                errors.append(
                    f"messages[{i}] tool content must be an array of tool-result objects"
                )
            continue

        if not isinstance(content, list):
            errors.append(f"messages[{i}].content must be a string or array")
            continue

        for item in content:
            item_errors = _validate_item(item, i, role)
            errors.extend(item_errors)
            if item_errors:
                continue
            if isinstance(item, dict):
                cid = item.get("toolCallId")
                if item.get("type") == "tool-call" and cid:
                    if cid in call_ids:
                        errors.append(f"duplicate tool-call id: {cid}")
                    call_ids.add(cid)
                    active_calls[cid] = item["toolName"]
                if item.get("type") == "tool-result" and cid:
                    if cid in result_ids:
                        errors.append(f"duplicate tool-result id: {cid}")
                    if active_calls.get(cid) != item["toolName"]:
                        errors.append(
                            f"tool-result {cid} must match a preceding contiguous call group and tool name"
                        )
                    else:
                        active_calls.pop(cid)
                    result_ids.add(cid)
                if role == "tool" and item.get("type") != "tool-result":
                    errors.append(
                        f"messages[{i}] tool content may only contain tool-result objects"
                    )

    if not has_user:
        errors.append("messages must contain at least one user message")
    if not has_assistant:
        errors.append("messages must contain at least one assistant message")

    if errors:
        raise ValidationError(errors)

    try:
        trajectory_item_types(messages)
    except ValueError as error:
        raise ValidationError([str(error)]) from error

    orphans = call_ids - result_ids
    if orphans:
        warnings.append(
            f"tool-call ids without matching tool-result: {sorted(orphans)}"
        )

    tokens = _estimate_tokens(messages)
    if tokens > TOKEN_WARN_THRESHOLD:
        warnings.append(
            f"estimated input tokens ({tokens}) exceeds warning threshold "
            f"({TOKEN_WARN_THRESHOLD})"
        )

    return {
        "message_count": len(messages),
        "estimated_tokens": tokens,
        "tool_calls": len(call_ids),
        "tool_results": len(result_ids),
        "warnings": warnings,
    }
