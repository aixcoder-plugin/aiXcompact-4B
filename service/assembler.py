from __future__ import annotations

import re
from typing import Any

from . import config
from . import token_count
from .parser import ParseResult, PruneDecision
from .prompt import collect_tools
from .messages import normalize_tool_results

_PRUNING_BLOCK_RE = re.compile(
    r"<(?:tool_pruning|aix_pruning|aix_tool_pruning|aix_decisions)>[\s\S]*?</(?:tool_pruning|aix_pruning|aix_tool_pruning|aix_decisions)>"
)
_DANGLING_TEXT_MAX_CHARS = 150
_CASCADE_ACK_MAX_CHARS = 80
_DANGLING_ACTION_RE = re.compile(
    r"^(?:"
    r"(?:I will|I'll|Let's|Let me|Now|First|Next|Checking|Searching|Looking)\b.*"
    r"\b(?:check|search|look|see|find|run|execute|read|explore|review|inspect)\b"
    r"|Now let me\b.*"
    r"|Let me also\b.*"
    r"|Let me check\b.*"
    r"|Let me see\b.*"
    r"|Let me look\b.*"
    r"|Let me search\b.*"
    r")",
    re.IGNORECASE,
)
_CASCADE_ACK_RE = re.compile(
    r"^(?:ok(?:ay)?|got it|sure|alright|understood|makes sense|sounds good|noted|i see)"
    r"(?:[.!])?$",
    re.IGNORECASE,
)
_STATUS_SIGNAL_RE = re.compile(
    r"\b(?:"
    r"task(?: is)? not complete|not complete|passed|failed|error|errors|issue|problem|"
    r"summary|phase|results? show|remaining|next steps?|blocker|verified|verification|"
    r"reproduced|fix works|implementation appears|core implementation"
    r")\b",
    re.IGNORECASE,
)
_SUMMARY_PREFIX = (
    "本会话是从之前因上下文耗尽而中断的对话继续而来，"
    "下面的摘要涵盖了该对话较早部分的内容。"
)


def _strip_tool_pruning_text(text: str) -> str:
    result = _PRUNING_BLOCK_RE.sub("", text)
    result = re.sub(r"\n{3,}", "\n\n", result)
    return result.strip()


def _normalize_text(text: str) -> str:
    return " ".join(text.split()).strip()


def _is_text_item(item: Any) -> bool:
    return (
        isinstance(item, dict)
        and item.get("type") == "text"
        and isinstance(item.get("text"), str)
    )


def _is_primary_dangling_text(text: str) -> bool:
    normalized = _normalize_text(text)
    if not normalized:
        return False
    if len(normalized) > _DANGLING_TEXT_MAX_CHARS:
        return False
    if _STATUS_SIGNAL_RE.search(normalized):
        return False
    if normalized.endswith(":"):
        return True
    return bool(_DANGLING_ACTION_RE.match(normalized))


def _is_cascade_ack_text(text: str) -> bool:
    normalized = _normalize_text(text)
    if not normalized:
        return False
    if len(normalized) > _CASCADE_ACK_MAX_CHARS:
        return False
    if _STATUS_SIGNAL_RE.search(normalized):
        return False
    return bool(_CASCADE_ACK_RE.match(normalized))


def _remove_dangling_prefix_text(
    kept: list[Any],
) -> tuple[int, int]:
    """Remove short dangling text blocks immediately before a removed tool call."""
    removed = 0
    candidates = 0
    if not kept or not _is_text_item(kept[-1]):
        return removed, candidates

    if _is_primary_dangling_text(kept[-1]["text"]):
        candidates += 1
        kept.pop()
        removed += 1
    else:
        return removed, candidates

    while kept and _is_text_item(kept[-1]):
        if not _is_cascade_ack_text(kept[-1]["text"]):
            break
        candidates += 1
        kept.pop()
        removed += 1

    return removed, candidates


def find_dangling_texts(messages: list[dict]) -> list[dict[str, Any]]:
    """Find likely dangling transition texts left in assembled output."""
    findings: list[dict[str, Any]] = []
    for msg_idx, msg in enumerate(messages):
        if msg.get("role") != "assistant":
            continue
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for item_idx, item in enumerate(content):
            if not _is_text_item(item):
                continue
            text = item["text"]
            if not _is_primary_dangling_text(text):
                continue
            next_item = content[item_idx + 1] if item_idx + 1 < len(content) else None
            if isinstance(next_item, dict) and next_item.get("type") == "tool-call":
                continue
            findings.append(
                {
                    "message_index": msg_idx,
                    "item_index": item_idx,
                    "text": _normalize_text(text),
                }
            )
    return findings


def _strip_tool_pruning_from_post(msgs: list[dict]) -> list[dict]:
    """Remove <tool_pruning> blocks from assistant text (mirrors TS version)."""
    out = []
    for msg in msgs:
        if msg.get("role") != "assistant":
            out.append(msg)
            continue
        c = msg.get("content")
        if isinstance(c, str):
            cleaned = _strip_tool_pruning_text(c)
            out.append({**msg, "content": cleaned} if cleaned != c else msg)
            continue
        if not isinstance(c, list):
            out.append(msg)
            continue
        changed = False
        items = []
        for item in c:
            if not isinstance(item, dict):
                items.append(item)
                continue
            if item.get("type") == "text" and isinstance(item.get("text"), str):
                cleaned = _strip_tool_pruning_text(item["text"])
                if cleaned != item["text"]:
                    changed = True
                    items.append({**item, "text": cleaned})
                else:
                    items.append(item)
            else:
                items.append(item)
        out.append({**msg, "content": items} if changed else msg)
    return out


def _resolve_call_id(decision: PruneDecision, tools: list[dict]) -> str | None:
    """Resolve a decision's call_id against the actual tool list."""
    for t in tools:
        if t["callID"] == decision.call_id:
            return decision.call_id

    m = re.match(r"^call_(\d+)$", decision.call_id)
    if m:
        idx = int(m.group(1)) - 1
        if 0 <= idx < len(tools):
            return tools[idx]["callID"]

    return None


def _has_tool_call(content: Any) -> bool:
    return isinstance(content, list) and any(
        isinstance(item, dict) and item.get("type") == "tool-call" for item in content
    )


def _global_id_for_message(msg: dict, fallback: int) -> str:
    return str(msg.get("global_id") or fallback)


def _replace_tool_results(msg: dict, replacement: str | None) -> dict:
    if replacement is None:
        return msg
    content = msg.get("content")
    if not isinstance(content, list):
        return msg
    changed = False
    items: list[Any] = []
    for item in content:
        if not isinstance(item, dict) or item.get("type") != "tool-result":
            items.append(item)
            continue
        changed = True
        items.append(
            {
                **item,
                "output": {
                    "type": "text",
                    "value": replacement,
                },
            }
        )
    return {**msg, "content": items} if changed else msg


def _summary_message(summary_text: str) -> dict[str, Any]:
    return {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": f"{_SUMMARY_PREFIX}\n\n{summary_text}",
            }
        ],
    }


def _assemble_aix(pre: list[dict], parsed: ParseResult) -> dict:
    """Assemble output for the global_id-based aix schema."""
    pre = normalize_tool_results(pre)
    pick = {d.call_id: d for d in parsed.decisions}
    kept_messages: list[dict] = []
    fallback_global_id = 0
    removed_items = 0
    modified_items = 0
    missing_decisions = 0
    idx = 0

    while idx < len(pre):
        msg = pre[idx]
        if msg.get("role") == "system":
            idx += 1
            continue

        fallback_global_id += 1
        global_id = _global_id_for_message(msg, fallback_global_id)

        content = msg.get("content")
        if msg.get("role") == "assistant" and _has_tool_call(content):
            tool_calls = [
                item
                for item in (content if isinstance(content, list) else [])
                if isinstance(item, dict) and item.get("type") == "tool-call"
            ]
            result_msg = (
                pre[idx + 1]
                if idx + 1 < len(pre) and pre[idx + 1].get("role") == "tool"
                else None
            )

            per_call: list[tuple[str, str, PruneDecision | None]] = []
            for tc_idx, tc in enumerate(tool_calls):
                if tc_idx > 0:
                    fallback_global_id += 1
                gid = global_id if tc_idx == 0 else str(fallback_global_id)
                tc_call_id = str(tc.get("toolCallId", ""))
                d = pick.get(gid)
                if d is None:
                    missing_decisions += 1
                per_call.append((gid, tc_call_id, d))

            all_remove = len(per_call) > 0 and all(
                d is not None and d.action == "remove_all" for _, _, d in per_call
            )

            if all_remove:
                removed_items += len(per_call)
            else:
                remove_ids: set[str] = set()
                replace_map: dict[str, str] = {}
                for _, tc_call_id, d in per_call:
                    if d is not None and d.action == "remove_all":
                        remove_ids.add(tc_call_id)
                        removed_items += 1
                    elif (
                        d is not None and d.action == "pruner" and d.pruned_observation
                    ):
                        replace_map[tc_call_id] = d.pruned_observation
                        modified_items += 1

                if remove_ids:
                    new_asst_content = [
                        item
                        for item in (
                            content if isinstance(content, list) else [content]
                        )
                        if not (
                            isinstance(item, dict)
                            and item.get("type") == "tool-call"
                            and str(item.get("toolCallId", "")) in remove_ids
                        )
                    ]
                    kept_messages.append({**msg, "content": new_asst_content})
                else:
                    kept_messages.append(msg)

                if result_msg is not None:
                    r_content = result_msg.get("content")
                    if isinstance(r_content, list) and (remove_ids or replace_map):
                        new_r_content: list[Any] = []
                        for item in r_content:
                            if (
                                not isinstance(item, dict)
                                or item.get("type") != "tool-result"
                            ):
                                new_r_content.append(item)
                                continue
                            tc_id = str(item.get("toolCallId", ""))
                            if tc_id in remove_ids:
                                continue
                            if tc_id in replace_map:
                                new_r_content.append(
                                    {
                                        **item,
                                        "output": {
                                            "type": "text",
                                            "value": replace_map[tc_id],
                                        },
                                    }
                                )
                            else:
                                new_r_content.append(item)
                        if new_r_content:
                            kept_messages.append(
                                {**result_msg, "content": new_r_content}
                            )
                    else:
                        kept_messages.append(result_msg)

            idx += 2 if result_msg is not None else 1
            continue

        decision = pick.get(global_id)
        if decision is None:
            missing_decisions += 1
        action = decision.action if decision else "keep_all"
        if action == "remove_all":
            removed_items += 1
        else:
            kept_messages.append(msg)
        idx += 1

    summary_text = parsed.summary or parsed.raw
    result = [
        _summary_message(summary_text),
        *kept_messages,
    ]

    stats = {
        "total_items": fallback_global_id,
        "total_tools": len(collect_tools(pre)),
        "decisions": len(parsed.decisions),
        "keep_all": sum(1 for d in parsed.decisions if d.action == "keep_all"),
        "pruner": sum(1 for d in parsed.decisions if d.action == "pruner"),
        "remove_all": sum(1 for d in parsed.decisions if d.action == "remove_all"),
        "keep": sum(1 for d in parsed.decisions if d.action == "keep_all"),
        "keep_with_replaced_tool_result": sum(
            1 for d in parsed.decisions if d.action == "pruner"
        ),
        "remove": sum(1 for d in parsed.decisions if d.action == "remove_all"),
        "removed_items": removed_items,
        "modified_items": modified_items,
        "missing_item_decisions": missing_decisions,
        "summary_fallback_raw": parsed.summary is None,
        "schema": "aix",
    }
    return {"messages": result, "stats": stats}


def _strip_global_ids(messages: list[dict]) -> list[dict]:
    """Do not carry per-compaction decision IDs into the next trajectory."""
    return [
        {key: value for key, value in message.items() if key != "global_id"}
        for message in messages
    ]


def assemble(
    pre: list[dict],
    parsed: ParseResult,
    messages: list[dict],
) -> dict:
    """Assemble compressed output.

    Args:
        pre: original model messages (before compaction)
        parsed: parsed LLM output with decisions
        messages: original input messages for tool collection

    Returns:
        dict with 'messages' (compressed) and 'stats'.
    """
    if getattr(parsed, "schema", "legacy") == "aix":
        result = _assemble_aix(pre, parsed)
    else:
        result = _assemble_legacy(pre, parsed, messages)
    result["messages"] = _strip_global_ids(result["messages"])
    return result


def _iter_units(pre: list[dict]) -> list[list[dict]]:
    """Group messages into ordered units, keeping an assistant tool-call message
    paired with its following tool-result message so the pair is never split."""
    pre = normalize_tool_results(pre)
    units: list[list[dict]] = []
    idx = 0
    while idx < len(pre):
        msg = pre[idx]
        if msg.get("role") == "system":
            idx += 1
            continue
        content = msg.get("content")
        if (
            msg.get("role") == "assistant"
            and _has_tool_call(content)
            and idx + 1 < len(pre)
            and pre[idx + 1].get("role") == "tool"
        ):
            units.append([msg, pre[idx + 1]])
            idx += 2
        else:
            units.append([msg])
            idx += 1
    return units


def build_retention_fallback(
    pre: list[dict],
    parsed: ParseResult,
    input_tokens: int,
    target_ratio_pct: int,
) -> dict:
    """Deterministic fallback used when retention leaves the valid band —
    under-compression (retention >= R1/max_retention) or over-compression
    (retention < R0/min_retention). Discards the model's per-item decisions and
    rebuilds the output as the model summary plus the most-recent items that
    fit within an R2 token budget (``target_ratio_pct`` percent of
    ``input_tokens``)."""
    has_real_summary = parsed.summary is not None
    summary_text = parsed.summary if has_real_summary else parsed.raw
    budget = round(input_tokens * target_ratio_pct / 100)
    summary_msg = _summary_message(summary_text)
    summary_tokens = token_count.count_message_tokens(summary_msg)

    # A missing <aix_summary> falls back to the full raw model output, which can
    # be arbitrarily large and would blow the budget (defeating the guarantee).
    # Truncate that (non-summary) text to fit. A real <aix_summary> is kept whole.
    summary_truncated = False
    if summary_tokens > budget and not has_real_summary:
        suffix = "\n…[truncated to fit compression budget]"
        # reserve room for the prefix, the suffix, the "\n\n" joiner, and a small
        # safety margin to absorb ceil() rounding in the token estimate.
        reserve = len(_SUMMARY_PREFIX) + len(suffix) + 2 + 16
        max_chars = max(0, int(budget * config.CHARS_PER_TOKEN) - reserve)
        summary_text = summary_text[:max_chars] + suffix
        summary_msg = _summary_message(summary_text)
        summary_tokens = token_count.count_message_tokens(summary_msg)
        summary_truncated = True

    units = _iter_units(pre)
    kept_units: list[list[dict]] = []
    used = summary_tokens
    summary_over_budget = summary_tokens >= budget
    if not summary_over_budget:
        for unit in reversed(units):
            unit_tokens = sum(token_count.count_message_tokens(m) for m in unit)
            if used + unit_tokens > budget:
                break
            kept_units.append(unit)
            used += unit_tokens
    kept_units.reverse()
    kept_messages = [m for unit in kept_units for m in unit]

    result = _strip_global_ids([summary_msg, *kept_messages])
    stats = {
        "schema": "retention_fallback",
        "total_items": len(units),
        "total_tools": len(collect_tools(pre)),
        "decisions": len(parsed.decisions),
        "keep_all": 0,
        "pruner": 0,
        "remove_all": 0,
        "keep": 0,
        "keep_with_replaced_tool_result": 0,
        "remove": 0,
        "removed_items": len(units) - len(kept_units),
        "modified_items": 0,
        "missing_item_decisions": 0,
        "summary_fallback_raw": parsed.summary is None,
        "fallback_kept_items": len(kept_units),
        "budget_tokens": budget,
        "summary_over_budget": summary_over_budget,
        "summary_truncated": summary_truncated,
    }
    return {"messages": result, "stats": stats}


def _assemble_legacy(
    pre: list[dict],
    parsed: ParseResult,
    messages: list[dict],
) -> dict:

    tools = collect_tools(messages)

    pick: dict[str, PruneDecision] = {}
    for d in parsed.decisions:
        real = _resolve_call_id(d, tools)
        if real:
            pick[real] = d

    dangling_removed = 0
    dangling_candidates = 0

    tool_msgs: list[dict] = []
    for msg in pre:
        content = msg.get("content")
        if not isinstance(content, list):
            continue

        kept: list[Any] = []
        for item in content:
            if not isinstance(item, dict):
                kept.append(item)
                continue
            cid = item.get("toolCallId")
            if not cid:
                kept.append(item)
                continue

            d = pick.get(cid)
            if d is None:
                continue

            if d.action == "remove_all":
                removed, candidates = _remove_dangling_prefix_text(kept)
                dangling_removed += removed
                dangling_candidates += candidates
                continue

            if item.get("type") == "tool-call":
                kept.append(item)
                continue
            if item.get("type") != "tool-result":
                kept.append(item)
                continue

            if d.action == "pruner" and d.pruned_observation:
                kept.append(
                    {
                        **item,
                        "output": {
                            "type": "text",
                            "value": d.pruned_observation,
                        },
                    }
                )
            else:
                kept.append(item)

        has_tool = any(
            isinstance(k, dict) and k.get("type") in ("tool-call", "tool-result")
            for k in kept
        )
        if has_tool:
            tool_msgs.append({**msg, "content": kept})

    summary_text = parsed.summary or parsed.raw
    post: list[dict] = [
        _summary_message(summary_text),
    ]

    result = [*post, *tool_msgs]

    stats = {
        "total_tools": len(tools),
        "decisions": len(parsed.decisions),
        "keep_all": sum(1 for d in parsed.decisions if d.action == "keep_all"),
        "pruner": sum(1 for d in parsed.decisions if d.action == "pruner"),
        "remove_all": sum(1 for d in parsed.decisions if d.action == "remove_all"),
        "summary_fallback_raw": parsed.summary is None,
        "dangling_text_removed": dangling_removed,
        "dangling_text_candidates": dangling_candidates,
    }

    return {"messages": result, "stats": stats}
