"""Shared message grouping for rendering, decisions and retention fallback."""


def tool_calls(message: dict) -> list[dict]:
    content = message.get("content")
    if message.get("role") != "assistant" or not isinstance(content, list):
        return []
    return [
        item
        for item in content
        if isinstance(item, dict) and item.get("type") == "tool-call"
    ]


def normalize_tool_results(messages: list[dict]) -> list[dict]:
    """Join contiguous result messages so every call group has one result group.

    The validator checks IDs and adjacency first. Payloads are preserved, and
    the caller's messages/content arrays are never modified.
    """
    normalized = []
    index = 0
    while index < len(messages):
        message = messages[index]
        normalized.append(message)
        index += 1
        if not tool_calls(message):
            continue
        results = []
        while index < len(messages) and messages[index].get("role") == "tool":
            results.append(messages[index])
            index += 1
        if len(results) == 1:
            normalized.append(results[0])
        elif results:
            normalized.append(
                {
                    **results[0],
                    "content": [
                        item for result in results for item in result["content"]
                    ],
                }
            )
    return normalized


def trajectory_item_types(messages: list[dict]) -> dict[str, bool]:
    """Map each rendered global ID to whether it represents a tool pair."""
    messages = normalize_tool_results(messages)
    items = {}
    counter = 0
    index = 0
    while index < len(messages):
        message = messages[index]
        index += 1
        if message.get("role") == "system":
            continue
        calls = tool_calls(message)
        counter += 1
        first_id = str(message.get("global_id") or counter)
        for offset in range(max(1, len(calls))):
            if offset:
                counter += 1
            gid = first_id if offset == 0 else str(counter)
            if gid in items:
                raise ValueError(f"duplicate rendered global_id: {gid}")
            items[gid] = bool(calls)
        if calls and index < len(messages) and messages[index].get("role") == "tool":
            index += 1
    return items
