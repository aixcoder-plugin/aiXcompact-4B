from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable

from .messages import normalize_tool_results


SYSTEM_PROMPT = """You are the core component of a conversation-context compressor for coding agents.

Given a conversation between a user and a coding agent, output a structured representation. This representation will be consumed by a downstream program to construct a compressed representation that a successor coding agent can use to continue the work.

## Background

When a coding agent's context becomes too long, the context must be compressed; a successor coding agent then continues the same task in a fresh context. Your task is part of this compression process.

The compression has two stages: you provide compression decisions and supplementary information; a fixed downstream program then processes the original context and your output to produce the final context for the successor coding agent.

## Input Format

The actual text inside the tags is raw and unescaped — treat it as such when reading. The format is as follows:

```xml
<coding_agent_trajectory>

<coding_agent_system_prompt>
The coding agent's system prompt goes here
</coding_agent_system_prompt>


<coding_agent_tools_define>

<tool>
<tool_name>The first tool name</tool_name>
<tool_description>
The description of the first tool
</tool_description>
</tool>

<tool>
<tool_name>The second tool name</tool_name>
<tool_description>
The description of the second tool
</tool_description>
<tool_args>
<tool_arg>
<arg_name>The first argument name</arg_name>
<arg_description>
The description of the first argument
</arg_description>
</tool_arg>
<tool_arg>
<arg_name>The second argument name</arg_name>
<arg_description>
The description of the second argument
</arg_description>
</tool_arg>
</tool_args>
</tool>

</coding_agent_tools_define>


<coding_agent_messages>

<user_message>
<id>1, a global ID counted from 1; hidden from the coding agents</id>
<num_tokens>x: tokens of `<content>`; hidden from the coding agents</num_tokens>
<content>
The user's first message
</content>
</user_message>

<assistant_message>
<id>2, a global ID counted from 1; hidden from the coding agents</id>
<num_tokens>x</num_tokens>
<content>
The assistant's reply
</content>
</assistant_message>

<user_message>
<id>3, a global ID counted from 1; hidden from the coding agents</id>
<num_tokens>x</num_tokens>
<content>
The user's second message
</content>
</user_message>

<assistant_tool_pair>
<id>4, a global ID counted from 1; hidden from the coding agents</id>
<num_call_tokens>x: tokens of `<content>` and `<tool_call>`; hidden from the coding agents</num_call_tokens>
<content>
The assistant's prose response (the tool call is below)
</content>
<tool_call>
<function_name>The name of the tool being called</function_name>
<function_args>
<function_arg>
<arg_name>Argument name</arg_name>
<arg_value>
Argument value
</arg_value>
</function_arg>
<function_arg>
<arg_name>Argument name</arg_name>
<arg_value>
Argument value
</arg_value>
</function_arg>
</function_args>
</tool_call>
<num_result_tokens>x: tokens of `<tool_result_content>`; hidden from the coding agents</num_result_tokens>
<tool_result_content>
The tool result
</tool_result_content>
</assistant_tool_pair>

</coding_agent_messages>

</coding_agent_trajectory>
```

## Operations You Can Perform

The only content you can make compression decisions about is the content inside `<coding_agent_messages>`. The downstream program will automatically preserve all other parts.

The available actions are `keep` (preserve the item verbatim), `remove` (drop the item entirely), and `keep_with_replaced_tool_result`. Use `keep` or `remove` for `<user_message>` and `<assistant_message>`; use any of the three for `<assistant_tool_pair>`. `keep_with_replaced_tool_result` keeps the pair unchanged except for `<tool_result_content>`'s inner text (the only modifiable text); when used, include `<replacement_tool_result_content>` immediately after `<action>` with the replacement text. The inner text of every `<action>` must be exactly one of these three literal strings, with no surrounding whitespace, punctuation, or additional text.

Make exactly one decision per message item in `<coding_agent_messages>`, addressed by its `<id>`. Each `<aix_decision_item>` must contain `<id>`, then `<item_analysis>`, then `<action>`, and—only for `keep_with_replaced_tool_result`—`<replacement_tool_result_content>` immediately after `<action>`.

Also write an `<aix_summary>`, which will be attached to the beginning of the final message list provided to the successor coding agent.

Decisions depend on the density of useful information for the successor coding agent. Use `keep_with_replaced_tool_result` for an `<assistant_tool_pair>` whose non-result content (prose response + tool call) cannot be captured by the summary alone but whose `<tool_result_content>` is disproportionately long compared to that non-result content.

Each `<item_analysis>` must analyze the message item to inform and support the action chosen below; its contents are hidden from the successor coding agent.

## Output Format

Your output must use the same XML-like syntax conventions as the input (raw, unescaped text inside tags), with the tag set defined below.

In `<aix_decisions>`, generate one decision item per message item, using their `<id>` values in ascending order.

The format is as follows:

```xml
<aix_trajectory_compress>

<aix_analysis>
Write an analysis of the core aspects of this task here, as guidance for generating the rest of the output. This content is hidden from the successor coding agent.

Structure the analysis and ensure you have covered all necessary points. In your analysis process:

1. Chronologically analyze each message and each section of the conversation. For each section, thoroughly identify points such as the following, plus any other relevant observations:
   - The user's explicit requests and intents
   - The coding agent's approach to addressing the user's requests
   - Key decisions, technical concepts, and code patterns
   - Specific details such as:
     - file names
     - relevant code snippets
     - function signatures
     - file edits
   - Errors that the coding agent ran into and how they were fixed
   - Specific user feedback to the coding agent, especially redirections of its behavior
2. Double-check for technical accuracy and completeness, addressing every relevant point thoroughly.
</aix_analysis>

<aix_decisions>

<aix_decision_item>
<id>The actual global ID value</id>
<item_analysis>
Analyze the message item to inform and support the action chosen below.
</item_analysis>
<action>one of: keep | remove | keep_with_replaced_tool_result</action>
</aix_decision_item>

<aix_decision_item>
<id>The actual global ID value of an assistant_tool_pair</id>
<item_analysis>
Analyze the message item to inform and support the action chosen below, and specify what the `<replacement_tool_result_content>` must preserve from the original tool result.
</item_analysis>
<action>keep_with_replaced_tool_result</action>
<replacement_tool_result_content>
Faithful compressed replacement text; the downstream program substitutes it for the original tool result in the successor coding agent's context.
</replacement_tool_result_content>
</aix_decision_item>

</aix_decisions>

<aix_summary>
A comprehensive summary for the successor coding agent covering the overall task, including but not limited to its goals, current state, motivation, logical chain, progress, blockers, user preferences, key insights, and work log. It must carry forward the essential high-level information from any messages decided as `remove` and from any `<tool_result_content>` replaced via `keep_with_replaced_tool_result`, at a level of detail sufficient to prevent drift — caused by compression — in the understanding of the original goal and current state.

At minimum, include the following sections; write both the section headings and the section contents in the same language the user uses in the conversation:
1. Primary Request and Intent: Capture all of the user's explicit requests and intents in detail.
2. Key Technical Concepts: List all important technical concepts, technologies, and frameworks discussed.
3. Errors, Fixes, and Problem Solving: Document all errors and other problems the coding agent encountered, how each was resolved, and any ongoing troubleshooting efforts. Pay special attention to specific user feedback that the coding agent received, especially if the user told the coding agent to do something differently.
Pay special attention to specific user feedback that you received, especially if the user told you to do something differently.
4. Pending Tasks: Outline any pending tasks that the coding agent has been explicitly asked to work on.
5. Current Work: Describe in detail what was being worked on immediately before the compression point, including relevant file paths and code snippets.
</aix_summary>

</aix_trajectory_compress>
```
"""

USER_PROMPT_MARKER = "Below is the trajectory to be compressed this time:"
TRAJECTORY_PLACEHOLDER = "<coding_agent_trajectory>\n</coding_agent_trajectory>"
COMPRESSION_TARGET_PLACEHOLDER = "__COMPRESSION_TARGET__"
USER_PROMPT = f"""{USER_PROMPT_MARKER}

```xml
{TRAJECTORY_PLACEHOLDER}
```

{COMPRESSION_TARGET_PLACEHOLDER}

Please provide the compression for this trajectory. The format must strictly follow the specification, starting with `<aix_trajectory_compress>` and ending with `</aix_trajectory_compress>`."""

STUDENT_SYSTEM_PROMPT = SYSTEM_PROMPT
STUDENT_USER_PROMPT = USER_PROMPT


def _snip(s: str, size: int = 80) -> str:
    flat = s.replace("\n", "\\n")
    if len(flat) <= size:
        return flat
    return flat[:size] + "[truncated]"


def _snip_json(v: Any, size: int = 80) -> str:
    try:
        raw = json.dumps(v, ensure_ascii=False)
    except (TypeError, ValueError):
        raw = str(v)
    return _snip(raw, size)


def _raw_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return _raw_text(content)
    parts: list[str] = []
    for item in content:
        if not isinstance(item, dict):
            continue
        if item.get("type") == "text":
            text = item.get("text", "")
            if text:
                parts.append(str(text))
    return "\n".join(parts)


def _tool_result_text(message: dict | None, call_id: str) -> str:
    if not isinstance(message, dict) or message.get("role") != "tool":
        return ""
    content = message.get("content")
    if not isinstance(content, list):
        return _raw_text(content)
    for item in content:
        if not isinstance(item, dict) or item.get("type") != "tool-result":
            continue
        if item.get("toolCallId") != call_id:
            continue
        output = item.get("output", {})
        if isinstance(output, dict):
            return _raw_text(output.get("value", output))
        return _raw_text(output)
    return ""


def _append_text_element(lines: list[str], tag: str, text: Any) -> None:
    lines.append(f"<{tag}>")
    raw = _raw_text(text)
    if raw:
        lines.append(raw)
    lines.append(f"</{tag}>")


def _append_inline_text_element(lines: list[str], tag: str, text: Any) -> None:
    lines.append(f"<{tag}>{_raw_text(text)}</{tag}>")


_LENGTH_TOKENIZER: Any | None = None
_LENGTH_TOKENIZER_PATH: Path | None = None


def length_tokenizer_path() -> Path:
    """Use the model bundle by default; allow a local-file override on split hosts."""
    override = os.getenv("LENGTH_TOKENIZER_PATH", "").strip()
    if override:
        return Path(override).expanduser().resolve()
    model_path = os.getenv("MODEL_PATH", "").strip()
    if not model_path:
        raise ValueError(
            "Set MODEL_PATH to the merged model directory containing tokenizer.json"
        )
    return (Path(model_path).expanduser() / "tokenizer.json").resolve()


def _load_length_tokenizer() -> Any:
    global _LENGTH_TOKENIZER, _LENGTH_TOKENIZER_PATH
    path = length_tokenizer_path()
    if _LENGTH_TOKENIZER is not None and _LENGTH_TOKENIZER_PATH == path:
        return _LENGTH_TOKENIZER
    if not path.is_file():
        raise FileNotFoundError(
            f"Tokenizer file not found: {path}. Check MODEL_PATH or LENGTH_TOKENIZER_PATH."
        )
    from tokenizers import Tokenizer

    # Load tokenization rules only, never model weights. Cache once per path.
    tokenizer = Tokenizer.from_file(str(path))
    _LENGTH_TOKENIZER = tokenizer
    _LENGTH_TOKENIZER_PATH = path
    return tokenizer


def model_token_length(text: str) -> int:
    """Count trajectory tokens with the same tokenizer file as the served model."""
    if not text:
        return 0
    return len(_load_length_tokenizer().encode(text, add_special_tokens=False).ids)


def _format_length(length: int) -> str:
    return f"Estimated length of approximately {length} tokens"


def _append_length_element(
    lines: list[str],
    tag: str,
    text: str,
    length_counter: Callable[[str], int],
) -> None:
    _append_inline_text_element(lines, tag, _format_length(length_counter(text)))


def compute_visible_message_tokens(
    trajectory_xml: str,
    length_counter: Callable[[str], int] | None = None,
) -> int:
    """Count tokens of the ``<coding_agent_messages>`` section in *trajectory_xml*."""
    counter = length_counter or model_token_length
    start_tag = "<coding_agent_messages>"
    end_tag = "</coding_agent_messages>"
    s = trajectory_xml.find(start_tag)
    e = trajectory_xml.find(end_tag)
    if s == -1 or e == -1:
        return counter(trajectory_xml)
    return counter(trajectory_xml[s : e + len(end_tag)])


def _build_compression_target_text(
    visible_message_tokens: int,
    target_compression_ratio_pct: int,
) -> str:
    target_tokens = round(visible_message_tokens * target_compression_ratio_pct / 100)
    return (
        f"The content visible to the coding agent in the original "
        f"`<coding_agent_messages>` costs about {visible_message_tokens} tokens; "
        f"the goal is to reduce it to about {target_compression_ratio_pct}% of that, "
        f"approximately {target_tokens} tokens."
    )


DEFAULT_TOOL_DEFINITIONS: dict[str, dict[str, Any]] = {
    "bash": {
        "name": "bash",
        "description": "Execute a bash command in the terminal.",
        "args": [{"name": "command", "description": "The bash command to execute."}],
    },
    "execute_bash": {
        "name": "execute_bash",
        "description": "Execute a bash command in the terminal.",
        "args": [{"name": "command", "description": "The bash command to execute."}],
    },
    "finish": {
        "name": "finish",
        "description": "Signals completion of the current task or conversation.",
        "args": [
            {"name": "message", "description": "Final message to send to the user."}
        ],
    },
    "str_replace_editor": {
        "name": "str_replace_editor",
        "description": "Custom editing tool for viewing, creating, and editing plain-text files.",
        "args": [
            {"name": "command", "description": "The editor operation to perform."},
            {"name": "path", "description": "The target file or directory path."},
            {
                "name": "file_text",
                "description": "Full file text for create operations.",
            },
            {"name": "old_str", "description": "Text to replace."},
            {"name": "new_str", "description": "Replacement text."},
            {
                "name": "insert_line",
                "description": "Line number for insert operations.",
            },
            {
                "name": "view_range",
                "description": "Optional line range for view operations.",
            },
        ],
    },
    "think": {
        "name": "think",
        "description": "Record internal reasoning or planning without changing files.",
        "args": [{"name": "thought", "description": "The thought to record."}],
    },
}


def _observed_tool_names(messages: list[dict]) -> list[str]:
    names: list[str] = []
    seen: set[str] = set()
    for msg in messages:
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for item in content:
            if not isinstance(item, dict) or item.get("type") != "tool-call":
                continue
            name = str(item.get("toolName") or "").strip()
            if name and name not in seen:
                seen.add(name)
                names.append(name)
    return names


def _definition_for_missing_tool(name: str) -> dict[str, Any]:
    if name in DEFAULT_TOOL_DEFINITIONS:
        return dict(DEFAULT_TOOL_DEFINITIONS[name])
    if name.startswith("str_replace_editor."):
        base = dict(DEFAULT_TOOL_DEFINITIONS["str_replace_editor"])
        base["name"] = name
        base["description"] = (
            DEFAULT_TOOL_DEFINITIONS["str_replace_editor"]["description"]
            + f"\nThis entry documents the `{name}` operation observed in the trajectory."
        )
        return base
    return {
        "name": name,
        "description": f"Tool `{name}` was observed in the trajectory, but no source tool schema was available.",
        "args": [{"name": "raw", "description": "Raw arguments passed to the tool."}],
    }


def get_tool_definition(name: str) -> dict[str, Any]:
    """Get a tool definition by name. Public API for main.py."""
    return _definition_for_missing_tool(name)


def _format_tool_definitions(
    tool_definitions: list[dict[str, Any]] | None,
    *,
    observed_tool_names: list[str] | None = None,
) -> list[str]:
    lines = ["<coding_agent_tools_define>"]
    tools: list[dict[str, Any]] = list(tool_definitions or [])
    defined_names = {
        str(tool.get("name") or tool.get("tool_name") or "").strip()
        for tool in tools
        if isinstance(tool, dict)
    }
    for name in observed_tool_names or []:
        if name not in defined_names:
            tools.append(_definition_for_missing_tool(name))
            defined_names.add(name)

    for tool in tools:
        name = str(tool.get("name") or tool.get("tool_name") or "").strip()
        if not name:
            continue
        lines.extend(["", "<tool>"])
        _append_inline_text_element(lines, "tool_name", name)
        description = _raw_text(tool.get("description", ""))
        if tool.get("parameters") is not None:
            # Keep the complete schema, including root constraints and $defs,
            # inside the existing tool-description field of the model protocol.
            description = (
                f"{description}\n\nInput JSON Schema:\n{_raw_text(tool['parameters'])}"
            ).strip()
        _append_text_element(lines, "tool_description", description)
        args = tool.get("args") or tool.get("tool_args") or []
        if args:
            lines.append("<tool_args>")
            for arg in args:
                if not isinstance(arg, dict):
                    continue
                arg_name = str(arg.get("name") or arg.get("arg_name") or "").strip()
                if not arg_name:
                    continue
                lines.append("<tool_arg>")
                _append_inline_text_element(lines, "arg_name", arg_name)
                _append_text_element(
                    lines, "arg_description", arg.get("description", "")
                )
                lines.append("</tool_arg>")
            lines.append("</tool_args>")
        lines.append("</tool>")
    lines.extend(["", "</coding_agent_tools_define>"])
    return lines


def _format_function_args(lines: list[str], tool_input: Any) -> None:
    if isinstance(tool_input, dict):
        items = tool_input.items()
    else:
        items = [("raw", tool_input)]
    lines.append("<function_args>")
    for arg_name, arg_value in items:
        lines.append("<function_arg>")
        _append_inline_text_element(lines, "arg_name", arg_name)
        _append_text_element(lines, "arg_value", arg_value)
        lines.append("</function_arg>")
    lines.append("</function_args>")


def _tool_call_length_text(
    content_text: str, function_name: str, tool_input: Any
) -> str:
    parts = [content_text, function_name]
    if isinstance(tool_input, dict):
        for arg_name, arg_value in tool_input.items():
            parts.append(str(arg_name))
            parts.append(_raw_text(arg_value))
    else:
        parts.append(_raw_text(tool_input))
    return "\n".join(part for part in parts if part)


def build_coding_agent_trajectory_xml(
    messages: list[dict],
    *,
    coding_agent_system_prompt: str = "",
    tool_definitions: list[dict[str, Any]] | None = None,
    length_counter: Callable[[str], int] | None = None,
) -> str:
    """Render the compressor input history in the documented trajectory XML shape."""
    messages = normalize_tool_results(messages)
    count_length = length_counter or model_token_length
    lines: list[str] = [
        "<coding_agent_trajectory>",
        "",
        "<coding_agent_system_prompt>",
    ]
    if coding_agent_system_prompt:
        lines.append(coding_agent_system_prompt)
    lines.extend(["</coding_agent_system_prompt>", "", ""])
    lines.extend(
        _format_tool_definitions(
            tool_definitions,
            observed_tool_names=_observed_tool_names(messages),
        )
    )
    lines.extend(["", "", "<coding_agent_messages>"])

    idx = 0
    fallback_global_id = 0
    while idx < len(messages):
        msg = messages[idx]
        role = msg.get("role", "")
        if role == "system":
            idx += 1
            continue

        fallback_global_id += 1
        global_id = str(msg.get("global_id") or fallback_global_id)
        content = msg.get("content")
        if role == "assistant" and isinstance(content, list):
            tool_calls = [
                item
                for item in content
                if isinstance(item, dict) and item.get("type") == "tool-call"
            ]
            if tool_calls:
                result_msg = (
                    messages[idx + 1]
                    if idx + 1 < len(messages)
                    and messages[idx + 1].get("role") == "tool"
                    else None
                )
                content_text = _content_text(content)
                for tc_idx, call in enumerate(tool_calls):
                    if tc_idx > 0:
                        fallback_global_id += 1
                    gid = global_id if tc_idx == 0 else str(fallback_global_id)
                    call_id = str(call.get("toolCallId") or gid)
                    pair_content_text = content_text if tc_idx == 0 else ""
                    tool_name = str(call.get("toolName", ""))
                    tool_input = call.get("input", {})
                    lines.extend(["", "<assistant_tool_pair>"])
                    _append_inline_text_element(lines, "id", gid)
                    _append_length_element(
                        lines,
                        "num_call_tokens",
                        _tool_call_length_text(
                            pair_content_text, tool_name, tool_input
                        ),
                        count_length,
                    )
                    _append_text_element(lines, "content", pair_content_text)
                    lines.append("<tool_call>")
                    _append_inline_text_element(lines, "function_name", tool_name)
                    _format_function_args(lines, tool_input)
                    lines.append("</tool_call>")
                    tool_result_text = _tool_result_text(result_msg, call_id)
                    _append_length_element(
                        lines,
                        "num_result_tokens",
                        tool_result_text,
                        count_length,
                    )
                    _append_text_element(
                        lines,
                        "tool_result_content",
                        tool_result_text,
                    )
                    lines.append("</assistant_tool_pair>")
                if result_msg is not None:
                    idx += 2
                else:
                    idx += 1
                continue

        if role == "assistant":
            lines.extend(["", "<assistant_message>"])
            _append_inline_text_element(lines, "id", global_id)
            content_text = _content_text(content)
            _append_length_element(lines, "num_tokens", content_text, count_length)
            _append_text_element(lines, "content", content_text)
            lines.append("</assistant_message>")
        else:
            lines.extend(["", "<user_message>"])
            _append_inline_text_element(lines, "id", global_id)
            content_text = _content_text(content)
            _append_length_element(lines, "num_tokens", content_text, count_length)
            _append_text_element(lines, "content", content_text)
            lines.append("</user_message>")
        idx += 1

    lines.extend(["", "</coding_agent_messages>", "", "</coding_agent_trajectory>"])
    return "\n".join(lines)


def build_compressor_user_prompt(
    trajectory_xml: str,
    prompt_template: str | None = None,
    *,
    visible_message_tokens: int | None = None,
    target_compression_ratio_pct: int | None = None,
) -> str:
    """Insert the rendered trajectory into the documented compressor user prompt."""
    template = (prompt_template or USER_PROMPT).strip()

    if COMPRESSION_TARGET_PLACEHOLDER in template:
        if (
            visible_message_tokens is not None
            and target_compression_ratio_pct is not None
        ):
            target_text = _build_compression_target_text(
                visible_message_tokens,
                target_compression_ratio_pct,
            )
        else:
            target_text = ""
        template = template.replace(COMPRESSION_TARGET_PLACEHOLDER, target_text)
        while "\n\n\n" in template:
            template = template.replace("\n\n\n", "\n\n")

    if TRAJECTORY_PLACEHOLDER in template:
        inner_xml = _trajectory_inner_xml(trajectory_xml)
        filled = f"<coding_agent_trajectory>{inner_xml}</coding_agent_trajectory>"
        return template.replace(TRAJECTORY_PLACEHOLDER, filled)
    if "__CODING_AGENT_TRAJECTORY__" in template:
        return template.replace("__CODING_AGENT_TRAJECTORY__", trajectory_xml)
    return f"{template}\n\n```xml\n{trajectory_xml}\n```".strip()


def _trajectory_inner_xml(trajectory_xml: str) -> str:
    """Return only the content inside the outer coding_agent_trajectory tag."""
    stripped = trajectory_xml.strip()
    start = "<coding_agent_trajectory>"
    end = "</coding_agent_trajectory>"
    if stripped.startswith(start) and stripped.endswith(end):
        return stripped[len(start) : -len(end)]
    return stripped


def build_compaction_model_input(
    messages: list[dict],
    prompt: str,
    *,
    coding_agent_system_prompt: str = "",
    tool_definitions: list[dict[str, Any]] | None = None,
    include_system: bool = False,
    compressor_system_prompt: str = SYSTEM_PROMPT,
    length_counter: Callable[[str], int] | None = None,
    visible_message_tokens: int | None = None,
    target_compression_ratio_pct: int | None = None,
) -> str:
    trajectory_xml = build_coding_agent_trajectory_xml(
        messages,
        coding_agent_system_prompt=coding_agent_system_prompt,
        tool_definitions=tool_definitions,
        length_counter=length_counter,
    )
    user_prompt = build_compressor_user_prompt(
        trajectory_xml,
        prompt or USER_PROMPT,
        visible_message_tokens=visible_message_tokens,
        target_compression_ratio_pct=target_compression_ratio_pct,
    )
    parts: list[str] = []
    if include_system:
        parts.append(f"[System]\n{compressor_system_prompt}")
        parts.append(f"[User]\n{user_prompt}")
    else:
        parts.append(user_prompt)
    return "\n\n".join(part for part in parts if part)


def collect_tools(messages: list[dict]) -> list[dict]:
    """Extract tool calls from messages (mirrors compaction.ts collectTools)."""
    result: list[dict] = []
    idx = 0
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for item in content:
            if not isinstance(item, dict):
                continue
            if item.get("type") != "tool-call":
                continue
            idx += 1
            call_id = item.get("toolCallId", "")
            name = item.get("toolName", "")
            inp = item.get("input", {})
            title = _snip_json(inp, 80)
            result.append({"idx": idx, "callID": call_id, "tool": name, "title": title})
    return result


def build_prompt(messages: list[dict]) -> str:
    """Build the compaction user prompt."""
    return USER_PROMPT


def build_student_prompt(messages: list[dict]) -> str:
    """Build the SFT student prompt."""
    return STUDENT_USER_PROMPT
