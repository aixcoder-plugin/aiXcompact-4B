from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class PruneDecision:
    call_id: str
    name: str
    action: str  # legacy-normalized: "keep_all" | "pruner" | "remove_all"
    reason: str
    pruned_observation: str | None = None
    pruned_observation_analysis: str | None = None
    item_analysis: str | None = None


@dataclass
class ParseResult:
    analysis: str | None = None
    summary: str | None = None
    decisions: list[PruneDecision] = field(default_factory=list)
    raw: str = ""
    schema: str = "legacy"
    root_valid: bool = False
    errors: list[str] = field(default_factory=list)


class ProtocolError(ValueError):
    """The model output cannot safely be applied to the original messages."""


class _ProtocolReader:
    """Read structural fields in order, treating payloads as raw text."""

    def __init__(self, text: str):
        self.text = text
        self.pos = 0

    def starts(self, tag: str) -> bool:
        while self.pos < len(self.text) and self.text[self.pos].isspace():
            self.pos += 1
        return self.text.startswith(tag, self.pos)

    def expect(self, tag: str) -> None:
        if not self.starts(tag):
            raise ProtocolError(f"expected {tag} at offset {self.pos}")
        self.pos += len(tag)

    def field(self, name: str) -> str:
        self.expect(f"<{name}>")
        start = self.pos
        close = f"</{name}>"
        # CDATA explicitly protects occurrences of a field's own delimiter.
        if self.text[start:].lstrip().startswith("<![CDATA["):
            cdata_start = self.text.index("<![CDATA[", start) + len("<![CDATA[")
            cdata_end = self.text.find("]]>", cdata_start)
            if cdata_end < 0:
                raise ProtocolError(f"unterminated CDATA in {name}")
            self.pos = cdata_end + 3
            self.expect(close)
            return self.text[cdata_start:cdata_end].strip()
        end = self.text.find(close, start)
        if end < 0:
            raise ProtocolError(f"missing {close}")
        self.pos = end + len(close)
        return self.text[start:end].strip()


def _parse_current(text: str) -> ParseResult:
    result = ParseResult(raw=text, schema="aix")
    reader = _ProtocolReader(text)
    try:
        reader.expect("<aix_trajectory_compress>")
        result.analysis = reader.field("aix_analysis")
        reader.expect("<aix_decisions>")
        while not reader.starts("</aix_decisions>"):
            reader.expect("<aix_decision_item>")
            gid = reader.field("id")
            analysis = reader.field("item_analysis")
            action = reader.field("action")
            if action not in {"keep", "remove", "keep_with_replaced_tool_result"}:
                raise ProtocolError("invalid decision action")
            if not gid or not analysis:
                raise ProtocolError("empty decision id or item_analysis")
            replacement = None
            if action == "keep_with_replaced_tool_result":
                replacement = reader.field("replacement_tool_result_content")
                if not replacement:
                    raise ProtocolError("empty replacement content")
            reader.expect("</aix_decision_item>")
            result.decisions.append(
                PruneDecision(
                    call_id=gid,
                    name="",
                    action=normalize_action(action),
                    reason="",
                    pruned_observation=replacement,
                    pruned_observation_analysis=analysis
                    if replacement is not None
                    else None,
                    item_analysis=analysis,
                )
            )
        reader.expect("</aix_decisions>")
        result.summary = reader.field("aix_summary")
        reader.expect("</aix_trajectory_compress>")
        if text[reader.pos :].strip():
            raise ProtocolError("unexpected text after protocol root")
        result.root_valid = True
    except ProtocolError as error:
        result.errors.append(str(error))
    return result


def validate_output(result: ParseResult, expected_items: dict[str, bool]) -> None:
    """Gate all destructive assembly on a complete, aligned task protocol."""
    if result.errors:
        raise ProtocolError("; ".join(result.errors))
    if not result.root_valid or not result.analysis or not result.summary:
        raise ProtocolError(
            "complete root, analysis and non-empty summary are required"
        )
    ids = [decision.call_id for decision in result.decisions]
    if ids != list(expected_items):
        raise ProtocolError(
            "decision IDs must cover each input item exactly once in order"
        )
    for decision in result.decisions:
        if decision.action == "pruner" and not expected_items[decision.call_id]:
            raise ProtocolError("tool-result replacement is only valid for tool pairs")


_ANALYSIS_RE = re.compile(r"<analysis>([\s\S]*?)</analysis>")
_SUMMARY_RE = re.compile(r"<summary>([\s\S]*?)</summary>")
_TOOL_PRUNING_RE = re.compile(r"<tool_pruning>([\s\S]*?)</tool_pruning>")
_TOOL_RE = re.compile(r'<tool\s+call_id="([^"]+)"\s+name="([^"]+)">([\s\S]*?)</tool>')
_REASON_RE = re.compile(r"<reason>([\s\S]*?)</reason>")
_ACTION_RE = re.compile(r"<action>([\s\S]*?)</action>")
_PRUNED_RE = re.compile(r"<pruned_observation>([\s\S]*?)</pruned_observation>")
_AIX_ANALYSIS_RE = re.compile(r"<aix_analysis>([\s\S]*?)</aix_analysis>")
_AIX_SUMMARY_RE = re.compile(r"<aix_summary>([\s\S]*?)</aix_summary>")
_AIX_ITEM_RE = re.compile(
    r"<aix_(?:(?:tool_)?pruning_item|decision_item)>([\s\S]*?)</aix_(?:(?:tool_)?pruning_item|decision_item)>"
)
_GLOBAL_ID_RE = re.compile(r"<id>([\s\S]*?)</id>")
_MODIFIED_RE = re.compile(
    r"<(?:modified_tool_result_as|replacement_tool_result_content)>([\s\S]*?)</(?:modified_tool_result_as|replacement_tool_result_content)>"
)
_MODIFIED_ANALYSIS_RE = re.compile(
    r"<modified_tool_result_analysis>([\s\S]*?)</modified_tool_result_analysis>"
)
_ITEM_ANALYSIS_RE = re.compile(r"<item_analysis>([\s\S]*?)</item_analysis>")

VALID_ACTIONS = {"keep_all", "pruner", "remove_all"}
NEW_TO_LEGACY_ACTION = {
    "keep": "keep_all",
    "modify_tool_result_as": "pruner",
    "keep_with_replaced_tool_result": "pruner",
    "remove": "remove_all",
}
LEGACY_TO_NEW_ACTION = {
    "keep_all": "keep",
    "pruner": "keep_with_replaced_tool_result",
    "remove_all": "remove",
}
_CDATA_START = "<![CDATA["
_CDATA_END = "]]>"


def _clean_text(text: str) -> str:
    return text.strip().removeprefix(_CDATA_START).removesuffix(_CDATA_END).strip()


def _clean_global_id(text: str) -> str:
    cleaned = _clean_text(text)
    if not cleaned:
        return ""
    first_line = cleaned.splitlines()[0].strip()
    parts = first_line.split()
    return parts[0] if parts else first_line


def normalize_action(action: str) -> str | None:
    cleaned = _clean_text(action)
    if cleaned in VALID_ACTIONS:
        return cleaned
    return NEW_TO_LEGACY_ACTION.get(cleaned)


def action_to_aix(action: str) -> str:
    return LEGACY_TO_NEW_ACTION.get(action, action)


def _aix_pruning_block(text: str) -> str | None:
    starts = ("<aix_pruning>", "<aix_tool_pruning>", "<aix_decisions>")
    ends = ("</aix_pruning>", "</aix_tool_pruning>", "</aix_decisions>")
    candidates: list[tuple[int, str]] = []
    for start_tag in starts:
        start = text.find(start_tag)
        if start >= 0:
            candidates.append((start, start_tag))
    if not candidates:
        return None
    start, start_tag = min(candidates, key=lambda item: item[0])
    content_start = start + len(start_tag)
    end_positions = [
        text.find(end_tag, content_start)
        for end_tag in ends
        if text.find(end_tag, content_start) >= 0
    ]
    if not end_positions:
        return None
    return text[content_start : min(end_positions)]


def _parse_aix_pruning(block: str) -> list[PruneDecision]:
    decisions: list[PruneDecision] = []
    for item_m in _AIX_ITEM_RE.finditer(block):
        body = item_m.group(1)
        gid_m = _GLOBAL_ID_RE.search(body)
        action_m = _ACTION_RE.search(body)
        if not gid_m or not action_m:
            continue
        call_id = _clean_global_id(gid_m.group(1))
        action = normalize_action(action_m.group(1))
        if not call_id or not action:
            continue
        item_analysis = None
        ia_m = _ITEM_ANALYSIS_RE.search(body)
        if ia_m:
            item_analysis = _clean_text(ia_m.group(1))
        modified = None
        modified_analysis = None
        if action == "pruner":
            modified_m = _MODIFIED_RE.search(body)
            if modified_m:
                modified = _clean_text(modified_m.group(1))
            analysis_m = _MODIFIED_ANALYSIS_RE.search(body)
            if analysis_m:
                modified_analysis = _clean_text(analysis_m.group(1))
            if item_analysis and not modified_analysis:
                modified_analysis = item_analysis
        decisions.append(
            PruneDecision(
                call_id=call_id,
                name="",
                action=action,
                reason="",
                pruned_observation=modified,
                pruned_observation_analysis=modified_analysis,
                item_analysis=item_analysis,
            )
        )
    return decisions


def parse(text: str) -> ParseResult:
    """Parse LLM XML output (mirrors compaction.ts parseOutput)."""
    if text.lstrip().startswith("<aix_trajectory_compress>"):
        return _parse_current(text)
    result = ParseResult(raw=text)

    m = _AIX_ANALYSIS_RE.search(text) or _ANALYSIS_RE.search(text)
    if m:
        result.analysis = _clean_text(m.group(1))

    m = _AIX_SUMMARY_RE.search(text) or _SUMMARY_RE.search(text)
    if m:
        result.summary = _clean_text(m.group(1))

    aix_block = _aix_pruning_block(text)
    if aix_block is not None:
        result.schema = "aix"
        result.decisions.extend(_parse_aix_pruning(aix_block))
        return result

    m = _TOOL_PRUNING_RE.search(text)
    if not m:
        return result

    block = m.group(1)
    for tm in _TOOL_RE.finditer(block):
        call_id = tm.group(1)
        name = tm.group(2)
        body = tm.group(3)

        reason_m = _REASON_RE.search(body)
        reason = _clean_text(reason_m.group(1)) if reason_m else ""

        action_m = _ACTION_RE.search(body)
        if not action_m:
            continue
        action = normalize_action(action_m.group(1))
        if not action:
            continue

        pruned = None
        if action == "pruner":
            pruned_m = _PRUNED_RE.search(body)
            if pruned_m:
                pruned = _clean_text(pruned_m.group(1))

        result.decisions.append(
            PruneDecision(
                call_id=call_id,
                name=name,
                action=action,
                reason=reason,
                pruned_observation=pruned,
            )
        )

    return result
