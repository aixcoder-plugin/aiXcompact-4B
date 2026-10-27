#!/usr/bin/env python3
"""Render the exact model API request or a synthetic SFT row."""

import argparse
import json
from pathlib import Path
from service import llm, prompt

p = argparse.ArgumentParser()
p.add_argument("--input", type=Path, default=Path("examples/request.json"))
p.add_argument("--sft", action="store_true")
args = p.parse_args()
body = json.loads(args.input.read_text())
xml = prompt.build_coding_agent_trajectory_xml(
    body["messages"],
    coding_agent_system_prompt=body.get("system_prompt", ""),
    tool_definitions=body.get("tool_defs"),
)
user = prompt.build_compressor_user_prompt(
    xml,
    visible_message_tokens=prompt.compute_visible_message_tokens(xml),
    target_compression_ratio_pct=body.get("target_ratio_pct", 25),
)
messages = llm.build_api_messages(user)
if args.sft:
    messages.append(
        {"role": "assistant", "content": Path("examples/response.xml").read_text()}
    )
    result = {
        "id": "synthetic-parser-fix-001",
        "messages": messages,
        "feature_type": "synthetic",
    }
else:
    result = {
        "model": "aiXcompact",
        "messages": messages,
        "temperature": 0.7,
        "top_p": 0.95,
        "max_tokens": 32768,
        "chat_template_kwargs": {"enable_thinking": False},
    }
print(json.dumps(result, ensure_ascii=False))
