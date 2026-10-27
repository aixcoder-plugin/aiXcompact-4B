#!/usr/bin/env python3
"""Call the compaction service with a synthetic coding conversation."""

import argparse
import json
from pathlib import Path
import uuid
import httpx

parser = argparse.ArgumentParser()
parser.add_argument("--url", default="http://127.0.0.1:12003")
parser.add_argument(
    "--input", type=Path, default=Path(__file__).with_name("request.json")
)
args = parser.parse_args()
body = json.loads(args.input.read_text())
body["request_id"] = str(uuid.uuid4())
with httpx.Client(timeout=1800) as client:
    response = client.post(args.url.rstrip("/") + "/compact", json=body)
    response.raise_for_status()
    print(json.dumps(response.json(), ensure_ascii=False, indent=2))
