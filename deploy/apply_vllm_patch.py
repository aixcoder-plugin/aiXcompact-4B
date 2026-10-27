#!/usr/bin/env python3
"""Apply the release patch only to the exact vLLM 0.22.1 source files."""

import hashlib
import importlib.metadata
import json
from pathlib import Path
import re
import sys


def main():
    if importlib.metadata.version("vllm") != "0.22.1":
        raise SystemExit("This patch requires exactly vLLM 0.22.1")
    root = Path(importlib.metadata.distribution("vllm").locate_file(""))
    patches = Path(sys.argv[1])
    manifest = json.loads((patches / "vllm-files.json").read_text())
    states = []
    for entry in manifest:
        digest = hashlib.sha256((root / entry["path"]).read_bytes()).hexdigest()
        states.append(
            "before"
            if digest == entry["before_sha256"]
            else "after"
            if digest == entry["after_sha256"]
            else "unknown"
        )
    if all(s == "after" for s in states):
        print("Patch already installed and verified")
        return
    if not all(s == "before" for s in states):
        raise SystemExit("Refusing modified/mixed source files: " + repr(states))
    patch = patches / "vllm-0.22.1-gdn-ngram.patch"
    # Official serving images do not include git. Apply these hash-pinned,
    # text-only hunks strictly, without offsets or fuzzy context matching.
    replacements = {}
    for section in patch.read_text().split("diff --git ")[1:]:
        lines = section.splitlines(keepends=True)
        name = lines[0].strip().split(" b/", 1)[1]
        if name not in {e["path"] for e in manifest}:
            raise SystemExit("Unexpected patch path")
        original = (root / name).read_text().splitlines(keepends=True)
        result, cursor, pos = [], 0, 1
        while pos < len(lines):
            if not lines[pos].startswith("@@ "):
                pos += 1
                continue
            start = int(re.match(r"@@ -(\d+)", lines[pos]).group(1)) - 1
            result.extend(original[cursor:start])
            cursor = start
            pos += 1
            while pos < len(lines) and not lines[pos].startswith("@@ "):
                line = lines[pos]
                if line[0] in " -":
                    if cursor >= len(original) or original[cursor] != line[1:]:
                        raise SystemExit("Patch context mismatch: " + name)
                    cursor += 1
                if line[0] in " +":
                    result.append(line[1:])
                pos += 1

        result.extend(original[cursor:])
        replacements[name] = "".join(result).encode()
    if set(replacements) != {e["path"] for e in manifest}:
        raise SystemExit("Patch file list mismatch")
    for entry in manifest:
        if (
            hashlib.sha256(replacements[entry["path"]]).hexdigest()
            != entry["after_sha256"]
        ):
            raise SystemExit("Patched content checksum mismatch")
    for name, content in replacements.items():
        (root / name).write_bytes(content)
    for entry in manifest:
        assert (
            hashlib.sha256((root / entry["path"]).read_bytes()).hexdigest()
            == entry["after_sha256"]
        )
    print(f"{len(manifest)} patched files verified")


if __name__ == "__main__":
    main()
