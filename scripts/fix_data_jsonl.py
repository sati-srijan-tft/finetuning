#!/usr/bin/env python3
"""
Fix an existing data.jsonl where some messages have content as an array
instead of a string, causing PyArrow schema errors in LLaMA-Factory.

Converts:
  {"role": "user", "content": [{"audio": "path.wav"}, {"text": "prompt"}]}
To:
  {"role": "user", "content": "<audio>prompt"}
  + top-level "audios": ["path.wav"]

Usage:
    python scripts/fix_data_jsonl.py LLaMA-Factory/data/data.jsonl
"""

import json
import sys
from pathlib import Path


def fix_sample(sample: dict) -> dict:
    audios = list(sample.get("audios", []))

    for msg in sample.get("messages", []):
        content = msg.get("content")
        if not isinstance(content, list):
            continue

        audio_paths = []
        text_parts = []
        for item in content:
            if "audio" in item:
                audio_paths.append(item["audio"])
                text_parts.insert(0, "<audio>")
            elif "image" in item:
                text_parts.insert(0, "<image>")
            elif "text" in item:
                text_parts.append(item["text"])
            elif "video" in item:
                text_parts.insert(0, "<video>")

        msg["content"] = "".join(text_parts)
        audios.extend(audio_paths)

    if audios:
        sample["audios"] = audios
    return sample


def main():
    path = Path(sys.argv[1] if len(sys.argv) > 1 else "LLaMA-Factory/data/data.jsonl")
    if not path.exists():
        print(f"ERROR: {path} not found")
        sys.exit(1)

    backup = path.with_suffix(".jsonl.bak")
    print(f"Reading {path} ...")

    with path.open("r", encoding="utf-8") as f:
        samples = [json.loads(line) for line in f if line.strip()]

    array_content_count = sum(
        1 for s in samples
        for m in s.get("messages", [])
        if isinstance(m.get("content"), list)
    )
    print(f"  {len(samples)} samples, {array_content_count} messages with array content")

    if array_content_count == 0:
        print("Nothing to fix — all content fields are already strings.")
        sys.exit(0)

    path.rename(backup)
    print(f"  Backup → {backup}")

    fixed = [fix_sample(s) for s in samples]

    with path.open("w", encoding="utf-8") as f:
        for s in fixed:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"Fixed {len(fixed)} samples → {path}")
    print("Re-run 03_run_stage1_training.sh.")


if __name__ == "__main__":
    main()
