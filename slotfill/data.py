"""Read the processed JSONL splits."""

from __future__ import annotations

import json
from pathlib import Path

# Repository root (two levels up from this file: slotfill/data.py -> slotfill/ -> repo).
# Every script resolves paths from here, so they work no matter which folder you run them from.
ROOT = Path(__file__).resolve().parent.parent
# Where the frozen train/test split and the label audit live.
PROCESSED = ROOT / "data" / "processed"


def read_jsonl(path: str | Path) -> list[dict]:
    # JSONL = one JSON object per line. Blank lines are skipped so a trailing newline is harmless.
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: str | Path, rows: list[dict]) -> None:
    # Create the parent folder if needed (e.g. results/ on a fresh clone).
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            # ensure_ascii=False keeps non-ASCII characters readable instead of \uXXXX escapes.
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def target_json(gold: dict) -> str:
    """The exact string the model is trained to emit. Key order is fixed."""
    # A fixed key order means the fine-tuned model always learns one exact output layout,
    # which makes its answers predictable for a downstream import.
    return json.dumps({k: gold[k] for k in ("company", "date", "address", "total")}, ensure_ascii=False)
