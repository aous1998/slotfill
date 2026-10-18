"""Audit the gold labels against the OCR text the model actually sees.

    python scripts/audit_labels.py

SROIE labels were typed from the receipt image, but the model only gets the OCR
transcript, which has its own errors ("SDN BND" for "SDN BHD"). If a gold value
is not recoverable from the OCR, no model can get it exactly right. This script
measures that, per field, and writes:

  data/processed/audit_report.json        support rates (an accuracy ceiling)
  data/processed/audit_flags.jsonl        every doc where a gold value is not in the OCR
  data/processed/manual_review_sample.csv 50 random test docs for a human spot-check
"""

from __future__ import annotations

import csv
import json
import random
import re
import sys
from difflib import SequenceMatcher
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from slotfill.data import PROCESSED, read_jsonl, write_jsonl  # noqa: E402
from slotfill.normalize import normalize_date, normalize_total, text_key  # noqa: E402

FUZZY = 0.9


def total_supported(gold: str, ocr: str) -> bool:
    return any(normalize_total(tok) == gold for tok in re.findall(r"(?:RM\s*)?\d[\d,]*\.\d{1,2}", ocr))


def date_supported(gold: str, ocr: str) -> bool:
    candidates = re.findall(
        r"\d{1,4}[-/.]\d{1,2}[-/.]\d{2,4}|\d{1,2}[-/ ][A-Za-z]{3,9}[-/ ]\d{2,4}|[A-Za-z]{3,9} \d{1,2},? \d{4}|\d{8}", ocr)
    return any(normalize_date(c) == gold for c in candidates)


def text_supported(gold: str | None, ocr: str) -> tuple[bool, float]:
    """Exact (normalised) substring, else best fuzzy match over windows of OCR lines."""
    if gold is None:
        return True, 1.0
    g = text_key(gold)
    raw_lines = ocr.splitlines()
    lines = [text_key(line) for line in raw_lines]
    joined = text_key(" ".join(raw_lines))
    if g in joined or g.replace(" ", "") in joined.replace(" ", ""):
        return True, 1.0
    best = 0.0
    for i in range(len(lines)):
        for j in range(i + 1, min(i + 7, len(lines) + 1)):
            best = max(best, SequenceMatcher(None, g, text_key(" ".join(raw_lines[i:j]))).ratio())
    return False, best


def audit(rows: list[dict]) -> tuple[dict, list[dict]]:
    counts = {"total": 0, "date": 0, "company": 0, "company_fuzzy": 0, "address": 0, "address_fuzzy": 0, "all_exact": 0}
    flags = []
    for r in rows:
        g, ocr = r["gold"], r["ocr"]
        t, d = total_supported(g["total"], ocr), date_supported(g["date"], ocr)
        c, c_sim = text_supported(g["company"], ocr)
        a, a_sim = text_supported(g["address"], ocr)
        counts["total"] += t
        counts["date"] += d
        counts["company"] += c
        counts["company_fuzzy"] += c or c_sim >= FUZZY
        counts["address"] += a
        counts["address_fuzzy"] += a or a_sim >= FUZZY
        counts["all_exact"] += t and d and c and a
        missing = [f for f, ok in (("total", t), ("date", d), ("company", c), ("address", a)) if not ok]
        if missing:
            flags.append({"id": r["id"], "missing_in_ocr": missing, "company_sim": round(c_sim, 3),
                          "address_sim": round(a_sim, 3), "gold": g})
    n = len(rows)
    return {k: round(v / n, 4) for k, v in counts.items()} | {"n": n}, flags


def main() -> None:
    report, all_flags = {}, []
    for split in ("train", "test"):
        rows = read_jsonl(PROCESSED / f"{split}.jsonl")
        report[split], flags = audit(rows)
        all_flags += [f | {"split": split} for f in flags]

    (PROCESSED / "audit_report.json").write_text(json.dumps(report, indent=2) + "\n")
    write_jsonl(PROCESSED / "audit_flags.jsonl", all_flags)

    test = read_jsonl(PROCESSED / "test.jsonl")
    sample = random.Random(7).sample(test, 50)
    with open(PROCESSED / "manual_review_sample.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "company", "date", "address", "total", "label_correct(y/n)", "note"])
        for r in sample:
            g = r["gold"]
            w.writerow([r["id"], g["company"], g["date"], g["address"], g["total"], "", ""])

    print(json.dumps(report, indent=2))
    print(f"{len(all_flags)} docs flagged -> data/processed/audit_flags.jsonl")


if __name__ == "__main__":
    main()
