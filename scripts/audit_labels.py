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

# Make the repo root importable so `from slotfill...` works when run as `python scripts/x.py`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from slotfill.data import PROCESSED, read_jsonl, write_jsonl  # noqa: E402
from slotfill.normalize import normalize_date, normalize_total, text_key  # noqa: E402

# Similarity (0..1) above which a near-match counts as "fuzzy supported", e.g. one OCR typo.
FUZZY = 0.9


def total_supported(gold: str, ocr: str) -> bool:
    # Find every money-looking number in the OCR ("RM 1,007.50", "60.3") and check
    # whether any of them normalises to the gold total.
    return any(normalize_total(tok) == gold for tok in re.findall(r"(?:RM\s*)?\d[\d,]*\.\d{1,2}", ocr))


def date_supported(gold: str, ocr: str) -> bool:
    # Find every date-looking string in the OCR, in all the formats normalize_date
    # understands, and check whether any of them normalises to the gold date.
    candidates = re.findall(
        r"\d{1,4}[-/.]\d{1,2}[-/.]\d{2,4}|\d{1,2}[-/ ][A-Za-z]{3,9}[-/ ]\d{2,4}|[A-Za-z]{3,9} \d{1,2},? \d{4}|\d{8}", ocr)
    return any(normalize_date(c) == gold for c in candidates)


def text_supported(gold: str | None, ocr: str) -> tuple[bool, float]:
    """Exact (normalised) substring, else best fuzzy match over windows of OCR lines."""
    if gold is None:
        return True, 1.0  # "no address" is always achievable
    g = text_key(gold)
    raw_lines = ocr.splitlines()
    lines = [text_key(line) for line in raw_lines]
    joined = text_key(" ".join(raw_lines))
    # Exact: the gold text appears in the OCR (also checked with all spaces removed,
    # since OCR often splits or merges words).
    if g in joined or g.replace(" ", "") in joined.replace(" ", ""):
        return True, 1.0
    # Not exact: slide a window of 1-6 consecutive OCR lines (addresses span several
    # lines) and keep the best similarity score.
    best = 0.0
    for i in range(len(lines)):
        for j in range(i + 1, min(i + 7, len(lines) + 1)):
            best = max(best, SequenceMatcher(None, g, text_key(" ".join(raw_lines[i:j]))).ratio())
    return False, best


def audit(rows: list[dict]) -> tuple[dict, list[dict]]:
    # Count, per field, how many docs have their gold value recoverable from the OCR.
    counts = {"total": 0, "date": 0, "company": 0, "company_fuzzy": 0, "address": 0, "address_fuzzy": 0, "all_exact": 0}
    flags = []
    for r in rows:
        g, ocr = r["gold"], r["ocr"]
        t, d = total_supported(g["total"], ocr), date_supported(g["date"], ocr)
        c, c_sim = text_supported(g["company"], ocr)
        a, a_sim = text_supported(g["address"], ocr)
        # Booleans add as 0/1, so these are running counts.
        counts["total"] += t
        counts["date"] += d
        counts["company"] += c
        counts["company_fuzzy"] += c or c_sim >= FUZZY
        counts["address"] += a
        counts["address_fuzzy"] += a or a_sim >= FUZZY
        counts["all_exact"] += t and d and c and a  # all four recoverable: the per-doc ceiling
        # Record which fields are missing from the OCR, for later inspection.
        missing = [f for f, ok in (("total", t), ("date", d), ("company", c), ("address", a)) if not ok]
        if missing:
            flags.append({"id": r["id"], "missing_in_ocr": missing, "company_sim": round(c_sim, 3),
                          "address_sim": round(a_sim, 3), "gold": g})
    n = len(rows)
    # Convert counts to rates (0..1) and include the number of docs.
    return {k: round(v / n, 4) for k, v in counts.items()} | {"n": n}, flags


def main() -> None:
    # Audit both splits and collect all flagged docs, tagged with their split.
    report, all_flags = {}, []
    for split in ("train", "test"):
        rows = read_jsonl(PROCESSED / f"{split}.jsonl")
        report[split], flags = audit(rows)
        all_flags += [f | {"split": split} for f in flags]

    (PROCESSED / "audit_report.json").write_text(json.dumps(report, indent=2) + "\n")
    write_jsonl(PROCESSED / "audit_flags.jsonl", all_flags)

    # A fixed random sample of 50 test docs for a human to check against the images.
    # The last two columns are left empty for the reviewer to fill in.
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
