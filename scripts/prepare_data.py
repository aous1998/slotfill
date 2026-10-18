"""Download SROIE, normalise the human labels and write a frozen vendor-disjoint split.

    python scripts/prepare_data.py              # clones the data, writes data/processed/
    python scripts/prepare_data.py --raw-dir X  # use an existing checkout with data/box and data/key

Why a vendor-disjoint split: SROIE has 626 labelled receipts but only 236
vendors, and one vendor alone has 45 receipts. A random split would put the
same vendors (same layout, same name, same address) in train and test, and the
fine-tuned model would score well by memorising them. Holding out whole vendors
measures what the business actually needs: handling layouts it has not seen.
Vendors with more than 12 receipts always stay in train, so the test set is
spread across many layouts instead of being dominated by two or three.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from slotfill.data import PROCESSED, ROOT, write_jsonl  # noqa: E402
from slotfill.normalize import normalize_date, normalize_text, normalize_total, text_key  # noqa: E402
from slotfill.schema import Receipt  # noqa: E402

SROIE_REPO = "https://github.com/zzzDavid/ICDAR-2019-SROIE.git"


def fetch(raw_dir: Path) -> str:
    if not (raw_dir / "data" / "key").exists():
        raw_dir.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--depth", "1", "--filter=blob:none", "--sparse", SROIE_REPO, str(raw_dir)], check=True)
        subprocess.run(["git", "-C", str(raw_dir), "sparse-checkout", "set", "data/box", "data/key"], check=True)
    try:
        return subprocess.run(["git", "-C", str(raw_dir), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def read_text(path: Path) -> str:
    data = path.read_bytes()
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return data.decode("latin-1")


def read_ocr(path: Path) -> str:
    """Each line is x1,y1,...,x4,y4,transcript; the transcript itself may contain commas."""
    lines = []
    for line in read_text(path).splitlines():
        parts = line.split(",", 8)
        if len(parts) == 9 and parts[8].strip():
            lines.append(parts[8].strip())
    return "\n".join(lines)


def build_record(doc_id: str, raw: dict, ocr: str) -> tuple[dict | None, str | None]:
    gold = {
        "company": normalize_text(raw.get("company")),
        "date": normalize_date(raw.get("date", "")),
        "address": normalize_text(raw.get("address")),
        "total": normalize_total(raw.get("total", "")),
    }
    for field in ("company", "date", "total"):
        if not gold[field]:
            return None, f"unreadable_{field}"
    if not ocr:
        return None, "empty_ocr"
    Receipt.model_validate(gold)  # gold must satisfy the same schema we grade against
    return {"id": doc_id, "vendor": text_key(gold["company"]), "ocr": ocr, "gold": gold, "raw_label": raw}, None


def vendor_split(records: list[dict], test_size: int, seed: int, max_vendor_docs: int) -> tuple[list[dict], list[dict]]:
    by_vendor: dict[str, list[dict]] = {}
    for r in records:
        by_vendor.setdefault(r["vendor"], []).append(r)
    vendors = sorted(by_vendor)
    random.Random(seed).shuffle(vendors)
    test_vendors: set[str] = set()
    n = 0
    for v in vendors:  # greedy fill: take a vendor only if it still fits
        size = len(by_vendor[v])
        # Very large vendors stay in train so a couple of layouts can't dominate the test set.
        if size <= max_vendor_docs and n + size <= test_size:
            test_vendors.add(v)
            n += size
        if n == test_size:
            break
    train = [r for r in records if r["vendor"] not in test_vendors]
    test = [r for r in records if r["vendor"] in test_vendors]
    return train, test


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw-dir", type=Path, default=ROOT / "data" / "raw" / "sroie")
    ap.add_argument("--test-size", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-vendor-docs-in-test", type=int, default=12)
    args = ap.parse_args()

    commit = fetch(args.raw_dir)
    key_dir, box_dir = args.raw_dir / "data" / "key", args.raw_dir / "data" / "box"

    records, dropped = [], Counter()
    for key_path in sorted(key_dir.glob("*.json")):
        doc_id = key_path.stem
        box_path = box_dir / f"{doc_id}.csv"
        if not box_path.exists():
            dropped["missing_ocr"] += 1
            continue
        rec, reason = build_record(doc_id, json.loads(read_text(key_path)), read_ocr(box_path))
        if rec is None:
            dropped[reason] += 1
            print(f"drop {doc_id}: {reason}  raw={json.loads(read_text(key_path))}")
            continue
        records.append(rec)

    train, test = vendor_split(records, args.test_size, args.seed, args.max_vendor_docs_in_test)
    assert not {r["vendor"] for r in train} & {r["vendor"] for r in test}, "vendor leakage"

    write_jsonl(PROCESSED / "train.jsonl", train)
    write_jsonl(PROCESSED / "test.jsonl", test)
    manifest = {
        "source": SROIE_REPO,
        "source_commit": commit,
        "labelled_docs": len(records) + sum(dropped.values()),
        "kept_docs": len(records),
        "dropped": dict(dropped),
        "split": "vendor-disjoint",
        "seed": args.seed,
        "max_vendor_docs_in_test": args.max_vendor_docs_in_test,
        "train_docs": len(train),
        "test_docs": len(test),
        "train_vendors": len({r["vendor"] for r in train}),
        "test_vendors": len({r["vendor"] for r in test}),
        "test_sha256": hashlib.sha256((PROCESSED / "test.jsonl").read_bytes()).hexdigest(),
    }
    (PROCESSED / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
