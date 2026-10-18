"""Guards on the committed, frozen split."""

import hashlib
import json

from slotfill.data import PROCESSED, read_jsonl
from slotfill.schema import Receipt


def test_split_is_vendor_disjoint_and_frozen():
    train, test = read_jsonl(PROCESSED / "train.jsonl"), read_jsonl(PROCESSED / "test.jsonl")
    manifest = json.loads((PROCESSED / "manifest.json").read_text())
    assert len(test) == manifest["test_docs"] == 200
    assert not {r["vendor"] for r in train} & {r["vendor"] for r in test}
    assert not {r["id"] for r in train} & {r["id"] for r in test}
    assert hashlib.sha256((PROCESSED / "test.jsonl").read_bytes()).hexdigest() == manifest["test_sha256"]


def test_every_gold_label_passes_the_schema():
    for split in ("train", "test"):
        for r in read_jsonl(PROCESSED / f"{split}.jsonl"):
            Receipt.model_validate(r["gold"])
