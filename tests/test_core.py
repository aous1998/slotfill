import json

import pytest

from slotfill.data import target_json
from slotfill.metrics import aggregate, cost_per_1k, percentile, score_example
from slotfill.normalize import normalize_date, normalize_total, text_key
from slotfill.schema import validate_output
from slotfill.training import IGNORE_INDEX, split_train_dev, tokenize_example

GOLD = {"company": "BOOK TA .K (TAMAN DAYA) SDN BHD", "date": "2018-12-25",
        "address": "NO.53 55,57 & 59, JALAN SAGU 18, TAMAN DAYA, 81100 JOHOR BAHRU, JOHOR.", "total": "9.00"}


@pytest.mark.parametrize("raw,expected", [
    ("25/12/2018", "2018-12-25"),
    ("12-01-19", "2019-01-12"),
    ("05 MAR 2018", "2018-03-05"),
    ("24-MAR-2018", "2018-03-24"),
    ("2018-03-23", "2018-03-23"),
    ("20180304", "2018-03-04"),
    ("25032018", "2018-03-25"),
    ("(06/12/2016)", "2016-12-06"),
    ("OCT 3, 2016", "2016-10-03"),
    ("12/28/2017", "2017-12-28"),  # day-first impossible -> month-first
    ("11.02.18", "2018-02-11"),
    ("31/02/2018", None),
    ("hello", None),
])
def test_normalize_date(raw, expected):
    assert normalize_date(raw) == expected


@pytest.mark.parametrize("raw,expected", [
    ("9.00", "9.00"), ("RM 29.70", "29.70"), ("$8.20", "8.20"), ("1,007.50", "1007.50"),
    ("43.7", "43.70"), ("-1.73", None), ("", None),
])
def test_normalize_total(raw, expected):
    assert normalize_total(raw) == expected


def test_text_key_ignores_spacing_and_case():
    assert text_key("Jalan Sagu 18 , Taman Daya.") == text_key("JALAN SAGU 18,TAMAN DAYA")


def test_validate_output():
    assert validate_output(target_json(GOLD)).valid
    assert validate_output("```json\n" + target_json(GOLD) + "\n```").error == "json"
    assert validate_output("[1,2]").error == "not_object"
    assert validate_output(json.dumps(GOLD | {"date": "25/12/2018"})).error == "schema"
    assert validate_output(json.dumps(GOLD | {"date": "2018-02-31"})).error == "schema"
    assert validate_output(json.dumps(GOLD | {"total": "RM9.00"})).error == "schema"
    assert validate_output(json.dumps(GOLD | {"currency": "MYR"})).error == "schema"
    assert validate_output(json.dumps(GOLD | {"address": None})).valid
    no_addr = {k: v for k, v in GOLD.items() if k != "address"}
    assert validate_output(json.dumps(no_addr)).error == "schema"


def test_score_example():
    perfect = score_example(GOLD, target_json(GOLD))
    assert perfect["valid"] and perfect["all_fields"]

    near = score_example(GOLD, json.dumps(GOLD | {"company": "book ta .k (taman daya) sdn bhd", "total": "9.10"}))
    assert near["company"] and not near["total"] and not near["all_fields"]
    assert 0 < near["total_sim"] < 1

    invalid = score_example(GOLD, "Sure! " + target_json(GOLD))
    assert not invalid["valid"] and not any(invalid[f] for f in ("company", "date", "address", "total"))


def test_aggregate_and_cost():
    rows = [score_example(GOLD, target_json(GOLD)), score_example(GOLD, "nope")]
    agg = aggregate(rows)
    assert agg["schema_valid"] == 0.5 and agg["all_fields"] == 0.5
    assert agg["invalid_by_reason"] == {"json": 1}

    preds = [{"input_tokens": 1000, "output_tokens": 100}] * 2
    api = cost_per_1k({"pricing": {"input_per_mtok": 4.0, "output_per_mtok": 20.0}}, preds)
    assert api == pytest.approx((1000 * 4 + 100 * 20) / 1e6 * 1000)
    gpu = cost_per_1k({"pricing": {"gpu_hourly_usd": 0.36}, "wall_seconds": 10}, preds)
    assert gpu == pytest.approx(0.36 * 10 / 3600 / 2 * 1000)
    assert percentile([1, 2, 3, 4], 0.5) == 2.5


class FakeTokenizer:
    """Character-level stand-in for a HF tokenizer with a ChatML-style template."""

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        s = "".join(f"<{m['role']}>{m['content']}</s>" for m in messages)
        return s + ("<assistant>" if add_generation_prompt else "")

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(c) for c in text]}


def test_tokenize_masks_prompt():
    tok = FakeTokenizer()
    ex = tokenize_example(tok, "OCR LINE", GOLD, max_len=10_000)
    answer = target_json(GOLD) + "</s>"
    n_answer = len(answer)
    assert ex["labels"][-n_answer:] == [ord(c) for c in answer]
    assert all(t == IGNORE_INDEX for t in ex["labels"][:-n_answer])
    assert tokenize_example(tok, "OCR LINE", GOLD, max_len=50) is None


def test_split_train_dev_is_deterministic():
    rows = [{"id": str(i)} for i in range(100)]
    a = split_train_dev(rows, 0.1, None, 1)
    b = split_train_dev(rows, 0.1, None, 1)
    assert a == b and len(a[1]) == 10 and len(a[0]) == 90
    assert len(split_train_dev(rows, 0.1, 25, 1)[0]) == 25
