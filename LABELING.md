# Labelling process

## Where the labels come from

All labels are **human annotations** from the ICDAR 2019 SROIE competition
(Task 3, key information extraction from scanned receipts), as published in
[zzzDavid/ICDAR-2019-SROIE](https://github.com/zzzDavid/ICDAR-2019-SROIE)
(MIT licence, commit `27be427`). Each receipt has:

- `data/box/<id>.csv`: OCR lines with their bounding boxes. Only the text goes to the model.
- `data/key/<id>.json`: four fields typed by an annotator from the receipt image: `company`, `date`, `address`, `total`.

**No label was generated or corrected by an LLM.** That avoids the main trap in
this kind of project: training on synthetic labels from the same model family
you are trying to beat, then grading against them.

## Normalisation (scripted, `slotfill/normalize.py`)

Annotators copied values exactly as printed. The target schema needs canonical
values, so every label goes through one deterministic pass:

| Field | Raw examples | Canonical |
|---|---|---|
| date | `25/12/2018`, `12-01-19`, `05 MAR 2018`, `20180304`, `OCT 3, 2016` | ISO `YYYY-MM-DD` |
| total | `RM 29.70`, `$8.20`, `1,007.50`, `43.7` | `"29.70"`: digits, 2 decimals, no currency |
| company, address | free text | whitespace collapsed, otherwise verbatim |

Rules for ambiguous cases:

- Malaysian receipts print dates **day-first**, so `05/03/2018` → `2018-03-05`.
- A date is read month-first only when day-first is impossible (`12/28/2017`).
- 8-digit dates are tried as `YYYYMMDD` first, then `DDMMYYYY`.
- Years outside 2000–2030 are rejected.

The raw labels contain 17 distinct date formats and 15 total formats. The
parsing is unit-tested in `tests/test_core.py`.

## Dropped records

626 labelled receipts → **624 kept**. Two were dropped, and the reason is logged by `prepare_data.py`:

| id | reason |
|---|---|
| 033 | `total` label is empty |
| 347 | `total` label is negative (`-1.73`, a refund slip) |

Every kept gold record passes the same `Receipt` schema the models are graded
against. `tests/test_data_split.py` checks this.

## Split: vendor-disjoint, frozen

624 receipts come from only ~230 vendors, and the largest vendor has 44
receipts. With a random split, the test set would contain the same layouts,
names and addresses as training, and a fine-tuned model would look great by
memorising them.

So the split is **by vendor**:

- **Test: 200 receipts, 124 vendors, none of which appear in training.**
- Vendors with more than 12 receipts always stay in train, so no single layout dominates the test set.
- Train: 424 receipts, 108 vendors. 10% of these are carved off as a dev set for eval loss during training.
- Seed 42. The test file's sha256 is stored in `data/processed/manifest.json`, and a test fails if the file changes.

The test split is never read by `train.py`.

## Audit: how much can the OCR actually support? (`scripts/audit_labels.py`)

Labels were typed from the image, but the model only sees the OCR transcript,
which has its own errors (e.g. OCR reads `SDN BND` where the receipt and the
label say `SDN BHD`). The audit checks whether each gold value can be found in
the OCR text:

| Field | Train | Test | Notes |
|---|---|---|---|
| total | 100% | 100% | |
| date | 99.8% | 100% | |
| company (exact) | 98.4% | 96.0% | 100% within 0.9 string similarity |
| address (exact) | 90.1% | 85.0% | 100% within 0.9 similarity: OCR typos, label punctuation |
| **all four exact** | **88.4%** | **82.0%** | |

**What this means for the results:** about 18% of test receipts contain at
least one gold value that does not appear verbatim in the input. Exact-match
accuracy on company and address is therefore capped below 100% for *any*
model. To make that visible, `evaluate.py` also reports a similarity score
(`sim_*` in `results.json`) next to exact match. The 85 flagged documents are
listed in `data/processed/audit_flags.jsonl`.

## Human spot-check

`audit_labels.py` writes `data/processed/manual_review_sample.csv`: 50 random
test receipts with columns for a reviewer's verdict. To spot-check, open each
receipt image from the SROIE repo (`data/img/<id>.jpg`) and mark
`label_correct`.

## Known limitations

- **624 receipts, not 1,000+.** The public SROIE release with labels and OCR
  has 626 training receipts. The official test keys were released separately.
  Adding them, or another labelled set such as CORD, is the natural next step.
  The pipeline takes any `data/box` + `data/key` folder via `--raw-dir`.
- **Receipts, not freight invoices.** The extraction problem has the same shape:
  noisy OCR, many layouts, a strict schema. The fields are simpler than a
  carrier invoice's (no line items, no carrier/load IDs).
- **Similar layouts across vendors.** Some vendors share layouts, such as
  `MR D.I.Y. (JOHOR)` and `MR. D.I.Y. (M)`. The split is by company name, so
  this residual similarity is not removed.
