"""Scoring: schema validity, field accuracy, latency and cost.

An invalid output scores zero on every field: if the accounting import rejects
the record, it does not matter that three of four fields were right.
"""

from __future__ import annotations

import math
import random
from difflib import SequenceMatcher

from .normalize import normalize_date, normalize_total, text_key
from .schema import FIELDS, validate_output


def _field_key(field: str, value) -> str:
    # Turn a gold or predicted value into a comparison key so that harmless differences
    # don't count as errors ("60.3" vs "60.30", "jalan  sagu" vs "JALAN SAGU").
    if value is None:
        return ""
    if field == "date":
        # Fall back to the plain text key if the date can't be parsed.
        return normalize_date(str(value)) or text_key(value)
    if field == "total":
        return normalize_total(str(value)) or text_key(value)
    return text_key(value)


def score_example(gold: dict, raw_output: str | None) -> dict:
    # Score one document: is the output valid, and which fields are exactly right?
    v = validate_output(raw_output)
    row = {"valid": v.valid, "error": v.error}
    for f in FIELDS:
        if not v.valid:
            # Invalid output: every field is wrong, no partial credit.
            row[f] = False
            row[f"{f}_sim"] = 0.0
            continue
        g, p = _field_key(f, gold[f]), _field_key(f, v.data.get(f))
        row[f] = g == p  # exact match after normalisation
        # Similarity 0..1 (how close it got), useful for error analysis; not used for accuracy.
        row[f"{f}_sim"] = 1.0 if g == p else SequenceMatcher(None, g, p).ratio()
    # The headline metric: all four fields right on the same document.
    row["all_fields"] = all(row[f] for f in FIELDS)
    return row


def percentile(values: list[float], q: float) -> float:
    # Linear-interpolation percentile (q=0.95 -> p95). Used for latency.
    if not values:
        return float("nan")
    s = sorted(values)
    k = (len(s) - 1) * q  # fractional position in the sorted list
    lo, hi = math.floor(k), math.ceil(k)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def bootstrap_ci(flags: list[bool], n_boot: int = 2000, seed: int = 0) -> tuple[float, float]:
    """95% percentile-bootstrap CI for a rate. 200 test docs means ±~5pp, so say so."""
    # Resample the test set with replacement 2000 times, compute the rate each time, and
    # take the 2.5th and 97.5th percentiles. This shows how much the score could move by chance.
    if not flags:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    n = len(flags)
    means = sorted(sum(rng.choices(flags, k=n)) / n for _ in range(n_boot))
    return means[int(0.025 * n_boot)], means[int(0.975 * n_boot) - 1]


def aggregate(rows: list[dict]) -> dict:
    # Turn the per-document rows into one summary for the results table.
    n = len(rows)
    out = {"n": n, "schema_valid": sum(r["valid"] for r in rows) / n}
    for f in FIELDS:
        out[f"acc_{f}"] = sum(r[f] for r in rows) / n  # exact-match accuracy per field
        out[f"sim_{f}"] = sum(r[f"{f}_sim"] for r in rows) / n  # average closeness per field
    out["field_acc_mean"] = sum(out[f"acc_{f}"] for f in FIELDS) / len(FIELDS)
    out["all_fields"] = sum(r["all_fields"] for r in rows) / n
    out["all_fields_ci95"] = bootstrap_ci([r["all_fields"] for r in rows])
    out["schema_valid_ci95"] = bootstrap_ci([r["valid"] for r in rows])
    # Count why outputs were invalid (bad JSON, schema failure, empty...).
    errors: dict[str, int] = {}
    for r in rows:
        if r["error"]:
            errors[r["error"]] = errors.get(r["error"], 0) + 1
    out["invalid_by_reason"] = errors
    return out


def cost_per_1k(meta: dict, preds: list[dict]) -> float | None:
    """API runs: token prices. Self-hosted runs: GPU $/hour x wall-clock for the batch."""
    pricing = meta.get("pricing") or {}
    n = len(preds)
    if not n:
        return None
    if "gpu_hourly_usd" in pricing:
        # Self-hosted: you pay for GPU time, so cost = hourly rate x hours the batch took,
        # divided by docs processed, scaled to 1,000 docs.
        return pricing["gpu_hourly_usd"] * meta["wall_seconds"] / 3600 / n * 1000
    if "input_per_mtok" in pricing:
        # API: you pay per token, priced per million input and output tokens.
        tin = sum(p.get("input_tokens") or 0 for p in preds)
        tout = sum(p.get("output_tokens") or 0 for p in preds)
        usd = tin / 1e6 * pricing["input_per_mtok"] + tout / 1e6 * pricing["output_per_mtok"]
        return usd / n * 1000
    return None  # no pricing recorded for this run
