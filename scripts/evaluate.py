"""Score prediction files against the held-out gold labels and write the comparison table.

    python scripts/evaluate.py \
        --run "Claude Opus 5.5 (prompted)=results/preds_claude.jsonl" \
        --run "Qwen2.5-3B base (prompted)=results/preds_base.jsonl" \
        --run "Qwen2.5-3B + LoRA=results/preds_lora.jsonl"

Writes results/results.json, results/results.md and results/errors_<run>.jsonl.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

# Make the repo root importable so `from slotfill...` works when run as `python scripts/x.py`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from slotfill.data import PROCESSED, read_jsonl, write_jsonl  # noqa: E402
from slotfill.metrics import aggregate, cost_per_1k, percentile, score_example  # noqa: E402
from slotfill.schema import FIELDS  # noqa: E402


# Score one prediction file: returns a summary row for the table plus the list of failures.
def evaluate_run(name: str, path: Path, gold_by_id: dict[str, dict]) -> tuple[dict, list[dict]]:
    preds = read_jsonl(path)
    # Every test doc must have a prediction. Otherwise a system could look better by skipping hard docs.
    missing = set(gold_by_id) - {p["id"] for p in preds}
    if missing:
        raise SystemExit(f"{name}: {len(missing)} test docs have no prediction; rerun predict.py on the full split")
    # predict.py writes <name>.meta.json next to each prediction file (model, timing, pricing).
    meta_path = path.with_suffix(".meta.json")
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}

    rows, errors = [], []
    for p in preds:
        if p["id"] not in gold_by_id:
            continue
        gold = gold_by_id[p["id"]]
        # Validity + per-field correctness for this one document (see slotfill/metrics.py).
        row = score_example(gold, p.get("raw_output"))
        # A request that crashed is labelled as such, so it isn't confused with a bad model answer.
        if p.get("api_error"):
            row["error"] = "api_error"
        rows.append(row)
        # Keep every document with at least one wrong field, for error analysis afterwards.
        if not row["all_fields"]:
            errors.append({"id": p["id"], "error": row["error"],
                           "wrong": [f for f in FIELDS if not row[f]],
                           "gold": gold, "raw_output": p.get("raw_output")})

    # Latency only over requests that succeeded; failed requests would distort it.
    lat = [p["latency_s"] for p in preds if p.get("latency_s") is not None and not p.get("api_error")]
    # Accuracy numbers from aggregate(), plus latency percentiles and cost per 1,000 docs.
    summary = aggregate(rows) | {
        "name": name,
        "file": str(path),
        "model": meta.get("model"),
        "latency_p50_s": percentile(lat, 0.50),
        "latency_p95_s": percentile(lat, 0.95),
        "concurrency": meta.get("concurrency"),
        "cost_per_1k_usd": cost_per_1k(meta, preds),
    }
    return summary, errors


# 0.873 -> "87.3%"
def fmt_pct(x: float) -> str:
    return f"{100 * x:.1f}%"


# Build the comparison table (one row per system) as Markdown, as shown in the README.
def to_markdown(summaries: list[dict]) -> str:
    head = ("| System | Schema valid | Field acc (mean) | company | date | address | total | All 4 fields (95% CI) "
            "| Cost / 1k docs | p50 / p95 latency |")
    lines = [head, "|" + "---|" * 10]
    for s in summaries:
        lo, hi = s["all_fields_ci95"]
        c = s["cost_per_1k_usd"]
        # Show cents for cheap runs (e.g. $0.012) and dollars otherwise.
        cost = "n/a" if c is None else f"${c:.2f}" if c >= 1 else f"${c:.3f}"
        lines.append(
            f"| {s['name']} | {fmt_pct(s['schema_valid'])} | {fmt_pct(s['field_acc_mean'])} "
            f"| {fmt_pct(s['acc_company'])} | {fmt_pct(s['acc_date'])} | {fmt_pct(s['acc_address'])} "
            f"| {fmt_pct(s['acc_total'])} | {fmt_pct(s['all_fields'])} ({fmt_pct(lo)}–{fmt_pct(hi)}) "
            f"| {cost} | {s['latency_p50_s']:.2f}s / {s['latency_p95_s']:.2f}s |")
    n = summaries[0]["n"] if summaries else 0
    lines.append("")
    lines.append(f"n = {n} held-out receipts from vendors never seen in training. Invalid outputs score 0 on every field. "
                 "Latency is per request at the concurrency recorded in each run's .meta.json.")
    return "\n".join(lines) + "\n"


# "Qwen2.5-3B + LoRA (424 docs)" -> "qwen2_5_3b_lora_424_docs", safe to use in a file name.
def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # Each --run is "Display name=path/to/preds.jsonl"; pass one per system to compare.
    ap.add_argument("--run", action="append", required=True, help='"Display name=path/to/preds.jsonl"')
    ap.add_argument("--split", type=Path, default=PROCESSED / "test.jsonl")
    ap.add_argument("--out-dir", type=Path, default=Path("results"))
    args = ap.parse_args()

    # The correct answers, looked up by document id.
    gold_by_id = {r["id"]: r["gold"] for r in read_jsonl(args.split)}
    summaries = []
    for spec in args.run:
        # Split "name=path" at the first "=".
        name, _, path = spec.partition("=")
        summary, errors = evaluate_run(name.strip(), Path(path.strip()), gold_by_id)
        summaries.append(summary)
        write_jsonl(args.out_dir / f"errors_{slug(summary['name'])}.jsonl", errors)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    # Write the table both as JSON (for scripts) and Markdown (for people), and print it.
    (args.out_dir / "results.json").write_text(json.dumps(summaries, indent=2) + "\n")
    md = to_markdown(summaries)
    (args.out_dir / "results.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
