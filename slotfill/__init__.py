"""Slotfill: LoRA fine-tuned receipt/invoice extraction vs a prompted frontier model."""
# This package holds the shared building blocks used by every script in scripts/:
#   data.py      reading/writing the JSONL splits and the exact training target string
#   normalize.py turning messy printed dates, totals and text into one canonical form
#   schema.py    the strict JSON schema an output must satisfy to count as "valid"
#   prompts.py   the single prompt every system (small, large, fine-tuned) receives
#   training.py  tokenisation with the prompt masked out of the loss
#   metrics.py   per-document scoring, aggregation, confidence intervals and cost
