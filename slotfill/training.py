"""Turn (OCR, gold) pairs into token ids with the prompt masked out of the loss.

Only the assistant's JSON answer contributes to the loss; the model is not
trained to reproduce the system prompt or the OCR text. Kept free of torch
imports so it can be unit-tested anywhere.
"""

from __future__ import annotations

import random

from .data import target_json
from .prompts import build_messages

# Hugging Face's convention: label positions set to -100 are ignored by the loss.
IGNORE_INDEX = -100


def tokenize_example(tokenizer, ocr: str, gold: dict, max_len: int) -> dict | None:
    # Render the conversation twice with the model's own chat template:
    #   prompt_text = system + user + the "assistant is about to speak" marker
    #   full_text   = the same, plus the correct JSON answer
    prompt_text = tokenizer.apply_chat_template(build_messages(ocr), tokenize=False, add_generation_prompt=True)
    full_text = tokenizer.apply_chat_template(build_messages(ocr, target_json(gold)), tokenize=False)
    # The masking below assumes the prompt is an exact prefix of the full text. If a
    # template ever breaks that, fail loudly instead of silently masking the wrong tokens.
    if not full_text.startswith(prompt_text):
        raise ValueError("chat template does not render the prompt as a prefix of the full conversation")

    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
    if len(full_ids) > max_len:
        return None  # drop rather than truncate: a cut-off answer teaches invalid JSON

    # Labels: -100 for every prompt token (no loss there), the real token ids for the answer.
    # So the model is only graded on producing the JSON, not on copying the OCR.
    labels = [IGNORE_INDEX] * len(prompt_ids) + full_ids[len(prompt_ids):]
    return {"input_ids": full_ids, "attention_mask": [1] * len(full_ids), "labels": labels}


def split_train_dev(rows: list[dict], dev_frac: float, n_train: int | None, seed: int) -> tuple[list[dict], list[dict]]:
    """A small dev set carved out of *train* for eval loss. The test split is never touched here."""
    rows = rows[:]  # copy so shuffling doesn't change the caller's list
    random.Random(seed).shuffle(rows)  # seeded, so the dev set is the same every run
    n_dev = max(1, int(len(rows) * dev_frac))  # at least one dev doc
    dev, train = rows[:n_dev], rows[n_dev:]
    # Data-size ablation: keep only the first N training docs. The dev set is cut
    # first, so it is identical for the full run and the ablation.
    if n_train is not None:
        train = train[:n_train]
    return train, dev
