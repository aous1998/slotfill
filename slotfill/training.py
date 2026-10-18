"""Turn (OCR, gold) pairs into token ids with the prompt masked out of the loss.

Only the assistant's JSON answer contributes to the loss; the model is not
trained to reproduce the system prompt or the OCR text. Kept free of torch
imports so it can be unit-tested anywhere.
"""

from __future__ import annotations

import random

from .data import target_json
from .prompts import build_messages

IGNORE_INDEX = -100


def tokenize_example(tokenizer, ocr: str, gold: dict, max_len: int) -> dict | None:
    prompt_text = tokenizer.apply_chat_template(build_messages(ocr), tokenize=False, add_generation_prompt=True)
    full_text = tokenizer.apply_chat_template(build_messages(ocr, target_json(gold)), tokenize=False)
    if not full_text.startswith(prompt_text):
        raise ValueError("chat template does not render the prompt as a prefix of the full conversation")

    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
    if len(full_ids) > max_len:
        return None  # drop rather than truncate: a cut-off answer teaches invalid JSON

    labels = [IGNORE_INDEX] * len(prompt_ids) + full_ids[len(prompt_ids):]
    return {"input_ids": full_ids, "attention_mask": [1] * len(full_ids), "labels": labels}


def split_train_dev(rows: list[dict], dev_frac: float, n_train: int | None, seed: int) -> tuple[list[dict], list[dict]]:
    """A small dev set carved out of *train* for eval loss. The test split is never touched here."""
    rows = rows[:]
    random.Random(seed).shuffle(rows)
    n_dev = max(1, int(len(rows) * dev_frac))
    dev, train = rows[:n_dev], rows[n_dev:]
    if n_train is not None:
        train = train[:n_train]
    return train, dev
