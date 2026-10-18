"""Train a QLoRA adapter for receipt extraction.

    python scripts/train.py --rank 16 --alpha 32 --lr 2e-4 --out adapters/slotfill-r16
    python scripts/train.py --n-train 100 --out adapters/slotfill-n100   # data-size ablation

Backends:
    --backend unsloth  (default) Unsloth's FastLanguageModel: ~2x faster, fits a 3B model on a free T4.
    --backend peft     plain transformers + bitsandbytes + PEFT, no Unsloth dependency.

Both produce a standard PEFT adapter (adapter_config.json + adapter_model.safetensors)
that vLLM loads with --enable-lora. Hyperparameters, data counts and the loss
history are written to <out>/train_log.json. Training loss is logged for
debugging only: the result that matters comes from evaluate.py on the held-out set.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from slotfill.data import PROCESSED, read_jsonl  # noqa: E402
from slotfill.training import split_train_dev, tokenize_example  # noqa: E402

TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def load_unsloth(args):
    from unsloth import FastLanguageModel  # must be imported before transformers

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model, max_seq_length=args.max_len, load_in_4bit=True, dtype=None)
    model = FastLanguageModel.get_peft_model(
        model, r=args.rank, lora_alpha=args.alpha, lora_dropout=args.dropout,
        target_modules=TARGET_MODULES, bias="none",
        use_gradient_checkpointing="unsloth", random_state=args.seed)
    return model, tokenizer


def load_peft(args, bf16: bool):
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16 if bf16 else torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, quantization_config=bnb, device_map="auto")
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model = get_peft_model(model, LoraConfig(
        r=args.rank, lora_alpha=args.alpha, lora_dropout=args.dropout,
        target_modules=TARGET_MODULES, bias="none", task_type="CAUSAL_LM"))
    return model, tokenizer


class ListDataset:
    def __init__(self, items: list[dict]):
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> dict:
        return self.items[i]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--backend", choices=["unsloth", "peft"], default="unsloth")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--n-train", type=int, help="Use only the first N training docs (ablation).")
    ap.add_argument("--dev-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=3407)
    args = ap.parse_args()

    import torch

    bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    if args.backend == "unsloth":
        model, tokenizer = load_unsloth(args)
    else:
        model, tokenizer = load_peft(args, bf16)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    from transformers import DataCollatorForSeq2Seq, Trainer, TrainingArguments

    train_rows, dev_rows = split_train_dev(read_jsonl(PROCESSED / "train.jsonl"), args.dev_frac, args.n_train, args.seed)
    tok = lambda rows: [x for r in rows if (x := tokenize_example(tokenizer, r["ocr"], r["gold"], args.max_len))]  # noqa: E731
    train_ds, dev_ds = ListDataset(tok(train_rows)), ListDataset(tok(dev_rows))
    print(f"train {len(train_ds)}/{len(train_rows)} docs, dev {len(dev_ds)}/{len(dev_rows)} docs (rest exceeded --max-len)")

    targs = TrainingArguments(
        output_dir=str(args.out / "checkpoints"),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=5,
        weight_decay=0.0,
        logging_steps=5,
        eval_strategy="epoch",
        save_strategy="no",
        bf16=bf16,
        fp16=torch.cuda.is_available() and not bf16,
        optim="adamw_8bit",
        seed=args.seed,
        report_to="none",
        remove_unused_columns=False,
    )
    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        data_collator=DataCollatorForSeq2Seq(tokenizer, padding=True, label_pad_token_id=-100),
    )

    t0 = time.time()
    trainer.train()
    elapsed = time.time() - t0

    args.out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(args.out))
    tokenizer.save_pretrained(str(args.out))
    log = {
        "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "target_modules": TARGET_MODULES,
        "train_docs": len(train_ds),
        "dev_docs": len(dev_ds),
        "train_seconds": round(elapsed, 1),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "peak_gpu_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2) if torch.cuda.is_available() else None,
        "log_history": trainer.state.log_history,
    }
    (args.out / "train_log.json").write_text(json.dumps(log, indent=2) + "\n")
    print(f"saved adapter -> {args.out} ({elapsed / 60:.1f} min)")


if __name__ == "__main__":
    main()
