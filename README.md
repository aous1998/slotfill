# Slotfill

**LoRA fine-tuned receipt/invoice extraction, benchmarked against a prompted frontier LLM.**
Python · Hugging Face Transformers · PEFT · Unsloth · vLLM · Pydantic

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aous1998/slotfill/blob/main/notebooks/slotfill_colab.ipynb)

## The question

A back office processes tens of thousands of supplier documents a month across hundreds of layouts.
A frontier model extracts them well, but it costs a lot per document, and when its output
doesn't match the schema, the accounting import breaks.

**Can a small open model with a LoRA adapter keep to the schema at a fraction of the cost? And when should you just keep the prompt?**

This repo answers that with numbers on the same held-out set for every system:

| System | What it is |
|---|---|
| Claude Opus 5.5, prompted | Frontier API, same instructions, free-form output |
| Claude Opus 5.5, structured output | Same, with decoding constrained to the JSON schema |
| Qwen2.5-3B-Instruct, prompted | The small open model before tuning |
| Qwen2.5-3B + LoRA (424 docs) | The adapter, served by vLLM |
| Qwen2.5-3B + LoRA (100 docs) | Ablation: how much labelled data is actually needed |

## Results

> **Pending.** Run [`notebooks/slotfill_colab.ipynb`](notebooks/slotfill_colab.ipynb) on a free Colab T4
> (about 1 hour, about $2–8 of API credit for the baseline). Then paste `results/results.md` here.

| System | Schema valid | Field acc (mean) | All 4 fields (95% CI) | Cost / 1k docs | p95 latency |
|---|---|---|---|---|---|
| Claude Opus 5.5, prompted | – | – | – | – | – |
| Claude Opus 5.5, structured output | – | – | – | – | – |
| Qwen2.5-3B base, prompted | – | – | – | – | – |
| Qwen2.5-3B + LoRA r16 (424 docs) | – | – | – | – | – |
| Qwen2.5-3B + LoRA r16 (100 docs) | – | – | – | – | – |

How to read the table:

- **n = 200** held-out receipts from **124 vendors never seen in training**.
- An invalid output scores 0 on every field.
- With n = 200, differences under about 5 points are inside the confidence interval. Don't over-read them.
- About 18% of test receipts have a gold value that isn't verbatim in the OCR (see [LABELING.md](LABELING.md#audit-how-much-can-the-ocr-actually-support-scriptsaudit_labelspy)), so exact match tops out near 82–96% depending on the field.
- The headline is held-out accuracy, never training loss.

### Recommendation

> _Write this after the run, using the rule below. If the prompt wins, say so and ship the prompt._

Decision rule:

1. **Fine-tune wins** if, versus the structured-output baseline, the adapter is within the CI on all-4-fields accuracy, has equal or better schema validity, and is at least 10× cheaper per 1k docs at your volume. You then accept the cost of owning a GPU deployment and a retraining loop.
2. **The prompt wins** if structured output already gives ~100% validity and the adapter trails on accuracy by more than the CI. Pay the API cost and skip the infrastructure.
3. **Volume matters:** self-hosted cost is per GPU-hour, not per token. At low volume an idle GPU is pure overhead. `cost_per_1k` assumes the batch keeps the GPU busy.

## How it works

```
SROIE (human labels + OCR)
        │  scripts/prepare_data.py   normalise → vendor-disjoint split → frozen test set (sha256)
        │  scripts/audit_labels.py   how much of each gold field is recoverable from the OCR
        ▼
data/processed/{train,test}.jsonl
        │  scripts/train.py          QLoRA on Qwen2.5-3B (Unsloth or plain PEFT), loss on the answer only
        ▼
adapters/slotfill-r16/              standard PEFT adapter + train_log.json
        │  scripts/serve_vllm.sh     vllm serve --enable-lora: base + N adapters on one GPU
        ▼
scripts/predict.py                  same prompt, same 200 docs, for vLLM and the Claude API
        │                           records raw output, latency, tokens
        ▼
scripts/evaluate.py                 schema validity · field accuracy · CI · cost/1k · p95 latency
```

### The correctness signal

[`slotfill/schema.py`](slotfill/schema.py) defines a strict Pydantic model:

```python
class Receipt(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    company: str                        # non-empty
    date: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}$")   # and a real calendar date
    address: str | None
    total: str = Field(pattern=r"^\d+\.\d{2}$")         # "60.30", no currency
```

An output passes only if the raw text is one JSON object that validates: no markdown fences, no
extra keys, no `"RM 60.3"`. That is the bar an accounting import applies.

### Training setup (logged per run in `adapters/<name>/train_log.json`)

| Setting | Value |
|---|---|
| Base model | `Qwen/Qwen2.5-3B-Instruct`, 4-bit NF4 (QLoRA) |
| LoRA | r = 16, alpha = 32, dropout 0, all attention + MLP projections |
| Optimiser | AdamW 8-bit, lr 2e-4, cosine schedule, 5 warmup steps |
| Batch | 4 × 4 gradient accumulation = 16 |
| Epochs | 3 |
| Loss | assistant answer tokens only; prompt and OCR masked |
| Data | 424 train receipts (10% held as dev for eval loss); test never loaded |
| **Ablation** | same settings with 100 training receipts (`--n-train 100`) |

## Run it

**Colab (recommended):** open the notebook above and run top to bottom on a T4.

**Locally / rented GPU:**

```bash
git clone https://github.com/aous1998/slotfill.git && cd slotfill
pip install -r requirements.txt
pytest                                   # 28 tests, CPU only

# data is committed; to rebuild it from source:
python scripts/prepare_data.py && python scripts/audit_labels.py

# train (GPU env)
pip install -r requirements-train.txt
python scripts/train.py --out adapters/slotfill-r16
python scripts/train.py --n-train 100 --out adapters/slotfill-n100

# serve + predict (separate env: vLLM pins its own torch)
pip install -r requirements-serve.txt -r requirements.txt
bash scripts/serve_vllm.sh &
python scripts/predict.py --provider vllm --model Qwen/Qwen2.5-3B-Instruct --gpu-hourly 0.40 --out results/preds_base.jsonl
python scripts/predict.py --provider vllm --model slotfill-r16              --gpu-hourly 0.40 --out results/preds_lora_r16.jsonl

# frontier baseline
export ANTHROPIC_API_KEY=...
python scripts/predict.py --provider anthropic --model claude-opus-5-5 --out results/preds_claude.jsonl
python scripts/predict.py --provider anthropic --model claude-opus-5-5 --structured --out results/preds_claude_structured.jsonl

python scripts/evaluate.py \
  --run "Claude Opus 5.5, prompted=results/preds_claude.jsonl" \
  --run "Claude Opus 5.5, structured output=results/preds_claude_structured.jsonl" \
  --run "Qwen2.5-3B base, prompted=results/preds_base.jsonl" \
  --run "Qwen2.5-3B + LoRA r16 (424 docs)=results/preds_lora_r16.jsonl"
```

Use `--limit 10` on `predict.py` for a cheap smoke test.

## Repo layout

```
slotfill/            schema.py · normalize.py · prompts.py · metrics.py · training.py · data.py
scripts/             prepare_data · audit_labels · train · serve_vllm.sh · predict · evaluate
data/processed/      train.jsonl · test.jsonl · manifest.json · audit_report.json · audit_flags.jsonl
notebooks/           slotfill_colab.ipynb
tests/               normalisation, schema, metrics, masking, split integrity
LABELING.md          data source, normalisation rules, split, audit, limitations
```

## Design choices worth defending

- **Human labels only.** No LLM-generated training data, so the comparison isn't the model grading its own homework.
- **Vendor-disjoint test set.** A random split leaks layouts: one vendor has 44 receipts. The test set has 124 vendors the adapter has never seen.
- **Same prompt everywhere.** Any gap comes from fine-tuning, not prompt wording.
- **Strongest baseline included.** Structured output often fixes schema validity on its own. If it does, that is the honest answer.
- **Cost from measurements.** API cost uses the token counts the API returns. Self-hosted cost uses GPU $/hour × measured wall-clock.

## Data and licence

Code: MIT. Data: derived from the ICDAR 2019 SROIE dataset via
[zzzDavid/ICDAR-2019-SROIE](https://github.com/zzzDavid/ICDAR-2019-SROIE) (MIT), for research use.
