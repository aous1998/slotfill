# Slotfill

**Fine-tuning a small open model to extract invoice data, and testing whether it beats a prompted frontier LLM.**

Python · Hugging Face Transformers · PEFT · Unsloth · vLLM · Pydantic

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aous1998/slotfill/blob/main/notebooks/slotfill_colab.ipynb)

## The problem

Companies process thousands of invoices a month. A large LLM like Claude extracts the data well,
but it costs a lot per document. When its output doesn't match the expected format, the
accounting import breaks.

**Can a small 3B model with a LoRA adapter do the same job, in the exact format, for much less money?**

## What it does

1. **Takes** the OCR text of a receipt or invoice.
2. **Extracts** four fields into strict JSON: `company`, `date`, `address`, `total`.
3. **Compares** five systems on the same 200 held-out documents:
   - Claude Opus 5.5 with a prompt
   - Claude Opus 5.5 with structured (schema-constrained) output
   - Qwen2.5-3B with a prompt and no fine-tuning
   - Qwen2.5-3B + LoRA adapter trained on 424 documents
   - Qwen2.5-3B + LoRA adapter trained on 100 documents (ablation: how much data is needed?)
4. **Measures**, for each system:
   - schema validity
   - accuracy per field
   - cost per 1,000 documents
   - p95 latency

## How it works

```
Receipts (human labels)  →  clean + split  →  train LoRA adapter  →  serve with vLLM  →  compare vs Claude
   prepare_data.py                              train.py              serve_vllm.sh      predict.py + evaluate.py
```

- **Data:** 624 real receipts from the SROIE dataset, labelled by humans. No AI-generated labels.
- **Strict schema:** a Pydantic model. An output passes only if it is clean JSON with an ISO date and a total like `"60.30"`.
- **Training:** QLoRA on Qwen2.5-3B (r=16, alpha=32, lr 2e-4, 3 epochs). The loss counts only the answer, not the prompt. It runs on a free Colab T4.
- **Serving:** vLLM serves the base model and several adapters from one GPU.
- **Fair comparison:** every system gets the same prompt and the same test set. Cost is calculated from real token counts and GPU time.

## Key design decisions

- **Test on unseen vendors.** The dataset has only ~230 vendors, and one vendor has 44 receipts. With a random split, the model could memorise vendors. So the 200 test receipts come from 124 vendors the model never saw in training.
- **Audit the labels first.** Only 82% of test receipts have every correct value written exactly in the OCR text, because the OCR makes typos. So no model can score 100%. Knowing this ceiling keeps the results honest.
- **Include the strongest baseline.** Claude with structured output often fixes formatting on its own. If the prompt wins, the right answer is to ship the prompt, not the fine-tune.
- **Judge on held-out results, not training loss.** A falling loss curve proves nothing on its own.

## Run it

Open the notebook in Colab (badge above), choose a **T4 GPU**, add your `ANTHROPIC_API_KEY` as a Colab secret, and run all cells.
It trains, serves, runs the comparison and prints the results table.

Or locally:

```bash
pip install -r requirements.txt
pytest                                   # 28 tests, no GPU needed
python scripts/train.py --out adapters/slotfill-r16          # GPU
bash scripts/serve_vllm.sh                                   # GPU
python scripts/predict.py --provider vllm --model slotfill-r16 --out results/preds_lora.jsonl
python scripts/predict.py --provider anthropic --model claude-opus-5-5 --out results/preds_claude.jsonl
python scripts/evaluate.py --run "LoRA=results/preds_lora.jsonl" --run "Claude=results/preds_claude.jsonl"
```

## Project structure

```
slotfill/        schema, normalisation, prompt, metrics, training helpers
scripts/         prepare_data · audit_labels · train · serve_vllm · predict · evaluate
data/processed/  train/test split + label audit
notebooks/       end-to-end Colab notebook
tests/           unit tests
LABELING.md      where the data comes from and how it was cleaned
```

## License

Code: MIT. Data: derived from ICDAR 2019 SROIE via [zzzDavid/ICDAR-2019-SROIE](https://github.com/zzzDavid/ICDAR-2019-SROIE) (MIT).
