"""Run one system over the held-out set and record raw outputs, latency and tokens.

Self-hosted model behind vLLM (base model or a LoRA adapter, OpenAI-compatible API):
    python scripts/predict.py --provider vllm --model slotfill-r16 \
        --gpu-hourly 0.40 --out results/preds_lora.jsonl

Large open-weight baseline served by vLLM, with schema-constrained decoding:
    python scripts/predict.py --provider vllm --model Qwen/Qwen2.5-32B-Instruct-AWQ \
        --structured --gpu-hourly 1.30 --out results/preds_qwen32b_structured.jsonl

Prompted frontier baseline (Claude, needs ANTHROPIC_API_KEY):
    python scripts/predict.py --provider anthropic --model claude-opus-5-5 \
        --out results/preds_claude.jsonl
    # add --structured to also constrain decoding to the JSON schema

Every system gets the same system prompt and the same 200 documents.
Outputs are written verbatim; scoring happens in evaluate.py.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

# Make the repo root importable so `from slotfill...` works when run as `python scripts/x.py`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from slotfill.data import PROCESSED, read_jsonl, write_jsonl  # noqa: E402
from slotfill.prompts import SYSTEM_PROMPT, build_messages, user_message  # noqa: E402
from slotfill.schema import JSON_SCHEMA  # noqa: E402

# USD per million tokens (input, output), first-party API list prices.
# Used only to estimate API cost per 1,000 docs from the token counts each call returns.
ANTHROPIC_PRICES = {
    "claude-opus-5-5": (4.00, 20.00),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-haiku-5-5": (0.10, 0.50),
}


class VLLMClient:
    """Any OpenAI-compatible chat endpoint; in this project, `vllm serve`."""

    def __init__(self, args):
        # vLLM exposes the same HTTP API as OpenAI, so the official openai client talks to it.
        # AsyncOpenAI lets many requests be in flight at once.
        from openai import AsyncOpenAI

        # A local vLLM server doesn't check the key, but the client requires one, hence "EMPTY".
        self.client = AsyncOpenAI(base_url=args.base_url, api_key=args.api_key or "EMPTY")
        self.model = args.model
        self.max_tokens = args.max_tokens
        # Extra request options. Empty means free-form output (the model may write anything).
        self.extra: dict = {}
        if args.structured:
            # vLLM guided decoding: tokens outside the JSON schema are masked out.
            self.extra["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "receipt", "schema": JSON_SCHEMA, "strict": True},
            }

    async def __call__(self, ocr: str) -> dict:
        resp = await self.client.chat.completions.create(
            model=self.model,
            messages=build_messages(ocr),
            # Temperature 0 = always pick the most likely token, so runs are repeatable.
            temperature=0.0,
            max_tokens=self.max_tokens,
            # Adds response_format when --structured is on.
            **self.extra,
        )
        return {
            # Keep the reply exactly as the model wrote it; evaluate.py decides if it's valid.
            # Token counts are recorded for reference; self-hosted cost is based on GPU time instead.
            "raw_output": resp.choices[0].message.content,
            "input_tokens": resp.usage.prompt_tokens if resp.usage else None,
            "output_tokens": resp.usage.completion_tokens if resp.usage else None,
            "finish_reason": resp.choices[0].finish_reason,
        }


# Claude via the Anthropic API (optional; the A100 notebook doesn't use it).
class AnthropicClient:
    def __init__(self, args):
        import anthropic

        # Reads ANTHROPIC_API_KEY from the environment.
        self.client = anthropic.AsyncAnthropic()
        self.model = args.model
        self.max_tokens = args.max_tokens
        # Optional settings: how hard the model thinks (effort) and/or forced JSON-schema output.
        self.output_config: dict = {}
        if args.effort:
            self.output_config["effort"] = args.effort
        if args.structured:
            self.output_config["format"] = {"type": "json_schema", "schema": JSON_SCHEMA}

    async def __call__(self, ocr: str) -> dict:
        kwargs = {}
        if self.output_config:
            kwargs["output_config"] = self.output_config
        # Server-side refusal fallback: if the model declines, the API retries on a
        # fallback model inside the same call instead of returning nothing.
        resp = await self.client.beta.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message(ocr)}],
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            **kwargs,
        )
        # A refusal produces no usable answer; record None so it is scored as invalid.
        if resp.stop_reason == "refusal":
            text = None
        else:
            # The reply is a list of content blocks; join the text ones.
            text = "".join(b.text for b in resp.content if b.type == "text")
        return {
            "raw_output": text,
            "input_tokens": resp.usage.input_tokens,
            "output_tokens": resp.usage.output_tokens,
            "finish_reason": resp.stop_reason,
            "served_model": resp.model,
        }


# Send every document to the model, at most `concurrency` at a time, and time each one.
async def run(call, docs: list[dict], concurrency: int) -> list[dict]:
    # The semaphore caps how many requests run in parallel.
    sem = asyncio.Semaphore(concurrency)

    async def one(doc: dict) -> dict:
        async with sem:  # wait for a free slot
            t0 = time.perf_counter()
            try:
                out = await call(doc["ocr"])
                err = None
            except Exception as e:  # noqa: BLE001 - record and keep going; counted as invalid
                out, err = {"raw_output": None}, f"{type(e).__name__}: {e}"
            # Per-request latency: what one user waiting for one document would see.
            latency = time.perf_counter() - t0
            return {"id": doc["id"], "latency_s": round(latency, 4), "api_error": err, **out}

    # Start all documents together; the semaphore queues them. Order is preserved.
    results = await asyncio.gather(*(one(d) for d in docs))
    return list(results)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # Command-line options. Run `python scripts/predict.py --help` to see them all.
    ap.add_argument("--provider", choices=["vllm", "anthropic"], required=True)
    ap.add_argument("--model", required=True, help="vLLM: base model or --lora-modules name. Anthropic: model id.")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--split", type=Path, default=PROCESSED / "test.jsonl")
    ap.add_argument("--limit", type=int, help="Only the first N docs (smoke test).")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--max-tokens", type=int)
    # vLLM
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--api-key")
    ap.add_argument("--gpu-hourly", type=float, help="USD/hour of the GPU serving vLLM, for cost per 1k.")
    ap.add_argument("--structured", action="store_true", help="Constrain decoding to the JSON schema (both providers).")
    # Anthropic
    ap.add_argument("--effort", choices=["low", "medium", "high", "xhigh", "max"])
    args = ap.parse_args()

    # Load the test documents (or only the first N with --limit, for a quick check).
    docs = read_jsonl(args.split)[: args.limit]
    if args.provider == "vllm":
        # The JSON answer is short, so 512 output tokens is plenty for the small models.
        # Cost is GPU $/hour x wall-clock time, so record the hourly rate if given.
        args.max_tokens = args.max_tokens or 512
        call = VLLMClient(args)
        pricing = {"gpu_hourly_usd": args.gpu_hourly} if args.gpu_hourly else {}
    else:
        # Claude may think before answering, so it gets a larger token budget.
        # Cost comes from token counts x the list price for this model.
        args.max_tokens = args.max_tokens or 16000
        call = AnthropicClient(args)
        price = ANTHROPIC_PRICES.get(args.model)
        pricing = {"input_per_mtok": price[0], "output_per_mtok": price[1]} if price else {}

    # Run the whole batch. Its wall-clock time is what the GPU cost is based on.
    t0 = time.perf_counter()
    preds = asyncio.run(run(call, docs, args.concurrency))
    wall = time.perf_counter() - t0

    # Save raw predictions, plus a .meta.json describing how they were produced.
    write_jsonl(args.out, preds)
    meta = {
        "provider": args.provider,
        "model": args.model,
        "structured": args.structured,
        "effort": args.effort,
        "n": len(preds),
        "concurrency": args.concurrency,
        "wall_seconds": round(wall, 2),
        "pricing": pricing,
        "split": str(args.split.name),
    }
    args.out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    # Requests that raised an exception (timeouts, server errors) are counted and reported.
    errors = sum(p["api_error"] is not None for p in preds)
    print(f"{len(preds)} docs in {wall:.1f}s, {errors} API errors -> {args.out}")


if __name__ == "__main__":
    main()
