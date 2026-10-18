"""Run one system over the held-out set and record raw outputs, latency and tokens.

Self-hosted model behind vLLM (base model or a LoRA adapter, OpenAI-compatible API):
    python scripts/predict.py --provider vllm --model slotfill-r16 \
        --gpu-hourly 0.40 --out results/preds_lora.jsonl

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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from slotfill.data import PROCESSED, read_jsonl, write_jsonl  # noqa: E402
from slotfill.prompts import SYSTEM_PROMPT, build_messages, user_message  # noqa: E402
from slotfill.schema import JSON_SCHEMA  # noqa: E402

# USD per million tokens (input, output), first-party API list prices.
ANTHROPIC_PRICES = {
    "claude-opus-5-5": (4.00, 20.00),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-haiku-5-5": (0.10, 0.50),
}


class VLLMClient:
    """Any OpenAI-compatible chat endpoint; in this project, `vllm serve`."""

    def __init__(self, args):
        from openai import AsyncOpenAI

        self.client = AsyncOpenAI(base_url=args.base_url, api_key=args.api_key or "EMPTY")
        self.model = args.model
        self.max_tokens = args.max_tokens

    async def __call__(self, ocr: str) -> dict:
        resp = await self.client.chat.completions.create(
            model=self.model,
            messages=build_messages(ocr),
            temperature=0.0,
            max_tokens=self.max_tokens,
        )
        return {
            "raw_output": resp.choices[0].message.content,
            "input_tokens": resp.usage.prompt_tokens if resp.usage else None,
            "output_tokens": resp.usage.completion_tokens if resp.usage else None,
            "finish_reason": resp.choices[0].finish_reason,
        }


class AnthropicClient:
    def __init__(self, args):
        import anthropic

        self.client = anthropic.AsyncAnthropic()
        self.model = args.model
        self.max_tokens = args.max_tokens
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
        if resp.stop_reason == "refusal":
            text = None
        else:
            text = "".join(b.text for b in resp.content if b.type == "text")
        return {
            "raw_output": text,
            "input_tokens": resp.usage.input_tokens,
            "output_tokens": resp.usage.output_tokens,
            "finish_reason": resp.stop_reason,
            "served_model": resp.model,
        }


async def run(call, docs: list[dict], concurrency: int) -> list[dict]:
    sem = asyncio.Semaphore(concurrency)

    async def one(doc: dict) -> dict:
        async with sem:
            t0 = time.perf_counter()
            try:
                out = await call(doc["ocr"])
                err = None
            except Exception as e:  # noqa: BLE001 - record and keep going; counted as invalid
                out, err = {"raw_output": None}, f"{type(e).__name__}: {e}"
            latency = time.perf_counter() - t0
            return {"id": doc["id"], "latency_s": round(latency, 4), "api_error": err, **out}

    results = await asyncio.gather(*(one(d) for d in docs))
    return list(results)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
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
    # Anthropic
    ap.add_argument("--structured", action="store_true", help="Constrain decoding to the JSON schema.")
    ap.add_argument("--effort", choices=["low", "medium", "high", "xhigh", "max"])
    args = ap.parse_args()

    docs = read_jsonl(args.split)[: args.limit]
    if args.provider == "vllm":
        args.max_tokens = args.max_tokens or 512
        call = VLLMClient(args)
        pricing = {"gpu_hourly_usd": args.gpu_hourly} if args.gpu_hourly else {}
    else:
        args.max_tokens = args.max_tokens or 16000
        call = AnthropicClient(args)
        price = ANTHROPIC_PRICES.get(args.model)
        pricing = {"input_per_mtok": price[0], "output_per_mtok": price[1]} if price else {}

    t0 = time.perf_counter()
    preds = asyncio.run(run(call, docs, args.concurrency))
    wall = time.perf_counter() - t0

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
    errors = sum(p["api_error"] is not None for p in preds)
    print(f"{len(preds)} docs in {wall:.1f}s, {errors} API errors -> {args.out}")


if __name__ == "__main__":
    main()
