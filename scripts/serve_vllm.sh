#!/usr/bin/env bash
# Serve the base model with one or more LoRA adapters behind an OpenAI-compatible API.
#
#   bash scripts/serve_vllm.sh                      # every adapter under adapters/
#   BASE=Qwen/Qwen2.5-3B-Instruct bash scripts/serve_vllm.sh
#
# Requests with "model": "<adapter dir name>" use that adapter; "model": "$BASE"
# uses the base model with no adapter. One server, one GPU, several adapters.
set -euo pipefail

BASE="${BASE:-Qwen/Qwen2.5-3B-Instruct}"
PORT="${PORT:-8000}"
MAX_LORA_RANK="${MAX_LORA_RANK:-64}"

modules=()
for dir in adapters/*/; do
  [ -f "${dir}adapter_config.json" ] || continue
  name="$(basename "$dir")"
  modules+=("${name}=${dir%/}")
done
if [ ${#modules[@]} -eq 0 ]; then
  echo "no adapters found under adapters/ (run scripts/train.py first)" >&2
  exit 1
fi
echo "serving $BASE with adapters: ${modules[*]}"

# T4 has no bf16, so let vLLM pick (fp16 there, bf16 on A100/L4).
exec vllm serve "$BASE" \
  --port "$PORT" \
  --enable-lora \
  --max-lora-rank "$MAX_LORA_RANK" \
  --lora-modules "${modules[@]}" \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.85 \
  --dtype auto
