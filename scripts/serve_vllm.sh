#!/usr/bin/env bash
# Serve the base model with one or more LoRA adapters behind an OpenAI-compatible API.
#
#   bash scripts/serve_vllm.sh                      # every adapter under adapters/
#   BASE=Qwen/Qwen2.5-3B-Instruct bash scripts/serve_vllm.sh
#
# Requests with "model": "<adapter dir name>" use that adapter; "model": "$BASE"
# uses the base model with no adapter. One server, one GPU, several adapters.
set -euo pipefail  # stop on any error, unset variable or failed pipe

# Settings, each overridable from the environment (e.g. GPU_UTIL=0.9 bash scripts/serve_vllm.sh).
# BASE:          Hugging Face model id to serve
# PORT:          port for the OpenAI-compatible API (http://localhost:$PORT/v1)
# MAX_LORA_RANK: largest adapter rank vLLM reserves room for (our adapters use 16)
# GPU_UTIL:      share of GPU memory vLLM may take for weights + KV cache
BASE="${BASE:-Qwen/Qwen2.5-3B-Instruct}"
PORT="${PORT:-8000}"
MAX_LORA_RANK="${MAX_LORA_RANK:-64}"
GPU_UTIL="${GPU_UTIL:-0.85}"

# NO_LORA=1 serves a plain model (e.g. a large open-weight baseline) with no adapters.
if [ "${NO_LORA:-0}" = "1" ]; then
  echo "serving $BASE (no adapters)"
  # exec replaces this shell with vLLM, so stopping the script stops the server.
  exec vllm serve "$BASE" --port "$PORT" --max-model-len 4096 \
    --gpu-memory-utilization "$GPU_UTIL" --dtype auto
fi

# Collect every trained adapter: each folder under adapters/ that has an adapter_config.json
# becomes "name=path", e.g. "slotfill-r16=adapters/slotfill-r16". The name is what
# predict.py passes as --model to pick that adapter.
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

# Start the server. Flags:
#   --enable-lora / --lora-modules: load the adapters on top of the one base model
#   --max-model-len 4096: longest prompt + answer in tokens (receipts are well below this)
#   --dtype auto: pick the precision for the GPU
# T4 has no bf16, so let vLLM pick (fp16 there, bf16 on A100/L4).
exec vllm serve "$BASE" \
  --port "$PORT" \
  --enable-lora \
  --max-lora-rank "$MAX_LORA_RANK" \
  --lora-modules "${modules[@]}" \
  --max-model-len 4096 \
  --gpu-memory-utilization "$GPU_UTIL" \
  --dtype auto
