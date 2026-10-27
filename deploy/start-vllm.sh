#!/usr/bin/env bash
set -euo pipefail
: "${MODEL_PATH:?Set MODEL_PATH to the merged model directory}"
[[ -f "$MODEL_PATH/config.json" && -f "$MODEL_PATH/model.safetensors.index.json" ]] || { echo "Not a sharded merged model directory" >&2; exit 1; }
IMAGE="${IMAGE:-aixcoder-compact-vllm:0.22.1}"
PORT="${PORT:-8000}"
NAME="${CONTAINER_NAME:-aixcoder-compact-vllm}"
MODE="${DECODE_MODE:-ngram}"
args=(--model /model --served-model-name aiXcompact --host 0.0.0.0 --port 8000
      --tensor-parallel-size 1 --dtype bfloat16 --max-model-len "${MAX_MODEL_LEN:-262144}"
      --max-num-seqs "${MAX_NUM_SEQS:-1}" --max-num-batched-tokens "${MAX_BATCHED_TOKENS:-8192}"
      --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.90}" --enable-chunked-prefill)
case "$MODE" in
  ngram) args+=(--speculative-config '{"method":"ngram","num_speculative_tokens":32,"prompt_lookup_max":10,"prompt_lookup_min":5}');;
  standard) ;;
  *) echo "DECODE_MODE must be standard or ngram" >&2; exit 1;;
esac
[[ -z "${VLLM_API_KEY:-}" ]] || args+=(--api-key "$VLLM_API_KEY")
docker run -d --name "$NAME" --gpus "device=${GPU_ID:-0}" --ipc=host \
  --publish "${BIND_HOST:-127.0.0.1}:$PORT:8000" \
  --mount "type=bind,source=$(cd "$MODEL_PATH" && pwd),target=/model,readonly" \
  "$IMAGE" "${args[@]}"
