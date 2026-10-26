#!/usr/bin/env bash
# Usage: serve_llamacpp.sh <model.gguf> <port> <mode> [mmproj.gguf] [threads]
#   mode = cpu         : everything on CPU (LLM layers and vision encoder)
#          gpu         : everything offloaded to GPU
#          split-vgpu  : vision encoder on GPU, LLM layers on CPU
#          split-lgpu  : LLM layers on GPU, vision encoder on CPU
# Flag names (esp. --no-mmproj-offload) vary by llama.cpp build: check `llama-server --help`
# and record the exact build in meta.json via --server-info.
set -euo pipefail
MODEL=$1; PORT=$2; MODE=$3; MMPROJ=${4:-}; THREADS=${5:-$(nproc)}
# CTX = total context, NP = parallel slots (llama-server splits CTX across slots!). Use NP=1 for sequential runs;
# for concurrency-k runs set NP=k and CTX=k*per-request-need. Planner at 8k ctx needs CTX>=10000 with NP=1.
CTX=${CTX:-8192}; NP=${NP:-1}
ARGS=(-m "$MODEL" --port "$PORT" --host 127.0.0.1 -c "$CTX" -np "$NP" -t "$THREADS" --no-webui)
[[ -n "$MMPROJ" ]] && ARGS+=(--mmproj "$MMPROJ")
case "$MODE" in
  cpu)        ARGS+=(-ngl 0 --no-mmproj-offload) ;;
  gpu)        ARGS+=(-ngl 99) ;;
  split-vgpu) ARGS+=(-ngl 0) ;;
  split-lgpu) ARGS+=(-ngl 99 --no-mmproj-offload) ;;
  *) echo "bad mode"; exit 1 ;;
esac
# CPU-only runs: hide GPUs so nothing is silently offloaded
[[ "$MODE" == "cpu" ]] && export CUDA_VISIBLE_DEVICES=""
exec llama-server "${ARGS[@]}"
