#!/usr/bin/env bash
# Usage: scripts/run_baseline_cpu.sh <label> [max_pixels=500000] [n=100] [passes=3]
# Needs a llama-server already running on :8001. Extra provenance: SERVER_INFO_EXTRA="model, quant, threads"
set -euo pipefail
LABEL=${1:?label}; PX=${2:-500000}; N=${3:-100}; PASSES=${4:-3}
PID=$(pgrep -f llama-server | head -1 || true)
python -m ecpipe.stage_bench --stage grounding --base-url http://127.0.0.1:8001/v1 --model vlm \
  --data data/dev.jsonl --data-root data --n "$N" --passes "$PASSES" --coord-space pixel \
  --max-pixels "$PX" --label "$LABEL" ${PID:+--server-pid $PID} \
  --server-info "$(llama-server --version 2>&1 | head -1); ${SERVER_INFO_EXTRA:-}"
