#!/usr/bin/env bash
# Phase-1 matrix. Start ONE server at a time (so devices do not contend), run, kill, next.
# Edit the MODEL/MMPROJ paths. Each config is run for GROUND (dev split) and PLANNER.
set -euo pipefail
VLM=models/UI-TARS-7B-DPO-Q4_K_S.gguf; MMPROJ=models/UI-TARS-7B-mmproj.gguf
LLM=models/gemma-3-12b-it-Q4_K_M.gguf   # planner; pick what fits the shared machine
for MODE in cpu gpu split-vgpu; do
  ./scripts/serve_llamacpp.sh $VLM 8001 $MODE $MMPROJ & PID=$!; sleep 60   # wait for load; time this as cold start separately
  python -m ecpipe.stage_bench --stage grounding --base-url http://127.0.0.1:8001/v1 --model vlm \
     --data data/dev.jsonl --n 100 --passes 3 --label $MODE-q4ks --server-pid $PID \
     --server-info "llama.cpp $(llama-server --version 2>&1 | head -1)"
  kill $PID; wait $PID 2>/dev/null || true
done
