#!/usr/bin/env bash
# Usage: scripts/cold_start.sh <model.gguf> <mmproj.gguf> [mode=cpu] [threads=8]
# Prints seconds from launching the server to /health ok. Without root we cannot drop the OS page cache,
# so repeat loads are "warm-file-cache"; the first load right after a download is the closest to cold.
set -euo pipefail
MODEL=$1; MMPROJ=$2; MODE=${3:-cpu}; THREADS=${4:-8}
pkill -f llama-server || true; sleep 2
T0=$(date +%s.%N)
./scripts/serve_llamacpp.sh "$MODEL" 8001 "$MODE" "$MMPROJ" "$THREADS" > cold_start_server.log 2>&1 &
SP=$!
until curl -sf localhost:8001/health > /dev/null; do
  kill -0 $SP 2>/dev/null || { echo "server exited; see cold_start_server.log"; exit 1; }
  sleep 0.2
done
T1=$(date +%s.%N)
python3 -c "print(f'start-to-ready: {$T1 - $T0:.1f} s (mode=$MODE threads=$THREADS)')"
