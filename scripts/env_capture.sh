#!/usr/bin/env bash
# Pin the environment for reproducibility; commit the output under results/env/.
mkdir -p results/env
{ date -u; uname -a; lscpu; free -h; nvidia-smi 2>/dev/null; llama-server --version 2>&1 | head -3;
  pip freeze; git rev-parse HEAD 2>/dev/null; } > results/env/env_$(hostname)_$(date +%Y%m%d_%H%M).txt
