# Phase 1: Reproduction, Data and Results Guide

> **Phase 1 submission:** reproduction steps, data/model instructions (with hashes), and a map of every result folder are in [PHASE1.md](PHASE1.md). Report: `report/phase1_report.pdf`.

Where do time, memory and energy go in a two-stage computer-use agent (planner + grounding) on CPU vs GPU?
Report: `report/phase1_report.pdf`.

## 1. Hardware and pinned versions

| Machine | Hardware | llama.cpp | Used for |
|---|---|---|---|
| UCSD DataHub | 8 vCPU | build 11889 (`d854d98ba`) | CPU reference runs (`dh-*`, `B1-all-cpu`) |
| AWS `g6.2xlarge` (us-west-2) | NVIDIA L4 23 GB, 8 vCPU (4 physical cores), driver 595.112.07, CUDA 13.2 | build 12020 (`0ec4027a2`) | GPU runs and same-machine CPU-vs-GPU comparison (`aws-*`, `B1-all-cpu-aws`, `B2-*`, `B3-*`) |

Full environment captures: `results/env/`, `results/aws/env_capture.txt`, `results/aws/pip_freeze.txt`.

Build llama.cpp (AWS):
```bash
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp && git checkout 0ec4027a2
cmake -B build-cuda -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89 -DCMAKE_BUILD_TYPE=Release
cmake --build build-cuda -j8 --target llama-server
export PATH=$PWD/build-cuda/bin:$PATH
```

Python:
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
PYTHONPATH=. python tests/smoke_test.py | tail -3
```

## 2. Data

ScreenSpot (`rootsautomation/ScreenSpot`, 1,272 samples), pinned split 381 dev / 891 eval:
```bash
python scripts/prepare_screenspot.py --hf-id rootsautomation/ScreenSpot --out data --dev-frac 0.3
sha256sum data/dev.jsonl data/eval.jsonl
```
Expected SHA-256:
- `data/dev.jsonl`  `9626dc7f02f10ca812bb2ac914ff0e8ab6adc7e7f809bd108c224f0cc1f34a5d`
- `data/eval.jsonl` `d1a9b47acf6090e9c78002aa35667f3278fc32c1bb292077269c8bc278905262`

All tuning and measurement uses `dev`; `eval` is used once (final held-out check).

## 3. Models

```bash
mkdir -p models
hf download ggml-org/Qwen2.5-VL-3B-Instruct-GGUF Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf mmproj-Qwen2.5-VL-3B-Instruct-Q8_0.gguf --local-dir models
hf download Qwen/Qwen2.5-3B-Instruct-GGUF qwen2.5-3b-instruct-q4_k_m.gguf --local-dir models
sha256sum models/*.gguf
```

| Role | File | SHA-256 |
|---|---|---|
| Grounding VLM | `Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf` | `d02fe9b69ad8cadbbd228e387667af66612c44bed29ffc8eb1e7caf9ac486c12` |
| Vision projector | `mmproj-Qwen2.5-VL-3B-Instruct-Q8_0.gguf` | `980c9b2f78c04e6cff93d277ada09e768394f112d75db3b4e9dea8a69f9fb904` |
| Planner | `qwen2.5-3b-instruct-q4_k_m.gguf` | `626b4a6678b86442240e33df819e00132d3ba7dddfe1cdc4fbb18e0a9615c62d` |

Hashes are identical on DataHub and AWS.

Optional reference model (one extra data point only): `bartowski/UI-TARS-7B-DPO-GGUF` `UI-TARS-7B-DPO-Q4_K_S.gguf` (SHA-256 `f3227a383955bcb3ce88b396df40acafe405a14c95dd25658c01ea8098eac289`) with its mmproj.

## 4. Measurement protocol

- Prompt cache off (`--prompt-cache off`, default); one server slot (`-np 1`).
- Grounding: `--prompt structured --coord-space pixel` (Qwen2.5-VL outputs pixel coordinates; `norm1000` gives wrong accuracy).
- CPU mode hides the GPU (`CUDA_VISIBLE_DEVICES=""`, set by `serve_llamacpp.sh cpu`); verified by `gpu_mem_max_mb = 0`.
- First request reported separately (`first_s`) from warm steady state (`warm_*`).
- Repeats: grounding 100 x 3, planner 30/20/10 x 3 (ctx 500/2000/4000), pipeline 20 x 2. All failures recorded (`failed` column; all 0).
- Energy is GPU-only (NVML); CPU energy is not exposed on EC2.

## 5. Commands

Servers (`scripts/serve_llamacpp.sh <model> <port> <cpu|gpu|split-vgpu|split-lgpu> [mmproj] [threads]`):
```bash
# grounding on 8001 (use cpu instead of gpu for CPU runs)
./scripts/serve_llamacpp.sh models/Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf 8001 gpu models/mmproj-Qwen2.5-VL-3B-Instruct-Q8_0.gguf 8
# planner on 8002 (8001 for stand-alone planner benchmarks)
./scripts/serve_llamacpp.sh models/qwen2.5-3b-instruct-q4_k_m.gguf 8002 gpu "" 8
```

Grounding benchmark and pixel sweep (server on 8001):
```bash
for PX in 125000 250000 500000; do
  python -m ecpipe.stage_bench --stage grounding --base-url http://127.0.0.1:8001/v1 --model vlm \
    --data data/dev.jsonl --data-root data --n 100 --passes 3 --max-pixels $PX --coord-space pixel \
    --label aws-gpu-qwenvl-$((PX/1000))k --server-pid $(pgrep -f llama-server | head -1)
done
```

Held-out eval check:
```bash
python -m ecpipe.stage_bench --stage grounding --base-url http://127.0.0.1:8001/v1 --model vlm \
  --data data/eval.jsonl --data-root data --n 891 --passes 1 --max-pixels 500000 --coord-space pixel \
  --label aws-gpu-qwenvl-500k-EVAL --server-pid $(pgrep -f llama-server | head -1)
```

Planner context sweep (planner server on 8001):
```bash
for C in 500:30 2000:20 4000:10; do CTX=${C%%:*}; N=${C##*:}
  python -m ecpipe.stage_bench --stage planner --base-url http://127.0.0.1:8001/v1 --model planner \
    --n $N --passes 3 --ctx-tokens $CTX --max-tokens 48 --label aws-gpu-planner-ctx$CTX \
    --server-pid $(pgrep -f llama-server | head -1)
done
```

Pipeline placements (grounding server on 8001, planner on 8002, started in the modes below):

| Config | Grounding server | Planner server |
|---|---|---|
| `configs/B1_all_cpu_aws.yaml` (and `B1_all_cpu.yaml` on DataHub) | cpu | cpu |
| `configs/B2_ground_gpu_planner_cpu.yaml` | gpu | cpu |
| `configs/B3_all_gpu.yaml` | gpu | gpu |

```bash
python -m ecpipe.pipeline --config configs/B3_all_gpu.yaml --data data/dev.jsonl --data-root data --n 20 --passes 2
```

Summary table of every run:
```bash
python -m ecpipe.analyze --results results
```

## 6. Results map

| Folder in `results/` | What it is | Report |
|---|---|---|
| `dh-cpu-q4km-{125000,250000,500k}__grounding` | DataHub CPU grounding, pixel sweep | Table 1 (CPU column) |
| `dh-cpu-planner-ctx{500,2000,4000}__planner` | DataHub CPU planner sweep | Table 1 (CPU column) |
| `B1-all-cpu__pipeline` | DataHub all-CPU pipeline | Table 2 (reference row) |
| `aws-gpu-qwenvl-{125k,250k,500k}__grounding` | L4 grounding, pixel sweep | Table 1 |
| `aws-gpu-qwenvl-500k-EVAL__grounding` | L4 grounding on held-out eval split | Table 1 |
| `aws-gpu-planner-ctx{500,2000,4000}__planner` | L4 planner sweep | Table 1 |
| `B1-all-cpu-aws__pipeline`, `B2-ground-gpu_planner-cpu__pipeline`, `B3-all-gpu__pipeline` | Same-machine placements on AWS | Table 2 |
| `aws-gpu-ground-500k__grounding` | **UI-TARS-7B** (not Qwen) on L4, 500k px | Finding 6 (reference point) |
| `aws-gpu-sanity-500k__grounding` | 5-sample GPU sanity check (UI-TARS-7B) | not reported |
| `pilot/` | Early DataHub pilots, incl. the prompt-cache test (`dh-cpu-test` vs `dh-cpu-nocache`) | Pitfalls paragraph |
| `aws/` | Server logs, GPU power log, AWS summary, environment | provenance |

Notes:
- B1-AWS "energy" (~1,570 J) is idle GPU power only (GPU unused); idle draw ~27 W (see `nvidia-smi` in `results/aws/env_capture.txt`).
- A grounding run with the wrong coordinate convention (`norm1000`, accuracy 0.20) was deleted and rerun with `pixel`; it is described in the report's pitfalls paragraph.
