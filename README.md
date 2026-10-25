# ecpipe — device-aware agent-pipeline harness (CSE 291A, Option 2, Phase 1)

Question: for a GUI/computer-use agent pipeline (preprocess -> planner LLM -> grounding VLM -> act),
where should each stage run (CPU / GPU / NPU), and what does placement cost in latency, energy,
memory and accuracy? Phase 1 = reproducible baseline + per-stage and end-to-end measurements.

## Layout
- `ecpipe/client.py`      streaming OpenAI-compatible client (TTFT, decode tok/s, failures recorded not dropped)
- `ecpipe/sysmon.py`      CPU util, server RSS, GPU util/power/mem, RAPL energy; env capture
- `ecpipe/workloads.py`   prompts, coordinate parsing (norm1000/norm1/pixel), hit/distance scoring, synthetic planner context
- `ecpipe/stage_bench.py` one stage on one backend (grounding | planner), N samples x P passes, optional concurrency
- `ecpipe/pipeline.py`    end-to-end step with per-stage latency breakdown; placement = YAML (configs/)
- `ecpipe/analyze.py`     summary.md/csv + latency.png (first request separate, failures in denominator)
- `scripts/`              data prep (dev/eval split), llama.cpp launcher (cpu|gpu|split-vgpu|split-lgpu), env capture, phase-1 matrix
- `tests/smoke_test.py`   mock-server end-to-end test: `PYTHONPATH=. python tests/smoke_test.py`

## Quickstart
```bash
pip install -r requirements.txt
PYTHONPATH=. python tests/smoke_test.py                       # harness sanity check, no models
python scripts/prepare_screenspot.py --out data               # dev.jsonl / eval.jsonl (check bbox format!)
scripts/env_capture.sh
# terminal 1: scripts/serve_llamacpp.sh models/<vlm>.gguf 8001 cpu models/<mmproj>.gguf
# terminal 2:
python -m ecpipe.stage_bench --stage grounding --base-url http://127.0.0.1:8001/v1 --model vlm \
   --data data/dev.jsonl --n 100 --passes 3 --label cpu-q4ks --server-pid <PID>
python -m ecpipe.analyze
```
Rules baked in: tune on `dev.jsonl` only, report on `eval.jsonl`; pin model/runtime/dataset versions
in `--server-info`; first request reported separately from steady state; failures and timeouts
count as misses and appear in the `failed` column; repeat passes give variance (`pass_std_s`).

## Known gaps (do these before the deadline)
1. NPU path: add a backend only if the shared machine has one (ONNX Runtime EP / OpenVINO); the harness needs only an OpenAI-compatible server or a small adapter in `client.py`.
2. Model-load/cold-start time is not measured by the Python harness: time server start-to-ready in the launcher and report it.
3. Verify `parse_point` coordinate space for the exact model/GGUF you use (UI-TARS versions differ).

## Compute resources (course setup)
- **TritonAI API** (OpenAI-compatible; `OPENAI_BASE_URL`/`OPENAI_API_KEY` are preset in DataHub): use as a *remote* reference placement
  (`--base-url $OPENAI_BASE_URL --model <name>`). Latency includes network + queueing and the device is unknown, so label it
  "remote reference", not a CPU/GPU result. Key is read from env, never stored. Check models: `python scripts/triton_check.py [model]`.
- **AWS** (team $200 total, $20/day): CPU-only baseline on a compute instance, GPU on g6 (L4 24GB) / g6e (L40S 48GB). Fix and report
  thread counts (`-t`) and instance types; stop instances when idle. Verify current hourly prices before planning.
- **DataHub slice** (8 CPU, 16 GB RAM, 24 GB GPU slice of RTX PRO 6000): free for development and a second GPU datapoint.
  16 GB RAM caps CPU-resident models (7B Q4 ~5 GB; do not hold a 7B VLM and a 12B planner on CPU together).
  `nvidia-smi` power/util may reflect the whole shared GPU and RAPL is usually unreadable in containers: report energy as unavailable/unattributable.
