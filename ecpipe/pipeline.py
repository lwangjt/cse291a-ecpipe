"""End-to-end agent step pipeline with a configurable stage->server placement.

Per step: preprocess (CPU) -> planner (LLM server P) -> grounding (VLM server G) -> act (simulated).
A *placement* is simply which server (CPU-backed, GPU-backed, NPU-backed) each stage points to,
described in a YAML file; see configs/. Phase 2 hooks (marked PHASE2) are where an adaptive
placement policy and stage overlap plug in.
"""
import argparse
import json
import os
import time

import yaml

from .client import chat_stream, encode_image
from .sysmon import Monitor, capture_env
from .workloads import PROMPTS, load_jsonl, make_history_text, parse_output, score


def run_step(cfg, item, idx, seed, data_root):
    g, pl = cfg["stages"]["grounding"], cfg["stages"]["planner"]
    rec = dict(sample_id=item["id"], ok=False, error=None, stage_times={})
    t_all = time.perf_counter()
    # --- preprocess
    t = time.perf_counter()
    url, sent, orig = encode_image(os.path.join(data_root, item["image"]), g.get("max_pixels"))
    rec["stage_times"]["preprocess"] = time.perf_counter() - t
    # --- planner  (PHASE2: placement policy may re-route this call per step)
    ctx = cfg.get("workload", {}).get("ctx_tokens", 2000)
    hist = make_history_text(ctx, seed + idx)
    r = chat_stream(pl["base_url"], pl["model"], [{"role": "user", "content":
        f"You control a computer.\nHistory:\n{hist}\n\nGoal: {item['instruction']}\n"
        "Describe in one short sentence the UI element to interact with next."}],
        pl.get("max_tokens", 48), timeout=pl.get("timeout", 300))
    rec["stage_times"]["planner"] = r["latency_s"]
    rec["planner_ttft_s"], rec["planner_prompt_tokens"] = r["ttft_s"], r["prompt_tokens"]
    if not r["ok"]:
        rec["error"] = "planner:" + str(r["error"])
        rec["latency_s"] = time.perf_counter() - t_all
        return rec
    target = r["text"].strip() if cfg.get("planner_feeds_grounding") else item["instruction"]
    # --- grounding
    r2 = chat_stream(g["base_url"], g["model"], [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": url}},
        {"type": "text", "text": PROMPTS[g.get("prompt", "structured")].format(instruction=target)}]}],
        g.get("max_tokens", 64), timeout=g.get("timeout", 300))
    rec["stage_times"]["grounding"] = r2["latency_s"]
    rec["grounding_ttft_s"] = r2["ttft_s"]
    if not r2["ok"]:
        rec["error"] = "grounding:" + str(r2["error"])
        rec["latency_s"] = time.perf_counter() - t_all
        return rec
    # --- act (simulated: score instead of clicking)
    pt, kind = parse_output(r2["text"], g.get("coord_space", "norm1000"), orig, sent)
    rec["hit"], rec["dist_px"] = score(pt, item["bbox"])
    rec["parsed"] = pt is not None
    rec["kind"] = kind
    rec["ok"] = True
    rec["latency_s"] = time.perf_counter() - t_all
    return rec


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--data-root", default=".")
    p.add_argument("--n", type=int, default=50)
    p.add_argument("--passes", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", default="results")
    a = p.parse_args()
    cfg = yaml.safe_load(open(a.config))
    items = load_jsonl(a.data, a.n)
    out = os.path.join(a.out_dir, f"{cfg['name']}__pipeline")
    os.makedirs(out, exist_ok=True)
    recs = []
    with Monitor(pids=cfg.get("server_pids", [])) as mon:
        for ps in range(a.passes):
            for i, it in enumerate(items):
                r = run_step(cfg, it, i, a.seed, a.data_root)
                r.update(label=cfg["name"], stage="pipeline", phase="first" if (ps == 0 and i == 0) else "warm")
                r["pass"], r["idx"] = ps, i
                recs.append(r)
    with open(os.path.join(out, "records.jsonl"), "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    json.dump(dict(args=vars(a), config=cfg, env=capture_env(), monitor=mon.summary(),
                   n_requests=len(recs)), open(os.path.join(out, "meta.json"), "w"), indent=1)
    print(f"[{cfg['name']}] {sum(r['ok'] for r in recs)}/{len(recs)} steps ok -> {out}")


if __name__ == "__main__":
    main()
