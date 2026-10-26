"""Single-stage benchmark: one stage (grounding VLM or planner LLM) on one backend/device.

Example:
  python -m ecpipe.stage_bench --stage grounding --base-url http://127.0.0.1:8001/v1 \
      --model uitars-7b --data data/dev.jsonl --n 100 --passes 3 --label cpu-q4ks
Outputs results/<label>__<stage>/{records.jsonl, meta.json}.
"""
import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor

from .client import chat_stream, encode_image
from .sysmon import Monitor, capture_env
from .workloads import PROMPTS, load_jsonl, make_history_text, parse_output, score


def CP(a):
    """Per-request extras: disable llama.cpp prompt/KV reuse so repeated samples are not served from cache."""
    return {"cache_prompt": False} if a.prompt_cache == "off" else None


def one_request(a, item, tag):
    rec = dict(label=a.label, stage=a.stage, **tag)
    if a.stage == "grounding":
        t = time.perf_counter()
        url, sent, orig = encode_image(os.path.join(a.data_root, item["image"]), a.max_pixels)
        rec["preprocess_s"] = time.perf_counter() - t
        prompt = PROMPTS[a.prompt].format(instruction=item["instruction"])
        msgs = [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": url}},
            {"type": "text", "text": prompt}]}]
        r = chat_stream(a.base_url, a.model, msgs, a.max_tokens, timeout=a.timeout, extra=CP(a))
        pt, kind = parse_output(r["text"], a.coord_space, orig, sent) if r["ok"] else (None, None)
        hit, dist = score(pt, item["bbox"])
        rec.update(sample_id=item["id"], parsed=pt is not None, kind=kind, hit=hit, dist_px=dist,
                   sent_wh=list(sent))
    else:  # planner
        hist = make_history_text(a.ctx_tokens, a.seed + tag["idx"])
        msgs = [{"role": "user", "content":
                 f"You control a computer.\nHistory:\n{hist}\n\nDecide the next action. "
                 "Answer in one short sentence."}]
        r = chat_stream(a.base_url, a.model, msgs, a.max_tokens, timeout=a.timeout, extra=CP(a))
    rec.update({k: r[k] for k in ("ok", "error", "latency_s", "ttft_s", "decode_tps",
                                   "prompt_tokens", "completion_tokens", "server_timings",
                                   "t_start")})
    rec["output"] = r["text"][:200]
    return rec


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=["grounding", "planner"], required=True)
    p.add_argument("--base-url", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--label", required=True, help="e.g. cpu-q4ks, gpu-fp16, split-vision-gpu")
    p.add_argument("--data", help="grounding jsonl (from scripts/prepare_screenspot.py)")
    p.add_argument("--data-root", default=".")
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--passes", type=int, default=3, help="repeat for variance estimates")
    p.add_argument("--prompt", choices=list(PROMPTS), default="structured")
    p.add_argument("--coord-space", default="norm1000", choices=["norm1000", "norm1", "pixel"])
    p.add_argument("--max-pixels", type=int, default=None)
    p.add_argument("--ctx-tokens", type=int, default=2000)
    p.add_argument("--max-tokens", type=int, default=64)
    p.add_argument("--concurrency", type=int, default=1, help=">1 measures contention")
    p.add_argument("--timeout", type=float, default=300)
    p.add_argument("--server-pid", type=int, action="append", default=[])
    p.add_argument("--server-info", default="", help="server version/flags, free text")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--prompt-cache", choices=["off", "default"], default="off",
                   help="off: send cache_prompt=false (llama.cpp). Use default for remote/vLLM endpoints.")
    p.add_argument("--out-dir", default="results")
    a = p.parse_args()

    items = load_jsonl(a.data, a.n) if a.stage == "grounding" else [None] * a.n
    work = [(items[i], dict(**{"pass": ps}, idx=i,
                            phase="first" if (ps == 0 and i == 0) else "warm"))
            for ps in range(a.passes) for i in range(len(items))]
    out = os.path.join(a.out_dir, f"{a.label}__{a.stage}")
    os.makedirs(out, exist_ok=True)
    recs = []
    with Monitor(pids=a.server_pid) as mon:
        t0 = time.perf_counter()
        if a.concurrency == 1:
            for it, tag in work:
                recs.append(one_request(a, it, tag))
        else:
            recs.append(one_request(a, *work[0]))      # 'first' request alone
            with ThreadPoolExecutor(a.concurrency) as ex:
                recs += list(ex.map(lambda w: one_request(a, *w), work[1:]))
        wall = time.perf_counter() - t0
    with open(os.path.join(out, "records.jsonl"), "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    meta = dict(args=vars(a), env=capture_env({"server_info": a.server_info}),
                monitor=mon.summary(), wall_s=wall, n_requests=len(recs),
                throughput_rps=len(recs) / wall)
    json.dump(meta, open(os.path.join(out, "meta.json"), "w"), indent=1)
    ok = sum(r["ok"] for r in recs)
    print(f"[{a.label}/{a.stage}] {ok}/{len(recs)} ok, wall {wall:.1f}s -> {out}")


if __name__ == "__main__":
    main()
