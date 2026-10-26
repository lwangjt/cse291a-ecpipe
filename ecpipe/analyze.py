"""Aggregate results/*/records.jsonl into a markdown/CSV table and a latency plot.

Conventions (match the course rubric): the very first request is reported separately
(cold-ish), failures/timeouts stay in the denominator, variability = std of per-pass means.
"""
import argparse
import csv
import glob
import json
import os
from collections import defaultdict

import numpy as np


def pct(x, q):
    return float(np.percentile(x, q)) if len(x) else float("nan")


def summarize(d):
    recs = [json.loads(l) for l in open(os.path.join(d, "records.jsonl"))]
    meta = json.load(open(os.path.join(d, "meta.json")))
    label, stage = recs[0]["label"], recs[0]["stage"]
    ok = [r for r in recs if r["ok"]]
    warm = [r for r in ok if r["phase"] == "warm"]
    first = [r["latency_s"] for r in recs if r["phase"] == "first"]
    per_pass = defaultdict(list)
    for r in warm:
        per_pass[r["pass"]].append(r["latency_s"])
    pass_means = [np.mean(v) for v in per_pass.values()]
    lat = [r["latency_s"] for r in warm]
    row = dict(label=label, stage=stage, n=len(recs), failed=len(recs) - len(ok),
               first_s=first[0] if first else float("nan"),
               warm_mean_s=float(np.mean(lat)) if lat else float("nan"),
               warm_p50_s=pct(lat, 50), warm_p95_s=pct(lat, 95),
               pass_std_s=float(np.std(pass_means)) if len(pass_means) > 1 else float("nan"))
    tt = [r.get("ttft_s") or r.get("grounding_ttft_s") for r in warm]
    tt = [x for x in tt if x]
    row["ttft_p50_s"] = pct(tt, 50)
    tps = [r["decode_tps"] for r in warm if r.get("decode_tps")]
    row["decode_tps_p50"] = pct(tps, 50)
    if any("hit" in r for r in recs):    # failures count as misses
        row["acc_all"] = sum(bool(r.get("hit")) for r in recs) / len(recs)          # lenient (point or box center)
        row["acc_strict"] = sum(bool(r.get("hit")) and r.get("kind") == "point" for r in recs) / len(recs)
        row["box_output_rate"] = sum(r.get("kind") == "box" for r in recs) / len(recs)
        row["unparsed_rate"] = sum(not r.get("parsed") for r in recs) / len(recs)
        row["median_dist_px"] = pct([r["dist_px"] for r in ok if r.get("dist_px") is not None], 50)
    m = meta.get("monitor", {})
    nok = max(1, len(ok))
    ce, ge = m.get("cpu_energy_j"), m.get("gpu_energy_j")
    row["energy_j_per_req"] = ((ce or 0) + (ge or 0)) / nok if (ce is not None or ge is not None) else None
    row["rss_max_mb"] = m.get("server_rss_max_mb")
    row["gpu_mem_max_mb"] = m.get("gpu_mem_max_mb")
    row["throughput_rps"] = meta.get("throughput_rps")
    if stage == "pipeline":
        for k in ("preprocess", "planner", "grounding"):
            v = [r["stage_times"].get(k) for r in warm if r["stage_times"].get(k)]
            row[f"{k}_mean_s"] = float(np.mean(v)) if v else float("nan")
    return row


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="results")
    a = p.parse_args()
    rows = [summarize(os.path.dirname(f)) for f in sorted(glob.glob(f"{a.results}/*/records.jsonl"))]
    if not rows:
        print("no results"); return
    cols = sorted({k for r in rows for k in r}, key=lambda k: (k not in ("label", "stage"), k))
    with open(f"{a.results}/summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rows)
    fmt = lambda v: "" if v is None else (f"{v:.3g}" if isinstance(v, float) else str(v))
    with open(f"{a.results}/summary.md", "w") as f:
        f.write("| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n")
        for r in rows:
            f.write("| " + " | ".join(fmt(r.get(c)) for c in cols) + " |\n")
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 3.5))
        names = [f"{r['label']}\n{r['stage']}" for r in rows]
        ax.bar(names, [r["warm_mean_s"] for r in rows],
               yerr=[0 if np.isnan(r["pass_std_s"]) else r["pass_std_s"] for r in rows], capsize=3)
        ax.set_ylabel("warm mean latency (s)"); plt.xticks(fontsize=7); plt.tight_layout()
        plt.savefig(f"{a.results}/latency.png", dpi=150)
    except Exception as e:  # noqa: BLE001
        print("plot skipped:", e)
    print(open(f"{a.results}/summary.md").read())


if __name__ == "__main__":
    main()
