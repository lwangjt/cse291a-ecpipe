"""Export ScreenSpot-style data to images + jsonl with a DEV/EVAL split (course requires both).

  python scripts/prepare_screenspot.py --hf-id rootsautomation/ScreenSpot --out data --dev-frac 0.3

VERIFY the bbox convention on the printed sample before trusting any accuracy number
(--bbox-format xyxy|xywh; normalized boxes are detected automatically).
"""
import argparse, json, os, random
from datasets import load_dataset

ap = argparse.ArgumentParser()
ap.add_argument("--hf-id", default="rootsautomation/ScreenSpot")
ap.add_argument("--split", default="test")
ap.add_argument("--bbox-format", default="xyxy", choices=["xyxy", "xywh"])
ap.add_argument("--dev-frac", type=float, default=0.3)
ap.add_argument("--limit", type=int, default=None)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", default="data")
a = ap.parse_args()

os.makedirs(f"{a.out}/images", exist_ok=True)
ds = load_dataset(a.hf_id, split=a.split)
rows = []
for i, ex in enumerate(ds):
    if a.limit and i >= a.limit:
        break
    img = ex["image"].convert("RGB"); w, h = img.size
    b = list(ex["bbox"])
    if a.bbox_format == "xywh":
        b = [b[0], b[1], b[0] + b[2], b[1] + b[3]]
    if max(b) <= 1.5:                          # normalized -> pixels
        b = [b[0] * w, b[1] * h, b[2] * w, b[3] * h]
    path = f"images/{i}.png"; img.save(f"{a.out}/{path}")
    rows.append(dict(id=i, image=path, instruction=ex["instruction"], bbox=b, w=w, h=h,
                     data_type=ex.get("data_type"), source=ex.get("data_source")))
random.Random(a.seed).shuffle(rows)
k = int(len(rows) * a.dev_frac)
for name, part in (("dev", rows[:k]), ("eval", rows[k:])):
    with open(f"{a.out}/{name}.jsonl", "w") as f:
        for r in part: f.write(json.dumps(r) + "\n")
print(f"dev={k} eval={len(rows)-k}; sample:", rows[0])
