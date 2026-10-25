"""Workloads: GUI-grounding samples (dev/eval split) and synthetic planner contexts."""
import json
import math
import random
import re

PROMPTS = {
    "naive": "Output the (x, y) coordinates of: {instruction}",
    "structured": (
        "You are a GUI agent. You are given a screenshot and a description of one UI element.\n"
        "Rules: locate the single best-matching element; return the CENTER point of that element; "
        "do not explain.\n"
        "Output format (exactly): click(start_box='(x,y)')\n"
        "Element: {instruction}"
    ),
}

_NUM = r"(-?\d+(?:\.\d+)?)"
_PATTERNS = [re.compile(r"\(\s*" + _NUM + r"\s*,\s*" + _NUM + r"\s*\)"),
             re.compile(r"\[\s*" + _NUM + r"\s*,\s*" + _NUM + r"\s*\]"),
             re.compile(r"<point>\s*" + _NUM + r"\s+" + _NUM + r"\s*</point>")]


def load_jsonl(path, n=None):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    return rows[:n] if n else rows


def parse_point(text, coord_space, orig_wh, sent_wh):
    """Parse model output to a point in ORIGINAL-image pixels, or None if unparseable.

    coord_space: 'norm1000' (UI-TARS-1.0/Qwen2-VL style 0..1000), 'norm1' (0..1),
                 'pixel' (pixels of the image actually SENT to the model).
    Unparseable output counts as a miss; keep the failure visible in the records.
    """
    for pat in _PATTERNS:
        m = pat.search(text or "")
        if m:
            x, y = float(m.group(1)), float(m.group(2))
            break
    else:
        return None
    ow, oh = orig_wh
    sw, sh = sent_wh
    if coord_space == "norm1000":
        return x / 1000 * ow, y / 1000 * oh
    if coord_space == "norm1":
        return x * ow, y * oh
    if coord_space == "pixel":
        return x * ow / sw, y * oh / sh
    raise ValueError(coord_space)


def score(point, bbox):
    """ScreenSpot-style hit (point inside bbox) and distance (px) to bbox center."""
    if point is None:
        return False, None
    x, y = point
    x1, y1, x2, y2 = bbox
    hit = x1 <= x <= x2 and y1 <= y <= y2
    return hit, math.hypot(x - (x1 + x2) / 2, y - (y1 + y2) / 2)


_VOCAB = ("open close save search settings file edit view tab window menu button icon "
          "dialog scroll type click select copy paste folder document sheet cell link "
          "toolbar sidebar panel confirm cancel next back refresh download upload").split()


def make_history_text(approx_tokens, seed):
    """Deterministic fake agent history of ~approx_tokens tokens (1 token ~ 0.75 word).
    Real prompt_tokens is taken from server usage; this only sets the order of magnitude."""
    rng = random.Random(seed)
    lines, words = [], 0
    k = 1
    while words < approx_tokens * 0.75:
        line = f"Step {k}: " + " ".join(rng.choice(_VOCAB) for _ in range(rng.randint(8, 16)))
        lines.append(line)
        words += len(line.split())
        k += 1
    return "\n".join(lines)
