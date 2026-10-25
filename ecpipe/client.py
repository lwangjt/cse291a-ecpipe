"""Streaming client for OpenAI-compatible servers (llama.cpp llama-server, vLLM, Ollama, LM Studio).

Using one wire protocol for every backend keeps the *measurement* identical across
CPU / GPU / NPU placements: only the server behind the URL changes.
"""
import base64
import io
import json
import os
import time

import requests
from PIL import Image


def encode_image(path, max_pixels=None):
    """Load, optionally downscale (aspect-preserving), PNG+base64 encode.

    Returns (data_url, (sent_w, sent_h), (orig_w, orig_h)).
    The cost of this function is the 'preprocess' stage in the pipeline.
    """
    img = Image.open(path).convert("RGB")
    ow, oh = img.size
    if max_pixels and ow * oh > max_pixels:
        s = (max_pixels / (ow * oh)) ** 0.5
        img = img.resize((max(1, int(ow * s)), max(1, int(oh * s))), Image.BICUBIC)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/png;base64,{b64}", img.size, (ow, oh)


def _headers(base_url):
    """Auth for remote endpoints (e.g. TritonAI). The key is read from the environment, never
    from code/args, and is NOT sent to local servers."""
    key = os.environ.get("ECPIPE_API_KEY")
    local = base_url.startswith(("http://127.", "http://localhost"))
    if not key and not local:
        key = os.environ.get("OPENAI_API_KEY")
    return {"Authorization": f"Bearer {key}"} if key else {}


def chat_stream(base_url, model, messages, max_tokens=128, temperature=0.0,
                timeout=600.0, extra=None):
    """One streamed chat completion. NEVER raises: failures are returned as ok=False
    so that timeouts/errors are recorded rather than silently dropped."""
    payload = {
        "model": model, "messages": messages, "max_tokens": max_tokens,
        "temperature": temperature, "stream": True,
        "stream_options": {"include_usage": True},
    }
    if extra:
        payload.update(extra)
    out = dict(ok=False, error=None, text="", latency_s=None, ttft_s=None,
               decode_tps=None, prompt_tokens=None, completion_tokens=None,
               server_timings=None, t_start=time.time())
    t0 = time.perf_counter()
    n_chunks, t_first = 0, None
    try:
        url = base_url.rstrip("/") + "/chat/completions"
        hdr = _headers(base_url)
        r = requests.post(url, json=payload, headers=hdr, stream=True, timeout=(10, timeout))
        if r.status_code == 400 and "stream_options" in payload:   # some servers reject it
            r.close()
            payload.pop("stream_options")
            r = requests.post(url, json=payload, headers=hdr, stream=True, timeout=(10, timeout))
        with r:
            r.raise_for_status()
            for raw in r.iter_lines(decode_unicode=True):
                if time.perf_counter() - t0 > timeout:
                    raise TimeoutError(f"exceeded {timeout}s")
                if not raw or not raw.startswith("data:"):
                    continue
                data = raw[5:].strip()
                if data == "[DONE]":
                    break
                ch = json.loads(data)
                if ch.get("usage"):
                    out["prompt_tokens"] = ch["usage"].get("prompt_tokens")
                    out["completion_tokens"] = ch["usage"].get("completion_tokens")
                if ch.get("timings"):          # llama.cpp reports server-side timings
                    out["server_timings"] = ch["timings"]
                for c in ch.get("choices", []):
                    piece = (c.get("delta") or {}).get("content")
                    if piece:
                        if t_first is None:
                            t_first = time.perf_counter()
                        out["text"] += piece
                        n_chunks += 1
        t_end = time.perf_counter()
        out["latency_s"] = t_end - t0
        out["ttft_s"] = (t_first - t0) if t_first else None
        ntok = out["completion_tokens"] or n_chunks
        out["completion_tokens"] = ntok
        if t_first and ntok > 1 and t_end > t_first:
            out["decode_tps"] = (ntok - 1) / (t_end - t_first)
        out["ok"] = bool(out["text"])
        if not out["ok"]:
            out["error"] = "empty_response"
    except Exception as e:  # noqa: BLE001 - we want every failure recorded
        out["latency_s"] = time.perf_counter() - t0
        out["error"] = f"{type(e).__name__}: {e}"[:300]
    return out
