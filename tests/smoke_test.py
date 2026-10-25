"""End-to-end smoke test against a mock OpenAI-compatible server (no models needed)."""
import json, os, subprocess, sys, tempfile, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from PIL import Image

class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def do_POST(self):
        self.rfile.read(int(self.headers["Content-Length"]))
        self.send_response(200); self.send_header("Content-Type", "text/event-stream"); self.end_headers()
        for w in ["click(start_box='(500,", "500)')"]:
            time.sleep(0.02)
            self.wfile.write(b"data: " + json.dumps({"choices": [{"delta": {"content": w}}]}).encode() + b"\n\n")
            self.wfile.flush()
        self.wfile.write(b"data: " + json.dumps({"choices": [], "usage": {"prompt_tokens": 10, "completion_tokens": 6}}).encode() + b"\n\ndata: [DONE]\n\n")

srv = ThreadingHTTPServer(("127.0.0.1", 18999), H)
threading.Thread(target=srv.serve_forever, daemon=True).start()
d = tempfile.mkdtemp()
os.makedirs(f"{d}/images")
with open(f"{d}/dev.jsonl", "w") as f:
    for i in range(4):
        Image.new("RGB", (800, 600), "white").save(f"{d}/images/{i}.png")
        f.write(json.dumps(dict(id=i, image=f"images/{i}.png", instruction="the OK button",
                                bbox=[350, 250, 450, 350] if i % 2 == 0 else [0, 0, 10, 10])) + "\n")
url = "http://127.0.0.1:18999/v1"
env = dict(os.environ, PYTHONPATH=".")
def run(*args): subprocess.run([sys.executable, "-m", *args], check=True, env=env)
run("ecpipe.stage_bench", "--stage", "grounding", "--base-url", url, "--model", "m", "--data", f"{d}/dev.jsonl",
    "--data-root", d, "--n", "4", "--passes", "2", "--label", "mock", "--out-dir", f"{d}/res")
run("ecpipe.stage_bench", "--stage", "planner", "--base-url", url, "--model", "m", "--n", "4", "--passes", "2",
    "--label", "mock", "--concurrency", "2", "--out-dir", f"{d}/res")
open(f"{d}/cfg.yaml", "w").write(f"""name: mockpipe
stages:
  planner: {{base_url: "{url}", model: m}}
  grounding: {{base_url: "{url}", model: m}}
""")
run("ecpipe.pipeline", "--config", f"{d}/cfg.yaml", "--data", f"{d}/dev.jsonl", "--data-root", d, "--n", "4",
    "--passes", "2", "--out-dir", f"{d}/res")
run("ecpipe.analyze", "--results", f"{d}/res")
