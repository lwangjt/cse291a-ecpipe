"""List models on the TritonAI endpoint and time one tiny request. Needs OPENAI_BASE_URL / OPENAI_API_KEY
(set automatically in UCSD DataHub; export them yourself elsewhere). Never commit the key."""
import os, sys
sys.path.insert(0, ".")
import requests
from ecpipe.client import chat_stream
base = os.environ["OPENAI_BASE_URL"]
h = {"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"}
print([m["id"] for m in requests.get(base.rstrip("/") + "/models", headers=h, timeout=30).json().get("data", [])])
if len(sys.argv) > 1:
    r = chat_stream(base, sys.argv[1], [{"role": "user", "content": "Say hi."}], 16)
    print({k: r[k] for k in ("ok", "error", "latency_s", "ttft_s", "text")})
