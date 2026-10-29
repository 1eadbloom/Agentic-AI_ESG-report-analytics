import os
import base64
import urllib.request
import urllib.error
from dotenv import load_dotenv

load_dotenv()

keys = ["LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY",
        "LANGFUSE_BASE_URL", "LANGFUSE_HOST", "LANGFUSE_OTEL_HOST"]
for k in keys:
    v = os.environ.get(k)
    print(k, "=", (v[:12] + "...") if v else "NOT SET")

pk = os.environ.get("LANGFUSE_PUBLIC_KEY", "")
sk = os.environ.get("LANGFUSE_SECRET_KEY", "")
host = (os.environ.get("LANGFUSE_BASE_URL")
        or os.environ.get("LANGFUSE_HOST") or "").rstrip("/")

if not (pk and sk and host):
    print("Missing key or host, stop.")
else:
    token = base64.b64encode(f"{pk}:{sk}".encode()).decode()
    req = urllib.request.Request(
        host + "/api/public/projects",
        headers={"Authorization": "Basic " + token},
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            print("HTTP", r.status, "-> keys and host match")
    except urllib.error.HTTPError as e:
        print("HTTP", e.code, "-> 401 means keys do not match this host")
    except Exception as e:
        print("Request failed:", type(e).__name__, e)