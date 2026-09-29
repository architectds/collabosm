"""Runs ON the Colab VM: one-glance stage + service health report."""
import os
import subprocess
import urllib.request

PORT = os.environ.get("PORT", "8090")
KEY_FILE = "/content/api-key.txt"


def sh(c):
    return subprocess.run(["bash", "-lc", c], capture_output=True, text=True).stdout


def models():
    """GET /v1/models with the box's key -- Strata's server wants it there too. Read here and
    sent as a header, never on a command line (a curl argument would show in `ps`)."""
    req = urllib.request.Request("http://127.0.0.1:%s/v1/models" % PORT)
    try:
        with open(KEY_FILE) as fh:
            req.add_header("Authorization", "Bearer " + fh.read().strip())
    except OSError:
        pass
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            return r.read().decode("utf-8", "replace").strip()
    except Exception as exc:                   # noqa: BLE001 - a report, not a check
        body = getattr(exc, "read", lambda: b"")()
        return (body.decode("utf-8", "replace").strip() if body else repr(exc))


status = "no STATUS file"
if os.path.exists("/content/STATUS"):
    status = open("/content/STATUS", errors="replace").read().strip()

health = sh("curl -s -o /dev/null -w %%{http_code} -m 5 http://127.0.0.1:%s/health" % PORT).strip()
listing = models()
# ours (api_server.py), or Strata's own server for a Strata recipe (strata.sh)
running = bool(sh("pgrep -f 'api_server.py|serve/server.py --engine strata'").strip())
gpu = sh("nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits").strip()
tunnel = sh("grep -oE 'https://[a-z0-9-]+\\.trycloudflare\\.com' /content/tunnel.log 2>/dev/null"
            " | head -1").strip()

print("stage:          " + status)
print("health:         " + health)
print("engine_running: " + str(running))
print("gpu_MiB:        " + gpu)
print("tunnel:         " + (tunnel or "(none)"))
if listing:
    print("models:         " + listing[:400])
