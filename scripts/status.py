"""Runs ON the Colab VM: one-glance stage + service health report."""
import os
import subprocess

PORT = os.environ.get("PORT", "8090")


def sh(c):
    return subprocess.run(["bash", "-lc", c], capture_output=True, text=True).stdout


status = "no STATUS file"
if os.path.exists("/content/STATUS"):
    status = open("/content/STATUS", errors="replace").read().strip()

health = sh("curl -s -o /dev/null -w %%{http_code} -m 5 http://127.0.0.1:%s/health" % PORT).strip()
models = sh("curl -s -m 5 http://127.0.0.1:%s/v1/models" % PORT).strip()
running = bool(sh("pgrep -f 'api_server.py'").strip())
gpu = sh("nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits").strip()
tunnel = sh("grep -oE 'https://[a-z0-9-]+\\.trycloudflare\\.com' /content/tunnel.log 2>/dev/null"
            " | head -1").strip()

print("stage:          " + status)
print("health:         " + health)
print("engine_running: " + str(running))
print("gpu_MiB:        " + gpu)
print("tunnel:         " + (tunnel or "(none)"))
if models:
    print("models:         " + models[:400])