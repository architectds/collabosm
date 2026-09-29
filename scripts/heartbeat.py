"""Runs ON the Colab VM, in the notebook kernel: one line of the box's health.

The frontend has this run every ~10 minutes while the box is in use
(scripts/colab_keepalive.py --heartbeat), and once more as soon as the tunnel
stops answering. Running it is the use Colab counts -- chat through the tunnel
is not (docs/MEASURED.md). The line is what is left to read when something goes
wrong. It is appended to /content/heartbeat.jsonl here and kept on the laptop
too (~/.collabosm/heartbeat.jsonl), since a reclaimed VM takes its disk with it.

It also answers, over Colab's own channel, the question the tunnel cannot:
when the public address stops answering, is it the tunnel, the model server,
or the machine? The VM's key is read only to ask the local server for its
activity; it is never printed or kept.

Everything runs inside one function, so nothing but that function's name is
left in the kernel's namespace, and the caller deletes that too. Prints exactly
one line, `HEARTBEAT_LINE ` followed by compact JSON.
"""


def _collabosm_heartbeat():
    import json
    import os
    import shutil
    import subprocess
    import time
    import urllib.error
    import urllib.request

    log = "/content/heartbeat.jsonl"
    port = int(os.environ.get("PORT") or 8090)
    t0 = time.time()

    def sh(cmd, timeout=5):
        try:
            return subprocess.run(["bash", "-lc", cmd], capture_output=True, text=True,
                                  timeout=timeout).stdout.strip()
        except Exception:
            return ""

    def num(s):
        try:
            return float(s)
        except (TypeError, ValueError):
            return None

    def read(path):
        try:
            with open(path, errors="replace") as fh:
                return fh.read().strip()
        except OSError:
            return ""

    def http(url, headers=None, timeout=5):
        """(status or None, body, ms)"""
        start = time.time()
        try:
            req = urllib.request.Request(url, headers=headers or {})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.status, r.read(65536), int((time.time() - start) * 1000)
        except urllib.error.HTTPError as e:
            return e.code, b"", int((time.time() - start) * 1000)
        except Exception:
            return None, b"", None

    beat = {"at": int(t0)}
    up = read("/proc/uptime").split()
    beat["uptime_s"] = int(float(up[0])) if up else None

    rows = sh("nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,"
              "temperature.gpu,power.draw --format=csv,noheader,nounits").splitlines()
    parts = [p.strip() for p in rows[0].split(",")] if rows else []
    beat["gpu"] = (dict(zip(("util_pct", "mem_used_mib", "mem_total_mib", "temp_c", "power_w"),
                            map(num, parts))) if len(parts) == 5 else None)

    mem = {}
    for line in read("/proc/meminfo").splitlines():
        k, _, v = line.partition(":")
        if k in ("MemTotal", "MemAvailable") and v.split():
            mem[k] = num(v.split()[0])
    beat["ram"] = ({"used_gib": round((mem["MemTotal"] - (mem.get("MemAvailable") or 0)) / 2 ** 20, 1),
                    "total_gib": round(mem["MemTotal"] / 2 ** 20, 1)} if mem.get("MemTotal") else None)
    try:
        beat["disk_free_gib"] = round(shutil.disk_usage("/content").free / 2 ** 30, 1)
    except OSError:
        beat["disk_free_gib"] = None

    # the model server, from inside the VM: no tunnel in the way
    # ours (api_server.py), or Strata's own server for a Strata recipe (scripts/strata.sh)
    server = {"proc": bool(sh("pgrep -f '/content/api_server.py|serve/server.py --engine strata'"))}
    code, _, ms = http("http://127.0.0.1:%d/health" % port)
    server.update(health=code, ms=ms)
    key = read("/content/api-key.txt")
    if key:
        code, body, _ = http("http://127.0.0.1:%d/v1/status" % port,
                             {"Authorization": "Bearer " + key}, timeout=6)
        if code == 200:
            try:
                st = json.loads(body)
                act = st.get("activity") or {}
                server.update(requests=act.get("requests"), in_flight=act.get("in_flight"),
                              last_request_at=act.get("last_request_at"))
            except ValueError:
                pass
    key = None
    beat["server"] = server

    # the tunnel: is cloudflared running, and does the public address come back to us
    url = read("/content/url.txt")
    tunnel = {"proc": bool(sh("pgrep -f 'cloudflared tunnel'")), "url": url or None}
    if url:
        code, _, ms = http(url.rstrip("/") + "/health", timeout=8)
        tunnel.update(public=code, ms=ms)
    beat["tunnel"] = tunnel

    beat["took_ms"] = int((time.time() - t0) * 1000)
    line = json.dumps(beat, separators=(",", ":"))
    try:
        with open(log, "a") as fh:
            fh.write(line + "\n")
    except OSError:
        pass
    print("HEARTBEAT_LINE " + line)


_collabosm_heartbeat()
del _collabosm_heartbeat
