#!/usr/bin/env bash
# collabosm serve (runs ON the Colab VM).
#
#   bash /content/serve.sh          (reads /content/collabosm.env itself)
#
# Starts the OpenAI-compatible API (scripts/api_server.py) with the measured-best
# ExLlamaV3 settings, then a cloudflared quick tunnel, and writes /content/STATUS.
# scripts/up.sh chains this after bootstrap.sh; run it by hand to restart the API.
set -uo pipefail

LOG=/content/serve.log
TUNNEL_LOG=/content/tunnel.log
STATUS=/content/STATUS
KEY_FILE=/content/api-key.txt
PY=${PY:-python3}

say() { echo "[serve $(date -u +%H:%M:%S)] $*"; }
status() { echo "$*" > "$STATUS"; }

# Exported, not merely set: api_server.py reads CACHE_SIZE, CPU_CACHE_GB, GCS ... from
# its environment. A plain `source` sets them for this shell only, so the log line
# below printed the intended values while the server quietly ran on its defaults
# (262144 tokens, no pinned-RAM tier, gcs 4096).
if [[ -f /content/collabosm.env ]]; then set -a; source /content/collabosm.env; set +a; fi
: "${PORT:=8090}"

[[ -f "$KEY_FILE" ]] || $PY - <<'PY'
import pathlib, secrets
pathlib.Path("/content/api-key.txt").write_text("sk-collabosm-" + secrets.token_hex(10))
PY
KEY=$(cat "$KEY_FILE")

# stop any previous instance (this script is safe to re-run)
# Older serve.sh processes must go too: otherwise a superseded instance watches its
# own child get killed and prints "api_server died" over the new run's log.
for pid in $(pgrep -f 'bash /content/serve.sh' 2>/dev/null); do
  [ "$pid" = "$$" ] || kill "$pid" 2>/dev/null || true
done
pkill -f 'api_serve[r].py' 2>/dev/null || true
pkill -f 'cloudflare[d]' 2>/dev/null || true
sleep 2

say "starting the API for ${RECIPE:-no recipe} (${MODEL_ID:-?}: cache=${CACHE_SIZE:-262144} cq=${CACHE_QUANT:-4} ccs=${CPU_CACHE_GB:-0}GB rcs=${RECURRENT_CACHE_GB:-4}GB ndt=${NDT:-4} gcs=${GCS:-4096} vision=${VISION:-0} yarn=${YARN_FACTOR:-0})"
status "stage=loading"
nohup $PY -u /content/api_server.py --port "$PORT" > "$LOG" 2>&1 &

for _ in $(seq 1 120); do
  code=$(curl -s -o /dev/null -w '%{http_code}' -m 5 "http://127.0.0.1:$PORT/health" || true)
  [[ "$code" == "200" ]] && break
  if ! pgrep -f '/content/api_server.py' >/dev/null; then
    say "!! api_server died"; tail -80 "$LOG"; status "stage=serve_failed"; exit 1
  fi
  sleep 5
done
[[ "${code:-}" == "200" ]] || { say "!! never became healthy"; tail -80 "$LOG"; status "stage=serve_timeout"; exit 1; }
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader | sed 's/^/  /'

[[ -x /content/cloudflared ]] || curl -fsSL -o /content/cloudflared \
  https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 \
  && chmod +x /content/cloudflared
# TUNNEL_TOKEN=<token from `cloudflared tunnel token <name>`> gives a STABLE hostname
# (a named tunnel with its public hostname already pointing at http://localhost:$PORT).
# Without it we get a quick tunnel, whose hostname changes on every restart.
if [[ -n "${TUNNEL_TOKEN:-}" ]]; then
  say "starting the NAMED tunnel (stable hostname)"
  nohup /content/cloudflared tunnel --no-autoupdate run --token "$TUNNEL_TOKEN" > "$TUNNEL_LOG" 2>&1 &
  URL="${PUBLIC_URL:-}"
  sleep 5
else
  nohup /content/cloudflared tunnel --url "http://127.0.0.1:$PORT" --no-autoupdate > "$TUNNEL_LOG" 2>&1 &
  URL=""
  for _ in $(seq 1 45); do
    URL=$(grep -oE 'https://[a-z0-9-]+\.trycloudflare\.com' "$TUNNEL_LOG" | head -1 || true)
    [[ -n "$URL" ]] && break
    sleep 4
  done
fi

# Everything the frontend needs to couple, in one file: it is fetched with a single
# `colab download` (the contents API -- `colab exec` goes through the kernel and can
# hang). Written before STATUS says ready, so a reader never sees a stale URL.
$PY - "$URL" "$PORT" "${RECIPE:-}" "${MODEL_ID:-}" <<'PY'
import json, pathlib, sys, time
url, port, recipe, model = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]
key = pathlib.Path("/content/api-key.txt").read_text().strip()
pathlib.Path("/content/endpoint.json").write_text(json.dumps(
    {"url": url or None, "port": port, "key": key, "at": int(time.time()),
     "recipe": recipe or None, "model": model or None}))
PY

if [[ -n "$URL" ]]; then
  printf '%s\n' "$URL" > /content/url.txt
  status "stage=ready url=$URL port=$PORT"
  say "ready: $URL   (api key: $KEY)"
else
  status "stage=ready_no_tunnel port=$PORT"
  say "ready locally on :$PORT; no tunnel url (check $TUNNEL_LOG)"
fi