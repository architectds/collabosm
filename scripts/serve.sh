#!/usr/bin/env bash
# collabosm serve (runs ON the Colab VM).
#
#   source /content/collabosm.env && bash serve.sh
#
# Starts the OpenAI-compatible API (scripts/api_server.py) with the measured-best
# ExLlamaV3 settings, then a cloudflared quick tunnel, and writes /content/STATUS.
set -uo pipefail

LOG=/content/serve.log
TUNNEL_LOG=/content/tunnel.log
STATUS=/content/STATUS
KEY_FILE=/content/api-key.txt
PY=${PY:-python3}

say() { echo "[serve $(date -u +%H:%M:%S)] $*"; }
status() { echo "$*" > "$STATUS"; }

[[ -f /content/collabosm.env ]] && source /content/collabosm.env
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

say "starting the API (cache=${CACHE_SIZE:-262144} cq=${CACHE_QUANT:-4} ccs=${CPU_CACHE_GB:-0}GB ndt=${NDT:-4} gcs=${GCS:-4096})"
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
nohup /content/cloudflared tunnel --url "http://127.0.0.1:$PORT" --no-autoupdate > "$TUNNEL_LOG" 2>&1 &
URL=""
for _ in $(seq 1 45); do
  URL=$(grep -oE 'https://[a-z0-9-]+\.trycloudflare\.com' "$TUNNEL_LOG" | head -1 || true)
  [[ -n "$URL" ]] && break
  sleep 4
done

if [[ -n "$URL" ]]; then
  printf '%s\n' "$URL" > /content/url.txt
  status "stage=ready url=$URL port=$PORT"
  say "ready: $URL   (api key: $KEY)"
else
  status "stage=ready_no_tunnel port=$PORT"
  say "ready locally on :$PORT; no tunnel url (check $TUNNEL_LOG)"
fi