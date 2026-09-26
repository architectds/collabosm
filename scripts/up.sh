#!/usr/bin/env bash
# collabosm up: get a working EXL3 Flash-Next endpoint on a Colab A100-80GB.
#
#   bash scripts/up.sh
#   SESSION=mybox CACHE_SIZE=524288 bash scripts/up.sh
#
# Everything is derived from this script's own location, so the checkout is portable.
# There is no Drive, no Colab secret and no personal path in the critical path.
#
# Order of operations is chosen so that the expensive things happen last:
#   1. restore/verify the box   (a 40 GB box is rejected in ~1 min, before any download)
#   2. upload the kit + bootstrap launch script
#   3. bootstrap: runtime, then weights from Hugging Face at a pinned revision
#   4. serve      : load the model, publish it through the tunnel (chained after 3 ON the VM)
#   5. wait until the API is healthy AND serve.sh has published its URL
#
# Exit codes: 0 READY | 1 upload/bootstrap failed | 2-6 from restore.py | 7 timed out
#             8 serve.sh failed | 9 healthy on the VM but no tunnel URL
set -uo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)
COLAB=${COLAB:-$(command -v colab 2>/dev/null || echo "$HOME/.local/bin/colab")}
COLAB_PY=${COLAB_PY:-$(ls -d "$HOME"/.local/share/uv/tools/google-colab-cli/bin/python 2>/dev/null | head -1)}

SESSION=${SESSION:-collabosm}
RUNTIME=${RUNTIME:-wheel}          # wheel | source   (see manifest.json)
CACHE_SIZE=${CACHE_SIZE:-500224}    # measured best on an 80 GB card (see docs/CONCURRENCY.md)
CACHE_QUANT=${CACHE_QUANT:-4}
CPU_CACHE_GB=${CPU_CACHE_GB:-32}    # pinned RAM second-tier KV page cache, 0 = off
RECURRENT_CACHE_GB=${RECURRENT_CACHE_GB:-24}
NDT=${NDT:-4}
GCS=${GCS:-8192}                    # biggest prefill lever measured (2,806 -> 3,882 t/s)
PORT=${PORT:-8090}
WAIT_MIN=${WAIT_MIN:-50}

say() { printf "[up %s] %s\n" "$(date -u +%H:%M:%S)" "$*"; }

# ---------------------------------------------------------------- 1. the box
say "restoring/creating the A100-80GB box (session: $SESSION)"
if [ -n "$COLAB_PY" ]; then
  "$COLAB_PY" "$SCRIPT_DIR/restore.py" -n "$SESSION" || {
    rc=$?; say "!! restore failed (rc=$rc). 6 = a 40 GB box was drawn and stopped; just re-run."
    exit "$rc"; }
else
  python3 "$SCRIPT_DIR/restore.py" -n "$SESSION" || exit $?
fi

# ------------------------------------------------------- 2. push the toolkit
say "uploading the toolkit"
# api_server.py MUST be here: serve.sh launches /content/api_server.py, and this kit
# shipped for a while without uploading it, so a fresh clone reached
# "!! never became healthy" with nothing obviously wrong. Fail loudly instead.
for f in api_server.py bootstrap.sh serve.sh status.py probe_gpu.py; do
  timeout 180 $COLAB upload -s "$SESSION" "$SCRIPT_DIR/$f" "/content/$f" >/dev/null 2>&1 \
    && say "  ok   $f" || { say "  FAIL $f (upload) - refusing to continue"; exit 1; }
done

# ----------------------------------------------------------- 3. bootstrap it
say "bootstrapping (runtime=$RUNTIME)"
cat > /tmp/collabosm_env.sh <<ENV
export RUNTIME=$(printf '%q' "$RUNTIME")
export CACHE_SIZE=$(printf '%q' "$CACHE_SIZE")
export CACHE_QUANT=$(printf '%q' "$CACHE_QUANT")
export CPU_CACHE_GB=$(printf '%q' "$CPU_CACHE_GB")
export RECURRENT_CACHE_GB=$(printf '%q' "$RECURRENT_CACHE_GB")
export NDT=$(printf '%q' "$NDT")
export GCS=$(printf '%q' "$GCS")
export PORT=$(printf '%q' "$PORT")
ENV
timeout 120 $COLAB upload -s "$SESSION" /tmp/collabosm_env.sh /content/collabosm_env.sh >/dev/null 2>&1
# serve.sh is chained after bootstrap.sh on the VM, so it runs only if bootstrap
# succeeded. For a long time nothing started it at all: this script then waited its
# full 50 minutes for a /health that could never answer. `exec` matters too:
# serve.sh stops older copies of itself by matching "bash /content/serve.sh", and
# without exec this launcher shell's own command line would match.
cat > /tmp/start_bootstrap.py <<'PY'
import subprocess
print(subprocess.run("nohup bash -lc 'source /content/collabosm_env.sh && bash /content/bootstrap.sh "
                     "&& exec bash /content/serve.sh' > /content/bootstrap.log 2>&1 & echo BOOTSTRAPPING",
                     shell=True, capture_output=True, text=True).stdout)
PY
timeout 200 $COLAB exec -s "$SESSION" --timeout 150 -f /tmp/start_bootstrap.py 2>&1 | grep -o 'BOOTSTRAPPING' | head -1

say "waiting up to ${WAIT_MIN} min for the endpoint"
deadline=$(( $(date +%s) + WAIT_MIN * 60 ))
while [ "$(date +%s)" -lt "$deadline" ]; do
  sleep 45
  out=$($COLAB exec -s "$SESSION" --timeout 60 -f "$SCRIPT_DIR/status.py" 2>/dev/null)
  echo "$out" | sed 's/^/  /'
  stage=$(echo "$out" | sed -n 's/^stage: *stage=\([a-z_]*\).*/\1/p' | head -1)
  # READY needs the published URL as well as a healthy API: /health answers a few
  # seconds before serve.sh has the tunnel hostname, and a READY without a URL
  # leaves every client -- and the frontend -- with nothing to point at.
  if [ "$stage" = "ready" ] && echo "$out" | grep -q "health:         200"; then
    say "READY"
    echo "$out" | grep -E '^tunnel:|^models:' | sed 's/^/  /'
    say "when you are done: bash $ROOT/scripts/down.sh   (stopping is the whole point)"
    exit 0
  fi
  case "$stage" in
    serve_failed|serve_timeout)
      say "!! serve.sh reported stage=$stage - see /content/serve.log on the VM"
      exit 8 ;;
    ready_no_tunnel)
      say "!! the API is healthy on the VM but no tunnel URL was published - see /content/tunnel.log"
      exit 9 ;;
  esac
  if $COLAB exec -s "$SESSION" --timeout 60 -f "$SCRIPT_DIR/bootstrap_ok.py" 2>/dev/null | grep -q "bootstrap_failed"; then
    say "!! bootstrap reported failure - see /content/bootstrap.log on the VM"
    exit 1
  fi
done
say "!! timed out after ${WAIT_MIN} min"
exit 7