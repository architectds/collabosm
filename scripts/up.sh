#!/usr/bin/env bash
# collabosm up: get a working EXL3 endpoint on Colab, for one recipe of recipes.json.
#
#   bash scripts/up.sh                                   # the default recipe
#   RECIPE=a100-40g/qwen38-27b bash scripts/up.sh        # another card + model
#   SESSION=mybox CACHE_SIZE=524288 bash scripts/up.sh   # override one setting, once
#   python3 scripts/recipe.py list                       # what the registry holds
#
# The recipe decides everything: the card (accelerator, shape, the VRAM a box must
# show before anything is downloaded), the model (repo, pinned revision, directory),
# and how it loads (cache, KV quant, tiers, draft depth, chunk size, vision, YaRN).
# A variable set by the caller wins over the recipe for that run.
#
# Everything is derived from this script's own location, so the checkout is portable.
# There is no Drive, no Colab secret and no personal path in the critical path.
#
# Order of operations is chosen so that the expensive things happen last:
#   1. restore/verify the box   (a box below the recipe's VRAM is rejected in ~1 min)
#   2. upload the kit + bootstrap launch script
#   3. bootstrap: runtime, then weights from Hugging Face at a pinned revision
#   4. serve      : load the model, publish it through the tunnel (chained after 3 ON the VM)
#   5. wait until the API is healthy AND serve.sh has published its URL
#
# Exit codes: 0 READY | 1 upload/bootstrap failed | 2 bad recipe or args | 3-6 from restore.py
#             7 timed out | 8 serve.sh failed | 9 healthy on the VM but no tunnel URL
set -uo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)
COLAB=${COLAB:-$(command -v colab 2>/dev/null || echo "$HOME/.local/bin/colab")}
COLAB_PY=${COLAB_PY:-$(ls -d "$HOME"/.local/share/uv/tools/google-colab-cli/bin/python 2>/dev/null | head -1)}
PYTHON=${PYTHON:-$(command -v python3 2>/dev/null || command -v python)}

SESSION=${SESSION:-collabosm}
RECIPE=${RECIPE:-a100-80g/qwen38-fn}
RUNTIME=${RUNTIME:-wheel}          # wheel | source   (see manifest.json)
PORT=${PORT:-8090}
WAIT_MIN=${WAIT_MIN:-50}
export RECIPE RUNTIME PORT

say() { printf "[up %s] %s\n" "$(date -u +%H:%M:%S)" "$*"; }

# ------------------------------------------------------------- 0. the recipe
# Everything the caller did not set comes from recipes.json (scripts/recipe.py).
recipe_env=$("$PYTHON" "$SCRIPT_DIR/recipe.py" env "$RECIPE") || {
  say "!! recipe $RECIPE cannot run (see above; list them with: python3 scripts/recipe.py list)"
  exit 2; }
eval "$recipe_env"
say "recipe $RECIPE: $ACCELERATOR shape=$SHAPE (>= ${MIN_VRAM_GIB} GiB) · $MODEL_REPO@${MODEL_REVISION:0:12}"
say "  cache=${CACHE_SIZE:-?} cq=${CACHE_QUANT:-?} ccs=${CPU_CACHE_GB:-0}GB rcs=${RECURRENT_CACHE_GB:-?}GB ndt=${NDT:-?} gcs=${GCS:-?} vision=${VISION:-0} yarn=${YARN_FACTOR:-0}"

# ---------------------------------------------------------------- 1. the box
say "restoring/creating the $ACCELERATOR box, shape $SHAPE (session: $SESSION)"
box_args=(-n "$SESSION" --accelerator "$ACCELERATOR" --shape "$SHAPE" --min-vram-gib "$MIN_VRAM_GIB")
if [ -n "$COLAB_PY" ]; then
  "$COLAB_PY" "$SCRIPT_DIR/restore.py" "${box_args[@]}" || {
    rc=$?; say "!! restore failed (rc=$rc). 6 = a box below ${MIN_VRAM_GIB} GiB was drawn and stopped; just re-run."
    exit "$rc"; }
else
  python3 "$SCRIPT_DIR/restore.py" "${box_args[@]}" || exit $?
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
work=$(mktemp -d)                  # not fixed /tmp names: two runs must not share them
trap 'rm -rf "$work"' EXIT
# the recipe's launch settings as they stand now, overrides included
"$PYTHON" "$SCRIPT_DIR/recipe.py" vmenv "$RECIPE" > "$work/collabosm_env.sh" || {
  say "!! could not write the launch environment for $RECIPE"; exit 2; }
# Checked: without it bootstrap never starts on a fresh box (and this script waits its
# full 50 paid minutes), and on a re-attached box the previous run's file is sourced,
# launching the old recipe under the new one's name.
timeout 120 $COLAB upload -s "$SESSION" "$work/collabosm_env.sh" /content/collabosm_env.sh >/dev/null 2>&1 \
  || { say "!! could not upload the launch environment - refusing to continue"; exit 1; }
# serve.sh is chained after bootstrap.sh on the VM, so it runs only if bootstrap
# succeeded. For a long time nothing started it at all: this script then waited its
# full 50 minutes for a /health that could never answer. `exec` matters too:
# serve.sh stops older copies of itself by matching "bash /content/serve.sh", and
# without exec this launcher shell's own command line would match. A re-attached box
# still has the last run's STATUS and endpoint.json: they go first, so neither this
# script nor the frontend can take the old service for the new one.
cat > "$work/start_bootstrap.py" <<'PY'
import subprocess
print(subprocess.run("rm -f /content/STATUS /content/endpoint.json /content/url.txt; "
                     "nohup bash -lc 'source /content/collabosm_env.sh && bash /content/bootstrap.sh "
                     "&& exec bash /content/serve.sh' > /content/bootstrap.log 2>&1 & echo BOOTSTRAPPING",
                     shell=True, capture_output=True, text=True).stdout)
PY
started=$(timeout 200 $COLAB exec -s "$SESSION" --timeout 150 -f "$work/start_bootstrap.py" 2>&1 \
          | grep -o 'BOOTSTRAPPING' | head -1)
[ "$started" = BOOTSTRAPPING ] || {
  say "!! the bootstrap did not start on the VM - refusing to wait 50 min for it"; exit 1; }
echo "$started"

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