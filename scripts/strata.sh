# collabosm: a Strata recipe on the VM. Sourced by bootstrap.sh and serve.sh, which
# define say, status, fail, PORT, MODEL_DIR and (serve.sh) LOG and KEY.
#
# Strata (github.com/Niko1221/Strata; the recipe pins a commit of a fork) is its own
# engine and its own OpenAI-compatible server, for Qwen3.8-Flash-Next: the experts it
# cannot keep in VRAM are computed by the CPU from pinned RAM. Its setup.py does the
# install -- the ready-made engine, CUDA's pip libraries, the MTP draft layer, a
# config. What collabosm adds: the pinned commit, the engine zip checked against the
# recipe's SHA-256 before setup.py sees it, the GGUF at the recipe's pinned revision
# (in parallel, where setup.py would read one stream), CUDA 12 (STRATA_CUDA=12: the
# fork's option for drivers older than 580), the recipe's engine flags, and the server
# started behind the box's key, which goes in through the environment, not argv.

STRATA_DIR=${STRATA_DIR:-/content/strata}
STRATA_ENGINE_DIR=/content/strata-engine      # the checked zip; setup.py copies it from here

strata_install() {
  status "stage=runtime"
  say "Strata ${STRATA_COMMIT:0:12} from $STRATA_REPO"
  if [ "$(git -C "$STRATA_DIR" rev-parse HEAD 2>/dev/null)" != "$STRATA_COMMIT" ]; then
    # another commit into the same checkout: its .venv and its downloads stay put
    [ -d "$STRATA_DIR/.git" ] || git init -q "$STRATA_DIR"
    git -C "$STRATA_DIR" fetch -q --depth 1 "$STRATA_REPO" "$STRATA_COMMIT" \
      && git -C "$STRATA_DIR" checkout -q -f FETCH_HEAD || { say "!! could not fetch Strata"; return 1; }
  fi
  # setup.py keeps an installed engine of the same version, so a new build (the image
  # encoder added, say) would be passed over: the engine the recipe pins is the one there
  if [ "$(cat "$STRATA_DIR/engine/.collabosm-sha256" 2>/dev/null)" != "$STRATA_SHA256" ]; then
    rm -rf "$STRATA_DIR/engine"
  fi
  local zip="$STRATA_ENGINE_DIR/strata-linux-x64.zip"
  mkdir -p "$STRATA_ENGINE_DIR"
  if ! echo "$STRATA_SHA256  $zip" | sha256sum -c --status - 2>/dev/null; then
    curl -fsSL -o "$zip" "${STRATA_PREBUILT_URL}strata-linux-x64.zip" || { say "!! engine download"; return 1; }
    echo "$STRATA_SHA256  $zip" | sha256sum -c --status - \
      || { say "!! the engine is not the one the recipe pins (SHA-256)"; rm -f "$zip"; return 1; }
  fi
  say "engine checked: ${STRATA_SHA256:0:16}"
  # setup.sh wants a Python whose venv can bring pip (ensurepip; Debian ships it apart
  # as python3-venv) and otherwise calls `sudo apt-get`. Colab runs as root and has no
  # business being apt-installed into, so the .venv is made here -- setup.sh keeps a
  # .venv that has pip -- with get-pip.py when ensurepip is missing.
  local venv="$STRATA_DIR/.venv"
  if ! "$venv/bin/python" -m pip --version >/dev/null 2>&1; then
    rm -rf "$venv"
    python3 -m venv "$venv" >/dev/null 2>&1 || {
      rm -rf "$venv"
      say "no ensurepip in $(python3 --version 2>&1): the .venv gets pip from get-pip.py"
      python3 -m venv --without-pip "$venv" \
        && curl -fsSL https://bootstrap.pypa.io/get-pip.py | "$venv/bin/python" - -q
    } || { say "!! could not make Strata's Python environment"; return 1; }
  fi
  say "Python for Strata: $("$venv/bin/python" --version 2>&1), $("$venv/bin/python" -m pip --version 2>&1 | cut -d' ' -f1-2)"
  # setup.py prints its own steps; what it still downloads goes to MODEL_DIR (the MTP
  # layer), and --prebuilt as a folder means it copies our zip, not a URL's
  status "stage=weights"
  strata_weights || return 1
  ( cd "$STRATA_DIR" && STRATA_CUDA="${STRATA_CUDA:-12}" bash setup.sh --family qwen \
      --model "$STRATA_MODEL" --vision "${STRATA_VISION:-none}" --context "${STRATA_CONTEXT:-262144}" \
      --kv "${STRATA_KV:-int8}" --data-dir "$MODEL_DIR" --port "$PORT" \
      --prebuilt "$STRATA_ENGINE_DIR/" --yes --no-start ) || { say "!! Strata setup failed"; return 1; }
  [ -n "$(strata_config)" ] || { say "!! Strata setup wrote no config"; return 1; }
  echo "$STRATA_SHA256" > "$STRATA_DIR/engine/.collabosm-sha256"
  strata_engine_args "$(strata_config)" || return 1
}

# The GGUF and the image encoder, before setup.py: at the recipe's pinned revision, with
# Hugging Face's Xet transfers (many ranges of a file at once). setup.py reads each file
# as one HTTP stream from main, and its 84 GB took most of a 22-minute first start. They
# go where setup.py looks (<data>/models/<quant>/, the encoder in <data>/models/), each
# with the .done mark that makes setup.py keep a file instead of fetching it.
strata_weights() {
  local models="$MODEL_DIR/models" vision="${STRATA_VISION:-none}"
  local want="$MODEL_REPO@$MODEL_REVISION $STRATA_MODEL vision=$vision"
  if [ "$(cat "$models/.collabosm-revision" 2>/dev/null)" = "$want" ]; then
    say "weights already present: $want"
    return 0
  fi
  say "downloading $want -> $models"
  local t0; t0=$(date +%s)
  python -c "import hf_xet" 2>/dev/null || python -m pip install -q hf_xet >/dev/null 2>&1 \
    || say "  (no hf_xet: one HTTP stream per file)"
  HF_XET_HIGH_PERFORMANCE=1 python - "$MODEL_REPO" "$MODEL_REVISION" "$models" "$STRATA_MODEL" "$vision" <<'PY' \
    || { say "!! model download (HF_HUB_DISABLE_XET=1 is the fallback when Xet stalls)"; return 1; }
import pathlib, sys
from huggingface_hub import snapshot_download
repo, rev, models, quant, vision = sys.argv[1:]
models = pathlib.Path(models)
snapshot_download(repo_id=repo, revision=rev, local_dir=str(models), max_workers=8,
                  allow_patterns=[quant + "/*.gguf"] + (["mmproj-*.gguf"] if vision != "none" else []))
got = sorted((models / quant).glob("*.gguf")) + (sorted(models.glob("mmproj-*.gguf")) if vision != "none" else [])
if len(got) < 2 + (vision != "none"):
    sys.exit("expected two %s shards%s, found %s" % (quant, " and the image encoder" if vision != "none" else "",
                                                     [p.name for p in got]))
for p in got:
    p.with_name(p.name + ".done").write_text("collabosm " + rev)
    print("  %s  %.1f GB" % (p.name, p.stat().st_size / 1e9))
PY
  echo "$want" > "$models/.collabosm-revision"
  say "weights in $(( $(date +%s) - t0 ))s"
}

# STRATA_ENGINE_ARGS: engine flags the recipe measured better than setup.py's choice
# (e.g. "--ple-io mmap --ple-row-cache 16777216"), merged into the config's args: a flag
# already there gets the recipe's value, a new one is added, "-FLAG" removes one.
strata_engine_args() {
  [ -n "${STRATA_ENGINE_ARGS:-}" ] || return 0
  "$STRATA_DIR/.venv/bin/python" - "$1" "$STRATA_ENGINE_ARGS" <<'PY' || { say "!! STRATA_ENGINE_ARGS: bad"; return 1; }
import json, shlex, sys
path, extra = sys.argv[1], shlex.split(sys.argv[2])
cfg = json.load(open(path, encoding="utf-8-sig"))
args = cfg["args"]
def find(flag):
    return args.index(flag) if flag in args else -1
i = 0
while i < len(extra):
    flag = extra[i]
    has_value = i + 1 < len(extra) and not extra[i + 1].startswith("-")
    if flag.startswith("--"):
        at = find(flag)
        if has_value:
            if at >= 0 and at + 1 < len(args) and not args[at + 1].startswith("-"):
                args[at + 1] = extra[i + 1]
            elif at >= 0:
                args.insert(at + 1, extra[i + 1])
            else:
                args += [flag, extra[i + 1]]
        elif at < 0:
            args.append(flag)
    elif flag.startswith("-") and find("-" + flag) >= 0:          # "-FLAG": remove --FLAG and its value
        at = find("-" + flag)
        del args[at:at + (2 if at + 1 < len(args) and not args[at + 1].startswith("-") else 1)]
    i += 2 if has_value else 1
json.dump(cfg, open(path, "w", encoding="utf-8"), indent=1)
print("engine args:", " ".join(args))
PY
}

strata_config() { ls -t "$STRATA_DIR"/strata-*.json 2>/dev/null | head -1; }

strata_start() {
  local cfg; cfg=$(strata_config)
  [ -n "$cfg" ] || { say "!! no Strata config: did bootstrap.sh run?"; return 1; }
  say "Strata server: $(basename "$cfg"), 127.0.0.1:$PORT"
  ( cd "$STRATA_DIR" && STRATA_API_KEY="$KEY" exec nohup .venv/bin/python -u serve/server.py \
      --engine strata --config "$cfg" --port "$PORT" ) > "$LOG" 2>&1 &
}

STRATA_PROC='serve/server.py --engine strata'

# The n-gram table (the second shard, 28.8 GB) is read 16 rows per token, one 4 KB
# O_DIRECT read each. /content is overlayfs on a loop device over Colab's PersistentDisk,
# which serves ~2,500 random reads a second: real text then reads its prompt at 240-410
# t/s, the GPU idle for 72% of it. The loop device caches its backing file (dio=0), so one
# O_DIRECT pass over the table leaves it in RAM once -- a buffered read would cache it twice
# and push itself out -- and every lookup after is a ~27 us memory copy: 905-1,376 t/s,
# and decode 42 -> 61 t/s (measured on the A100-40G, 2026-09-29). The engine's own start
# reads 47 GB of experts through the same cache and pushes the table out, so this runs
# after every engine start, the first and any restart (Strata restarts a stopped engine
# inside the next request, its /health up throughout), once the engine's memory has
# stopped growing: in the background, chat works during the ~90 s, just slower.
strata_warm() {
  local table; table=$(ls "$MODEL_DIR/models/$STRATA_MODEL/"*-00002-of-00002.gguf 2>/dev/null | head -1)
  [ -n "$table" ] || { say "no n-gram table to warm under $MODEL_DIR/models/$STRATA_MODEL"; return 0; }
  pkill -f 'strata_war[m]_loop' 2>/dev/null || true
  setsid nohup bash -c '
    strata_warm_loop() {
      local last="" prev=0 pid rss t0
      while :; do
        pid=$(pgrep -f "/engine/strat[a] --serve" | head -1)
        rss=$(ps -o rss= -p "${pid:-0}" 2>/dev/null | tr -d " ")
        # loaded: over 10 GB resident and under 100 MB more than 20 s ago
        if [ -n "$pid" ] && [ "$pid" != "$last" ] && [ "${rss:-0}" -gt 10000000 ] && \
           [ $(( ${rss:-0} - prev )) -lt 102400 ]; then
          t0=$(date +%s)
          dd if="$1" of=/dev/null bs=16M iflag=direct status=none
          echo "[warm $(date -u +%H:%M:%S)] n-gram table in RAM in $(( $(date +%s) - t0 ))s (engine $pid)"
          last=$pid
        fi
        prev=${rss:-0}
        sleep 20
      done
    }
    strata_warm_loop "$1"' strata-warm "$table" >> /content/strata-warm.log 2>&1 < /dev/null &
  say "n-gram table: kept in RAM after each engine start (/content/strata-warm.log)"
}
