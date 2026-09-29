# collabosm: a Strata recipe on the VM. Sourced by bootstrap.sh and serve.sh, which
# define say, status, fail, PORT, MODEL_DIR and (serve.sh) LOG and KEY.
#
# Strata (github.com/Niko1221/Strata; the recipe pins a commit of a fork) is its own
# engine and its own OpenAI-compatible server, for Qwen3.8-Flash-Next: the experts it
# cannot keep in VRAM are computed by the CPU from pinned RAM. Its setup.py does the
# install -- the ready-made engine, CUDA's pip libraries, the GGUF from Hugging Face,
# the MTP draft layer, a config. What collabosm adds: the pinned commit, the engine
# zip checked against the recipe's SHA-256 before setup.py sees it, CUDA 12
# (STRATA_CUDA=12: the fork's option for drivers older than 580), and the server
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
  # setup.py prints its own steps; everything it downloads goes to MODEL_DIR (the GGUF,
  # the MTP layer), and --prebuilt as a folder means it copies our zip, not a URL's
  status "stage=weights"
  ( cd "$STRATA_DIR" && STRATA_CUDA="${STRATA_CUDA:-12}" bash setup.sh --family qwen \
      --model "$STRATA_MODEL" --vision "${STRATA_VISION:-none}" --context "${STRATA_CONTEXT:-262144}" \
      --kv "${STRATA_KV:-int8}" --data-dir "$MODEL_DIR" --port "$PORT" \
      --prebuilt "$STRATA_ENGINE_DIR/" --yes --no-start ) || { say "!! Strata setup failed"; return 1; }
  [ -n "$(strata_config)" ] || { say "!! Strata setup wrote no config"; return 1; }
  echo "$STRATA_SHA256" > "$STRATA_DIR/engine/.collabosm-sha256"
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
