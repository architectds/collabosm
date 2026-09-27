# collabosm — for agents

How to install collabosm and use it on someone's behalf, for an AI coding agent (Codex, Claude
Code, …). The user guide is [README.md](README.md) and the internals are
[docs/DEVELOPER.md](docs/DEVELOPER.md). This file is the short version, plus the rules that matter
when you act for the user.

## What it is

A local app on `http://127.0.0.1:3020`. It rents a Colab GPU with the user's Google account, serves
a model on it, and exposes that model as an OpenAI-compatible API at `http://127.0.0.1:3020/v1`, for
its own chat page and for any program on this computer.

## Rules

1. **A GPU costs the user money** (Colab compute units, from its first second). Never start one
   without the user's explicit yes for that start. Say which card and model, and the CU per hour:
   the `confirm_required` answer below has them in `warning` (`card`, `model`, `cu_per_hour`,
   `eta_min`).
2. **Stop what you started** when the work is done (`POST /control/stop`), and say so. The app also
   stops a GPU nobody has used for 20 minutes.
3. **Never print, log or commit the GPU's API key.** Programs on this computer do not need it (see
   [Use the model](#use-the-model)).
4. **Signing in to Google is the user's to do**: a consent page in their browser. Ask them; never
   type credentials anywhere.
5. Do not restart the app while a GPU is starting (a `stage` other than `idle`, `ready`, `stopped`
   or `failed`).

## Install

Needs Python 3.12 or newer, and git (or the repository as a ZIP).

```bash
git clone https://github.com/architectds/collabosm.git
cd collabosm
python frontend/server.py --no-browser
```

- Keep it running in the background. It is up when `GET http://127.0.0.1:3020/control/status`
  answers.
- Leave out `--no-browser` when the user should see the page: a normal start opens it in their
  browser.
- The first normal start also puts a desktop shortcut down. `--no-shortcut` skips that, and
  `python frontend/shortcut.py` makes it again. On Windows the shortcut starts the app with no
  window; its output then goes to `~/.collabosm/server.log`.

Then the one-time Colab setup, in the page's **Colab** section or through the API below:
1. `install`: puts Google's Colab CLI into collabosm's own folder.
2. `connect`: opens Google's consent page, which the user approves (rule 4).
3. Wait until `colab.stage` in the status is `connected`, and `balance.cu` shows compute units.

## Drive it: the control API

Plain HTTP on `127.0.0.1:3020`. A POST must be `Content-Type: application/json`, with a body of at
least `{}`, and must not carry another site's `Origin` header; a command-line client sends none.

| call | what it does |
|---|---|
| `GET /control/status` | everything, including `stage`, `note`, `colab.stage`, `balance`, `recipes`, `live_model`, `tunnel` and `heartbeat` |
| `POST /control/colab/install`, `…/connect`, `…/cancel`, `…/check`, `…/disconnect` | the Colab setup steps |
| `POST /control/select {"recipe": "<id>"}` | answers `confirm_required` with the cost; nothing starts yet |
| `POST /control/select {"recipe": "<id>", "confirm": true}` | **starts the GPU, and its billing** (rule 1) |
| `POST /control/cancel` | withdraws a pending confirmation; nothing was started |
| `POST /control/stop` | stops the GPU, and its billing |
| `POST /control/couple` | reconnects to the running GPU, for example after its tunnel changed |
| `POST /control/quit` | closes the app itself. A running GPU is **not** stopped: stop it first (rule 2) |

- A recipe is a card plus a model. They are listed under `recipes` in the status, or by
  `python scripts/recipe.py list`.
- A start moves `stage` through `requesting` → `uploading` → `bootstrapping` → `loading` → `ready`,
  which takes about 6 to 11 minutes.
- When something goes wrong, `note` and `log_tail` in the status say what.

## Use the model

While `stage` is `ready`:

| setting | value |
|---|---|
| base URL | `http://127.0.0.1:3020/v1` |
| API key | any value, for example `sk-local` or `local_key` from the status (what the page copies); the app replaces it with the real key |
| model | the running one: `live_model` in the status, or `GET /v1/models` |

- **Endpoints:** `/v1/chat/completions` and `/v1/responses` both work, streaming or not, with
  tools, including Codex's `apply_patch`.
- **Configuration:** Codex and ModelDock setup are in [README.md](README.md#5-use-the-model-in-other-programs).
- **Use this address, not the tunnel.** Point programs here, not at the GPU's `….trycloudflare.com`
  address: that one changes with every GPU and needs the key.

## When something breaks

- `GET /control/status`:
  - `note`: what the app is doing or what went wrong;
  - `tunnel.cause`, while the tunnel is down: which part stopped;
  - `heartbeat`: the GPU's own view of its health.
- `~/.collabosm/heartbeat.jsonl`: a line of the GPU's health every 10 minutes while it is in use.
  It survives the GPU.
- `~/.collabosm/ledger.json`: every GPU session, what it cost, and why it ended.
- Colab takes back a GPU with no kernel activity after about 20 minutes. The app's heartbeat keeps
  one that is in use; traffic through the tunnel alone does not.
