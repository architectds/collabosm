# collabosm

**Run a large AI model on your own Google Colab GPU, and use it from your own computer.**

collabosm rents a GPU in *your* Google Colab account, loads an open AI model onto it
(Qwen3.8-Flash-Next by default), and gives you two things on your computer:

- **A chat page with a control panel.** Start and stop the GPU, see what it costs, and chat with
  the model — images included.
- **A local API address**, `http://127.0.0.1:3020/v1`, that works like OpenAI's API. Coding agents
  such as Codex, chat apps such as Open WebUI, or your own scripts can use the model through it.

The model runs on Google's GPU, not on your computer; your computer only runs a small app. You pay
Google for the GPU time, in Colab *compute units*. collabosm itself is free and open source.

```
 your computer                                     your Google Colab account
┌───────────────────────────────┐                ┌──────────────────────────────┐
│ browser ── chat + panel       │                │  A100 GPU                    │
│ Codex, Open WebUI, scripts ─┐ │   encrypted    │  the model (ExLlamaV3)       │
│                             ▼ │    tunnel      │  an OpenAI-compatible API    │
│        collabosm app  ◄───────┼───────────────►│                              │
│    http://127.0.0.1:3020      │                │  rented and stopped by the   │
└───────────────────────────────┘                │  app, billed by Google       │
                                                 └──────────────────────────────┘
```

---

## What you need

- **A Google account with Colab compute units.** That means a paid Colab plan (Pro, Pro+) or
  pay-as-you-go units. The free tier cannot rent the A100 GPUs this uses, and Colab's rules require
  paid units for this kind of use. Buy them at [colab.research.google.com/signup](https://colab.research.google.com/signup).
- **A computer with Windows, macOS or Linux** and **Python 3.12 or newer**
  ([python.org](https://www.python.org/downloads/), or [uv](https://docs.astral.sh/uv/)).
  On Windows you do **not** need WSL. The app installs everything else it needs.
- **About 11 minutes** each time you start a GPU: the model is downloaded and loaded fresh on
  every new GPU.

## Install and open it

```bash
git clone https://github.com/architectds/collabosm.git
cd collabosm
python frontend/server.py
```

(No git? Download the ZIP from GitHub, unpack it, and run the last line inside the folder.
Depending on your system, the command may be `python3` or `py` instead of `python`.)

Then open **http://127.0.0.1:3020** in your browser. Keep the terminal window open: closing it stops
the app — but, careful, *not* the GPU (see [Stopping](#stop-it-when-you-are-done)).

## First run: connect your Colab (once)

The right-hand panel starts with a **Colab** section. It walks you through three steps, one button
each:

1. **Colab CLI → Install.** Installs Google's official Colab command-line tool into the app's own
   folder (`~/.collabosm/colab-cli`). About a minute; nothing else on your computer changes.
2. **Google account → Connect Google account.** Your browser opens Google's sign-in page. Choose
   your account and click **Allow**, then come back to the app. The page names Google's own Colab /
   Cloud tool rather than collabosm — that is expected: you are signing in to Google's Colab tool,
   which asks for Colab, Drive (only files this app uses) and Cloud access. The sign-in is saved
   only on your computer.
3. **Compute units.** Shows your Colab balance. If it is zero, buy compute units first.

When all three are green you can fold the section away; the app remembers. If a step ever stops
working — the sign-in expired, for example — the section says so and shows the button to fix it.

## Start a GPU and chat

1. In **GPUs**, each card has a model menu. Pick one and press **Start**.
2. A card shows what it will cost *before* anything is rented: compute units per hour, minutes until
   ready, the cost of loading, and your balance afterwards. **Nothing is billed until you press
   "Start — billing begins".** "Cancel" costs nothing.
3. Wait while the panel moves through **Request → Upload → Install → Load → Ready**.
4. Chat on the left, like any chat app. You can attach images: the default model can see them.

| GPU | Model | Status | Ready in | Speed | Context |
|---|---|---|---|---|---|
| A100 80GB High-RAM | Qwen3.8-Flash-Next | tested | ~11 min | ~3,900 tokens/s reading, ~97 tokens/s writing | up to ~500K tokens |
| A100 40GB | Qwen3.8-27B | not tested yet — its numbers are estimates | ~6 min | — | up to ~262K tokens |

One conversation is answered at a time. The 80GB speeds were measured with up to 114K tokens of
context; its 500K limit and image input are set up, but not yet measured together on the card.

## Stop it when you are done

**The GPU is billed for every minute it runs**, whether or not you use it. So:

- **Press Stop** (top bar) when you are done.
- **Closing the browser tab or the app does not stop the GPU.** If you quit the app with a GPU
  running, start the app again: it finds the unclosed session and offers Stop.
- Safety nets, in case you forget: the app stops the GPU by itself after **20 minutes without
  chat**, and after **6 hours** in any case. A start that fails is stopped too.
- While you are chatting, the app keeps Colab from reclaiming the GPU as idle.

## What it costs

| GPU | Compute units per hour | About (USD) per hour |
|---|---|---|
| A100 80GB High-RAM | 6.77 (Colab's own rate) | $0.68 |
| A100 40GB | 5.37 (estimate) | $0.54 |

For example, 200 compute units last about 29 hours on the 80GB card. Loading counts too: about
11 minutes of GPU time each start. The panel's **Compute units** section shows your real balance
and burn rate, read from Colab, and keeps a record of every session in `~/.collabosm/ledger.json`.

## Use it from other apps

While a GPU is running, other programs on your computer can use the model through the app:

| setting | value |
|---|---|
| Base URL | `http://127.0.0.1:3020/v1` |
| API key | anything — the app adds the real key itself |
| Model | `qwen3.8-flash-next-exl3` |

It speaks both OpenAI dialects, **Chat Completions** and **Responses**, with streaming and tool
calls. While no GPU is running it answers "No instance is running".

**Codex** — add this to `~/.codex/config.toml`, and set the environment variable
`COLLABOSM_API_KEY` to any value:

```toml
model = "qwen3.8-flash-next-exl3"
model_provider = "collabosm"

[model_providers.collabosm]
name = "collabosm"
base_url = "http://127.0.0.1:3020/v1"
wire_api = "responses"
env_key = "COLLABOSM_API_KEY"
```

(For Codex's file-editing tool, `apply_patch`, Codex also needs a model catalog — see
[docs/DEVELOPER.md](docs/DEVELOPER.md#tool-calls-end-to-end-through-codex).)

**Open WebUI** — Admin Settings → Connections → OpenAI → base URL `http://127.0.0.1:3020/v1`, any
key. Notes in [docs/OPEN-WEBUI.md](docs/OPEN-WEBUI.md).

## The panel at a glance

| part | what it shows |
|---|---|
| top bar | what is running, the language (EN / 中 / 日), **Stop**, and the button that hides the panel (Ctrl/⌘+B) |
| Colab | the set-up steps: the Colab tool, your Google sign-in, your balance |
| status | the running GPU: stage and progress, its address, the idle countdown, and the machine's GPU memory, RAM and disk |
| Compute units | your balance and the burn rate, straight from Colab |
| GPUs | the GPUs and models you can start, with measured speeds (a dashed mark means "estimate") |
| Speed | how fast the last reply was read and written |
| Connection | what Colab says is running on your account right now |
| Log | what the app has been doing |

Hover over an icon to see what it means.

## Try it without paying

```bash
python frontend/server.py --mock
```

Everything in the panel works, and nothing is rented or billed. To also chat with a stand-in
model, start `python scripts/dev_stub.py --port 8099` in a second terminal, and the app with
`python frontend/server.py --mock --backend http://127.0.0.1:8099`.

## Questions and problems

| what you see | what to do |
|---|---|
| The page does not open | Is `python frontend/server.py` still running in its terminal? If port 3020 is taken, add `--port 3021` and open that port. |
| "Looking for the Colab CLI…" for a while | On Windows with WSL, the first check after a while can take ~40 seconds. |
| "Installing needs Python 3.12 or newer" | Install Python 3.12+ or uv, then press **Check again**. (macOS's built-in `python3` is too old.) |
| The Google page did not open | Use **"No tab? Open the sign-in page"** in the panel. |
| "This sign-in no longer works" | Press **Connect again**. |
| "No compute units left" | Buy compute units in Colab. |
| **Start** is greyed out | Finish the Colab section first. |
| Chat says "No instance is running" | No GPU is running: start one and wait for Ready. |
| "drew a card below the recipe's VRAM" | Colab handed out a smaller GPU than the model needs; the app gave it back at once (~0.13 compute units). Press **Start** again. |
| "The VM is gone" | Colab reclaimed the GPU, or it was stopped elsewhere. Billing for it is closed. |
| "the stop could not sign in" | Press **Connect again** in the Colab section, then **Stop** again. Until then the GPU may still be billing. |
| Is anything still billing? | **Connection** lists what Colab is running on your account. |

**Do I need WSL on Windows?** No. If you already use the Colab tool inside WSL, the app finds it
and uses it; otherwise everything runs natively.

**Does the model run on my computer?** No. It runs on the rented GPU; your computer only runs the
small app and your browser.

## Command line (optional)

The panel does all of this; the same steps also run from a terminal. Use the Python the app
installed: `~/.collabosm/colab-cli/Scripts/python` on Windows, `~/.collabosm/colab-cli/bin/python`
elsewhere (shown as `<python>` below).

```bash
<python> scripts/provision.py up          # start the default GPU and model
<python> scripts/provision.py down        # STOP it
<python> scripts/provision.py sessions    # what is running on your Colab account
python scripts/recipe.py list             # the GPU + model combinations
```

## How it works, and your privacy

- The app uses **Google's official Colab tool** to rent the GPU, upload a small server to it, and
  stop it. Your sign-in stays on your computer (`~/.config/colab-cli/token.json`); **Disconnect** in
  the panel revokes it at Google and deletes it.
- The model is downloaded from Hugging Face onto the rented machine at a fixed version, and served
  with [ExLlamaV3](https://github.com/turboderp-org/exllamav3).
- The machine publishes its API through an encrypted Cloudflare tunnel that needs a key. The app
  keeps that key and adds it for you, so it never reaches your browser, and the local address
  (`127.0.0.1:3020`) stays the same even though the tunnel changes with every GPU.
- Everything runs in your own Google account; nothing goes to the authors of this project.
- Colab's terms: use it for yourself. Colab does not allow running a service for other people on
  its machines.

## More

- [docs/DEVELOPER.md](docs/DEVELOPER.md) — how it works inside: measurements, the API, the tests
- [docs/RUNBOOK.md](docs/RUNBOOK.md) — operating notes and the pitfalls behind them
- [docs/MEASURED.md](docs/MEASURED.md) and [docs/CONCURRENCY.md](docs/CONCURRENCY.md) — the numbers
  and how many conversations a GPU can hold

## Licence

The code is MIT (see `LICENSE`). The model weights are not part of this repository and carry their
own licence — see `NOTICE`.
