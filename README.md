# collabosm

Run **Qwen3.8-Flash-Next** (125B-A6B MoE, hybrid Gated-DeltaNet + full attention, 262144 native
context) on **one Colab A100-80GB High-RAM**, and get an OpenAI-compatible endpoint you can point
a client at.

Measured on the box this repo was built against:

| what | value |
|---|---|
| prefill (30K prompt) | **2,806 t/s** |
| prefill at `-gcs 4096` (30K prompt) | **3,360 t/s** |
| prefill at `-gcs 8192` (30K prompt) | **3,882 t/s** |
| decode, MTP `ndt=4`, 30K context | **97.4 t/s** |
| decode, MTP `ndt=4`, 114K context | **90.1 t/s** |
| decode, no MTP | 56 t/s |
| KV cache | 10,752 B/token (2.63 GiB at 262K, 5.01 GiB at 500K) |
| resume a fully-evicted 32K conversation | **0.58 s** instead of a 9.72 s re-prefill (16.8x) |
| cost | 6.77 CU/h (A100 High-RAM, Colab's own rate for the box) ≈ $0.68/h, ≈29.5 h per 200 CU |

ExLlamaV3 is the only engine in this repo. See `docs/MEASURED.md` for provenance and caveats, and
`docs/CONCURRENCY.md` for how many sessions the card can actually hold.

## Requirements

- A Colab plan that can allocate an **A100** (this kit was developed on a Pro tier with ~200 CU/month).
- **HIGH_RAM** specifically. A 40 GB standard A100 **cannot load this model at all** — about 63.6 GiB
  of weights must be VRAM-resident. `scripts/restore.py` requests it explicitly and refuses a 40 GB box.
- Python 3.12+ (or `uv`) and the Colab CLI, `google-colab-cli` -- the frontend's Colab guide installs
  it; by hand, `uv tool install google-colab-cli`. Windows (with or without WSL), macOS and Linux all
  work: everything that runs on your machine is Python (see [Where it runs](#where-it-runs)).
  (`gh` too, if you want to fork/push.)
- ~110 GiB of Colab disk for the weights.

## Quickstart

```bash
python scripts/provision.py up                                # the default recipe: Flash-Next on an A100-80G
python scripts/provision.py up --recipe a100-40g/qwen38-27b   # another card + model from recipes.json
python scripts/recipe.py list                                 # what the registry holds
python scripts/provision.py down   # STOP THE VM. On a metered plan this is the most important command.
```

Run them with the Colab CLI's interpreter (`~/.collabosm/colab-cli` when the frontend installed it,
else `~/.local/share/uv/tools/google-colab-cli`); `bash scripts/up.sh` and `bash scripts/down.sh`
still work and do exactly this. `up` prints the tunnel URL when the endpoint answers `/health`. Point your client at
`<url>/v1` with the API key it prints (also at `/content/api-key.txt` on the VM).

Everything a launch needs comes from the recipe; a variable you set yourself wins for that run:

```bash
SESSION=mybox CACHE_SIZE=524288 CPU_CACHE_GB=8 python scripts/provision.py up
```

| variable | Flash-Next / A100-80G | meaning |
|---|---|---|
| `SESSION` | `collabosm` | local session name (not part of a recipe) |
| `CACHE_SIZE` | `500224` | total KV tokens across all jobs (multiple of 256) |
| `CACHE_QUANT` | `4` | KV bits (`4` = Q4, `2`-`8` allowed) |
| `CPU_CACHE_GB` | `32` | **pinned-RAM second-tier KV page cache** (0 = off). Sized at 92K tokens/GB; do not treat it as free - it is allocated as pinned memory in full (32 + 24 = ~56 GB next to a 36.4 GiB n-gram table) |
| `RECURRENT_CACHE_GB` | `24` | host-RAM store for Gated-DeltaNet checkpoints (~2048-token interval, 116 MB each) |
| `GCS` | `8192` | generator chunk size — the biggest prefill lever we found |
| `NDT` | `4` | MTP draft depth |
| `VISION` | `1` | load the pack's vision tower (image input) |
| `EXL3_VISION_PINNED` | `1` | keep the tower's weights in pinned host RAM instead of VRAM (an ExLlamaV3 switch) |
| `YARN_FACTOR` | `2` | YaRN over the native 262,144 positions (0 = native only); see below |
| `CONCURRENCY` | `1` | reported; requests are still served one at a time |
| `RUNTIME` | `wheel` | `wheel` (prebuilt, no compile) or `source` |
| `TUNNEL_TOKEN` | unset | named-tunnel token: gives a **stable hostname** instead of a quick tunnel. Pair with `PUBLIC_URL` for the URL shown in status |

These settings have not yet been loaded together: the verified load in `docs/MEASURED.md` ran
`-gcs 4096 -ccs 16 -rcs 16`, without vision or YaRN. `GET /v1/status` reports what a running server
actually got (`recipe`, `launch`, `context`, `vision`, `concurrency`).

`up` exits `0` on READY — a healthy API **and** a published tunnel URL — and otherwise with
`1` (upload/bootstrap), `2` (a recipe or argument that cannot run), `3`-`6` (from `restore.py`),
`7` (timed out), `8` (`serve.sh` failed) or `9` (healthy on the VM, but no tunnel URL).

## Recipes

`recipes.json` is the one list of what can run where: a recipe is a card (`gpus`), a model
(`models`) and the launch settings for the pair. `scripts/recipe.py` resolves an id into the
environment `up` runs with: the card's shape and the VRAM a box must show before anything is
downloaded (`restore.py`), the model's repo, pinned revision and directory (`bootstrap.sh`), and how
it loads (`serve.sh` -> `api_server.py`). The frontend lists the same entries and sends only the id,
so the rail and a hand-run `up` cannot disagree. A new pairing is a new entry, not a new script.

Every number carries its evidence, `{v, measured, src}`, and a recipe's `status` says how far it has
been taken: `verified` (run on that card, numbers measured) or `unmeasured` (the launch path exists,
the numbers are estimates). Only recipes expected to run are listed; a pair that cannot fit is not.

| recipe | card | model | status | notes |
|---|---|---|---|---|
| `a100-80g/qwen38-fn` | A100-80G High-RAM, 6.77 CU/h (measured) | Qwen3.8-Flash-Next 4.05 bpw | verified | 500K cache, YaRN x2, vision, n-gram table in host RAM |
| `a100-40g/qwen38-27b` | A100-40G, 5.37 CU/h (Colab's figure) | Qwen3.8-27B 3.50 bpw, 15.4 GB | unmeasured | 262K native, vision, no n-gram table; ~22 GiB of 39 estimated |

The 40 GB card is requested by sending no shape at all (Colab's default is the standard 40 GB shape
in every draw we logged); `shape=hm` is sent only for High-RAM.

**YaRN, and the ExLlamaV3 trap.** Both packs are 262,144 positions natively. The model cards extend
them by rewriting `text_config.rope_parameters` to `rope_type: "yarn"` with a `factor` over
`original_max_position_embeddings: 262144`. ExLlamaV3, however, ignores `factor` whenever
`original_max_position_embeddings` is present and derives it as `max_position_embeddings / original`
-- so the card's block alone is factor 1.0, a silent no-op. `api_server.py` (`apply_yarn`) therefore
also raises `max_position_embeddings` to `262144 x factor`, keeps the pack's own file as
`config.json.orig`, and puts it back when a recipe asks for no YaRN (static YaRN costs a little on
short prompts, the card warns). Quality past 262K is not measured yet.

## What is in here

| path | role |
|---|---|
| `recipes.json` | **the recipe registry**: cards, models, and the launch settings for each pair, every number marked measured or not |
| `scripts/recipe.py` | resolves a recipe id into the environment `up` launches with (`list`, `show`, `env`, `vmenv`, `check`) |
| `scripts/restore.py` | **the session restore script.** Re-attaches an orphaned VM from server truth, or creates one and actually requests the recipe's shape. Refuses/stops a box below the recipe's VRAM before spending anything. |
| `scripts/colab_keepalive.py` | tells Colab the box is in use (the frontend calls it only while it is), and re-registers the CLI's session record when the CLI drops it |
| `scripts/colab_ccu.py` | the account's real CU balance and burn rate, read from Colab |
| `scripts/colab_auth.py` | the CLI's own sign-in, for the frontend: `status`, `login` (loopback redirect), `logout` (revoke) |
| `scripts/probe_gpu.py` | runs on the VM; reports VRAM/RAM/cc/disk as one JSON line |
| `scripts/provision.py` | **up** (restore → upload → bootstrap → serve → wait for health), **down** (stop the VM, report what still bills), **fetch** (one small file from the VM, never `colab exec`), **sessions** -- the same on Windows, macOS, Linux and WSL |
| `scripts/colab_cmd.py` | the Colab CLI (`colab ...`) on any machine: on Windows it stands in for the two Unix-only modules its console imports |
| `scripts/up.sh`, `scripts/down.sh` | one-line shims onto `provision.py up` / `down`, so the old commands keep working |
| `scripts/bootstrap.sh` | runtime (pinned wheel) + weights (from HF at a pinned revision), idempotent |
| `scripts/serve.sh` | launch the API + cloudflared tunnel (runs on the VM) |
| `scripts/api_server.py` | minimal OpenAI-compatible server; **one long-lived Generator** |
| `scripts/status.py` | one-glance stage/health report |
| `scripts/dev_stub.py` | serves the real Handler on loopback with a stubbed engine: exercises the wire format with no GPU and no weights |
| `scripts/check_surface.py` | drives a running server over HTTP and judges the surface (deltas, terminal event, ids, sequence numbers) |
| `scripts/check_codex.py` | runs the real Codex CLI against `dev_stub.py --script` (manufactured model turns) and judges what Codex did with each tool-call syntax |
| `manifest.json` | every pinned revision, size and measured number |
| `docs/MEASURED.md` | the measurements, with provenance and caveats |
| `docs/RUNBOOK.md` | the traps, the protocols, the cost guardrails |

## Why the runtime and the model come from different places

The **runtime** is a pinned prebuilt wheel from the ExLlamaV3 GitHub release — no compile step, ~22 s,
bound to `(python, torch, cuda)`, so `bootstrap.sh` probes the image and picks the matching asset.
The **weights** come from Hugging Face at a pinned revision (~100 GiB, measured 4 min 40 s at
~380 MB/s anonymously; `HF_HUB_DISABLE_XET=1` is the documented fallback when the Xet path stalls).

Nothing here needs Google Drive, a Colab secret, or any personal path — a repo cannot ship your Drive,
and anything that depends on one is not shareable.

## Four traps this repo exists to spare you

1. **`colab new --gpu A100` is a lottery.** It never sends `shape`, so you get either 80 GB High-RAM or
   40 GB standard. Eleven consecutive unpatched attempts gave 40 GB. `restore.py` patches the URL
   builder to send `shape=hm` (`google-colab-cli#47`: the enum exists and `machineShape` is parsed, but
   it is never sent).
2. **Your local session record is disposable.** When the runtime proxy token lapses the CLI reports
   "No active sessions found" and deletes its bookkeeping — while the VM is still running and still
   billing. We hit this twice in one day, once mid-run with `/content` fully intact. `restore.py`
   re-attaches from `list_assignments()` instead of creating a second VM.
3. **Never detect an 80 GB card by looking for "80".** An A100 reports compute capability `sm_80`, so a
   40 GB box prints "80" everywhere. Compare `vram_GiB`.
4. **A new `Generator` is a cold cache.** `PageTable` and the CPU page cache are built inside
   `Generator.__init__`. A per-request Generator silently disables prompt caching *and* the pinned-RAM
   KV tier — the failure is invisible, you just quietly re-prefill everything. `api_server.py` holds one.

## Verified end to end

On 2026-09-25, on one account, with no manual steps beyond the scripts in here:

- `scripts/restore.py` **re-attached an orphaned assignment** after the Colab CLI dropped its local
  record for the third time that day (VM alive, still billing) and confirmed the box:
  79.3 GiB VRAM, 167.1 GiB RAM, cc 8.0, python 3.13.15.
- `scripts/serve.sh` loaded the 4.05 bpw pack at **`cache_size 500224` (500K per stream)** in
  259.5 s and reached `/health` 200 at **76,437 / 81,920 MiB** of VRAM.
- Over a public tunnel: `/v1/models` returned **401 without the key, 200 with it**, and a real
  chat completion came back in **3.1 s**.
- Later the same day the whole chain was re-verified with a **real client**: Codex CLI 0.144.6,
  `wire_api = "responses"`, through the quick tunnel to the live 4.05 bpw pack -
  `Reply with exactly: pong` came back `pong`, exit 0, 8,536 prompt tokens.

Known gaps, stated plainly:

- **Streaming is incremental now.** `collect()` drives `Generator.enqueue()` / `iterate()` and
  forwards every decode step, so both `/v1/chat/completions` and `/v1/responses` emit text while
  the model is still working. Verified locally through a real Codex CLI client: first delta at
  0.01 s, 11 deltas, terminal `response.completed` present, `sequence_number` monotonic. The old
  one-shot path survives as `_generate_blocking()` and is used only if a build's Job API differs.
- **Requests are serialised** by a lock — one Generator, one cache. "Multiple streams" today means
  queued, not parallel.
- **`max_batch_size > 1` is untested.** Every measurement ran `num_slots = 1`. ExLlamaV3 itself batches
  (continuous batching in `Generator`), but it allocates recurrent-state slots at load from `-ambs`
  (default 1), and `enqueue`/`iterate`/`cancel` are not thread-safe: real concurrency needs `-ambs N`
  plus one engine thread that owns the Generator, and then a measurement on the card.
- **Vision and YaRN are configured, not measured.** The 80G recipe loads the vision tower (weights in
  pinned host RAM) and YaRN x2; neither has run on the card together with the 500K cache yet.
- Session *assignment* is scripted; it is not yet a configurable hosting layer.

## Testing it without a GPU

The thing that keeps breaking is the *protocol*, and proving a protocol change used to cost a
session restore plus a four-minute weight load. Two scripts remove that cost:

```bash
python scripts/dev_stub.py --port 8099 --chunk 24 --delay 0.01   # real Handler, fake engine
python scripts/check_surface.py --base http://127.0.0.1:8099/v1  # 13 checks, exits 1 on failure
```

`dev_stub.py` imports the shipping `Handler` and replaces only `_engine_fragments()`, so the code
under test is the code that deploys. `check_surface.py` asserts what clients actually depend on:
deltas arrive before the end, a terminal event is sent, every delta carries `item_id` /
`output_index` / `content_index`, `sequence_number` is monotonic, and the chat and Responses views
of the same completion agree word for word.

Point a real client at it too -- this is the fastest way to learn that a stream is shaped wrong:

```toml
# CODEX_HOME/config.toml
model = "qwen3.8-flash-next-exl3"
model_provider = "collabosm"
preferred_auth_method = "apikey"

[model_providers.collabosm]
name = "collabosm"
base_url = "http://127.0.0.1:8099/v1"
wire_api = "responses"
env_key = "COLLABOSM_TEST_KEY"
```

Add `--think` to `dev_stub.py` to exercise the reasoning item lifecycle. Both shapes were accepted
by Codex CLI 0.144.6 (`codex exec --skip-git-repo-check 'Reply with exactly: pong'`).

### Tool calls, end to end through Codex

`dev_stub.py --script turns.json` plays a fixed list of model turns, one per generation: the exact
text a model writes, tool-call XML included, and why it stopped (`stop_token` or `max_new_tokens`).
`check_codex.py` uses that to put every call syntax the parser accepts through the real Codex CLI,
and then checks what Codex did: the command it ran, the file a patch wrote, the history it sent
back, and the prompt the model would be shown next.

```bash
python scripts/check_codex.py            # 15 scenarios, ~30 s, exits 1 on failure
python scripts/check_codex.py --list     # what each one covers
python scripts/check_codex.py patch cut --keep   # some of them; keep their files
```

| scenario | the model writes | what is checked |
|---|---|---|
| `xml` | one Qwen3-Coder XML call, sent in a single fragment | the command runs; the call and its output are replayed into the next prompt |
| `typed` | `timeout_ms`, `login`, and a command `1234` | a number stays a number, a boolean stays a boolean, a string-only `1234` stays a string |
| `verbatim` | a multi-line value with tags, quotes, `` ` ``, `$`, backslashes, unicode | the arguments and the file the command wrote match byte for byte |
| `json-body` | `{"name": ..., "arguments": {...}}` inside `<tool_call>` | it runs like the XML form |
| `no-close` | a call without `</tool_call>`, then one without `</function>` | both run |
| `patch` | freeform `apply_patch`: add a file, then update it | Codex applies both; the file is exact; the call goes out as `custom_tool_call` |
| `big-patch` | an 80-line patch, sent one character per fragment | the file is exact |
| `plan` | `update_plan` with an array of objects | Codex shows it as its to-do list |
| `namespace` | `close_agent` from Codex's `multi_agent_v1` group | the call goes back with its `namespace` and reaches the multi-agent handler |
| `serial` / `parallel` | two calls in one turn | one runs when `parallel_tool_calls` is false, both run when it is true |
| `cut` | a call the token ceiling cuts off mid-value | nothing runs; Codex takes `response.incomplete` as a failed attempt and asks again |
| `think` | a think block, then a call | the reasoning item Codex replays is the thought exactly, and the model sees it again |
| `image` | `view_image` on a PNG (fake vision tower) | the picture reaches the model inside the tool result |
| `text-edges` | a plain answer full of `<`, with tools offered | the text arrives whole, with no call |

Codex offers our model the freeform `apply_patch` tool only when a model catalog says so, so each
scenario writes one (`model_catalog_json`) and a scratch `CODEX_HOME`. Approvals are off and there
is no sandbox, so the only commands Codex runs are the scripted ones (`echo`, and writing files in a
scratch directory). The pack's `chat_template.jinja` is fetched into `_dev_model/` from the pinned
revision the first time.

One ambiguity is inherent to the XML format, and llama-server has it too: a value ends at the first
`</parameter>` that is followed by `<parameter=`, `</function>` or `</tool_call>`. A value that
contains that exact sequence is cut short there. `</parameter>` followed by anything else stays part
of the value, and `verbatim`, `patch` and `big-patch` check that.

## How many sessions fit

Measured fit on this card (details and assumptions in `docs/CONCURRENCY.md`):

| context per stream | max concurrent live sessions | bound by |
|---|---:|---|
| 500K | **1** | KV |
| 262K | 2 | KV |
| 131K | 5 | both |
| 32K | 11 | recurrent state |
| 16K | 14 | recurrent state |

Each live slot costs **~546 MiB** of Gated-DeltaNet state before any KV, so concurrency saturates
at ~20 slots as contexts get short (16K: 14 slots, 8K: 17) and 8-14 is the practical band. The
pinned-RAM tier does **not** raise those numbers - eviction only touches unreferenced pages, and it
is not a swap device for a live stream - but it does let ~1.47M tokens of *idle* conversation be
resumed in well under a second instead of re-prefilled. `docs/CONCURRENCY.md` works through why the
RAM tier cannot extend live capacity.

## Licence

Our scripts: MIT (see `LICENSE`). The weights are **not** shipped here and carry their own licence —
see `turboderp/Qwen3.8-Flash-Next-exl3` and `NOTICE`. ExLlamaV3 is MIT.

## API surface

The server is ours (`scripts/api_server.py`) - ExLlamaV3 ships no HTTP server. It answers
both OpenAI dialects, and **every** error path is JSON (an HTML error page is a bug, not a
client problem):

| endpoint | notes |
|---|---|
| `POST /v1/chat/completions` | `choices[].message.content`, `finish_reason` = `stop`/`length`, `usage.prompt_tokens_details.cached_tokens`; `stream: true` returns SSE chunks + `[DONE]` (+ usage with `stream_options.include_usage`) |
| `GET /v1/status` | read-only contract: launch parameters, cache size, uptime, **vision availability** and the image-input policy |
| `POST /v1/responses` | Responses surface: `status`, `output[].content[].text`, `output_text`, `usage.{input,output,total}_tokens`; `stream: true` emits the full lifecycle below |
| `GET /v1/models` | the one model id, with `created` |
| `GET /health` | plain `ok`, unauthenticated |
| anything else | JSON 404 / 405 (never HTML) |

Accepted: `max_tokens` and `max_completion_tokens`, `temperature`, `top_p`, `stop`
(string or list), `stream`, `stream_options.include_usage`.

A streamed `/v1/responses` sends `sequence_number` on every event, in this order:
`response.created` -> `response.in_progress` -> *(if thinking is on)*
`output_item.added`(reasoning) -> `reasoning_text.delta` -> `reasoning_text.done` ->
`output_item.done` -> `output_item.added`(message) -> `content_part.added` ->
`output_text.delta` xN -> `output_text.done` -> `content_part.done` -> `output_item.done` ->
`response.completed`. Deltas always carry `item_id`, `output_index` and `content_index`, and the
ids in the terminal event are the ids the stream announced -- Codex binds deltas to an announced
item and rejects the stream otherwise. Text is forwarded **while** the engine decodes: the server
enqueues a `Job` and drains `Generator.iterate()`, rather than chunking up a finished completion.
If a build's incremental path yields nothing, the server falls back to the blocking generator and
logs `incremental path produced no text (new=... eos=...)` -- it never answers empty.

**Thinking** is off by default, because that is what plain chat clients expect. Turn it on
with `enable_thinking: true` or `reasoning_effort: xhigh|medium|low` (the pack's own template
takes both); the trace is returned in `message.reasoning_content` and never leaks into
`content`.

## The frontend

**This repo ships its own frontend** -- the one local client -- and still does not *write* a chat
app. `frontend/` is our shell (`shell.html`, `server.py`, `control.py`: the only files we own) wrapped around a
llama.cpp Web UI vendored byte for byte -- our diff against upstream is zero. Everything lives in
**one right-hand column** -- the Colab guide, session, card/recipe picker, money, speed, connection, log -- and the
WebUI sits in an iframe beside it; ⌘B / Ctrl+B collapses the column. The shell is colour-blocked
against the WebUI -- warm charcoal by day beside its white page, bone paper by night beside its
near-black one, switching when the WebUI's own theme does -- and speaks **English, 中文 and 日本語**
(switch in the header; `control.py` sends codes and numbers, and `shell.html` owns every word). Run it with
`python frontend/server.py` (port 3020); see [frontend/UPSTREAM.md](frontend/UPSTREAM.md) for the
pinned upstream commit and how to rebuild it.

`frontend/control.py` is the real control plane behind that column. It is the only thing in the
kit that can spend money, so it is also where the guardrails are:

| what | how |
|---|---|
| Colab setup | the guide at the top of the column (`frontend/colab_setup.py`): no CLI, no sign-in, no compute units -- each with its button; Start is refused until it is done |
| cards | `select()` runs `scripts/provision.py up` with the Colab CLI's interpreter -- natively, or inside WSL when the CLI is there |
| money | a CU ledger in `~/.collabosm/ledger.json`; the column shows used / left against `--budget-cu` |
| confirmation | a bare click returns `confirm_required` with CU/h, ETA and what the load itself costs; only `confirm: true` starts the job (and the billing); `/control/cancel` drops it |
| stopping | Stop runs `provision.py down`; **idle auto-stop** after `--idle-stop-min` (20) with no chat traffic, plus a `--max-session-h` (6) ceiling; a job that **fails** after `assign` is stopped too, and a session a previous frontend never closed can be stopped from the column |
| chat | once ready, `/v1/*` is proxied to the live tunnel with the VM key injected; when nothing is live the frontend answers **503** instead of pretending |
| provisioning | always through `scripts/restore.py`, so an orphaned VM is adopted, never duplicated |

Rehearse the whole flow with no card and no CU:

```bash
python frontend/server.py --fake-provision   # real control plane, fake provisioning (~24 s), rehearsal ledger
python frontend/server.py --mock             # the same, with a ledger that is never written
python scripts/dev_stub.py --port 8099 &     # optional: something to chat with while rehearsing
python frontend/server.py --mock --backend http://127.0.0.1:8099
COLLABOSM_FAKE_FAIL=serve python frontend/server.py --mock   # rehearse a failure + its auto-stop
```

Both modes print the same log lines `up` does (`[restore] box: ...`, `stage=weights`,
the tunnel URL, `READY`), so the rail, the stages and the stop path are all exercised for free.
Only the call to `provision.py` itself is replaced.

### First run: the Colab guide

A new user has no Colab CLI and no sign-in, so the column opens with a small guide -- expanded on
the first launch, and folded or not after that as the user leaves it:

| step | when it is not done yet |
|---|---|
| Colab CLI | **Install** puts `google-colab-cli==0.6.0` in a venv of its own (`~/.collabosm/colab-cli`); it needs Python 3.12+ or `uv`, and says so when neither is here |
| Google account | **Connect Google account** opens Google's consent page in the browser; one Allow and the token is written where the CLI keeps it (`~/.config/colab-cli/token.json`) -- a CLI that is already signed in is simply found connected. A revoked or expired sign-in, or one without the Colab scope, asks to connect again |
| compute units | the balance, read from Colab; none left says an A100 needs a paid plan |

The sign-in is the CLI's own: `scripts/colab_auth.py` runs with the CLI's interpreter and uses its
OAuth client, scopes and token file. The CLI asks for a code to be pasted back, because it must work
without a browser; the frontend runs where the browser is, so it takes the loopback redirect the
client is registered for (`http://localhost:<port>/`, PKCE), and nothing but Allow is asked of the
user. Google's consent page shows Google's own SDK app -- that is the CLI's client -- asking for
Colab, Drive (only files this app uses) and Cloud Platform.

### Where it runs

Everything that runs on your machine is Python, started with the Colab CLI's own interpreter:
the sign-in (`colab_auth.py`), the balance, the keep-alive, and `provision.py` -- up, down, the VM's
files, the session list. `provision.py` drives the Colab CLI itself, through `colab_cmd.py`. No step
needs bash, WSL or GNU coreutils (macOS has no `timeout`, which the old `up.sh` leaned on); the
scripts that run *on the VM* (`bootstrap.sh`, `serve.sh`) are the only shell left, and the VM is Linux.

| machine | what the guide finds or installs | notes |
|---|---|---|
| Windows, no WSL | a native CLI in `~/.collabosm/colab-cli` | the `colab` command dies on its Unix-only console (`termios`); `colab_cmd.py` stands in for the two modules it imports, and every command this kit uses works |
| Windows with WSL | the CLI inside WSL, if it is there | looked for without booting WSL (`wsl --list`); a missing distro, or no WSL at all, falls through to native |
| macOS, Linux | a native CLI | Python 3.12+ from python.org, Homebrew or `uv`; the system `python3` of macOS is too old, and the guide says so |

Where the CLI is looked for: inside WSL first when that distro exists (the setup this kit grew up
on), then natively -- `$COLLABOSM_COLAB_PY`, `~/.collabosm/colab-cli`, a `uv tool install`, or the
frontend's own Python. The files it uploads go to the VM with LF endings even when a Windows editor
left CRLF in them.

```bash
COLLABOSM_FAKE_COLAB=missing python frontend/server.py --mock   # rehearse the guide from nothing
COLLABOSM_FAKE_COLAB=expired python frontend/server.py --mock   # ... or from a dead sign-in
COLLABOSM_COLAB_WSL=0 python frontend/server.py                 # walk a fresh machine's real path
```

### Pointing other clients at it

Codex, Open WebUI, a script: give them the frontend, not the VM. It proxies `/v1/*` to whichever box
is live, with the VM's key injected, and answers 503 while none is.

```
base URL : http://127.0.0.1:3020/v1
API key  : anything (the frontend injects the real one)
model    : qwen3.8-flash-next-exl3
```

Both dialects stream through it line by line -- `/v1/responses` and `/v1/chat/completions`, plus
`/v1/models` -- and `check_surface.py` passes 13/13 through it; Codex runs through it unchanged.

**Why a local proxy instead of pointing a client straight at the endpoint**

- the bearer key stays in this process and is never handed to a browser,
- the tunnel's hostname changes with every VM, and this address does not,
- the page is same-origin with the proxy, so there is no CORS surface at all.

There is deliberately **no** `Access-Control-Allow-Origin: *` here: a wildcard on a proxy that
injects a bearer key would let any web page you happen to visit read what your GPU says. That alone
does not stop a page from *sending* a `text/plain` or form POST, though -- those need no preflight --
so every POST must also be `application/json`, come from this origin (or from no browser at all), and
name this proxy in `Host`, which shuts out DNS rebinding too; `/control/*`, where a POST can start
billing, is held to the same rule. Server-side clients are unaffected (they call from their own
process); if a browser-hosted UI ever needs in, add an explicit origin allowlist, not a wildcard.

### The metrics contract

The OpenAI protocol has no field for prefill or decode throughput, so no off-the-shelf UI can show
it. `api_server.py` sends llama.cpp's own `timings` on a stream's last chunk, and the frontend reads
prefill and decode from there as the stream passes; without them it keeps only the time to the first
token and derives no rate from characters. A wrong number presented as a measurement is worse than
no number.

Any other OpenAI-compatible client works too: `uv tool install open-webui`, then
Settings -> Connections -> OpenAI API with the values above. That path is now **deprecated** -- it
costs ~3.9 GB on disk and needs Python 3.11 + torch -- see [docs/OPEN-WEBUI.md](docs/OPEN-WEBUI.md)
for the measured numbers if you still want it.

Two gaps to know about before pointing anything at it:

- **`temperature` and `top_p` are accepted and ignored** by `api_server.py`: it builds the job with
  `sampler=None` and never forwards them. A UI slider for them is a fake knob until the sampler is
  wired. Thinking (`enable_thinking`, `reasoning_effort`), `max_tokens` and `stop` do work.
- **Images: implemented, and off by default.** See [Images](#images) below. The old behaviour --
  an image part rendered as text into the prompt, so the request "worked" and the picture was
  ignored -- is gone: an image is now embedded, or the request is refused with a reason.

## Images

The pack is multimodal (`vision_config`, `image_token_id`, `preprocessor_config.json`). The reason
this server could not see images is specific and worth knowing: **`model_init.init()` loads only the
`text` component (or `mtp`)**, so the vision tower has to be loaded as its own component. That is
what `VISION=1` does:

```bash
VISION=1 ...            # load ExModel(component="vision") alongside the text model
IMAGE_URLS=1 ...        # allow remote http(s) image URLs (see below)
MAX_IMAGE_BYTES=12582912
MAX_IMAGES=8
```

Then an ordinary OpenAI image part works in both dialects:

```json
{"role": "user", "content": [
  {"type": "text", "text": "What colour is this image?"},
  {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
]}
```

Mechanically: the image is embedded once, and the embedding's `text_alias` is inserted into the
prompt where the image belongs. `tokenizer.encode(..., encode_special_tokens=True, embeddings=[...])`
expands that alias into the placeholder span, and the same embeddings ride along on the `Job`. Text
prompts take the plain path exactly as before -- special-token encoding is only switched on when
embeddings are present.

**Deliberately not a URL fetcher by default.** This server is published through a tunnel, so
"fetch whatever URL the caller sent" would be an SSRF primitive aimed at the VM's own metadata
service, and "open whatever path the caller sent" would be a local file read. So:

- `data:` URLs only, unless `IMAGE_URLS=1`.
- with `IMAGE_URLS=1`, the host is resolved and every address must be **globally routable** --
  loopback, private, link-local and metadata addresses are refused, and redirects are not followed
  (one hop only, so a redirect cannot bounce to an internal host after the check).
- filesystem paths are never accepted, with or without vision, and the reference policy is checked
  *before* the capability check so the error you get is about your request, not about this box.
- an image that cannot be embedded is a **400** (`type: vision_unavailable`), never a silent
  text-only answer.

**Vision is on in both recipes, and still unmeasured.** The 80G box already sits at 76.4 / 81.9 GiB
at a 500K cache, so its recipe also sets `EXL3_VISION_PINNED=1`: ExLlamaV3 then keeps the tower's
linear weights in pinned host RAM instead of VRAM (the card has ~70 GB of host RAM to spare), which
costs a PCIe copy per image and no resident VRAM. The 27B pack carries its tower (bf16) inside its
shards and fits it in VRAM easily. Neither has been loaded on its card yet -- watch the number on the
first real load. `GET /v1/status` says which state a running server is in, so a client (or ModelDock's
`supportsVision`, or the frontend's WebUI through `/props`) can stop guessing.

Verify either way -- the tool accepts both outcomes, because a server without vision must refuse:

```bash
python scripts/check_vision.py --base http://127.0.0.1:8090/v1 --key <key>
python scripts/check_vision.py --base <tunnel>/v1 --key <key>
```

## How it is served

There is one shipping path, and its last hop is a **Cloudflare tunnel** -- the VM has no inbound
address of its own, so the tunnel is what makes the API reachable from this machine:

```
bootstrap.sh  ->  runtime + weights on the VM
serve.sh      ->  api_server.py on 127.0.0.1:8090          (model load: ~264 s, 76.4/81.9 GiB)
              ->  /content/cloudflared tunnel --url http://127.0.0.1:8090
              ->  https://<host>.trycloudflare.com/v1       <- every client points here
```

`serve.sh` writes the URL into `/content/STATUS` and prints it. The bearer key is generated once
into `/content/api-key.txt` and is **stable across restarts**. Clients therefore need:

```
base_url  = https://<host>.trycloudflare.com/v1
api_key   = contents of /content/api-key.txt
model     = qwen3.8-flash-next-exl3
```

For **ModelDock** that is a custom endpoint in `~/.modeldock/custom-endpoints.json`:

```json
[
  {
    "modelId": "qwen3.8-flash-next-exl3",
    "baseUrl": "https://<host>.trycloudflare.com/v1",
    "apiKey": "<key from /content/api-key.txt>",
    "label": "A100 80G exl3 (tunnel)",
    "supportsVision": true,
    "transport": "responses",
    "contextWindow": 0
  }
]
```

`transport: "responses"` matters: ModelDock then speaks the Responses dialect to this server, which
is the path that was broken and is now fixed. For Codex CLI against the same endpoint, use the
provider block in [Testing it without a GPU](#testing-it-without-a-gpu) or point
`base_url` at the tunnel - the rest is identical.

### The one sharp edge: the hostname

A **quick tunnel changes hostname on every restart**, so after each `serve.sh` the `baseUrl` in
ModelDock is stale and every request 404s until it is updated (and ModelDock is relaunched). Two
ways out:

- **Named tunnel - recommended, and already wired.** Put `TUNNEL_TOKEN` in `/content/collabosm.env`
  (plus `PUBLIC_URL` if you want `status.py` to print the address) and the hostname is fixed for
  the life of the tunnel. ModelDock then never needs editing again.
- Re-read the URL from `/content/STATUS` after each start and update ModelDock by hand.

The API itself does not care: on loopback it is the same server, which is why all protocol work is
done locally with `python scripts/check_surface.py --base http://127.0.0.1:8090/v1 --key <key>`.
