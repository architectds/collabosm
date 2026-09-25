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
| resume a fully-evicted 32K conversation | **0.58 s** instead of 15.62 s |
| cost | 7.52 CU/h (A100 High-RAM) ≈ $0.75/h, ≈26.6 h per 200 CU |

ExLlamaV3 is the only engine in this repo. See `docs/MEASURED.md` for provenance and caveats, and
`docs/CONCURRENCY.md` for how many sessions the card can actually hold.

## Requirements

- A Colab plan that can allocate an **A100** (this kit was developed on a Pro tier with ~200 CU/month).
- **HIGH_RAM** specifically. A 40 GB standard A100 **cannot load this model at all** — about 63.6 GiB
  of weights must be VRAM-resident. `scripts/restore.py` requests it explicitly and refuses a 40 GB box.
- `uv tool install google-colab-cli` (and `gh` if you want to fork/push).
- ~110 GiB of Colab disk for the weights.

## Quickstart

```bash
bash scripts/up.sh          # restore/create the box, install runtime, fetch weights, serve
bash scripts/down.sh        # STOP THE VM. On a metered plan this is the most important command.
```

`up.sh` prints the tunnel URL when the endpoint answers `/health`. Point your client at
`<url>/v1` with the API key it prints (also at `/content/api-key.txt` on the VM).

Useful knobs, all via environment:

```bash
SESSION=mybox CACHE_SIZE=524288 CPU_CACHE_GB=8 bash scripts/up.sh
```

| variable | default | meaning |
|---|---|---|
| `SESSION` | `collabosm` | local session name |
| `CACHE_SIZE` | `500224` | total KV tokens across all jobs (multiple of 256) |
| `CACHE_QUANT` | `4` | KV bits (`4` = Q4, `2`-`8` allowed) |
| `CPU_CACHE_GB` | `32` | **pinned-RAM second-tier KV page cache** (0 = off). Sized at 92K tokens/GB; do not treat it as free - it is allocated as pinned memory in full (32 + 24 = ~56 GB next to a 36.4 GiB n-gram table) |
| `RECURRENT_CACHE_GB` | `24` | host-RAM store for Gated-DeltaNet checkpoints (~2048-token interval, 116 MB each) |
| `GCS` | `4096` | generator chunk size — the biggest prefill lever we found |
| `NDT` | `4` | MTP draft depth |
| `RUNTIME` | `wheel` | `wheel` (prebuilt, no compile) or `source` |
| `TUNNEL_TOKEN` | unset | named-tunnel token: gives a **stable hostname** instead of a quick tunnel. Pair with `PUBLIC_URL` for the URL shown in status |

## What is in here

| path | role |
|---|---|
| `scripts/restore.py` | **the session restore script.** Re-attaches an orphaned VM from server truth, or creates one and actually requests the HIGH_RAM shape. Refuses/stops a 40 GB box before spending anything. |
| `scripts/probe_gpu.py` | runs on the VM; reports VRAM/RAM/cc/disk as one JSON line |
| `scripts/up.sh` | restore → upload → bootstrap → serve → wait for health |
| `scripts/down.sh` | stop the VM and report what is still billing |
| `scripts/bootstrap.sh` | runtime (pinned wheel) + weights (from HF at a pinned revision), idempotent |
| `scripts/serve.sh` | launch the API + cloudflared tunnel (runs on the VM) |
| `scripts/api_server.py` | minimal OpenAI-compatible server; **one long-lived Generator** |
| `scripts/status.py` | one-glance stage/health report |
| `scripts/dev_stub.py` | serves the real Handler on loopback with a stubbed engine: exercises the wire format with no GPU and no weights |
| `scripts/check_surface.py` | drives a running server over HTTP and judges the surface (deltas, terminal event, ids, sequence numbers) |
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
- **`max_batch_size > 1` is untested.** Every measurement ran `num_slots = 1`.
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

## Client, and using a real WebUI

`collabosm.py` is the local client. Stdlib only, no install step:

```bash
python collabosm.py config --init --endpoint https://<host>.trycloudflare.com/v1 --api-key <key>
python collabosm.py ui        # http://127.0.0.1:8790  (page + proxy)
python collabosm.py chat "hello"          # same path, terminal
python collabosm.py update-ui             # pull ui/ from this repo and cache it
```

It implements **exactly two dialects**, the two Codex uses: `/v1/responses` and
`/v1/chat/completions` (plus `/v1/models`, which every OpenAI-compatible client probes first).
Anything else returns a JSON 404 that says so.

**Why a local proxy instead of pointing a client straight at the endpoint**

- the bearer key stays in this process and is never handed to a browser,
- the page is same-origin with the proxy, so there is no CORS surface at all,
- the UI can be updated independently of the VM.

There is deliberately **no** `Access-Control-Allow-Origin: *` here: a wildcard on a proxy that
injects a bearer key would let any web page you happen to visit spend your GPU through it. Server-side
UIs do not need it (they call upstream from their own backend); if a browser-hosted UI ever does,
add an explicit origin allowlist, not a wildcard.

### Two windows, named

The client serves three pages, and the launcher opens the two that matter so it is
obvious at a glance which is which:

```bash
python collabosm.py start       # start the client if needed, then open both windows
python collabosm.py startup     # do the same at every login (Startup folder shortcut)
python collabosm.py startup --remove
```

| page | title | what it is |
|---|---|---|
| `/` | `collabosm` | a landing page with both entries, for when you open it by hand |
| `/chat` | `collabosm — chat` | conversations, streamed, thinking collapsed |
| `/status` | `collabosm — status` | endpoint state, live generation, and measured prefill / decode / ttft |

`start` is deliberately tolerant of an unconfigured or unreachable endpoint: the
client comes up anyway so the status page can say *what* is missing. Bringing the
A100 up or down stays with `scripts/up.sh` / `scripts/down.sh`; this client never
spends money on its own.

If you would rather chat in Open WebUI, point the chat window elsewhere and keep
this dashboard:

```bash
python collabosm.py config --chat-url http://127.0.0.1:8080
```

### The metrics contract

The OpenAI protocol has no field for prefill or decode throughput, so no off-the-shelf UI can show
it. The client times the stream as it passes and derives it:

- `prefill ~= prompt_tokens / time-to-first-token` (includes queueing and prefill)
- `decode ≈ output_tokens / (total − time-to-first-token)`

`GET /local/metrics` returns `{last, live, turns}`. Tokens come from the upstream's own `usage` when
it sends one and are flagged `"estimated": true` otherwise; a non-streamed turn reports no rates at
all, because without a first-token time there is no prefill/decode boundary to measure. A wrong
number presented as a measurement is worse than no number.

### Pointing an existing WebUI at it

Any OpenAI-compatible client works; give it the proxy, not the VM:

```
base URL : http://127.0.0.1:8790/v1
API key  : anything (the proxy injects the real one)
model    : qwen3.8-flash-next-exl3
```

For Open WebUI, which is the natural fit: `uv tool install open-webui`, then
Settings -> Connections -> OpenAI API with the values above. It talks to the proxy from its own
backend, uses `/v1/chat/completions` (not the Responses dialect, which is Codex's), and brings
multi-conversation itself -- which is why this repo does not try to be a chat app.
The measured recipe -- where it installs, which environment variables matter, what the first
run does, and what it costs on a metered A100 -- is in [docs/OPEN-WEBUI.md](docs/OPEN-WEBUI.md).

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

**Why `VISION=1` is not the default yet:** this box already sits at 76.4 / 81.9 GiB, and the tower's
VRAM cost has not been measured here. Flip it on a real load, watch the number, then make it the
default in `serve.sh`. `GET /v1/status` says which state a running server is in, so a client (or
ModelDock's `supportsVision`) can stop guessing.

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
