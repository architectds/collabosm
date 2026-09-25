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
