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
| `CACHE_SIZE` | `262144` | total KV tokens across all jobs (multiple of 256) |
| `CACHE_QUANT` | `4` | KV bits (`4` = Q4, `2`-`8` allowed) |
| `CPU_CACHE_GB` | `0` | **pinned-RAM second-tier KV page cache** (0 = off) |
| `RECURRENT_CACHE_GB` | `4` | host-RAM store for Gated-DeltaNet checkpoints |
| `GCS` | `4096` | generator chunk size — the biggest prefill lever we found |
| `NDT` | `4` | MTP draft depth |
| `RUNTIME` | `wheel` | `wheel` (prebuilt, no compile) or `source` |

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

Known gaps, stated plainly:

- **No token-by-token streaming.** A `stream: true` request is delivered as one delta followed by
  `[DONE]`, so clients work but text appears all at once.
- **Requests are serialised** by a lock — one Generator, one cache. "Multiple streams" today means
  queued, not parallel.
- **`max_batch_size > 1` is untested.** Every measurement ran `num_slots = 1`.
- Session *assignment* is scripted; it is not yet a configurable hosting layer.

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
