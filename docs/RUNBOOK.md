# Runbook

Operating notes for collabosm. Everything here is a thing that either bit us or is easy to get wrong.

## The 40 GB lottery

`colab new --gpu A100` does **not** send `shape`, so you get one of two machines:

| shape | VRAM | RAM | CU/h | can it run this model? |
|---|---:|---:|---:|---|
| HIGH_RAM (`shape=hm`) | 79.3 GiB | 167 GB | 7.52 | **yes** |
| standard | ~39 GiB | 83 GB | 5.37 | **no** — about 63.6 GiB of weights must be VRAM-resident |

It is not a coin flip in your favour: **eleven consecutive unpatched attempts returned the 40 GB
machine.** `scripts/restore.py` patches `Client._build_assign_url` to add `params["shape"] = "hm"`
(`google-colab-cli#47` — `Shape.HIGH_RAM` exists and `machineShape` is parsed, but the field is never
sent), then verifies and **stops** an unsuitable box so the retry costs ~0.13 CU instead of ~10.

**Never detect the card by searching for "80".** An A100's compute capability is `sm_80`, so a 40 GB
box prints "80" in every capability string and will pass a naive test. Compare `vram_GiB` (a 40 GB box
measures ~39).

## Your local session record is disposable — the VM is not

Twice in one day the CLI reported `No active sessions found` while the VM was **alive and billing**,
because the runtime proxy token lapses and the CLI deletes its local bookkeeping. Once it happened
mid-run with `/content` fully intact, and recovery cost seconds.

`scripts/restore.py` handles this by reading server truth (`list_assignments()`) and re-attaching
rather than creating a second VM. It also refuses to adopt an endpoint already claimed by another
local session name, because two names pointing at one VM is how you stop the wrong machine.

Recovery by hand, if you ever need it:

```bash
cat ~/.config/colab-cli/sessions.json   # local record (often {})
colab sessions                          # server truth — what is actually billing
python ~/.local/share/uv/tools/google-colab-cli/bin/python scripts/restore.py -n mybox
```

## The `-ccs` pinned-RAM KV tier: verified A/B

`-ccs` (GB, default 0) is a **second-tier page cache in pinned system memory**, off by default. It is
RAM, not the pagefile: there is no disk-backed KV path in ExLlamaV3, and pinned memory cannot be
swapped out.

Identical prompts, identical eviction pressure (104 live pages evicted in both runs), only `-ccs`
differing:

| step | `-ccs 8` | `-ccs 0` |
|---|---|---|
| A cold (32,026 tok) | 15.62 s | 9.85 s |
| A again | 32,000 cached (99.9%), 0.45 s | 32,000 cached (99.9%), 0.44 s |
| B different (60,031 tok, evicts A) | 18.33 s, 104 pages pushed | 18.12 s, 104 evicted, no tier |
| **A after eviction** | **32,000 cached (99.9%), 0.58 s** | **0 cached, 9.72 s (full re-prefill)** |
| A once more | 0.39 s | 32,000 cached, 0.40 s |
| `stashes_stranded` | **0** | 6 |

Two conclusions: **0.58 s versus 9.72 s to resume an evicted conversation (16.8x)**, and with no tier
the Gated-DeltaNet checkpoints get orphaned (`stashes_stranded: 6`) because their anchor KV pages were
evicted with nowhere to go. The ccs=8 run's *cold* number looks worse only because it was the first
generation after load and absorbed the one-time kernel autotune; the B rows agree (18.33 vs 18.12 s).

Caveats worth knowing before you turn it on:

- The `-ccs` number is a **ceiling**: `max_slots = budget // slot_size`, allocated lazily as pages are
  pushed. Our slot is 2,981,888 B (2.84 MiB) because a slot holds one page image across the **main and
  draft** caches, so `-ccs 8` is 2,880 slots ≈ 737K tokens of warm prefix.
- It is **evict-only**. Pages enter the tier when the VRAM pool repurposes them; active pages must be
  in VRAM. It does not let one live conversation exceed `CACHE_SIZE`.
- Not supported in tensor-parallel mode (irrelevant here — TP is not implemented for this
  architecture at all).

## KV sizing, and split versus unified

`CACHE_SIZE` is the **total** across concurrent jobs, not per job — the engine's own contract is
`seq_len * batch_size <= max_num_tokens`. exllamav3 is unified by construction: there is no per-job
slice mode. So size it as a sum (`-cs 524288` for one 500K conversation with spare), never
"`-cs 500000` x N".

At 10,752 B/token (Q4):

| context | Q4 KV |
|---|---:|
| 262K | 2.63 GiB |
| 500K | 5.01 GiB |
| 1M | 10.0 GiB |

With ~63.6 GiB of weights and ~5.65 GiB of fixed workspace, the measured cache budget on an 80 GB card
is ~9-10 GiB ≈ **900K-1M tokens total**. One 500K stream fits with ~4.5 GiB spare; two do not.

## `-gcs` is the biggest prefill lever, and the first run lies

| `-gcs` | 1024 | 2048 | 4096 | 8192 |
|---|---:|---:|---:|---:|
| prefill t/s (second run) | 2,204 | 2,806 | 3,360 | **3,882** |
| prefill t/s (first run) | — | — | — | **1,674** |

The first generation at any new chunk size pays a one-time kernel autotune. A single-run sweep
concludes the opposite of the truth. 16384 is untested.

## A new `Generator` is a cold cache

`PageTable` and `CPUPageCache` are constructed inside `Generator.__init__`. Build a Generator per
request and you silently lose prompt caching *and* the pinned-RAM tier — nothing errors, you just
re-prefill everything. `api_server.py` holds exactly one Generator, and `docs/MEASURED.md` records the
harness bug that hid this from us for an entire sweep (every row read `cached_tokens: 0` because the
harness made a new Generator per measurement).

## The recurrent checkpoint budget, not the KV tier, limits resumable depth

This model is hybrid: 36 Gated-DeltaNet layers keep a fixed-size state that does not grow with
context. A resumable prefix therefore needs **both** the KV pages **and** a matching recurrent
checkpoint anchored at that page hash — the engine explicitly refuses to restore pages past the last
usable checkpoint because replay prefill rewrites them anyway.

Measured: 37 recurrent layers = 572,129,280 B at `ndt=4`, so one checkpoint is ~116 MB
(348,108,992 B for 3, 698,277,984 for 6 — i.e. **116.4 MB each**). The default `-rcs 4` GB therefore
holds only ~35 checkpoints, roughly 70K tokens of anchored prefix at the default
`recurrent_checkpoint_interval` of 2048 tokens. For a 500K resumable history, budget `-rcs` in the tens
of GB. On a High-RAM box (167 GB, of which `-ngr` takes 36.4 GiB) there is room for that.

## Serving: the last hop is a tunnel

The VM has no inbound address, so `serve.sh` publishes the API through a Cloudflare tunnel, and
that URL is what every client uses:

```
api_server.py on 127.0.0.1:8090  ->  cloudflared --url http://127.0.0.1:8090
                                 ->  https://<host>.trycloudflare.com/v1
```

- The **key is stable** (`/content/api-key.txt`); the **hostname is not**, with a quick tunnel.
- After every restart, `~/.modeldock/custom-endpoints.json` (`baseUrl`, and `label` for legibility)
  points at a hostname that no longer exists, and every request 404s until it is updated and
  ModelDock is relaunched. This is the most common "the endpoint is broken" report, and it is not a
  server bug: check `/content/STATUS` first, then the configured `baseUrl`.
- Set `TUNNEL_TOKEN` (and `PUBLIC_URL` for `status.py`) in `/content/collabosm.env` for a stable
  hostname; then the endpoint config survives restarts untouched.
- The tunnel is only a transport. Before blaming it, reproduce on loopback inside the VM
  (`curl -s http://127.0.0.1:8090/health`) or locally against a stub
  (`scripts/dev_stub.py` + `scripts/check_surface.py`).
- `transport` must be `"responses"` in the ModelDock endpoint entry: the Responses dialect is the
  path Codex actually uses.

## Cost guardrails

- A100 High-RAM is **7.52 CU/h ≈ $0.75/h**. 200 CU ≈ **26.6 h/month**.
- A model load is ~2-6 min (132 s warm page cache, 380 s cold) and the whole bootstrap ~11 min from
  nothing, so *reloading to change one flag* is the main waste. Batch experiments into one load: the
  generator chunk size and the CPU tier are both settable without reloading (chunk size is a Generator
  argument), while `-cs`, `-cq`, `-ndt` and `-ccs` are load-time.
- Nothing in this kit starts a keep-alive daemon, on purpose. A keep-alive is what turns a 2 h session
  into a 24 h one. Run `bash scripts/down.sh` when you stop working, and check `colab sessions`.
- Colab idle-prunes an unattended VM after roughly 90 minutes; a long GPU run counts as activity.

## The frontend control plane (`frontend/control.py`)

The shell's right column is drawn from one object, and `control.py` is the real one. Facts that cost
time to learn:

- **The Colab CLI is Linux-only.** `uv tool install google-colab-cli` has no Windows build, so the
  Windows frontend shells out: `wsl.exe -d Ubuntu -- bash -lc "<cd /mnt/e/... && up.sh>"`. Two things
  that bite: `wsl.exe` prints *its own* messages as UTF-16 (set `WSL_UTF8=1`, or every line looks like
  `N A M E`), and the path it needs is the WSL one (`E:\models\collabosm` -> `/mnt/e/models/collabosm`).
  All `.sh` files here are LF; a CRLF script fails in WSL with `\r` errors.
- **Billing starts at the click, not at the load.** The ledger opens its record when the job is
  spawned (that is when `assign` happens), and closes it on `down`, failure or the safety stops. A
  drawn-and-rejected 40 GB box therefore costs ~0.13 CU, and the ledger says so.
- **The confirmation gate is the product.** `select()` without `confirm: true` returns
  `confirm_required` with CU/h, ETA and the cost of the load itself; a UI that skips the gate is a UI
  that spends CU on a mis-click.
- **Stages only move forward.** `up.sh` relays `scripts/status.py` rows, and a relayed
  `stage=probing` after `stage=loading` used to drag the rail backwards while the bar stayed at 60%.
  `_set_stage()` now refuses to regress, and the bar has a time-based floor because `up.sh` sleeps
  45 s between polls.
- **Two safety stops, both visible in the rail**: `--idle-stop-min` (default 20 min without chat
  traffic) and `--max-session-h` (default 6 h even with traffic). Chat traffic is anything the frontend
  proxies to `/v1/*`, so a long generation counts as activity.
- **Rehearse without CU.** `--fake-provision` keeps the real control plane and replaces only the WSL
  process with `frontend/fake_provision.py` (same log lines, ~24 s, no network). `--mock` swaps in the
  demo pacing. Both were used to verify the confirm gate, the stages, the ready card, the proxy and the
  stop path before a single CU was spent.
- **The upstream WebUI registers a service worker at scope `/`.** A shell cached by an older build came
  back as a *second* copy of the rail inside the iframe (and the old prototype on port 3010 was still
  serving that build). `shell.html` now unregisters service workers and drops caches on load; the stale
  prototype servers are gone. If a duplicate panel ever appears again, check the URL and the port
  first -- `http://127.0.0.1:<frontend-port>/` is the only current page.
- **One window.** `python collabosm.py start` opens the frontend shell when something answers on
  `--frontend-port` (default 3020) and only falls back to the older two-window chat/status pair when it
  does not.

## Installing the CLI, and the one thing that breaks it

```bash
uv tool install google-colab-cli
```

Do **not** run `uv tool upgrade google-colab-cli`: it pulls `jupyter-kernel-client` 1.0.2, which
removes `KernelClient` and breaks `colab exec` with `RuntimeError: Connection was lost`. Repair:

```bash
uv pip install --python ~/.local/share/uv/tools/google-colab-cli/bin/python jupyter-kernel-client==0.9.0
```

## Model notes

- `4.05bpw_h6_ng6` is a **revision (branch)**, not a subdirectory. Other branches: `6.05`, `5.05`,
  `3.05bpw_h5_ng5`, `2.05bpw_h4_ng4`. Pin the **sha** for reproducibility, not the branch name.
- The pack includes `vision_k6.safetensors` (561 MB), so vision is available; we have not benchmarked
  multimodal prefill.
- The 36.36 GiB n-gram/PLE table must live in host RAM (`-ngr`, i.e. `ngram_ram: true`). Streaming it
  from disk instead costs ~35 ms per speculative round and collapses decode to roughly a third.
- Weight streaming, cache tiers and the WS-2050 pinning (`-ccs`) are all host-RAM features; the
  measured 167 GB High-RAM box is what makes this model fit.
