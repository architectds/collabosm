# Measurements

All figures below were produced on **one Colab A100-80GB High-RAM** (79.3 GiB VRAM, 167 GB RAM, 12
vCPU) against `turboderp/Qwen3.8-Flash-Next-exl3` @ revision `4.05bpw_h6_ng6`
(sha `55a732e0c4c3d4614bc42b68493bb930d9b02c0a`, 107,463,600,896 B) with ExLlamaV3 1.5.1.
Runs reported here returned rc=0 with no errors unless stated.

## Prefill and decode, MTP `ndt=4`, Q4 KV, `-ngr`

Prompts were built to a target token count from the tokenizer and sent with a unique nonce, so
`cached_tokens = 0` and every number is a genuine cold prefill. "Second of pair" is the figure to
trust; both are shown.

| test | prompt tokens | prefill t/s (r0 / r1) | decode t/s (r0 / r1) | MTP accept |
|---|---:|---:|---:|---:|
| 4K pair | 4,143 | 2,518.8 / **2,498.4** | 74.7 / 75.0 | 41.7% |
| 30K pair | 30,256 | 2,805.2 / **2,805.8** | 74.8 / 75.0 | 41.7% |
| 114K pair | 114,097 | 2,707.9 / **2,793.3** | 74.3 / 74.5 | 41.7% |
| decode @85 ctx | 85 | — | **78.6** | 44.6% |
| decode @30K ctx | 30,254 | 2,809.4 | **97.4** | 63.9% |
| decode @114K ctx | 114,095 | 2,792.6 | **90.1** | 57.1% |

Without MTP the same setup measures a flat **~56 t/s** decode from context 0 to 130,816.

## Generator chunk size is the biggest prefill lever we found

30,261-token prompts, one value at a time, second run of each:

| `-gcs` | 1024 | 2048 | 4096 | 8192 |
|---|---:|---:|---:|---:|
| prefill t/s | 2,204 | 2,806 | 3,360 | **3,882** |

Two things to know:

- **The first run at any new chunk size pays a one-time kernel autotune.** `-gcs 8192` read 1,674 t/s
  on its first run and 3,882 t/s on its second. This is exactly why the pair matters; a single-run
  sweep would have concluded the opposite.
- We had been running 2048 everywhere (the library default), leaving 20-38% on the table.

We have **not** tested 16384. That is the obvious next experiment.

## A measurement-integrity warning about `eval/perf.py`

The official harness is not a direct long-context measurement. In `eval/perf.py`:

```python
pre_time = 0
if length >= chunk_size * 2:
    pre_time = (length // 2) / results[length // 2]   # inherited, not measured
results[length] = length / (pre_time + t.interval)
```

Every length >= 2 x chunk_size is inherited from a shorter measurement, which is why its published
curve looks flat from 4096 to 131072. With `-chunk_size 2048` only the rows below 4096 are measured;
the rest are extrapolation. Use it for shape, not for headline numbers — and prefer a runner that
sends a fresh prompt per length.

Its no-MTP result matches ours where it *is* measuring: 2,909-3,234 t/s prefill, ~56 t/s decode.

## Warm KV reuse, measured

This is the feature that makes an interactive server pleasant, and it is two separate mechanisms.

**In VRAM (prompt caching).** Same 32,026-token prompt twice inside one Generator:

| run | prompt tokens | cached | hit | wall |
|---|---:|---:|---:|---:|
| cold | 32,026 | 0 | 0% | 15.62 s |
| repeat | 32,026 | 32,000 | **99.9%** | **0.45 s** |

**From pinned host RAM (`-ccs`).** Same prompt, but a second, different 60,031-token request in
between to force the first one's pages out of VRAM:

| step | prompt | cached | hit | prefill t/s | wall | tier entries | live pages evicted | pages restored |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A cold | 32,026 | 0 | 0% | 2,100.8 | 15.62 s | 0 | 0 | 0 |
| A again | 32,026 | 32,000 | 99.9% | — | 0.45 s | 0 | 0 | 0 |
| B (different, 60K) | 60,031 | 0 | 0% | 3,362.0 | 18.33 s | 104 | 104 | 0 |
| **A after eviction** | 32,026 | 32,000 | **99.9%** | — | **0.58 s** | 208 | 208 | **104** |
| A once more | 32,026 | 32,000 | 99.9% | — | 0.39 s | 209 | 209 | 104 |

`alloc_tier_pages = 104` is the proof it came from host RAM and not from a surviving VRAM page: the
engine counts restores separately from VRAM hits. Against the fair baseline -- the 9.72 s re-prefill
of the `-ccs 0` control below -- the restore is **16.8x faster**. (Against the 15.62 s cold row it
reads 27x, but that row also paid the one-time kernel autotune.)

**Control: the identical sequence with `-ccs 0`.** Same prompts, same eviction pressure (104 live
pages evicted in both runs), only the tier differing:

| step | `-ccs 8` | `-ccs 0` |
|---|---|---|
| A cold | 15.62 s | 9.85 s |
| A again | 32,000 cached (99.9%), 0.45 s | 32,000 cached (99.9%), 0.44 s |
| B different (60,031 tok, evicts A) | 18.33 s, 104 pushed | 18.12 s, 104 evicted, no tier |
| **A after eviction** | **32,000 cached (99.9%), 0.58 s** | **0 cached (0%), 9.72 s** |
| A once more | 0.39 s | 32,000 cached, 0.40 s |
| `stashes_stranded` | **0** | 6 |

So the tier turns a 9.72 s re-prefill into a 0.58 s restore — **16.8x** — and without it the
Gated-DeltaNet checkpoints are orphaned by KV eviction (`stashes_stranded: 6`). The `-ccs 8` cold row
is worse only because it was the first generation after load and absorbed the one-time kernel autotune;
the B rows agree (18.33 vs 18.12 s). Design notes in `docs/RUNBOOK.md`.

## KV cache

The engine reports its own cache sizing, which matches the arithmetic exactly:

| quantity | value |
|---|---|
| `bytes_per_token` (reported) | **10,752.0** |
| layers with a KV cache | 12, all `CacheLayer_qsa_quant` |
| cache at 262,144 tokens | 2,818,572,288 B (2.63 GiB) |
| VRAM used with cache loaded | 73,589 / 81,920 MiB |
| recurrent (`GDNLayerState` + `PLELayerState`) | 572,395,568 B (546 MiB), constant with context |

Per full-attention layer per token: K 256 B + V 256 B + scales 64 B + QSA indexer planes 320 B = 896 B,
times 12 layers. The QSA planes stay fp16 regardless of `-cq`, so they are 36% of the cache and cap
the Q4 saving at ~62% versus fp16, not 75%.

| context | Q4 KV cache |
|---|---:|
| 262K | 2.63 GiB |
| 500K | 5.01 GiB |
| 1M | 10.0 GiB |

Sizing guidance and the split-versus-unified argument are in `docs/RUNBOOK.md`.

## Model geometry (from the pack's own `config.json`)

| item | value |
|---|---|
| layers | 48 — 36 linear attention (Gated DeltaNet) + 12 full attention |
| `full_attention_interval` | 4 |
| KV heads / head_dim | 2 / 256 |
| QSA indexer | head_dim 128, compress ratio 4, budget 2048 |
| native max position | 262144 |
| experts | 512, top-10 |
| vision | `vision_k6.safetensors` (561 MB) |
| n-gram/PLE table | `ngram_embedding.safetensors` (36.36 GiB, host RAM via `-ngr`) |
## Verified end to end, 2026-09-25

One account, one box, no manual steps beyond `scripts/`:

| step | result |
|---|---|
| `scripts/restore.py` | re-attached an **orphaned** assignment after the Colab CLI dropped its local record for the third time that day (VM alive and still billing); confirmed 79.3 GiB VRAM, 167.1 GiB RAM, cc 8.0, python 3.13.15 |
| `scripts/serve.sh` | loaded the 4.05 bpw pack at `cache_size 500224` in **259.5 s**; `/health` 200 at **76,437 / 81,920 MiB** VRAM |
| through the public tunnel | `/v1/models` **401 without the key, 200 with it**; a real chat completion answered in **3.1 s** |

Runtime flags for that run: `-cq 4 -ndt 4 -gcs 4096 -ccs 16 -rcs 16 -ngr`.

Two facts about the served model that cost real time to find:

- **It is a reasoning model with thinking on by default.** A `max_tokens` under ~1K returns a
  truncated reasoning trace rather than an answer; this looks like a broken server and is not one.
  (`api_server.py` has since turned thinking off unless a request asks for it.)
- **`Generator.generate()` returns `(completions, last_results)`.** Reading `text` out of
  `last_results` yields only the *final fragment* of the completion, so an early `api_server.py`
  answered `" inputs"` to a real question. The server now unpacks the tuple and asks for
  `completion_only=True`. This is the same class of bug as the per-request `Generator`: invisible,
  and it does not raise.

## What Colab charges, and what it counts as use (2026-09-25/26)

| item | value | source |
|---|---|---|
| A100-80G High-RAM rate | **6.77 CU/h** | `/tun/m/ccu-info` `consumptionRateHourly` with this box as the account's only assignment (older notes: 7.52) |
| unattended box reclaimed | **within 25 min** of the last `colab exec` | CLI history: last exec 02:35:37 UTC, assignment list empty at 03:00:05; the service was up and nothing sent a keep-alive |
| shape sent for a 40 GB card | none | 13 unpatched `assign` calls in `colab.log` came back `machineShape 0`; `shape=st` has never been sent |

Chat through the tunnel does not count as use; the kernel and the keep-alive ping do. That is why
the frontend keeps a box alive while it is in use (`docs/RUNBOOK.md`, cost guardrails).

## Capacity

The fitted VRAM line, the per-slot recurrent cost, the session table and the reasons the host-RAM
tier cannot extend live capacity are in `docs/CONCURRENCY.md`.
