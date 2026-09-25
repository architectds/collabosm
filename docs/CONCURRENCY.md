# How many sessions fit on one A100-80GB

There are two separate budgets, and the smaller one wins:

1. **VRAM**, which the KV cache pages compete for, and
2. **host RAM**, which the pinned KV page cache (`-ccs`), the recurrent checkpoint store
   (`-rcs`) and the n-gram table (`-ngr`) compete for.

On top of both sits a third cost that is easy to miss: every *live* slot carries its own
Gated-DeltaNet + PLE recurrent state, which is not a KV page and does not shrink with context.

All numbers below were measured on one Colab A100-80GB **High-RAM** (81,920 MiB reported VRAM,
167.1 GiB RAM) running `turboderp/Qwen3.8-Flash-Next-exl3` @ `4.05bpw_h6_ng6`, ExLlamaV3 1.5.1,
Q4 KV, `-ndt 4`, `-ngr`. Sources and caveats: `docs/MEASURED.md`.

## 1. The VRAM fit is a line

Two load points, both with exactly one live job:

| `cache_size` | VRAM used | vs. previous |
|---:|---:|---:|
| 262,144 | 73,589 MiB / 81,920 | - |
| 500,224 | 76,437 MiB / 81,920 | +2,848 MiB for +238,080 tokens |

That is **0.011963 MiB per cache token** (12.54 kB = 12.25 KiB), giving

```
used_MiB = 70,453 + cache_tokens x 0.011963
```

The engine's own `bytes_per_token` is **10,752 B** (10.50 KiB = 0.010263 MiB), so the measured
marginal slope is ~17% higher than nominal. The difference is consistent with prefill staging
scratch that grows with the configured cache; treat the fitted line as the empirical answer and
leave the margin in place.

With a **1 GiB** margin, the KV budget is therefore

```
H = 81,920 - 70,453 - 1,024 = 10,443 MiB  ->  10,443 / 0.011963 = ~872,000 tokens
```

That is **~870K tokens of KV in total**, split however you like across live jobs - subject to the
slot cost below. The line was fitted at *load*; prefill staging grows during a real request, so
**~600-700K joint is the safer operating point** than the 870K ceiling.

## 2. Every live slot costs ~546 MiB, before any KV

The recurrent state (`GDNLayerState` + `PLELayerState`) measures **572,395,568 B = 546 MiB** and is
constant with context. The 70,453 MiB baseline was taken with one live job, so it already contains
one slot's worth; the extra cost for `N` live jobs is `(N - 1) x 546 MiB`.

```
max live jobs = floor( (H + r) / (r + C x t) )
  H = 10,443 MiB usable VRAM margin
  r =    546 MiB recurrent state per live slot (ndt = 4)
  t =      0.011963 MiB per cache token
  C =      context (cache tokens) per stream
```

| context per stream | max live jobs | binding term |
|---:|---:|---|
| 500K | **1** | KV (a 2nd 500K stream needs 12,513 MiB) |
| 400K | **2** | KV |
| 262K | **2** (3 needs 11,043 MiB, 55 MiB over) | KV |
| 200K | **3** | KV |
| 131K | **5** | KV + recurrent together |
| 64K | **8** | recurrent |
| 32K | **11** | recurrent |
| 16K | **14** | recurrent |
| 8K | **17** | recurrent |

**Concurrency saturates at ~20 jobs** as context goes to zero (`10,988 / 546`), because slot state
alone then fills the card. The practical band is 8-14 slots for 16K-64K contexts.

## 3. Why "spill it to RAM above 600K joint" does not work for a live stream

The natural mental model is virtual memory: keep the working set in VRAM, page the rest out to host
RAM, and let RAM size set the real limit. That model is right about *capacity* and wrong about
*latency*, for three reasons.

**(a) The page pool is VRAM and its size is a load-time decision.** `cache_size` is the token pool
for all jobs together, allocated up front - which is why VRAM used at load tracks it linearly
(section 1). The pages the engine may evict to the tier are pages no running job references; that
is what makes the tier evict-only. There is no fault path that would let a live job pull its own
evicted pages back mid-generation, so exceeding joint capacity is an **allocation failure at
request time, not a slowdown**, and the pool cannot be grown without a reload (259.5 s measured).

**(b) Attention has no locality.** Every decode step reads the whole sequence's KV. At 114,095
tokens that is **1.23 GB per token**, and the measured 90.1 t/s means the card is already pulling
**~111 GB/s of KV** - HBM territory. A tier that has to move a stream across PCIe cannot serve
that rate:

| context | KV read per decode step | ceiling at a perfect 20 GB/s (PCIe 4.0 x16, zero overhead) |
|---:|---:|---:|
| 32K | 344 MB | ~58 t/s |
| 114K | 1.23 GB | ~16 t/s |
| 500K | 5.38 GB | ~4 t/s |

KV residency is therefore a **bandwidth** requirement, not just a capacity one - unlike the n-gram
table, whose per-step access is sparse, which is exactly why 36.4 GiB of it can live in host RAM
(`-ngr`) while the KV cannot.

**(c) The slot state cannot be paged at all.** 546 MiB per live slot must be resident in VRAM for
the whole life of the job, so no amount of RAM raises the ~20-slot ceiling.

## 4. What RAM *does* buy: swapping whole conversations

The tier is a swap device at the **conversation** level, and there it is genuinely worth 16.8x
(measured, `docs/MEASURED.md`): a fully evicted 32,000-token conversation restores in **0.58 s**
instead of **9.72 s** of re-prefill.

From that same measurement the tier moves 344 MB in 0.58 s = **~0.6 GB/s** effective, so:

| conversation | tier restore | re-prefill at 2,806 t/s | ratio |
|---|---:|---:|---:|
| 32K | 0.58 s | 11.4 s | 20x |
| 114K | ~2.1 s | 40.7 s | 20x |
| 500K | ~9 s | 179 s | 20x |

So the correct architecture is: size the VRAM pool to hold **one** maximal live stream, keep many
conversations parked in RAM, and pay ~0.5-9 s at each switch instead of paying a re-prefill. That is
a scheduler question (swap at request boundaries), not a hardware question.

## 5. Host RAM: the budget that is actually under-used

| item | cost | note |
|---|---:|---|
| total | 167.1 GiB | the High-RAM shape |
| n-gram / PLE table (`-ngr`) | 36.4 GiB | 39,040,193,720 B; sparse per-step access |
| python / torch / runtime | ~2 GiB | |
| `-ccs` (current) | 16 GB | pinned, cannot swap |
| `-rcs` (current) | 16 GB | recurrent checkpoints |
| staging / working set | ~2-4 GiB | multi-GiB chunks at `-gcs 8192` |
| **unallocated** | **~90 GiB** | the number worth arguing about |

Parked capacity is priced at 10,752 B/token, measured as **92K tokens per GB** of `-ccs`
(`-ccs 8` = 2,880 slots ~ 737K tokens):

| `-ccs` | parked prefix | equivalent to |
|---:|---:|---|
| 16 GB (current) | ~1.47M tokens | ~2.9 x 500K, or 45 x 32K |
| 64 GB | ~5.9M tokens | ~11 x 500K, or 180 x 32K |
| ~90 GB | ~8.3M tokens | ~16 x 500K, or 250 x 32K |

Raising `-ccs` costs no VRAM and does not change any number in section 2 - it only changes how many
conversations can be *warm* instead of re-prefilled. If the "the limit should be RAM, not VRAM"
intuition is to be cashed in anywhere, it is here: the current `-ccs 16` uses roughly 15% of the
host RAM that could be devoted to it.

**`-rcs` is the catch.** `-rcs` sizes *resumable depth*, not sessions: one Gated-DeltaNet checkpoint
is 116.4 MB and the interval is 2048 tokens, so **500K of resumable history needs ~244 checkpoints
~ 28 GB**. At `-rcs 16` a 500K conversation resumes with a partial re-prefill; full-depth resume
needs `-rcs` in the tens of GB, which competes with `-ccs` for the same RAM.

## 6. Where the real wall is, in one line

- **Joint live context:** ~870K tokens at load / **~600-700K** with prefill headroom. VRAM, fixed at
  load. RAM does not extend it.
- **Live session count:** ~20, and 8-14 for realistic contexts. VRAM, set by 546 MiB of slot state.
  RAM does not extend it.
- **Warm parked conversations:** RAM-bound, 92K tokens/GB. This is the only one of the three that
  buys more with more memory.
- **Switch cost:** ~0.6 GB/s of restore, i.e. ~0.58 s per 32K conversation.

## 7. Caveats, stated plainly

- **`max_batch_size > 1` has never been exercised.** Every measurement in this repo ran
  `num_slots = 1` (`model_init` defaults to `-ambs 1`). The cheap test is `-ambs 4` with four
  concurrent requests, looking for the extra `3 x 546 MiB ~ 1.6 GiB` of recurrent state in
  `nvidia-smi`. This is also the test that would show whether a *running* job's pages are ever
  evicted - the one assumption section 3(a) rests on.
- **The tier has never been driven to saturation.** 104-208 pages were pushed and restored; nobody
  has filled a 16 GB tier and watched the eviction policy choose.
- **"Fits" is not "faster".** One GPU shares compute and bandwidth across live jobs. Section 2 is a
  capacity answer, not a throughput answer.
- **The serialisation lock is still in place.** `api_server.py` holds one Generator and one cache,
  so today "multiple streams" means *queued*, and a conversation-level swap scheduler does not
  exist yet.
- **Cheapest lever for more slots: `-ndt 2`.** Speculative depth is what makes each slot cost
  546 MiB; at `ndt 2` it is ~343 MiB. That trades decode speed for capacity.
- **Vision has not been measured at these cache sizes**, and `-gcs 16384` is untested. At
  `-gcs 8192` staging alone is a multi-GiB transient, which is part of why the fitted slope exceeds
  nominal.

