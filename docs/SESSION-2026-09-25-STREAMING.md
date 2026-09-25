# collabosm session log — 2026-09-25: Responses streaming, resume point

**Status when this was written:** streaming protocol fixed and proven locally; deployed to the
A100 once more with a Job-construction fix plus a non-empty fallback. A100 session being shut
down at the end of this session to stop billing. **Resume by re-running `scripts/upserve.sh`
(or the usual restore + serve) and re-verifying — the last deploy's e2e was not confirmed.**

## The bug that was failing every Codex request

`/v1/responses` emitted `response.created` -> `response.output_text.delta` -> `response.completed`
with **no `item_id` / `output_index` / `content_index` on the deltas**, and the terminal event
carried a *different* message id than the stream had announced (`msg_...1` inline in the payload vs
`msg_...2` in the stream). Codex binds deltas to an announced item and needs a coherent terminal
event, so it aborted and retried: `st=0` on every `custom` request in `~/.modeldock/usage-events.jsonl`,
6 rapid attempts per turn, surfaced as "stream disconnected before completion".

Fixed by emitting the whole lifecycle with `sequence_number`:
`created` -> `in_progress` -> [reasoning item: added / reasoning_text.delta / .done / item.done] ->
message item / content_part.added -> deltas -> output_text.done -> content_part.done ->
output_item.done -> `completed`, all sharing one `rid` / `mid` / `rsn_id`.

## Second bug, found only after the first was fixed: empty answers

On the live A100 pack, the new incremental path ran but produced **one token**:
`[api] template=jinja prompt=29 cached=0 (0.0% hit) new=1 finish=stop` — so the stream was
protocol-correct but empty. Diagnosis is incomplete; the working theory is that a `Job` built with
fewer fields than `Generator.generate()` builds behaves differently. Mitigation deployed:
`_engine_fragments()` now mirrors `generate()`'s Job construction field for field, **and**
`collect()` falls back to the proven blocking `_generate_blocking()` whenever the incremental path
yields no text, logging `incremental path produced no text (new=... eos=...)` plus
`blocking fallback: new=... eos=... text=...`. That guarantees a non-empty answer either way;
whether streaming stays incremental on the A100 depends on which path the log shows.

## What is proven

Locally, no GPU, no tunnel — `scripts/dev_stub.py` (real Handler, fake engine) plus
`scripts/check_surface.py`:

- 13/13 checks pass for the incremental engine and for the `--think` variant.
- 13/13 with one deliberate exception for the `--silent` engine (nothing to stream): the blocking
  fallback still returns the full text, only the two "was it incremental" checks fail, as designed.
- **Real client, real parser**: Codex CLI 0.144.6 with `wire_api = "responses"` against the stub —
  exit 0, text rendered, both with and without a thinking block. Config used:

```toml
model = "qwen3.8-flash-next-exl3"
model_provider = "collabosm"
preferred_auth_method = "apikey"

[model_providers.collabosm]
name = "collabosm"
base_url = "http://127.0.0.1:8099/v1"
wire_api = "responses"
env_key = "COLLABOSM_TEST_KEY"
```

## Repository state

- `c3d4458` — pushed: incremental streaming, the Responses lifecycle fix, `dev_stub.py`,
  `check_surface.py`, README section "Testing it without a GPU".
- Uncommitted after that: the mirrored Job construction and the non-empty fallback in
  `scripts/api_server.py`, plus the `--silent` mode in `dev_stub.py`. **Commit these on resume**
  (they were verified locally, not yet on the A100).
- The VM keeps the server at `/content/api_server.py` (flat, not `/content/collabosm/...`), so
  deployment is a file write plus `serve.sh` — `/content/api_server.py.bak-deploy2` holds the
  previous revision.

## Operational facts worth keeping

- The Colab CLI **prunes the local session record while the VM keeps running**; `recover.py exl3
  A100` re-attaches from `list_assignments()`. Do not create a second VM.
- Quick tunnels change hostname on every restart. `~/.modeldock/custom-endpoints.json` must be
  updated (baseUrl + label + apiKey) after each restart, and ModelDock reloaded.
- The API key is stable across restarts in `/content/api-key.txt`. Do not print it.
- A blocking `curl` inside `colab exec` kills the kernel connection ("Connection was lost");
  detach with `nohup` and poll instead.
- Reload cost: ~264 s and 76,445 / 81,920 MiB. A failed experiment therefore costs about five
  minutes of A100 time, which is why the local harness exists.
- `TUNNEL_TOKEN` + `PUBLIC_URL` in `serve.sh` gives a stable hostname — worth using so the
  ModelDock endpoint does not need editing after every restart.

## Resume checklist

1. `recover.py exl3 A100` (or create) and start the service.
2. `python scripts/check_surface.py --base <tunnel>/v1 --key <key>` — expect 13/13 and a non-empty
   answer. If the log shows the blocking fallback firing, the incremental path still needs the
   real Job-field difference; the log line names the `eos_reason`.
3. Point a Codex CLI at the tunnel with the config above and confirm a streamed reply.
4. Update `~/.modeldock/custom-endpoints.json` with the new URL and key, then reload ModelDock.
5. Commit the two uncommitted `api_server.py` changes.