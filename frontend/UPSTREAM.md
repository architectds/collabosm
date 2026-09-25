# The vendored WebUI (`frontend/upstream/`)

`upstream/` is a production build of **llama.cpp's Web UI**. It is vendored byte
for byte: nothing in it is edited, patched or renamed, and `shell.html` +
`server.py` are the only frontend files this repo owns.

## Provenance

| what | value |
|---|---|
| upstream | `https://github.com/ggml-org/llama.cpp` |
| path | `tools/ui` |
| commit | `4b1a27fa0eb875bbca4f6cfe936e3d65adc685c0` (2026-09-25) |
| source size | 6.06 MB (sparse checkout of `tools/ui` only) |
| licence | MIT — text in `upstream/LICENSE`, copied from that same commit |
| toolchain | node v24.18.1 / npm 11.16.0 |

## How it was built

```sh
git clone --depth 1 --filter=blob:none --sparse https://github.com/ggml-org/llama.cpp.git
cd llama.cpp && git sparse-checkout set tools/ui && cd tools/ui
npm ci                                   # 11 min, 1051 packages, node_modules 484 MB
LLAMA_UI_OUT_DIR=/somewhere/dist npm run build    # 4 min 24 s
```

Both numbers were measured on 2026-09-25 on this machine. `npm run build` is
`build-pwa-assets && vite build`; it uses `@sveltejs/adapter-static`, so the
output is a plain static site.

## What is in the build

| path | size | note |
|---|---:|---|
| `index.html` | 12.3 KB | references the hashed assets below |
| `_app/immutable/bundle.WnBO6OxH.js` | 8.65 MB | the app; **hashed filename** |
| `_app/immutable/assets/bundle.DN2gXIUA.css` | 531 KB | styles |
| 48 `apple-splash-*.png` + pwa icons + `sw.js`/`workbox` | ~0.3 MB | PWA plumbing, unused by us |
| **total** | **9.43 MB / 70 files** | `SHA256SUMS.txt` lists every file |

We ship the whole build rather than trimming the PWA parts: 9.4 MB is nothing
next to the model, and `sw.js` is registered by the bundle — removing it turns a
clean load into a console error. If repo size ever matters, the trim is ~0.3 MB
and not worth the risk.

**No node is needed to use this repo.** The build is done once, here; users
download static files.

## The trap that cost us an hour

**Serve the WebUI in the ROOT path space, never under a prefix.**

The bundle fetches `/v1/*`, `/props`, `/tools` with *absolute* paths and its own
assets with *relative* ones (`./_app/...`). Under `/ui/` both go wrong at once,
and the app does not say so: it just renders the composer with
`disabled` + `cursor-not-allowed`, which reads like "the server is not
reachable". Symptom seen in the server log was `POST /ui/tools 404`.

`server.py` therefore serves the shell at `/` and the WebUI via `/?embed=1`,
which keeps every path resolving exactly as it does when the WebUI is served
alone.

## Rebuilding after an upstream update

1. Re-run the three commands above at the new commit.
2. Replace `upstream/` wholesale; regenerate `SHA256SUMS.txt`; update the commit
   hash at the top of this file.
3. Check `index.html` still has a `</head>` — not for injection (we no longer
   inject anything) but because the shell's iframe depends on the file being
   served as-is at the root.
4. Re-run `python frontend/server.py` and confirm the composer is *enabled*
   (type a character into it) before committing.

## Why we no longer inject an overlay

The first design injected `overlay/collabosm.{js,css}` into `index.html` with a
`<link>` + `<script>` pair. It worked, but it made our diff against upstream
non-zero and coupled us to their DOM. The shell keeps the diff at zero: the
WebUI is an iframe, and the card picker / progress / metrics live in our own
right-hand rail. A side benefit: the chat stream never has to be held open for
ten minutes while a box boots — progress is drawn in the rail, and the chat is
just there when the card is ready.