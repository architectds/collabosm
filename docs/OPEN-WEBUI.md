# Open WebUI against collabosm, measured

The division of labour is deliberate: **Open WebUI owns chat, this repo owns the adapter and the
dashboard.** Open WebUI already has multi-conversation, folders, search and a model picker; the
things that are genuinely missing -- wire-level prefill/decode, vision availability, the image
policy, which launch parameters the server is actually running -- are on `/status` and nowhere
else. So do not rebuild chat. Point Open WebUI at the proxy and keep the status page beside it.

## Install (E: drive, no C: growth)

C: was down to ~29 GB free, so every byte goes to E:. The uv tool environment, the binary and the
package cache all move; only the Python 3.11 interpreter is reused from where uv already had it.

```powershell
$uv = 'C:\Users\Chen Bao\AppData\Local\hermes\bin\uv.exe'
New-Item -ItemType Directory -Force -Path E:\open-webui, E:\uv-cache | Out-Null
$env:UV_TOOL_DIR      = 'E:\open-webui\tools'
$env:UV_TOOL_BIN_DIR  = 'E:\open-webui\bin'
$env:UV_CACHE_DIR     = 'E:\uv-cache'
& $uv tool install open-webui --python 3.11
```

Measured 2026-09-25: Open WebUI 0.11.4, **1.82 GB** in `E:\open-webui`, **2.07 GB** uv cache in
`E:\uv-cache`, C: free space unchanged. Python 3.11 is required -- 3.12 is still refused by the
dependency set.

## Run

`E:\open-webui\start-mock.ps1` starts it against the local mock. For a real endpoint change
`OPENAI_API_BASE_URL` to the proxy (`http://127.0.0.1:8790/v1`), never to the tunnel: the proxy
holds the bearer key, and the UI is same-origin with the status page.

```powershell
$env:DATA_DIR            = 'E:\open-webui\data'      # sqlite, uploads, cache -- stays on E:
$env:OPENAI_API_BASE_URL = 'http://127.0.0.1:8790/v1'
$env:OPENAI_API_KEY      = 'sk-anything'             # the proxy injects the real key
$env:WEBUI_SECRET_KEY    = '<per-install random>'
$env:OFFLINE_MODE        = 'True'                    # no HF model pulls behind your back
$env:WEBUI_NAME          = 'collabosm'
E:\open-webui\bin\open-webui.exe serve --host 127.0.0.1 --port 8080
```

First boot runs ~30 alembic migrations and takes about 3 minutes before `/health` answers
`{"status":true}`. First signup becomes the admin:

```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8080/api/v1/auths/signup `
  -ContentType application/json `
  -Body (@{ name='collabosm'; email='you@local.test'; password='<pw>' } | ConvertTo-Json)
```

Then: Admin Settings -> Connections -> OpenAI -> base URL `http://127.0.0.1:8790/v1`, any key.
The model appears as `qwen3.8-flash-next-exl3`.

## What was actually verified

Driven with Playwright against headless Chrome, not by hand:

1. attach a PNG in the composer -> thumbnail chip renders in the composer
2. send -> the image reaches the engine as an embedding, not as text: the engine-side record holds
   `"embeddings": 1` and the prompt ends `...one short sentence.<|image_pad|>`
3. the picture renders inside the user bubble in the transcript

This is the same path the EXL3 server refuses loudly (`400 vision_unavailable`) when `VISION=1` is
not set, so a silent text-only answer is not reachable from this UI.

## Costs and frictions worth knowing before a metered run

- **Every message triggers extra model calls.** Open WebUI runs title generation, follow-up
  suggestions and tag extraction in the background: four extra engine calls per turn were observed
  on a one-line prompt. On a metered A100 that is real money and real prefill time. Turn the task
  model off in Admin Settings -> Interface if you do not want them.
- **A release-notes modal opens on the first page load** and swallows clicks. Any scripted browser
  run has to dismiss or remove `[role=dialog]` first.
- **`temperature` / `top_p` are fake knobs here** -- `api_server.py` builds the job with
  `sampler=None` and drops them. Sliders exist in the UI; the values do nothing.
- **Vision capability is not toggled in the model config**; the upload path still delivers the
  image (verified above). If a future Open WebUI version starts gating on the capability flag and
  images stop arriving, that flag in Admin Settings -> Models is the first thing to set.
- CORS is `*` by default in this build. It only listens on loopback, but do not expose 8080.