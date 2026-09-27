#!/usr/bin/env python3
"""Minimal OpenAI-compatible server for ExLlamaV3 (collabosm).

Why this exists
---------------
exllamav3 v1.5.1 ships NO server: `examples/` contains chat.py / chat_console.py and
nothing that speaks HTTP, and TabbyAPI (the usual front end) does not expose the
parameters this setup depends on -- the pinned-RAM second-tier KV page cache
(`-ccs`), the recurrent checkpoint store (`-rcs`) and the generator chunk size
(`-gcs`). So this is a deliberately small server built on the library the rest of
the kit already uses.

Two details that are easy to get wrong and are load-bearing here:

1. ONE long-lived Generator. The PageTable and the CPU page cache are constructed
   inside Generator.__init__, so a Generator per request silently throws away prompt
   caching and the warm KV tier. This process builds exactly one and keeps it.
2. Requests are serialised. One Generator owns one cache of `cache_size` tokens, and
   `max_num_tokens` is the budget across concurrent jobs, so the lock is not a
   limitation of the server, it is the shape of the engine.

Surface: GET /health, GET /v1/models, POST /v1/chat/completions (stream and
non-stream). That is all on purpose -- this is how you point a client at the
measured-best configuration, not a production gateway.
"""
from __future__ import annotations

import argparse
import base64
import io
import ipaddress
import json
import os
import re
import secrets
import shutil
import socket
import subprocess
import sys
import threading
import time
import traceback
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MODEL_DIR = os.environ.get("MODEL_DIR", "/content/exl3")
MODEL_ID = os.environ.get("MODEL_ID", "qwen3.8-flash-next-exl3")
SRC = os.environ.get("EXL3_SRC", "/content/exllamav3-src")
if os.path.isdir(os.path.join(SRC, "examples")):
    sys.path.insert(0, os.path.join(SRC, "examples"))

from exllamav3 import Generator, model_init  # noqa: E402

try:
    from exllamav3 import Model as ExModel  # noqa: E402
except Exception:  # pragma: no cover - older builds
    ExModel = None

try:
    from exllamav3.generator import Job  # noqa: E402
except Exception:  # pragma: no cover - older builds
    try:
        from exllamav3.generator.job import Job  # noqa: E402
    except Exception:
        Job = None

try:
    from jinja2 import Template as JTemplate
except Exception:
    JTemplate = None

LOCK = threading.Lock()
GEN = None
TOK = None
ARGS = None
STOP_IDS = []
SERVER_STARTED = 0

# The vision tower is a SEPARATE component: model_init.init() loads only "text" (or
# "mtp"), so a multimodal pack answers as text-only unless we load it ourselves.
# Off by default: this box already sits at 76.4/81.9 GiB, and the tower's VRAM cost
# has not been measured here yet. VISION=1 enables it.
VISION = None
VISION_ERR = None
VISION_WANTED = os.environ.get("VISION", "0") == "1"

# Image input is data: URLs ONLY by default. This server is published through a
# tunnel, so fetching a caller-supplied URL would be an SSRF primitive aimed at the
# VM's own metadata service -- and reading a caller-supplied path would be a local
# file read. IMAGE_URLS=1 opts into remote fetch, still restricted to public hosts.
IMAGE_URLS = os.environ.get("IMAGE_URLS", "0") == "1"
MAX_IMAGE_BYTES = int(os.environ.get("MAX_IMAGE_BYTES", 12 * 1024 * 1024))
MAX_IMAGES = int(os.environ.get("MAX_IMAGES", 8))

# Thinking is not budgeted separately. The template's `reasoning_effort` picks how
# hard the model thinks, not how much it may spend; the only cap is the new-token
# allowance, shared with the answer. So the default has to be generous: 512 used to
# truncate long answers, and when it truncated a *thought* the half-finished
# reasoning came back as if it were the reply.
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", 32768))

# The pack's template takes enable_thinking (false pre-closes the think block, so the
# model answers directly). Off by default is what plain chat clients expect; a
# deployment that exists to serve a coding agent can set THINKING_DEFAULT=1 so every
# request that does not say otherwise gets the trace.
THINKING_DEFAULT = os.environ.get("THINKING_DEFAULT", "0").lower() in ("1", "true", "yes", "on")

# One throwaway request before the port opens (warm_up): the first prefill at a chunk
# size pays a one-time kernel autotune -- -gcs 8192 read 1,674 t/s on its first run and
# 3,882 on its second (docs/MEASURED.md) -- and the first decode captures its graphs.
# WARMUP=0 skips it.
WARMUP = os.environ.get("WARMUP", "1").lower() not in ("0", "false", "no", "off")

# The recipes.json entry this box was launched from (scripts/recipe.py -> up.sh ->
# bootstrap.sh -> serve.sh): reported, so a client knows which card+model it is.
RECIPE = os.environ.get("RECIPE") or None
# YaRN over the native window, the way the model card documents it (see apply_yarn).
# 0 = native context only.
YARN_FACTOR = float(os.environ.get("YARN_FACTOR", 0) or 0)
# Requests are serialised (module docstring, point 2). A recipe that asks for more
# is reported as asked-for and not honoured, rather than silently dropped.
CONCURRENCY = int(os.environ.get("CONCURRENCY", 1) or 1)

# What the server was launched with, for the read-only status contract.
LAUNCH = {}
ROPE = {}


class MMUnavailable(Exception):
    """A request carries an image this server cannot (or will not) embed."""


class ContextTooLong(Exception):
    """The prompt alone leaves no room in the cache for an answer."""

# --- auth ---------------------------------------------------------------------
# serve.sh creates /content/api-key.txt and then publishes the port through a
# quick tunnel. An open tunnel to an 80 GB GPU is not something to ship by
# accident, so the key is enforced whenever it exists. Set COLLABOSM_NO_AUTH=1
# for a loopback-only smoke test.
KEY_FILE = os.environ.get("KEY_FILE", "/content/api-key.txt")
NO_AUTH = os.environ.get("COLLABOSM_NO_AUTH", "0") == "1"
API_KEY = None
if not NO_AUTH and os.path.exists(KEY_FILE):
    try:
        API_KEY = open(KEY_FILE).read().strip() or None
    except Exception as exc:
        print("[api] could not read %s: %r" % (KEY_FILE, exc), flush=True)


def discover_stop_ids():
    """Stop tokens for this model.

    config.json in this pack declares eos_token_id = null; the real EOS list lives
    in the HF-style generation_config.json ([248046 <|im_end|>, 248044
    <|endoftext|>]), which exllamav3 does not read. Without stop conditions the only
    way generation ends is max_new_tokens, and the model will happily write the next
    turn's markup as ordinary text instead of stopping.
    """
    ids = set()
    path = os.path.join(MODEL_DIR, "generation_config.json")
    if os.path.exists(path):
        try:
            e = json.load(open(path)).get("eos_token_id")
            if isinstance(e, int):
                ids.add(int(e))
            elif isinstance(e, (list, tuple)):
                ids.update(int(x) for x in e)
        except Exception as exc:
            print("[api] generation_config.json unreadable: %r" % exc, flush=True)
    for tok in ("<|im_end|>", "<|endoftext|>"):
        try:
            enc = TOK.encode(tok, encode_special_tokens=True)
            vals = enc.flatten().tolist() if hasattr(enc, "flatten") else list(enc)
            ids.update(int(v) for v in vals)
        except Exception:
            pass
    return sorted(ids)


def clean_completion(text):
    """Cut the trailing markup a non-stopping model writes as plain text."""
    for marker in ("<|im_end|>", "<|endoftext|>", "<|im_start|>", "<|endofprompt|>"):
        i = text.find(marker)
        if i >= 0:
            text = text[:i]
    return text.strip()


THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"


def template_kwargs_from(req):
    """Thinking controls, as this pack's own template expects them.

    The template takes `enable_thinking` (false pre-closes the think block, so the
    model answers directly) and `reasoning_effort` (xhigh | medium | low). Default
    is THINKING_DEFAULT (OFF unless the deployment asked for it); turn it on with
    enable_thinking: true or a reasoning_effort value.

    Three shapes arrive in practice and all three are read here:

    * `enable_thinking` / `chat_template_kwargs.reasoning_effort` -- this pack's own,
    * a flat `reasoning_effort` -- what several OpenAI-compatible clients send,
    * `reasoning: {"effort": ...}` -- the OpenAI Responses shape, and the one Codex
      actually sends. Reading only the flat field is why a Codex turn ran with
      thinking off while the header said "reasoning effort: xhigh".
    """
    kw = {"enable_thinking": THINKING_DEFAULT}
    if isinstance(req.get("enable_thinking"), bool):
        kw["enable_thinking"] = req["enable_thinking"]

    def effort(eff):
        # OpenAI's scale (minimal/low/medium/high, and xhigh) onto this template's
        # three levels; "high" -- Codex's usual -- is thinking at the top level, not
        # thinking off
        if not isinstance(eff, str) or not eff:
            return
        eff = eff.strip().lower()
        if eff in ("none", "minimal"):
            kw["enable_thinking"] = False
            kw.pop("reasoning_effort", None)
            return
        kw["enable_thinking"] = True
        kw["reasoning_effort"] = {"low": "low", "medium": "medium",
                                  "high": "xhigh", "xhigh": "xhigh"}.get(eff, "medium")

    ck = req.get("chat_template_kwargs")
    if isinstance(ck, dict):
        if isinstance(ck.get("enable_thinking"), bool):
            kw["enable_thinking"] = ck["enable_thinking"]
        effort(ck.get("reasoning_effort"))
    effort(req.get("reasoning_effort"))
    rsn = req.get("reasoning")
    if isinstance(rsn, dict):
        effort(rsn.get("effort"))
    return kw


# Markers this pack can emit around or instead of an answer that are not text a
# client should ever see. The closing think tag is handled by split_thinking.
STRAY_MARKERS = ("<|end|>", "<|im_end|>", "<|im_start|>", "<|endoftext|>",
                 "<|eot_id|>", "<|start_header_id|>", "<|end_header_id|>")


def split_thinking(text, thinking_on=False):
    """(reasoning, answer), reported the way Qwen/DeepSeek endpoints do.

    Split on the LAST </think>: when thinking is disabled the template pre-closes
    the think block, and the model can echo another closing marker, so the first
    one is not necessarily the real boundary. Any leftover marker is stripped from
    both halves.

    With thinking on, the template has already written the opening <think>, so a
    reply with no closing tag is still inside the block -- mid-thought. That matters
    most for the streaming path, where a thought mislabelled as an answer has
    already been sent as answer text before the mistake can be seen.
    """
    t = text or ""
    i = t.rfind(THINK_CLOSE)
    if i >= 0:
        reasoning, answer = t[:i], t[i + len(THINK_CLOSE):]
    elif thinking_on or t.lstrip().startswith(THINK_OPEN):
        reasoning, answer = t, ""
    else:
        reasoning, answer = "", t
    for marker in (THINK_OPEN, THINK_CLOSE):
        reasoning = reasoning.replace(marker, "")
        answer = answer.replace(marker, "")
    for marker in STRAY_MARKERS:
        reasoning = reasoning.replace(marker, "")
        answer = answer.replace(marker, "")
    return reasoning.strip(), answer.strip()


def truncated(r):
    return (r.get("eos_reason") or "") == "max_new_tokens"


def prompt_opens_think(prompt):
    """Did the rendered prompt leave the model *inside* a think block?

    That is a fact about the prompt, not about the request's flags: this pack's
    template writes the opening <think> itself when thinking is on, and pre-closes
    the block when it is off, so the tail of the prompt is the authority on which
    state we are in. It is also the only signal that survives a request that asked
    for one thing and a template that rendered another.
    """
    return bool(re.search(r"<think>\s*$", prompt or ""))


def split_result(r, thinking_on):
    """(reasoning, answer) for a finished generation, honest about truncation.

    A reply that stopped on max_new_tokens without ever closing the think block is a
    half-finished thought. Handing that back as `output_text` with status
    "completed" is a lie the client cannot detect, so it stays in reasoning and the
    caller marks the response incomplete.
    """
    text = (r.get("text") or "")
    if thinking_on and truncated(r) and THINK_CLOSE not in text:
        return split_thinking(text, True)[0], ""
    return split_thinking(text, thinking_on)


def assistant_delta(text, reasoning):
    d = {"role": "assistant"}
    if reasoning:
        d["reasoning_content"] = reasoning
    d["content"] = text
    return d


def assistant_message(text, reasoning):
    msg = {"role": "assistant", "content": text}
    if reasoning:
        msg["reasoning_content"] = reasoning
    return msg


def _on(value, default=False) -> bool:
    if value is None or str(value).strip() == "":
        return default
    return str(value).strip().lower() in ("1", "true", "yes", "on")


def engine_argv(env, model_dir):
    """model_init's command line for one launch, and what it amounts to.

    Pure, so recipes.json can be checked without a GPU. The n-gram table and the MTP
    layer belong to the model, not the card: `-ngr` is only for a pack that ships
    ngram_embedding.safetensors (Flash-Next does, the 27B does not), and NGRAM
    unset means exactly that test.
    """
    gcs = int(env.get("GCS", 4096))
    ndt = int(env.get("NDT", 4))
    ccs = float(env.get("CPU_CACHE_GB", 0) or 0)
    rcs = float(env.get("RECURRENT_CACHE_GB", 4) or 0)
    ngram = str(env.get("NGRAM", "auto") or "auto").lower()
    ngram_on = (os.path.exists(os.path.join(model_dir, "ngram_embedding.safetensors"))
                if ngram == "auto" else _on(ngram))
    mtp_on = _on(env.get("MTP"), default=True)
    argv = ["serve", "-m", model_dir,
            "-cs", str(env.get("CACHE_SIZE", 262144)),
            "-cq", str(env.get("CACHE_QUANT", "4"))]
    if ngram_on:
        argv.append("-ngr")               # the n-gram/PLE table lives in host RAM
    if mtp_on:
        argv += ["-mtp", "-ndt", str(ndt)]
    argv += ["-gcs", str(gcs), "-mode", env.get("CHAT_MODE", "chatml")]
    if ccs:
        argv += ["-ccs", str(ccs)]        # pinned-RAM second-tier KV page cache
    if rcs:
        argv += ["-rcs", str(rcs)]        # GDN checkpoint store (host RAM)
    return argv, {
        "cache_size": int(env.get("CACHE_SIZE", 262144)),
        "cache_quant": str(env.get("CACHE_QUANT", "4")),
        "generator_chunk_size": gcs,
        "num_draft_tokens": ndt if mtp_on else 0,
        "cpu_cache_gb": ccs,
        "recurrent_cache_gb": rcs,
        "mode": env.get("CHAT_MODE", "chatml"),
        "mtp": mtp_on,
        "ngram": ngram_on,
    }


def apply_yarn(model_dir, factor):
    """Extend the context with YaRN, as the model card documents it: the text
    model's `rope_parameters` (older packs: `rope_scaling`) becomes rope_type "yarn"
    with `factor` over the native window, every other RoPE key kept (theta, partial
    rotary factor, the interleaved mRoPE sections), and max_position_embeddings
    raised to match. There is no command-line way: model_init has no RoPE flag.

    The pack's own config.json is kept once as config.json.orig and every launch is
    derived from it, so factor 0 (or a recipe without YaRN) puts the native RoPE
    back: static YaRN costs a little on short prompts, the card warns, and a box must
    not keep it after its recipe stops asking for it.
    """
    path = os.path.join(model_dir, "config.json")
    orig = path + ".orig"
    if not os.path.exists(path):
        return {"yarn": False, "why": "no config.json"}
    restored = factor <= 1 and os.path.exists(orig)
    if restored:
        shutil.copyfile(orig, path)
    with open(orig if os.path.exists(orig) else path, encoding="utf-8") as fh:
        cfg = json.load(fh)
    text = cfg["text_config"] if isinstance(cfg.get("text_config"), dict) else cfg
    # exllamav3 reads rope_scaling before rope_parameters (the first non-null wins),
    # so a pack that carries both is patched where it will be read
    key = ("rope_scaling" if text.get("rope_scaling")
           else "rope_parameters" if "rope_parameters" in text else "rope_scaling")
    params = dict(text.get(key) or {})
    native = int(text.get("max_position_embeddings") or 262144)
    if params.get("rope_type") == "yarn" and params.get("original_max_position_embeddings"):
        # already patched with no .orig beside it (a copied directory): its native
        # window is the recorded one, not the raised one -- or the factor compounds
        native = int(params["original_max_position_embeddings"])
    if factor <= 1:
        return dict({"yarn": False, "native": native, "max": native},
                    **({"restored": True} if restored else {}))
    if not os.path.exists(orig):
        shutil.copyfile(path, orig)
    params.update(rope_type="yarn", factor=float(factor),
                  original_max_position_embeddings=native)
    text[key] = params
    # exllamav3 (util/rope.py) derives the factor as max_position_embeddings /
    # original_max_position_embeddings whenever the latter is present, and ignores
    # `factor`: the card's block alone is 262144/262144 = 1.0, a silent no-op. So the
    # window is raised too, which every other reader agrees with.
    text["max_position_embeddings"] = int(native * factor)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(cfg, fh, indent=2)
    os.replace(tmp, path)
    return {"yarn": True, "factor": float(factor), "native": native,
            "max": int(native * factor), "key": key}


def build_engine():
    """Load once, with the flags the recipe carries (recipes.json)."""
    global GEN, TOK, ARGS, STOP_IDS, SERVER_STARTED
    SERVER_STARTED = int(time.time())
    parser = argparse.ArgumentParser(allow_abbrev=False)
    model_init.add_args(parser, cache=True, add_sampling_args=True,
                        add_draft_model_args=True,
                        default_cache_size=int(os.environ.get("CACHE_SIZE", 262144)),
                        default_autosplit_max_batch_size=1)
    for flags, kw in [
        (["-mode", "--mode"], dict(type=str, default="chatml")),
        (["-gcs", "--generator_chunk_size"], dict(type=int, default=4096)),
    ]:
        parser.add_argument(*flags, **kw)

    ROPE.update(apply_yarn(MODEL_DIR, YARN_FACTOR))
    print("[api] rope: %s" % ROPE, flush=True)
    if CONCURRENCY > 1:
        print("[api] CONCURRENCY=%d asked for, but requests are serialised: serving 1 at a "
              "time" % CONCURRENCY, flush=True)
    argv, launch = engine_argv(os.environ, MODEL_DIR)
    sys.argv = argv
    ARGS = parser.parse_args()
    gcs = launch["generator_chunk_size"]
    ccs, rcs = launch["cpu_cache_gb"], launch["recurrent_cache_gb"]
    LAUNCH.update(launch)
    LAUNCH.update({"recipe": RECIPE, "model": MODEL_ID, "vision": VISION_WANTED,
                   "yarn_factor": YARN_FACTOR if ROPE.get("yarn") else 0,
                   "concurrency": 1, "concurrency_requested": CONCURRENCY})

    t0 = time.time()
    loaded = model_init.init(ARGS)
    model, config, cache, tokenizer = loaded[0], loaded[1], loaded[2], loaded[3]
    draft_model, draft_cache = (loaded[4], loaded[6]) if len(loaded) >= 7 else (None, None)
    print("[api] model loaded in %.1fs, cache=%d tokens, draft=%s"
          % (time.time() - t0, cache.max_num_tokens, draft_model is not None), flush=True)

    # ONE Generator for the process lifetime -- see the module docstring.
    GEN = Generator(model=model, cache=cache, tokenizer=tokenizer,
                    draft_model=draft_model, draft_cache=draft_cache,
                    max_chunk_size=gcs,
                    cpu_cache_size=int(ccs * 1024 ** 3),
                    recurrent_cache_size=int(rcs * 1024 ** 3))
    TOK = tokenizer
    STOP_IDS = discover_stop_ids()
    print("[api] stop ids: %s" % STOP_IDS, flush=True)
    print("[api] generator ready (cpu tier: %s)"
          % (getattr(GEN, "cpu_page_cache", None) is not None), flush=True)

    global VISION, VISION_ERR
    prep = os.path.join(MODEL_DIR, "preprocessor_config.json")
    if not VISION_WANTED:
        print("[api] vision is OFF (set VISION=1 to load the tower). "
              "Image parts will be refused, never silently dropped.", flush=True)
    elif ExModel is None:
        VISION_ERR = "this exllamav3 build has no Model(component=...) API"
        print("[api] vision unavailable: %s" % VISION_ERR, flush=True)
    elif not os.path.exists(prep):
        VISION_ERR = ("no preprocessor_config.json in %s: this pack carries no "
                      "vision tower" % MODEL_DIR)
        print("[api] vision unavailable: %s" % VISION_ERR, flush=True)
    else:
        try:
            t_v = time.time()
            VISION = ExModel.from_config(config, component="vision")
            VISION.load(progressbar=False)
            print("[api] vision component loaded in %.1fs" % (time.time() - t_v),
                  flush=True)
        except Exception as exc:
            VISION = None
            VISION_ERR = repr(exc)
            print("[api] vision component FAILED to load: %s" % VISION_ERR, flush=True)


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """One hop only: a redirect must not be able to bounce us to an internal host
    after the address check has already passed."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _public_only(url):
    """Reject anything that is not an ordinary public address."""
    host = urllib.parse.urlsplit(url).hostname
    if not host:
        raise MMUnavailable("image URL has no host")
    try:
        infos = socket.getaddrinfo(host, None)
    except OSError as exc:
        raise MMUnavailable("cannot resolve image host %r: %r" % (host, exc))
    for info in infos:
        try:
            ip = ipaddress.ip_address(info[4][0])
        except ValueError:
            raise MMUnavailable("unparseable address for %r" % host)
        if not ip.is_global:
            raise MMUnavailable(
                "refusing to fetch %s: %s is not a public address (private, "
                "loopback, link-local or metadata)" % (url, ip))
    return url


def _fetch_image(url):
    _public_only(url)
    opener = urllib.request.build_opener(_NoRedirect)
    req = urllib.request.Request(url, headers={"User-Agent": "collabosm"})
    try:
        with opener.open(req, timeout=20) as resp:
            raw = resp.read(MAX_IMAGE_BYTES + 1)
    except Exception as exc:
        raise MMUnavailable("could not fetch image: %r" % exc)
    if len(raw) > MAX_IMAGE_BYTES:
        raise MMUnavailable("image exceeds MAX_IMAGE_BYTES (%d)" % MAX_IMAGE_BYTES)
    return raw


def _prepare_ref(ref):
    """Validate an image reference -> ("data", bytes) | ("url", url).

    Policy is checked here, deliberately BEFORE the capability check: a caller who
    sends a filesystem path should be told the reference is unsupported whether or
    not this server happens to have vision, and a remote URL must be refused even
    when no tower is loaded (otherwise a URL would be fetched and only then
    rejected). Only data: URLs are accepted unless IMAGE_URLS=1, and a caller can
    never make this server read its own disk or reach the cloud metadata service.
    """
    if isinstance(ref, dict):
        ref = ref.get("url") or ref.get("data") or ref.get("b64_json")
    if not isinstance(ref, str) or not ref.strip():
        raise MMUnavailable("image part carries no url/data")
    ref = ref.strip()
    if ref.startswith("data:"):
        _, _, payload = ref.partition(",")
        try:
            raw = base64.b64decode(payload, validate=False)
        except Exception as exc:
            raise MMUnavailable("image data URL is not valid base64: %r" % exc)
        if len(raw) > MAX_IMAGE_BYTES:
            raise MMUnavailable("image exceeds MAX_IMAGE_BYTES (%d)" % MAX_IMAGE_BYTES)
        return ("data", raw)
    if ref.startswith(("http://", "https://")):
        if not IMAGE_URLS:
            raise MMUnavailable(
                "remote image URLs are disabled (set IMAGE_URLS=1 to allow them). "
                "Send the image as a base64 data: URL instead.")
        return ("url", ref)
    raise MMUnavailable("unsupported image reference: expected a data: URL"
                        + (" or a public http(s) URL" if IMAGE_URLS else ""))


def _image_embedding(ref):
    """Embed one image -> (MMEmbedding, the prompt alias that stands in for it).

    The alias matters: it is inserted as text where the image belongs, and
    tokenizer.encode(..., embeddings=[...]) expands it into the placeholder span.
    """
    kind, payload = _prepare_ref(ref)
    if VISION is None:
        raise MMUnavailable(VISION_ERR or (
            "vision is not enabled on this server (VISION=1) -- refusing to answer "
            "as if the image were text"))
    try:
        from PIL import Image
    except Exception as exc:
        raise MMUnavailable("image input needs Pillow in the runtime: %r" % exc)
    raw = payload if kind == "data" else _fetch_image(payload)
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    # The tower runs on the same GPU, with the same tokenizer, as generation: under
    # the engine lock, not beside another request's decode (VRAM at these cache
    # sizes has no room for the two at once)
    with LOCK:
        try:
            ie = VISION.get_image_embeddings(tokenizer=TOK, image=img)
        except TypeError:
            ie = VISION.get_image_embeddings(TOK, img)
    alias = getattr(ie, "text_alias", None)
    if not alias:
        raise MMUnavailable("the vision model returned no prompt alias for this image")
    return ie, alias


def flatten_parts(content, embs):
    """Fold OpenAI content parts into one string, inlining image aliases."""
    if not isinstance(content, list):
        return content
    out = []
    for part in content:
        if isinstance(part, str):
            out.append(part)
            continue
        if not isinstance(part, dict):
            continue
        ptype = (part.get("type") or "").lower()
        if ptype in ("image_url", "input_image", "image") or "image_url" in part:
            if len(embs) >= MAX_IMAGES:
                raise MMUnavailable("too many images in one request (MAX_IMAGES=%d)"
                                    % MAX_IMAGES)
            ie, alias = _image_embedding(part.get("image_url")
                                         or part.get("image") or part.get("url"))
            embs.append(ie)
            out.append(alias)
            continue
        out.append(part.get("text") or "")
    return "".join(out)


def extract_images(req, embs):
    """Rewrite image parts in place so they survive the text pipeline.

    Covers both dialects: chat `content` parts and Responses `input` items. Raises
    MMUnavailable rather than dropping an image, because answering as if the picture
    were text is exactly the bug this replaced.
    """
    for key in ("messages", "input"):
        items = req.get(key)
        if not isinstance(items, list):
            continue
        for item in items:
            if not isinstance(item, dict):
                continue
            kind = item.get("type")
            # Messages only: a reasoning item's content is its thinking, replayed as
            # such (flattening it here made the replay find a string and skip it).
            if kind in (None, "message") and isinstance(item.get("content"), list):
                item["content"] = flatten_parts(item["content"], embs)
            # A tool's output can carry an image too (Codex's view_image): it goes to
            # the model like any other image instead of vanishing from the tool result.
            elif kind in ("function_call_output", "custom_tool_call_output") \
                    and isinstance(item.get("output"), list):
                item["output"] = flatten_parts(item["output"], embs)


_TEMPLATE = {"src": None, "compiled": None}


def _chat_template():
    """The pack's template, compiled the way transformers compiles chat templates:
    trimmed blocks, JSON without HTML escaping (tool schemas are full of < and >),
    and the raise_exception the template calls."""
    path = os.path.join(MODEL_DIR, "chat_template.jinja")
    if JTemplate is None or not os.path.exists(path):
        return None
    src = open(path, encoding="utf-8").read()
    if _TEMPLATE["src"] != src:
        import jinja2
        from jinja2.sandbox import ImmutableSandboxedEnvironment

        def raise_exception(message):
            raise jinja2.exceptions.TemplateError(message)

        env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
        env.filters["tojson"] = lambda x, indent=None, separators=None, sort_keys=False: \
            json.dumps(x, ensure_ascii=False, indent=indent, separators=separators,
                       sort_keys=sort_keys)
        env.globals["raise_exception"] = raise_exception
        env.globals["strftime_now"] = lambda fmt: time.strftime(fmt)
        _TEMPLATE.update(src=src, compiled=env.from_string(src))
    return _TEMPLATE["compiled"]


def render_chat(messages, template_kwargs=None, tools=None):
    """Prefer the model's own chat template; fall back to ChatML."""
    tpl = _chat_template()
    if tpl is not None:
        try:
            kw = {"messages": messages, "add_generation_prompt": True}
            if template_kwargs:
                kw.update(template_kwargs)
            if tools:
                kw["tools"] = tools
            out = tpl.render(**kw)
            if isinstance(out, str) and out.strip():
                return out, "jinja"
        except Exception as exc:
            print("[api] chat template failed (%r); falling back to ChatML" % exc, flush=True)
    parts = []
    for m in messages:
        parts.append("<|im_start|>%s\n%s<|im_end|>\n"
                     % (m.get("role", "user"), m.get("content", "") or ""))
    parts.append("<|im_start|>assistant\n")
    return "".join(parts), "chatml"


LAST = {}

# What the server has been asked to do, for the frontend's idle auto-stop: a client
# talking to the tunnel directly (Codex, ModelDock) is activity too, and only this
# process sees it. Generation requests only -- health and status polls are not use.
ACTIVITY = {"requests": 0, "last_request_at": None, "in_flight": 0}
ACTIVITY_LOCK = threading.Lock()
LAST_TIMINGS = {}


_MACHINE = {"at": 0.0, "data": None}


def _num(text):
    try:
        return float(text)
    except (TypeError, ValueError):
        return None                                  # nvidia-smi says "[N/A]"


def machine():
    """What the box has left -- VRAM, RAM, disk -- for the frontend's rail.

    Served here because this process is the one thing reachable through the
    tunnel: `colab exec` can hang, and `colab download` cannot run nvidia-smi.
    Cached for 20 s, since nvidia-smi is a subprocess and several clients may poll.
    """
    now = time.time()
    if _MACHINE["data"] is not None and now - _MACHINE["at"] < 20:
        return _MACHINE["data"]
    d = {"at": int(now)}
    try:
        row = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.used,memory.total,utilization.gpu,"
             "temperature.gpu,power.draw", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5).stdout.strip().splitlines()[0]
        name, used, total, util, temp, power = [x.strip() for x in row.split(",")]
        d["gpu"] = {"name": name, "used_mib": _num(used), "total_mib": _num(total),
                    "util_pct": _num(util), "temp_c": _num(temp), "power_w": _num(power)}
    except Exception:
        d["gpu"] = None
    try:
        info = {}
        with open("/proc/meminfo") as fh:
            for line in fh:
                k, _, v = line.partition(":")
                info[k] = int(v.split()[0]) * 1024
        d["ram"] = {"used_gib": round((info["MemTotal"] - info["MemAvailable"]) / 2 ** 30, 1),
                    "total_gib": round(info["MemTotal"] / 2 ** 30, 1)}
    except Exception:
        d["ram"] = None
    try:
        du = shutil.disk_usage("/content" if os.path.isdir("/content") else "/")
        d["disk"] = {"free_gib": round(du.free / 2 ** 30, 1), "total_gib": round(du.total / 2 ** 30, 1)}
    except Exception:
        d["disk"] = None
    try:
        with open("/proc/uptime") as fh:
            d["uptime_s"] = int(float(fh.read().split()[0]))
    except Exception:
        d["uptime_s"] = None
    _MACHINE.update(at=now, data=d)
    return d


def _timings(r, t_start, t_first, t_end):
    """llama.cpp's `timings` block -- the WebUI already knows how to show it.

    prompt_ms runs from the request to the first decoded fragment, so it includes
    queueing behind the lock and prefill; predicted_ms runs from there to the end.
    prompt_n excludes the cached prefix, as llama.cpp counts it.
    """
    cached = int(r.get("cached_tokens") or 0)
    prompt_n = max(0, int(r.get("prompt_tokens") or 0) - cached)
    new = int(r.get("new_tokens") or 0)
    prompt_ms = max(0.0, (t_first - t_start) * 1000.0)
    predicted_ms = max(0.0, (t_end - t_first) * 1000.0)
    return {"cache_n": cached,
            "prompt_n": prompt_n, "prompt_ms": round(prompt_ms, 1),
            "prompt_per_second": (round(prompt_n / prompt_ms * 1000.0, 1)
                                  if prompt_n and prompt_ms else None),
            "predicted_n": new, "predicted_ms": round(predicted_ms, 1),
            "predicted_per_second": (round(new / predicted_ms * 1000.0, 1)
                                     if new and predicted_ms else None)}


def _engine_fragments(prompt, max_tokens, temperature=None, top_p=None, stops=None,
                      embeddings=None):
    """Yield decoded text fragments from exllamav3 as they are produced.

    Real streaming goes through enqueue() + iterate(): iterate() reports a
    "streaming" stage result for every decode step, so the HTTP layer can forward
    text while the job is still running. generate() cannot do that -- it
    accumulates the completion internally and returns only once the job is done,
    which is why this server used to go silent and then emit one blob.

    Every exllamav3-version-specific detail lives in here. Callers only ever see a
    fragment iterator, and the terminal metrics land in LAST.
    """
    stops_all = list(STOP_IDS) + ["<|im_end|>"] + list(stops or [])
    LAST.clear()
    if embeddings:
        # Special-token encoding is on for images: the alias IS a special token, and
        # only this call expands it into the span the embeddings line up with.
        try:
            input_ids = TOK.encode(prompt, encode_special_tokens=True, add_bos=True,
                                   embeddings=embeddings)
        except TypeError:
            input_ids = TOK.encode(prompt, add_bos=True, embeddings=embeddings)
    else:
        try:
            input_ids = TOK.encode(prompt, encode_special_tokens=False, add_bos=True)
        except TypeError:
            input_ids = TOK.encode(prompt, add_bos=True)
    n_prompt = int(input_ids.shape[-1]) if hasattr(input_ids, "shape") else len(input_ids)
    LAST["prompt_tokens"] = n_prompt
    # exllamav3 reserves prompt + max_new_tokens + 1 + draft depth up front and
    # refuses a job that cannot fit the whole cache, so a generous default allowance
    # would make the top of the context unusable. Answer as much as fits instead.
    room = answer_room(n_prompt)
    if room is not None:
        if room < 1:
            raise ContextTooLong("the prompt is %d tokens; this server's context is %d tokens, "
                                 "which leaves no room for an answer"
                                 % (n_prompt, context_limit() or 0))
        max_tokens = min(max_tokens, room)
    # Mirror Generator.generate()'s own Job construction field for field. A Job
    # built with fewer fields behaves differently: it stopped after a single
    # token here, which is what an empty answer looks like from the client side.
    job = Job(input_ids=input_ids,
              max_new_tokens=max_tokens,
              min_new_tokens=0,
              stop_conditions=stops_all,
              sampler=None,
              filters=[],
              token_healing=False,
              decode_special_tokens=bool(embeddings),
              embeddings=list(embeddings or []),
              max_rq_tokens=None,
              stop_on_loop=None)
    serial = GEN.enqueue(job)
    finished = False
    try:
        while GEN.num_remaining_jobs():
            for r in GEN.iterate():
                if r.get("stage") == "error":
                    # A contained per-job failure. Surface it rather than returning a
                    # silently truncated completion.
                    finished = True
                    raise r["error"]
                if r.get("stage") != "streaming" or r.get("serial") != serial:
                    continue
                frag = r.get("text") or ""
                if frag:
                    yield frag
                if r.get("eos"):
                    finished = True
                    LAST["new_tokens"] = r.get("new_tokens")
                    LAST["eos_reason"] = r.get("eos_reason")
                    LAST["prompt_tokens"] = r.get("prompt_tokens") or n_prompt
                    # Without this, usage and the log line report a 0% prompt-cache hit
                    # on every request -- which is exactly what a cold Generator looks
                    # like, the one misdiagnosis this server is built to rule out.
                    LAST["cached_tokens"] = r.get("cached_tokens") or 0
    finally:
        if not finished:
            # The client went away mid-answer (or something upstream raised): cancel,
            # or the next request's loop runs this orphan to its end first -- up to
            # the whole allowance, minutes of decode nobody reads.
            try:
                GEN.cancel(job)
            except Exception as exc:
                print("[api] could not cancel an abandoned job: %r" % exc, flush=True)


def context_limit():
    """One conversation's ceiling: the cache (the budget across jobs) or the model's
    position window (native, or YaRN's), whichever is smaller. A cache larger than the
    window holds more conversations, not longer ones -- past the window RoPE has no
    positions left, and the answer is garbage. None if unknown."""
    total = getattr(getattr(GEN, "cache", None), "max_num_tokens", None)
    if not total:
        return None
    window = ROPE.get("max") or ROPE.get("native")
    return min(int(total), int(window)) if window else int(total)


def answer_room(n_prompt):
    """Tokens of answer that fit after this prompt, or None if unknown."""
    limit = context_limit()
    if not limit:
        return None
    ndt = int(LAUNCH.get("num_draft_tokens") or 0)
    # the job's page reservation: prompt + max_new + 1 + draft depth, rounded up to a
    # 256-token page -- keep one page of slack for the rounding
    return limit - int(n_prompt) - 1 - ndt - 256


def _generate_blocking(prompt, max_tokens, temperature=None, top_p=None, stops=None,
                       embeddings=None):
    """The old one-shot path, kept as the fallback for builds whose Job API differs."""
    kw = {"stop_conditions": list(STOP_IDS) + ["<|im_end|>"] + list(stops or [])}
    if embeddings:
        kw["embeddings"] = list(embeddings)
        kw["encode_special_tokens"] = True
        kw["decode_special_tokens"] = True
    if temperature is not None:
        kw["temperature"] = float(temperature)
    if top_p is not None:
        kw["top_p"] = float(top_p)
    with LOCK:
        try:
            try:
                comp, last = GEN.generate(prompt, max_new_tokens=max_tokens, add_bos=True,
                                          return_last_results=True,
                                          completion_only=True, **kw)
            except TypeError:
                try:
                    comp, last = GEN.generate(prompt, max_new_tokens=max_tokens, add_bos=True,
                                              return_last_results=True,
                                              completion_only=True)
                except TypeError:
                    comp, last = GEN.generate(prompt, max_new_tokens=max_tokens, add_bos=True,
                                              return_last_results=True)
                    if isinstance(comp, str) and comp.startswith(prompt):
                        comp = comp[len(prompt):]
        except Exception:
            raise
    if isinstance(comp, (list, tuple)):
        comp = comp[0] if comp else ""
    if not isinstance(last, dict):
        last = getattr(last, "__dict__", {}) or {}
    last = dict(last)
    last["text"] = clean_completion(comp or "")
    return last


def collect(prompt, max_tokens, temperature=None, top_p=None, stops=None,
            on_delta=None, embeddings=None):
    """The one engine entry point, for streaming and non-streaming alike.

    on_delta(cumulative_clean_text) is called while the model is still decoding;
    pass None for a one-shot generation. Returns the same metrics dict the old
    generate() did, so non-streaming callers are unchanged.
    """
    buf = []
    t_start = time.time()
    t_first = None
    try:
        with LOCK:
            frags = _engine_fragments(prompt, max_tokens, temperature, top_p, stops,
                                      embeddings)
            try:
                for frag in frags:
                    if t_first is None:
                        t_first = time.time()
                    buf.append(frag)
                    if on_delta is not None:
                        on_delta(clean_completion("".join(buf)))
            finally:
                # closed here, inside the lock: an abandoned job is cancelled by the
                # thread that owns the engine, never by a garbage collector later
                frags.close()
            # Snapshot while this request still owns the engine: the next request
            # clears LAST the moment it takes the lock.
            out = dict(LAST)
    except TypeError as exc:
        print("[api] enqueue path failed (%r); falling back to blocking generate()"
              % exc, flush=True)
        return _generate_blocking(prompt, max_tokens, temperature, top_p, stops,
                                  embeddings)
    if t_first is not None:
        # Only the incremental path has a first-token time; the blocking fallback
        # gets no timings rather than invented ones.
        out["timings"] = _timings(out, t_start, t_first, time.time())
        LAST_TIMINGS.clear()
        LAST_TIMINGS.update(out["timings"], at=int(time.time()))
    out["text"] = clean_completion("".join(buf))
    if not out["text"].strip():
        # Never answer empty. The incremental path can yield nothing if a build's
        # Job defaults differ, or if the model stops on token one; the blocking
        # path is the one that is known to work on this pack, so use it rather
        # than handing the client an empty completion.
        out["new_tokens"] = out.get("new_tokens")
        print("[api] incremental path produced no text (new=%s eos=%s); retrying "
              "with the blocking generator" % (out.get("new_tokens"),
                                               out.get("eos_reason")), flush=True)
        r = _generate_blocking(prompt, max_tokens, temperature, top_p, stops,
                               embeddings)
        print("[api] blocking fallback: new=%s eos=%s text=%r"
              % (r.get("new_tokens"), r.get("eos_reason"),
                 (r.get("text") or "")[:60]), flush=True)
        if on_delta is not None and (r.get("text") or "").strip():
            on_delta(r["text"])
        return r
    return out


def generate(prompt, max_tokens, temperature=None, top_p=None, stops=None,
             embeddings=None):
    """One-shot generation; kept because callers and diagnostics still use it."""
    return collect(prompt, max_tokens, temperature, top_p, stops,
                   embeddings=embeddings)


class _Delta:
    """Turn a cumulative completion into ordered reasoning / content / tool deltas.

    Thinking is split on the LAST </think> exactly as the non-streaming path does,
    so a client that streams and a client that does not end up with the same
    answer. A fragment that would require retracting bytes already sent is dropped
    instead: an SSE stream cannot unsend.

    With tools offered, the answer's text stops at the first <tool_call> and the
    calls stream as ("tool", event, index, payload) -- see ToolStream -- the way
    llama-server streams them: each call as soon as its name is written, then its
    arguments as they are generated, not all at once when the answer is over (a long
    call -- a whole file, a big patch -- used to leave the stream silent for minutes).
    """

    def __init__(self, thinking_on=False, kinds=None, parallel=True, raw_custom=True):
        self.thinking_on = thinking_on
        self.tools = (ToolStream(kinds, parallel, raw_custom) if kinds else None)
        self.sent_reasoning = ""
        self.sent_content = ""
        self.content_started = False

    def feed(self, accumulated, done=False, cut=False):
        reasoning, answer = split_thinking(accumulated, self.thinking_on)
        if not done and THINK_CLOSE not in accumulated:
            # mid-thought: a tail that could be the start of </think> is not thinking
            # yet -- with a tag split across fragments, its "<" used to go out as text --
            # and neither is the whitespace before it, which the finished thought loses
            # (streamed, it made the reasoning item a client replays end in "\n")
            reasoning = reasoning[:len(reasoning) - _held_tail(reasoning, THINK_CLOSE)].rstrip()
        calls = []
        if self.tools is not None:
            i = answer.find(TOOL_OPEN)
            if i >= 0:
                calls = self.tools.feed(answer[i:], final=done, cut=cut)
            # a tool call is not text: stop at it, and hold back what might be one
            # (until the answer is done: then a trailing '<' is text) -- and trailing
            # whitespace, which the answer loses if a call follows it
            answer = before_tool_call(answer, done)
            if not done and i < 0:
                answer = answer.rstrip()
        out = []
        if (not self.content_started
                and reasoning.startswith(self.sent_reasoning)
                and len(reasoning) > len(self.sent_reasoning)):
            out.append(("reasoning", reasoning[len(self.sent_reasoning):]))
            self.sent_reasoning = reasoning
        if answer.startswith(self.sent_content) and len(answer) > len(self.sent_content):
            if answer:
                self.content_started = True
            out.append(("content", answer[len(self.sent_content):]))
            self.sent_content = answer
        out.extend(("tool",) + ev for ev in calls)
        return out


# --- tools -------------------------------------------------------------------
# This pack's template renders `tools` into the system turn and teaches the model
# Qwen3-Coder's XML call format:
#
#     <tool_call>\n<function=NAME>\n<parameter=P>\nvalue\n</parameter>\n</function>\n</tool_call>
#
# Before this, `tools` never reached the template and nothing parsed that format,
# so a model that did try to call a tool had its call handed to the client as text.

TOOL_OPEN = "<tool_call>"
_FUNC_OPEN = re.compile(r"\s*<function=([^>\n]+)>")
_CALL_CLOSE = re.compile(r"\s*</tool_call>")
_JSON_BODY = re.compile(r"\s*(\{.*?\})\s*(?:</tool_call>|\Z)", re.S)


def _text_of(content):
    if isinstance(content, list):
        return "".join(p.get("text", "") for p in content if isinstance(p, dict))
    if content is None:
        return ""
    return content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)


def _arguments_dict(args):
    """The template iterates arguments with |items, so it needs a dict, while the
    wire carries a JSON string."""
    if isinstance(args, str):
        try:
            args = json.loads(args or "{}")
        except ValueError:
            return {"input": args}
    return args if isinstance(args, dict) else {}


def _tool_name(t):
    fn = t["function"] if isinstance(t.get("function"), dict) else t
    return fn.get("name") if t.get("type") == "function" else t.get("name")


def request_tools(req):
    """(tools in the template's chat-completions shape, name -> (kind, schema,
    namespace, name the client knows)).

    Chat sends {"type": "function", "function": {...}}; Responses sends flat
    function tools, Codex's freeform `custom` tools (apply_patch) -- which become
    one-string-parameter functions here and go back out as custom_tool_call items --
    and `namespace` groups (Codex's multi_agent_v1), whose tools are offered to the
    model as plain functions and go back out with their `namespace`; a name two
    groups share is offered as "<namespace>__<name>". Vendor-hosted tools
    (web_search, ...) cannot run here and are left out.
    """
    if req.get("tool_choice") == "none":
        return [], {}
    flat = []
    for t in req.get("tools") or []:
        if not isinstance(t, dict):
            continue
        if t.get("type") == "namespace" and isinstance(t.get("tools"), list):
            flat += [(t.get("name"), sub) for sub in t["tools"] if isinstance(sub, dict)]
        elif t.get("type") in ("function", "custom"):
            flat.append((None, t))
    seen = {}
    for _ns, t in flat:
        seen[_tool_name(t)] = seen.get(_tool_name(t), 0) + 1
    tools, kinds = [], {}
    for ns, t in flat:
        real = _tool_name(t)
        if not real:
            continue
        name = real if ns is None or seen[real] == 1 else "%s__%s" % (ns, real)
        if t.get("type") == "function":
            fn = t["function"] if isinstance(t.get("function"), dict) else t
            params = fn.get("parameters") or {"type": "object", "properties": {}}
            tools.append({"type": "function", "function": {
                "name": name, "description": fn.get("description") or "", "parameters": params}})
            kinds[name] = ("function", params, ns, real)
        else:
            desc = t.get("description") or ""
            fmt = t.get("format") if isinstance(t.get("format"), dict) else {}
            if fmt.get("definition"):
                desc += "\n\nThe input must follow this %s:\n%s" % (
                    fmt.get("syntax") or "grammar", fmt["definition"])
            params = {"type": "object", "required": ["input"],
                      "properties": {"input": {"type": "string",
                                               "description": "the tool's raw input"}}}
            tools.append({"type": "function", "function": {
                "name": name, "description": desc, "parameters": params}})
            kinds[name] = ("custom", params, ns, real)
    return tools, kinds


def _client_name(kinds, name):
    """(the name the client knows, its namespace or None) for a name the model used."""
    info = kinds.get(name) or ()
    return (info[3] if len(info) > 3 else name), (info[2] if len(info) > 2 else None)


def _model_name(kinds, name, namespace):
    """The reverse: a call in the client's history, named as the model was shown it."""
    for model, info in (kinds or {}).items():
        if len(info) > 3 and info[3] == name and info[2] == namespace:
            return model
    return name


def _schema_types(schema):
    """The JSON types a (possibly composite) schema allows; empty = anything."""
    if not isinstance(schema, dict):
        return set()                       # `true`, `false`, or junk: no constraint
    kinds = schema.get("type")
    out = set(kinds) if isinstance(kinds, list) else ({kinds} if isinstance(kinds, str) else set())
    for key in ("anyOf", "oneOf"):
        for sub in schema.get(key) or []:
            out |= _schema_types(sub)
    return out - {"null"}


def _json_type(v):
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "boolean"
    if isinstance(v, int):
        return "integer"
    if isinstance(v, float):
        return "number"
    return {str: "string", list: "array", dict: "object"}.get(type(v), "?")


def _schema_all_types(schema):
    """Like _schema_types, but keeping "null"."""
    if not isinstance(schema, dict):
        return set()
    kinds = schema.get("type")
    out = set(kinds) if isinstance(kinds, list) else ({kinds} if isinstance(kinds, str) else set())
    for key in ("anyOf", "oneOf"):
        for sub in schema.get(key) or []:
            out |= _schema_all_types(sub)
    return out


def _typed(value, schema):
    """A parameter's text as the JSON type its schema asks for, the way llama-server
    reads Qwen3-Coder values: a string-only parameter is its raw text; a parameter
    that cannot be a string is JSON; one that may be a string or something else is
    JSON when it parses to a type the schema allows (`null` -> null) and raw text
    otherwise ("1.0" for [string, null] stays "1.0")."""
    kinds = _schema_types(schema)
    if _schema_all_types(schema) == {"string"}:
        return value
    try:
        parsed = json.loads(value)
    except ValueError:
        low = value.strip().lower()
        return (low == "true") if "boolean" in kinds and low in ("true", "false") else value
    if "string" in kinds:
        allowed = _schema_all_types(schema)
        got = _json_type(parsed)
        fits = got in allowed or (got == "integer" and "number" in allowed)
        return parsed if fits and got != "string" else value
    if kinds == {"integer"} and isinstance(parsed, float) and parsed.is_integer():
        parsed = int(parsed)
    return parsed


_PARAM_OPEN = "<parameter="
_PARAM_END = "</parameter>"
_AFTER_PARAM = ("<parameter=", "</function>", "</tool_call>")
_FUNC_END = "</function>"


def _is_prefix_of_any(s, words):
    return any(w.startswith(s) for w in words)


def _held_tail(s, word):
    """Length of the longest suffix of `s` that is a proper prefix of `word`."""
    for k in range(min(len(s), len(word) - 1), 0, -1):
        if s.endswith(word[:k]):
            return k
    return 0


def _props_of(kinds, name):
    spec = (kinds.get(name) or (None, {}))[1]
    props = spec.get("properties") if isinstance(spec, dict) else None
    return props if isinstance(props, dict) else {}


def scan_tool_calls(text, kinds, final=False):
    """Every tool call in `text` (the answer from its first <tool_call> on), finished
    or still being written. One scanner for the stream and for the finished answer,
    so what was streamed is always a prefix of what is finally sent.

    Each call: {"name", "done", "pairs": [(key, typed value)] whose end is certain,
    "open": (key, value so far) or None}. A value ends only at a </parameter> that
    is followed by the next parameter, </function> or </tool_call> -- so a value that
    contains any of those tags is read whole. While streaming, a </parameter> not yet
    followed by anything decides nothing, and a tail of the value that could be the
    start of "\\n</parameter>" is held back: a value never has to be taken back. The
    older JSON body form is taken whole, once its </tool_call> (or the end) arrives.
    """
    calls, pos = [], 0
    while True:
        j = text.find(TOOL_OPEN, pos)
        if j < 0:
            break
        pos = j + len(TOOL_OPEN)
        m = _FUNC_OPEN.match(text, pos)
        if not m:
            b = _JSON_BODY.match(text, pos)
            if b and (final or b.group(0).rstrip().endswith("</tool_call>")):
                try:
                    obj = json.loads(b.group(1))
                except ValueError:
                    obj = None
                if isinstance(obj, dict) and obj.get("name"):
                    calls.append({"name": obj["name"], "done": True, "open": None,
                                  "pairs": list(_arguments_dict(obj.get("arguments")).items())})
                pos = b.end()
                continue
            if final:
                continue                     # not a call after all: look for the next
            break                            # the function name is not written yet
        call = {"name": m.group(1).strip(), "done": False, "pairs": [], "open": None}
        calls.append(call)
        props, pos = _props_of(kinds, call["name"]), m.end()
        seen = set()
        while True:
            rest = text[pos:].lstrip()
            ws = len(text) - pos - len(rest)
            if rest.startswith(_FUNC_END):
                call["done"] = True
                pos += ws + len(_FUNC_END)
                c = _CALL_CLOSE.match(text, pos)
                if c:
                    pos = c.end()
                break
            if not rest.startswith(_PARAM_OPEN):
                if final:
                    call["done"] = True      # ended without </function>: the model stopped
                break                        # a tag still being written, or nothing yet
            gt = rest.find(">")
            if gt < 0:
                break                        # the parameter's name is not written yet
            key = rest[len(_PARAM_OPEN):gt].strip()
            start = pos + ws + gt + 1
            if text.startswith("\n", start):
                start += 1
            end, search, pending = None, start, None
            while True:
                k = text.find(_PARAM_END, search)
                if k < 0:
                    break
                after = text[k + len(_PARAM_END):].lstrip()
                if after.startswith(_AFTER_PARAM) or (final and not after):
                    end = k
                    break
                if not after or _is_prefix_of_any(after, _AFTER_PARAM):
                    pending = k               # decided by what comes next
                    break
                search = k + 1                # part of the value: keep looking
            if end is None:
                hold = pending if pending is not None else len(text)
                held = text[start:hold]
                hold -= max(_held_tail(held, "\n" + _PARAM_END), _held_tail(held, _PARAM_END))
                if final:
                    # the value never closed: the call ends here, without half an argument
                    call["done"] = True
                elif key not in seen:
                    call["open"] = (key, text[start:max(start, hold)])
                break
            value = text[start:end]
            if value.endswith("\n"):
                value = value[:-1]
            if key not in seen:               # a repeated key keeps its first value
                seen.add(key)
                call["pairs"].append((key, _typed(value, props.get(key))))
            pos = end + len(_PARAM_END)
    return calls


def args_json(pairs, partial=False, open_=None):
    """Arguments as the JSON string clients receive (compact, like OpenAI and
    llama-server) -- or, while a call is streaming (partial=True), the prefix of it
    that can no longer change: the closed parameters, then an open string value
    (open_=(key, value so far)) without its closing quote, and no closing brace."""
    parts = [json.dumps(k, ensure_ascii=False) + ":"
             + json.dumps(v, ensure_ascii=False, separators=(",", ":")) for k, v in pairs]
    if not partial:
        return "{" + ",".join(parts) + "}"
    head = "{" + ",".join(parts)
    if open_ is not None:
        head += ("," if parts else "") + json.dumps(open_[0], ensure_ascii=False) + ":\"" \
            + json.dumps(open_[1], ensure_ascii=False)[1:-1]
    return head


def extract_tool_calls(text, kinds, cut=False):
    """(the answer before the first call, [{"name", "arguments"}]).

    Qwen3-Coder XML is what this template teaches; the older JSON body
    ({"name": ..., "arguments": {...}}) is taken too. A call that ends at
    </function> without </tool_call> still counts -- the model often stops there.
    One that never reached </function> is complete only if the generation was not
    cut off (cut=True: it stopped on the token ceiling): a cut call is missing its
    last argument, and sending it would run a tool on half its input. Only runs when
    the request offered tools, so a chat about XML is left alone.
    """
    i = text.find(TOOL_OPEN) if kinds else -1
    if i < 0:
        return text, []
    calls = []
    for c in scan_tool_calls(text[i:], kinds, final=not cut):
        if not c["done"]:
            break                             # cut off inside this call: drop it
        calls.append({"name": c["name"], "arguments": dict(c["pairs"])})
    return text[:i].rstrip(), calls


def _streams_as_string(kinds, name, key):
    kind = (kinds.get(name) or ("function",))[0]
    return kind == "custom" or "string" in _schema_types(_props_of(kinds, name).get(key))


class ToolStream:
    """Tool calls as they are written: each call once its name is known, then its
    arguments piece by piece, then done -- the pieces add up to exactly the final
    arguments. Custom (freeform) tools stream their raw input instead of JSON when
    raw_custom (the Responses API has them; Chat Completions only has functions).

    feed() returns ("start", i, call) / ("args", i, piece) / ("done", i, call) events,
    where call is {"name", "kind", "arguments" (dict), "text" (the JSON string, or
    the raw input for a custom tool)}. With parallel=False, calls after the first are
    not emitted: the client asked for one."""

    def __init__(self, kinds, parallel=True, raw_custom=True):
        self.kinds = kinds
        self.parallel = parallel
        self.raw_custom = raw_custom
        self.sent = []                        # per call: {"name", "text", "done"}

    def feed(self, text, final=False, cut=False):
        out = []
        calls = scan_tool_calls(text, self.kinds, final=final and not cut)
        if not self.parallel:
            calls = calls[:1]
        for i, c in enumerate(calls):
            kind = (self.kinds.get(c["name"]) or ("function",))[0]
            if kind == "custom" and not self.raw_custom:
                kind = "function"
            if i >= len(self.sent):
                self.sent.append({"name": c["name"], "text": "", "done": False})
                out.append(("start", i, {"name": c["name"], "kind": kind}))
            s = self.sent[i]
            if s["done"]:
                continue
            args = dict(c["pairs"])
            if kind == "custom":
                if c["done"]:
                    now = str(args.get("input", ""))
                elif c["open"] and c["open"][0] == "input":
                    now = c["open"][1]
                else:
                    now = s["text"]
            else:
                if c["done"]:
                    now = args_json(c["pairs"])
                else:
                    # only a string value can stream mid-way: a number or an object is
                    # typed when it closes, and appears whole
                    streamable = bool(c["open"]) and _streams_as_string(self.kinds, c["name"],
                                                                        c["open"][0])
                    now = args_json(c["pairs"], partial=True,
                                    open_=c["open"] if streamable else None)
            if now.startswith(s["text"]) and len(now) > len(s["text"]):
                out.append(("args", i, now[len(s["text"]):]))
                s["text"] = now
            if c["done"]:
                s["done"] = True
                out.append(("done", i, {"name": c["name"], "kind": kind, "arguments": args,
                                        "text": now}))
        return out


def before_tool_call(answer, done=False):
    """What of a streaming answer may be sent now: everything before a tool call,
    holding back a tail that could be the start of one ('<tool_c') -- until the
    answer is done, when a trailing '<' is just text and goes out."""
    i = answer.find(TOOL_OPEN)
    if i >= 0:
        return answer[:i].rstrip()
    if done:
        return answer
    for k in range(len(TOOL_OPEN) - 1, 0, -1):
        if answer.endswith(TOOL_OPEN[:k]):
            return answer[:-k]
    return answer


def template_roles(msgs):
    """Fold the leading system and developer turns into one system message.

    This template accepts `system` only as the first message and raises on any role
    it does not know -- and Codex sends `developer` turns. The raise made
    render_chat fall back to bare ChatML for every Codex request: no thinking
    control, no tools, and the model's markers leaking into the answer.

    Only the leading run is folded. A system or developer turn later in the
    conversation stays where it is, as a user turn: hoisting it to the top would
    rewrite the start of the prompt and throw away the whole prefix cache -- minutes
    of prefill at these context lengths.
    """
    head, rest = [], []
    for m in msgs:
        role = m.get("role")
        if role in ("system", "developer") and not rest:
            text = _text_of(m.get("content")).strip()
            if text:
                head.append(text)
        elif role in ("user", "assistant", "tool"):
            rest.append(m)
        else:
            rest.append(dict(m, role="user"))
    return ([{"role": "system", "content": "\n\n".join(head)}] if head else []) + rest


def chat_messages(msgs):
    """Chat messages as this template takes them: tool-call arguments as dicts,
    tool results as text, no null content, system/developer folded to the top."""
    out = []
    for m in msgs:
        if not isinstance(m, dict):
            continue
        m = dict(m)
        if m.get("role") == "assistant" and isinstance(m.get("tool_calls"), list):
            m["tool_calls"] = [{"id": tc.get("id"), "type": "function", "function": {
                "name": ((tc.get("function") or {}).get("name")) or "?",
                "arguments": _arguments_dict((tc.get("function") or {}).get("arguments"))}}
                for tc in m["tool_calls"] if isinstance(tc, dict)]
        if m.get("content") is None:
            m["content"] = ""
        if m.get("role") == "tool":
            m["content"] = _text_of(m.get("content"))
        out.append(m)
    return template_roles(out)


def responses_messages(req, kinds=None):
    """Map a Responses-API request onto chat messages the template accepts.

    function_call / custom_tool_call items become the assistant's tool_calls and
    their outputs become `tool` messages -- before, both turned into empty user
    turns, so the model never saw what its tools returned. A call that carries a
    `namespace` is named as the model was offered it (request_tools).
    """
    src = req.get("input")
    if src is None:
        src = req.get("messages")
    msgs = []

    def assistant_turn():
        # llama-server's merge rule: a reasoning item, a call or the assistant's text
        # joins the assistant turn just before it, so one step of an agent loop --
        # its thinking, its text, its calls -- is one assistant message again
        if msgs and msgs[-1]["role"] == "assistant":
            return msgs[-1]
        msgs.append({"role": "assistant", "content": ""})
        return msgs[-1]

    if isinstance(src, str):
        msgs.append({"role": "user", "content": src})
    elif isinstance(src, list):
        for item in src:
            if isinstance(item, str):
                msgs.append({"role": "user", "content": item})
                continue
            if not isinstance(item, dict):
                continue
            kind = item.get("type")
            if kind in ("function_call", "custom_tool_call"):
                args = (item.get("arguments") if kind == "function_call"
                        else {"input": item.get("input") or ""})
                assistant_turn().setdefault("tool_calls", []).append({
                    "id": item.get("call_id") or item.get("id"), "type": "function",
                    "function": {"name": _model_name(kinds, item.get("name") or "?",
                                                     item.get("namespace")),
                                 "arguments": _arguments_dict(args)}})
            elif kind in ("function_call_output", "custom_tool_call_output"):
                msgs.append({"role": "tool", "tool_call_id": item.get("call_id"),
                             "content": _text_of(item.get("output"))})
            elif kind == "reasoning":
                # The model's own thinking, handed back: the template puts it in front
                # of that step's calls, so a multi-step turn keeps its chain of thought
                # (llama-server does the same). Summary-only or encrypted items carry
                # no text to replay and are skipped.
                parts = item.get("content")
                thought = (parts if isinstance(parts, str) else
                           "".join(p.get("text", "") for p in parts if isinstance(p, dict))
                           if isinstance(parts, list) else "")
                if thought.strip():
                    turn = assistant_turn()
                    turn["reasoning_content"] = (turn.get("reasoning_content") or "") + thought
            elif kind in (None, "message"):
                c = item.get("content")
                text = _text_of(c) if c is not None else (item.get("text") or "")
                role = item.get("role") or "user"
                if role == "assistant" and msgs and msgs[-1]["role"] == "assistant" \
                        and not msgs[-1].get("content") and not msgs[-1].get("tool_calls"):
                    msgs[-1]["content"] = text       # its reasoning came just before it
                else:
                    msgs.append({"role": role, "content": text})
            # calls to vendor-hosted tools (web_search, ...) are not replayed
    if isinstance(req.get("instructions"), str) and req["instructions"].strip():
        msgs.insert(0, {"role": "system", "content": req["instructions"]})
    msgs = template_roles(msgs)
    if not msgs:
        msgs = [{"role": "user", "content": "hello"}]
    return msgs


def call_item(name, kinds, status="completed", arguments=None, text=None):
    """A Responses output item for one call: function_call (arguments as a JSON
    string) or custom_tool_call (raw input), under the name -- and namespace -- the
    client gave the tool."""
    real, ns = _client_name(kinds, name)
    call_id = "call_" + secrets.token_hex(12)
    if (kinds.get(name) or ("function",))[0] == "custom":
        item = {"type": "custom_tool_call", "id": "ctc_" + secrets.token_hex(12),
                "call_id": call_id, "name": real, "status": status,
                "input": text if text is not None else str((arguments or {}).get("input", ""))}
    else:
        item = {"type": "function_call", "id": "fc_" + secrets.token_hex(12),
                "call_id": call_id, "name": real, "status": status,
                "arguments": text if text is not None else args_json(list((arguments or {}).items()))}
    if ns:
        item["namespace"] = ns
    return item


def wire_calls(calls, kinds):
    """Parsed calls -> (chat tool_calls, Responses output items), sharing call ids.
    Arguments are compact JSON strings, as OpenAI and llama-server send them."""
    chat, items = [], []
    for c in calls:
        item = call_item(c["name"], kinds, arguments=c["arguments"])
        chat.append({"id": item["call_id"], "type": "function",
                     "function": {"name": c["name"],
                                  "arguments": args_json(list(c["arguments"].items()))}})
        items.append(item)
    return chat, items


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "collabosm"

    def log_message(self, fmt, *a):
        print("[api] " + (fmt % a), flush=True)

    def _send(self, code, payload, ctype="application/json"):
        body = payload if isinstance(payload, (bytes, str)) else json.dumps(payload)
        if isinstance(body, str):
            body = body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        if code >= 400:
            # Errors are terminal for the connection. Otherwise a client that was
            # told "no" reuses the socket, and any bytes we never read get parsed
            # as the next request line.
            self.send_header("Connection", "close")
            self.close_connection = True
        self.end_headers()
        self.wfile.write(body)

    # --- SSE framing -----------------------------------------------------------
    # A streamed body with no Content-Length and no Transfer-Encoding can only be
    # ended by closing the socket. Through a tunnel hop that is not reliable: the
    # origin closes, the edge keeps the client's read open, and the client waits
    # for an EOF that never comes ("stream disconnected before completion").
    # Explicit HTTP/1.1 chunked framing makes the end unambiguous at every hop.
    def _sse_start(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache, no-transform")
        self.send_header("Transfer-Encoding", "chunked")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self._chunked = True

    def _sse_write(self, text):
        data = text.encode("utf-8")
        if getattr(self, "_chunked", False):
            self.wfile.write(b"%x\r\n" % len(data) + data + b"\r\n")
        else:
            self.wfile.write(data)

    def _sse_event(self, name, payload):
        self._sse_write("event: %s\ndata: %s\n\n" % (name, json.dumps(payload)))

    def _sse_end(self):
        if getattr(self, "_chunked", False):
            self.wfile.write(b"0\r\n\r\n")
        self.wfile.flush()
        self.close_connection = True

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "authorization,content-type")
        self.send_header("Access-Control-Allow-Methods", "POST,GET,OPTIONS")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def _authorized(self):
        if NO_AUTH or not API_KEY:
            return True
        h = self.headers.get("Authorization", "") or ""
        return h == ("Bearer " + API_KEY) or h == API_KEY

    def send_error(self, code, message=None, explain=None):
        """Never answer with BaseHTTPRequestHandler's HTML error page.

        The HTML 501 a client saw was produced by THIS process, not by a second
        listener: an unread request body poisoned the keep-alive socket, and the
        next "request line" was the previous JSON body, which the base class
        reported as an unsupported method. Two listeners were never involved.
        Every error path here answers JSON, so a protocol failure can never be
        mistaken for a missing model.
        """
        try:
            self._send(code, {"error": {"message": str(message or "request error"),
                                        "type": "invalid_request_error"}})
        except Exception:
            pass

    def _reject_method(self):
        self._read_body()
        return self._send(405, {"error": {"message": "method not allowed: %s" % self.command,
                                          "type": "invalid_request_error"}})

    def do_PUT(self):
        return self._reject_method()

    def do_PATCH(self):
        return self._reject_method()

    def do_DELETE(self):
        return self._reject_method()

    def do_HEAD(self):
        self._read_body()
        if self.path.startswith("/health"):
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", "2")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            return
        return self._send(405, {"error": {"message": "use GET",
                                          "type": "invalid_request_error"}})

    def _read_body(self):
        """Consume the whole request body, however it is framed.

        Not consuming it is what broke every client probe: with HTTP/1.1
        keep-alive the stranded bytes are parsed as the next request line, which
        surfaces as `Unsupported method ('{"model":...}POST')` and HTTP 501.
        """
        te = (self.headers.get("Transfer-Encoding") or "").lower()
        if "chunked" in te:
            out = b""
            while True:
                line = self.rfile.readline(65536).strip()
                if not line:
                    break
                try:
                    n = int(line.split(b";")[0], 16)
                except ValueError:
                    break
                if n == 0:
                    self.rfile.readline(65536)
                    break
                out += self.rfile.read(n)
                self.rfile.readline(65536)
            return out
        try:
            n = int(self.headers.get("Content-Length") or 0)
        except (TypeError, ValueError):
            n = 0
        return self.rfile.read(n) if n > 0 else b""

    def do_GET(self):
        self._read_body()
        if self.path.startswith("/health"):
            return self._send(200, "ok", "text/plain")
        if not self._authorized():
            return self._send(401, {"error": {"message": "missing or bad API key",
                                              "type": "invalid_request_error"}})
        if self.path.startswith("/v1/models"):
            # MODEL_IDS lets a mux-style deployment advertise several served
            # models from one endpoint, which is what a WebUI model picker needs
            # to show machine/tier choices instead of a single hardcoded id.
            ids = [m.strip() for m in
                   os.environ.get("MODEL_IDS", MODEL_ID).split(",") if m.strip()]
            return self._send(200, {"object": "list", "data": [
                {"id": m, "object": "model", "created": SERVER_STARTED,
                 "owned_by": "collabosm"} for m in ids]})
        if self.path.startswith("/v1/status"):
            # Read-only contract: what this server can actually do, so a client does
            # not have to guess (and so a UI can grey out what is unavailable).
            cache_tokens = getattr(getattr(GEN, "cache", None), "max_num_tokens", None)
            return self._send(200, {
                "service": "collabosm",
                "model": MODEL_ID,
                "recipe": RECIPE,
                "started": SERVER_STARTED,
                "uptime_s": int(time.time()) - SERVER_STARTED if SERVER_STARTED else None,
                "cache_max_tokens": cache_tokens,
                # the context a request can use: the native window, or YaRN's
                "context": {"native": ROPE.get("native"), "yarn_factor": LAUNCH.get("yarn_factor", 0),
                            "max_positions": ROPE.get("max") or ROPE.get("native")},
                "concurrency": {"serving": 1, "requested": CONCURRENCY},
                "dialects": ["/v1/chat/completions", "/v1/responses"],
                "vision": {
                    "enabled": VISION_WANTED,
                    "available": VISION is not None,
                    "error": VISION_ERR,
                },
                "image_input": {
                    "data_urls": True,
                    "remote_urls": IMAGE_URLS,
                    "max_bytes": MAX_IMAGE_BYTES,
                    "max_images_per_request": MAX_IMAGES,
                },
                "generation": {
                    "max_new_tokens_default": MAX_NEW_TOKENS,
                    "thinking_default": THINKING_DEFAULT,
                    # There is no thinking budget to report: the template's
                    # reasoning_effort chooses how hard, max_output_tokens caps how
                    # long, and the two share the one allowance.
                    "thinking_budget": None,
                },
                "launch": LAUNCH,
                "activity": dict(ACTIVITY),
                "last_timings": dict(LAST_TIMINGS) or None,
                "machine": machine(),
            })
        if self.path in ("/", "/v1"):
            return self._send(200, {"service": "collabosm",
                                    "model": MODEL_ID,
                                    "endpoints": ["/v1/models", "/v1/chat/completions",
                                                  "/v1/responses", "/health"]})
        return self._send(404, {"error": {"message": "not found: %s" % self.path}})

    def do_POST(self):
        # Read first, always: every early return below used to strand the body.
        raw = self._read_body()
        if not (self.path.startswith("/v1/chat/completions")
                or self.path.startswith("/v1/responses")):
            return self._send(404, {"error": {"message": "not found: %s" % self.path}})
        if not self._authorized():
            return self._send(401, {"error": {"message": "missing or bad API key",
                                              "type": "invalid_request_error"}})
        try:
            req = json.loads(raw or b"{}")
        except Exception as exc:
            return self._send(400, {"error": {"message": "bad json: %r" % exc}})
        with ACTIVITY_LOCK:
            ACTIVITY["requests"] += 1
            ACTIVITY["in_flight"] += 1
            ACTIVITY["last_request_at"] = int(time.time())
        try:
            if self.path.startswith("/v1/responses"):
                return self._responses(req)
            return self._completions(req)
        finally:
            with ACTIVITY_LOCK:
                ACTIVITY["in_flight"] -= 1
                ACTIVITY["last_request_at"] = int(time.time())

    def _completions(self, req):
        msgs = req.get("messages") or []
        embs = []
        try:
            # Before anything reads the content: image parts become prompt aliases
            # plus the embeddings that stand behind them, or the request is refused.
            extract_images(req, embs)
        except MMUnavailable as exc:
            return self._send(400, {"error": {"message": str(exc),
                                              "type": "vision_unavailable"}})
        if not isinstance(msgs, list) or not msgs:
            return self._send(400, {"error": {"message": "messages must be a non-empty array",
                                              "type": "invalid_request_error",
                                              "param": "messages"}})
        tkw = template_kwargs_from(req)
        tools, kinds = request_tools(req)
        prompt, how = render_chat(chat_messages(msgs), tkw, tools)
        thinking_on = prompt_opens_think(prompt)
        # OpenAI renamed max_tokens -> max_completion_tokens; accept both.
        max_tokens = int(req.get("max_completion_tokens")
                         or req.get("max_tokens") or MAX_NEW_TOKENS)
        stream = bool(req.get("stream"))
        stop = req.get("stop")
        stops = [stop] if isinstance(stop, str) else list(stop or [])

        cid = "chatcmpl-%d" % int(time.time() * 1000)
        model = req.get("model", MODEL_ID)
        parallel = req.get("parallel_tool_calls") is not False

        def finish_reason(r):
            return "length" if (r.get("eos_reason") or "") == "max_new_tokens" else "stop"

        def report(r):
            pt_ = r.get("prompt_tokens") or 0
            ct_ = r.get("cached_tokens") or 0
            nt_ = r.get("new_tokens") or 0
            hit = (100.0 * ct_ / pt_) if pt_ else 0.0
            tm = r.get("timings") or {}
            print("[api] template=%s prompt=%d cached=%d (%.1f%% hit) new=%d finish=%s "
                  "prefill=%s t/s decode=%s t/s"
                  % (how, pt_, ct_, hit, nt_, finish_reason(r),
                     tm.get("prompt_per_second"), tm.get("predicted_per_second")), flush=True)

        def usage_for(r):
            pt_ = r.get("prompt_tokens") or 0
            ct_ = r.get("cached_tokens") or 0
            nt_ = r.get("new_tokens") or 0
            return {"prompt_tokens": pt_, "completion_tokens": nt_,
                    "total_tokens": pt_ + nt_,
                    "prompt_tokens_details": {"cached_tokens": ct_}}

        def chunk(delta, finish=None):
            return {"id": cid, "object": "chat.completion.chunk",
                    "created": int(time.time()), "model": model,
                    "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}

        if not stream:
            try:
                r = collect(prompt, max_tokens, req.get("temperature"),
                            req.get("top_p"), stops, embeddings=embs)
            except ContextTooLong as exc:
                return self._send(400, {"error": {"message": str(exc),
                                                  "type": "invalid_request_error",
                                                  "code": "context_length_exceeded"}})
            except Exception as exc:
                traceback.print_exc()
                return self._send(500, {"error": {"message": repr(exc)}})
            reasoning, text = split_result(r, thinking_on)
            text, calls = extract_tool_calls(text, kinds, cut=truncated(r))
            if not parallel:
                calls = calls[:1]
            report(r)
            message = assistant_message(text, reasoning)
            if calls:
                message["tool_calls"] = wire_calls(calls, kinds)[0]
                message["content"] = text or None
            body = {
                "id": cid, "object": "chat.completion", "created": int(time.time()),
                "model": model,
                # cut off on the token ceiling is "length" even with calls in it:
                # the client must not take a truncated turn for a finished one
                "choices": [{"index": 0,
                             "finish_reason": ("tool_calls" if calls and not truncated(r)
                                               else finish_reason(r)),
                             "message": message}],
                "usage": usage_for(r)}
            if r.get("timings"):
                body["timings"] = r["timings"]
            return self._send(200, body)

        # Genuinely incremental: text is forwarded while the engine decodes it,
        # rather than being chunked up after the completion is already finished.
        self._sse_start()
        delta = _Delta(thinking_on, kinds, parallel, raw_custom=False)
        started = [False]
        call_ids = {}                         # stream index -> call id
        done_calls = []

        def emit(deltas):
            for d in deltas:
                if not started[0]:
                    self._sse_write("data: " + json.dumps(
                        chunk({"role": "assistant", "content": ""})) + "\n\n")
                    started[0] = True
                if d[0] == "reasoning":
                    self._sse_write("data: " + json.dumps(
                        chunk({"reasoning_content": d[1]})) + "\n\n")
                elif d[0] == "content":
                    self._sse_write("data: " + json.dumps(
                        chunk({"content": d[1]})) + "\n\n")
                else:
                    # tool_calls deltas, the way llama-server and OpenAI stream them:
                    # the call's first delta carries its id, type and name; the rest
                    # carry argument pieces under the same index
                    what, i, payload = d[1], d[2], d[3]
                    if what == "start":
                        call_ids[i] = "call_" + secrets.token_hex(12)
                        tc = {"index": i, "id": call_ids[i], "type": "function",
                              "function": {"name": payload["name"], "arguments": ""}}
                    elif what == "args":
                        tc = {"index": i, "function": {"arguments": payload}}
                    else:
                        done_calls.append(payload)
                        continue
                    self._sse_write("data: " + json.dumps(
                        chunk({"tool_calls": [tc]})) + "\n\n")

        try:
            r = collect(prompt, max_tokens, req.get("temperature"), req.get("top_p"),
                        stops, on_delta=lambda acc: emit(delta.feed(acc)),
                        embeddings=embs)
        except Exception as exc:
            if not isinstance(exc, ContextTooLong):
                traceback.print_exc()
            err = ({"message": str(exc), "type": "invalid_request_error",
                    "code": "context_length_exceeded"} if isinstance(exc, ContextTooLong)
                   else {"message": repr(exc)})
            self._sse_write("data: " + json.dumps({"error": err}) + "\n\n")
            self._sse_write("data: [DONE]\n\n")
            return self._sse_end()
        emit(delta.feed(r.get("text") or "", done=True, cut=truncated(r)))
        report(r)
        final = chunk({}, "tool_calls" if done_calls and not truncated(r) else finish_reason(r))
        if r.get("timings"):
            # llama.cpp's field on the last chunk: the WebUI shows it under the
            # reply, and the frontend's rail reads it as it passes through.
            final["timings"] = r["timings"]
        self._sse_write("data: " + json.dumps(final) + "\n\n")
        so = req.get("stream_options")
        if isinstance(so, dict) and so.get("include_usage"):
            self._sse_write("data: " + json.dumps({
                "id": cid, "object": "chat.completion.chunk",
                "created": int(time.time()), "model": model, "choices": [],
                "usage": usage_for(r)}) + "\n\n")
        self._sse_write("data: [DONE]\n\n")
        self._sse_end()

    # ------------------------------------------------------------- Responses API
    def _responses(self, req):
        """Minimal OpenAI Responses surface.

        Clients probe both this and /v1/chat/completions; answering only one of
        them is what produced "supports neither Responses nor Chat Completions".

        Codex's Responses parser binds every delta to an item it was told about
        and will not accept a stream that ends without a terminal event, so the
        whole lifecycle is emitted: created -> in_progress -> [reasoning item] ->
        message item -> deltas -> done -> completed. Every event carries
        sequence_number, and the ids in the terminal event are the same ids the
        stream announced.
        """
        embs = []
        try:
            extract_images(req, embs)
        except MMUnavailable as exc:
            return self._send(400, {"error": {"message": str(exc),
                                              "type": "vision_unavailable"}})
        tools, kinds = request_tools(req)
        msgs = responses_messages(req, kinds)
        tkw = template_kwargs_from(req)
        prompt, how = render_chat(msgs, tkw, tools)
        thinking_on = prompt_opens_think(prompt)
        max_tokens = int(req.get("max_output_tokens") or req.get("max_tokens")
                         or MAX_NEW_TOKENS)
        stream = bool(req.get("stream"))
        now = int(time.time())
        model = req.get("model", MODEL_ID)
        base = now * 1000
        rid = "resp_%d" % base
        mid = "msg_%d" % (base + 1)
        rsn_id = "rs_%d" % (base + 2)

        parallel = req.get("parallel_tool_calls") is not False

        def reasoning_item(status, text):
            # The model's own thinking is the item's content (reasoning_text), as the
            # Responses API carries raw reasoning -- not a summary, which it is not.
            return {"id": rsn_id, "type": "reasoning", "status": status, "summary": [],
                    "content": ([{"type": "reasoning_text", "text": text}] if text else [])}

        def message_item(status, text):
            return {"id": mid, "type": "message", "role": "assistant", "status": status,
                    "content": ([{"type": "output_text", "text": text, "annotations": []}]
                                if text else [])}

        def payload_for(r, text, reasoning, calls=(), message=None):
            pt = r.get("prompt_tokens") or 0
            nt = r.get("new_tokens") or 0
            cut = truncated(r)
            output = []
            if reasoning:
                output.append(reasoning_item("completed", reasoning))
            if message if message is not None else (text or not calls):
                output.append(message_item("incomplete" if cut and not calls else "completed", text))
            output.extend(calls)
            return {
                "id": rid,
                "object": "response",
                "created_at": now,
                # A response that stopped on the token ceiling is not completed, and
                # saying so is the difference between "short answer" and "truncated".
                "status": "incomplete" if cut else "completed",
                "model": model,
                "output": output,
                "output_text": text,
                "parallel_tool_calls": parallel,
                "tool_choice": req.get("tool_choice") or "auto",
                "reasoning": {"effort": tkw.get("reasoning_effort") if tkw.get("enable_thinking") else None,
                              "summary": None},
                "usage": {"input_tokens": pt, "output_tokens": nt,
                          "total_tokens": pt + nt,
                          "input_tokens_details": {"cached_tokens": r.get("cached_tokens") or 0},
                          "output_tokens_details": {"reasoning_tokens": 0}},
                "incomplete_details": ({"reason": "max_output_tokens"} if cut else None),
                "error": None,
            }

        if not stream:
            try:
                r = collect(prompt, max_tokens, req.get("temperature"), req.get("top_p"),
                            embeddings=embs)
            except ContextTooLong as exc:
                return self._send(400, {"error": {"message": str(exc),
                                                  "type": "invalid_request_error",
                                                  "code": "context_length_exceeded"}})
            except Exception as exc:
                traceback.print_exc()
                return self._send(500, {"error": {"message": repr(exc)}})
            reasoning, text = split_result(r, thinking_on)
            text, calls = extract_tool_calls(text, kinds, cut=truncated(r))
            if not parallel:
                calls = calls[:1]
            items = wire_calls(calls, kinds)[1]
            print("[api] /v1/responses template=%s prompt=%s new=%s eos=%s calls=%s text=%r"
                  % (how, r.get("prompt_tokens"), r.get("new_tokens"),
                     r.get("eos_reason"), [c["name"] for c in calls], text[:60]), flush=True)
            return self._send(200, payload_for(r, text, reasoning, items))

        seq = [0]

        def ev(name, obj):
            body = dict(obj)
            body["type"] = name
            body["sequence_number"] = seq[0]
            seq[0] += 1
            self._sse_event(name, body)

        self._sse_start()
        rmeta = {"id": rid, "object": "response", "created_at": now, "model": model,
                 "status": "in_progress", "output": []}
        ev("response.created", {"response": rmeta})
        ev("response.in_progress", {"response": rmeta})

        delta = _Delta(thinking_on, kinds, parallel, raw_custom=True)
        # open: the item being written; idx: the next output_index; msg: whether a
        # message item went out; calls: stream index -> {"item", "oi"}
        state = {"open": None, "idx": 0, "msg": False}
        calls = {}

        def close_reasoning():
            if state["open"] != "reasoning":
                return
            ev("response.reasoning_text.done",
               {"item_id": rsn_id, "output_index": state["idx"], "content_index": 0,
                "text": delta.sent_reasoning})
            ev("response.output_item.done",
               {"output_index": state["idx"],
                "item": reasoning_item("completed", delta.sent_reasoning)})
            state["open"] = None
            state["idx"] += 1

        def open_message():
            if state["open"] == "message":
                return
            ev("response.output_item.added",
               {"output_index": state["idx"], "item": message_item("in_progress", "")})
            ev("response.content_part.added",
               {"item_id": mid, "output_index": state["idx"], "content_index": 0,
                "part": {"type": "output_text", "text": "", "annotations": []}})
            state["open"] = "message"
            state["msg"] = True

        def close_message(text, status="completed"):
            if state["open"] != "message":
                return
            ev("response.output_text.done",
               {"item_id": mid, "output_index": state["idx"], "content_index": 0,
                "text": text})
            ev("response.content_part.done",
               {"item_id": mid, "output_index": state["idx"], "content_index": 0,
                "part": {"type": "output_text", "text": text, "annotations": []}})
            ev("response.output_item.done",
               {"output_index": state["idx"], "item": message_item(status, text)})
            state["open"] = None
            state["idx"] += 1

        def on_call(what, i, payload):
            if what == "start":
                # the text before a call is finished once the call begins
                close_reasoning()
                close_message(delta.sent_content)
                item = call_item(payload["name"], kinds, status="in_progress", text="")
                calls[i] = {"item": item, "oi": state["idx"]}
                ev("response.output_item.added", {"output_index": state["idx"], "item": dict(item)})
                state["idx"] += 1
                return
            c = calls[i]
            item = c["item"]
            custom = item["type"] == "custom_tool_call"
            if what == "args":
                if custom:
                    item["input"] += payload
                    ev("response.custom_tool_call_input.delta",
                       {"item_id": item["id"], "call_id": item["call_id"],
                        "output_index": c["oi"], "delta": payload})
                else:
                    item["arguments"] += payload
                    ev("response.function_call_arguments.delta",
                       {"item_id": item["id"], "output_index": c["oi"], "delta": payload})
            elif what == "done":
                if custom:
                    item["input"] = payload["text"]
                    ev("response.custom_tool_call_input.done",
                       {"item_id": item["id"], "call_id": item["call_id"],
                        "output_index": c["oi"], "input": item["input"]})
                else:
                    item["arguments"] = payload["text"]
                    ev("response.function_call_arguments.done",
                       {"item_id": item["id"], "output_index": c["oi"], "name": item["name"],
                        "arguments": item["arguments"]})
                item["status"] = "completed"
                # the item a client acts on: a call that never reached this point (cut
                # off by the token ceiling) is never handed over to be run
                ev("response.output_item.done", {"output_index": c["oi"], "item": dict(item)})

        def emit(deltas):
            for d in deltas:
                if d[0] == "reasoning":
                    if state["open"] is None and not state["msg"] and not calls:
                        ev("response.output_item.added",
                           {"output_index": state["idx"],
                            "item": reasoning_item("in_progress", "")})
                        state["open"] = "reasoning"
                    if state["open"] == "reasoning":
                        ev("response.reasoning_text.delta",
                           {"item_id": rsn_id, "output_index": state["idx"],
                            "content_index": 0, "delta": d[1]})
                elif d[0] == "content":
                    close_reasoning()
                    open_message()
                    ev("response.output_text.delta",
                       {"item_id": mid, "output_index": state["idx"],
                        "content_index": 0, "delta": d[1]})
                else:
                    on_call(d[1], d[2], d[3])

        try:
            r = collect(prompt, max_tokens, req.get("temperature"), req.get("top_p"),
                        on_delta=lambda acc: emit(delta.feed(acc)), embeddings=embs)
        except Exception as exc:
            too_long = isinstance(exc, ContextTooLong)
            if not too_long:
                traceback.print_exc()
            ev("response.failed", {"response": {
                "id": rid, "object": "response", "created_at": now, "model": model,
                "status": "failed",
                "error": {"code": "context_length_exceeded" if too_long else "server_error",
                          "message": str(exc) if too_long else repr(exc)}}})
            return self._sse_end()

        cut = truncated(r)
        reasoning, text = split_result(r, thinking_on)
        text = extract_tool_calls(text, kinds, cut=cut)[0]
        emit(delta.feed(r.get("text") or "", done=True, cut=cut))
        close_reasoning()
        if state["open"] == "message":
            close_message(text, "incomplete" if cut and not calls else "completed")
        elif not state["msg"] and not calls:
            open_message()                      # a turn with nothing in it still answers
            close_message(text, "incomplete" if cut else "completed")
        items = [c["item"] for _, c in sorted(calls.items(), key=lambda kv: kv[1]["oi"])]
        for item in items:
            if item["status"] != "completed":
                item["status"] = "incomplete"   # cut off while it was being written
        print("[api] /v1/responses template=%s prompt=%s new=%s calls=%s%s"
              % (how, r.get("prompt_tokens"), r.get("new_tokens"),
                 [i.get("name") for i in items], " (cut)" if cut else ""), flush=True)
        # The terminal event says what the response is: a turn cut off by the token
        # ceiling ends in response.incomplete, as the Responses API defines it.
        ev("response.incomplete" if cut else "response.completed",
           {"response": payload_for(r, text, reasoning, items, message=state["msg"])})
        self._sse_end()

def warm_up():
    """Pay the first-use costs before anyone is waiting: one prompt of a full chunk
    and a few decoded tokens, so the prefill kernels autotune at LAUNCH's chunk size
    and the decode path (MTP drafts included) is captured. It is not a request: the
    activity counters never see it, and its timings are dropped, so the rail's speed
    shows a real reply. A failure is logged and serving goes on."""
    gcs = int(LAUNCH.get("generator_chunk_size") or 4096)
    t0 = time.time()
    try:
        # A little over one chunk, from words of one token each. (Numbers were five
        # times too long: this tokenizer splits them digit by digit, and the first box
        # warmed up on 39,850 tokens for 43.6 s instead of ~8,200.)
        words = ("the quick brown fox jumps over a lazy dog while seven old "
                 "wizards quietly judge the boxing match").split()
        n = gcs + 32
        text = " ".join(words[i % len(words)] for i in range(n))
        while TOK.encode(text).shape[-1] < gcs + 32:
            n += 256
            text = " ".join(words[i % len(words)] for i in range(n))
        r = collect(text, 8)
        print("[api] warm-up: %s-token prefill + %s tokens in %.1fs (first-use autotune paid "
              "here, not by the first request)" % (r.get("prompt_tokens"), r.get("new_tokens"),
                                                     time.time() - t0), flush=True)
    except Exception as exc:                          # noqa: BLE001 - never blocks serving
        print("[api] warm-up failed after %.1fs: %r -- serving anyway"
              % (time.time() - t0, exc), flush=True)
    LAST_TIMINGS.clear()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", 8090)))
    a = ap.parse_args()
    build_engine()
    if WARMUP:
        warm_up()
    srv = ThreadingHTTPServer((a.host, a.port), Handler)
    print("[api] listening on http://%s:%d" % (a.host, a.port), flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
