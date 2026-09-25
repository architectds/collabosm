'use strict';
const $ = (id) => document.getElementById(id);

function esc(s) {
  return String(s ?? '').replace(/[&<>"']/g, (c) =>
    ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}
const num = (v, d = 1) => (v === null || v === undefined) ? '—'
  : (Number.isInteger(v) ? String(v) : v.toFixed(d));
const clock = (t) => new Date(t * 1000).toLocaleTimeString();

function fact(k, v, mono = true) {
  return `<div class="card"><div class="k">${esc(k)}</div><div class="v"` +
         `${mono ? '' : ' style="font-family:inherit"'}>${esc(v)}</div></div>`;
}

function big(k, v, unit) {
  return `<div class="card"><div class="k">${esc(k)}</div><div class="v">${esc(v)}` +
         `${unit ? ` <span class="unit">${esc(unit)}</span>` : ''}</div></div>`;
}

async function getJSON(path) {
  const r = await fetch(path, { cache: 'no-store' });
  return r.json();
}

async function refresh() {
  let cfg = {}, health = {}, m = {};
  try { cfg = await getJSON('/local/config'); } catch {}
  try { health = await getJSON('/health'); } catch {}
  try { m = await getJSON('/local/metrics'); } catch {}
  let srv = {};
  try { srv = await getJSON('/v1/status'); } catch {}

  $('dot').className = 'dot ' + (health.ok ? 'ok' : 'bad');
  $('dot').title = health.ok ? 'proxy up' : 'proxy unreachable';

  $('facts').innerHTML =
    fact('endpoint', cfg.configured ? cfg.endpoint : 'NOT CONFIGURED') +
    fact('dialect', cfg.dialect || '—') +
    fact('model', cfg.model || '—') +
    fact('bearer key', cfg.has_key ? 'held by the client' : 'MISSING') +
    fact('turns this session', String((m.turns || []).length)) +
    fact('vision', srv.vision
      ? (srv.vision.available ? 'available' : (srv.vision.enabled ? 'failed to load' : 'off (VISION=1 to enable)'))
      : 'unknown') +
    fact('image URLs', srv.image_input
      ? (srv.image_input.remote_urls ? 'remote allowed' : 'data: URLs only')
      : 'unknown');

  const miss = (srv.launch && srv.launch.cache_size)
    ? `cache ${srv.launch.cache_size} · kv q${srv.launch.cache_quant} · gcs ${srv.launch.generator_chunk_size} · mtp ndt ${srv.launch.num_draft_tokens} · ccs ${srv.launch.cache_size ? srv.launch.cpu_cache_gb : 0}GB`
    : '';
  $('launch').textContent = miss;

  const live = m.live;
  const el = $('live');
  if (!cfg.configured) {
    el.className = 'live busy';
    el.textContent = 'no endpoint configured  —  ' +
      'python collabosm.py config --endpoint <url> --api-key <key>';
  } else if (live && live.events) {
    const age = (Date.now() / 1000 - live.started).toFixed(1);
    el.className = 'live busy';
    el.textContent = `generating…  ${live.chars} chars · ${live.events} events · ${age}s` +
                     (live.ttft !== null && live.ttft !== undefined
                        ? `  ·  first token at ${live.ttft.toFixed(2)}s` : '  ·  prefill…');
  } else {
    el.className = 'live';
    el.textContent = 'idle — no request in flight';
  }

  const t = m.last;
  $('last').innerHTML = t
    ? big('prefill', num(t.prefill_tps), 'tok/s') +
      big('decode', num(t.decode_tps), 'tok/s') +
      big('ttft', num(t.ttft_s, 3), 's') +
      big('tokens in', num(t.prompt_tokens)) +
      big('tokens out', num(t.output_tokens) + (t.estimated ? ' est' : '')) +
      big('wall clock', num(t.total_s, 2), 's')
    : big('prefill', '—') + big('decode', '—') + big('ttft', '—');

  const rows = (m.turns || []).slice().reverse().slice(0, 20);
  $('turns').innerHTML = rows.map((r) => `<tr>
      <td>${esc(clock(r.at))}</td>
      <td>${esc((r.path || '').replace('/v1/', ''))}</td>
      <td>${r.ttft_s === null || r.ttft_s === undefined ? '—' : num(r.ttft_s, 2) + 's'}</td>
      <td>${num(r.prefill_tps)}</td>
      <td>${num(r.decode_tps)}${r.estimated ? ' <span class="badge">est</span>' : ''}</td>
      <td>${num(r.prompt_tokens)}</td>
      <td>${num(r.output_tokens)}</td>
      <td>${num(r.total_s, 2)}s</td>
    </tr>`).join('');
}

refresh();
setInterval(refresh, 1200);