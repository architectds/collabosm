'use strict';

/* No persistence, on purpose: conversations live in this tab only. Storing
   transcripts would raise the question of which server build produced them
   (KV cache, quant, context), and we would have to invent an epoch scheme to keep
   old numbers honest. Reload starts clean instead. */

let cfg = null;
let convos = [];
let activeId = null;
let controller = null;
let pendingImages = [];     // data: URLs waiting to be sent with the next message
const MAX_IMAGES = 4;
const MAX_IMAGE_BYTES = 8 * 1024 * 1024;

const $ = (id) => document.getElementById(id);

/* ------------------------------------------------------------------ storage */
const SESSION = { dialect: null, model: null };

function save() {
  SESSION.dialect = $('dialect').value;
  SESSION.model = $('model').value;
}

function load() { return SESSION; }

function active() { return convos.find((c) => c.id === activeId); }

function newConvo() {
  const c = { id: 'c' + Date.now() + Math.random().toString(36).slice(2, 6),
              title: 'New conversation', created: Date.now(), messages: [] };
  convos.unshift(c);
  activeId = c.id;
  save(); renderList(); renderMessages();
  return c;
}

/* ------------------------------------------------------------------ render */
function esc(s) {
  return String(s ?? '').replace(/[&<>"']/g, (c) =>
    ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}

/* Markdown-lite: fenced blocks and inline code. No library, no build step. */
function md(text) {
  return String(text ?? '').split('```').map((seg, i) => {
    if (i % 2) return '<pre><code>' + esc(seg.replace(/^[A-Za-z0-9_+-]*\n/, '')) + '</code></pre>';
    return esc(seg).replace(/`([^`\n]+)`/g, '<code>$1</code>');
  }).join('');
}

function renderList() {
  const ul = $('list');
  ul.innerHTML = '';
  for (const c of convos) {
    const li = document.createElement('li');
    if (c.id === activeId) li.className = 'active';
    const span = document.createElement('span');
    span.textContent = c.title;
    const del = document.createElement('button');
    del.className = 'ghost'; del.textContent = '×'; del.title = 'delete';
    del.onclick = (e) => {
      e.stopPropagation();
      convos = convos.filter((x) => x.id !== c.id);
      if (activeId === c.id) activeId = convos[0] ? convos[0].id : null;
      if (!convos.length) newConvo();
      save(); renderList(); renderMessages();
    };
    li.append(span, del);
    li.onclick = () => { activeId = c.id; save(); renderList(); renderMessages(); };
    ul.append(li);
  }
}

function messageNode(m) {
  const wrap = document.createElement('div');
  wrap.className = 'msg ' + (m.role === 'user' ? 'user' : 'assistant') + (m.error ? ' error' : '');
  const who = document.createElement('div');
  who.className = 'who';
  who.textContent = m.role + (m.model ? ' · ' + m.model : '') + (m.dialect ? ' · ' + m.dialect : '');
  const body = document.createElement('div');
  body.className = 'body';
  if (m.pending) body.classList.add('cursor');
  wrap.append(who);

  if (m.reasoning) {
    const d = document.createElement('details');
    d.className = 'think';
    const s = document.createElement('summary');
    s.textContent = 'thinking';
    const rb = document.createElement('div');
    rb.className = 'body'; rb.innerHTML = md(m.reasoning);
    d.append(s, rb);
    wrap.append(d);
  }
  if (m.images && m.images.length) {
    const thumbs = document.createElement('div');
    thumbs.className = 'thumbs';
    for (const url of m.images) {
      const img = document.createElement('img');
      img.src = url;
      img.alt = 'attached image';
      thumbs.append(img);
    }
    wrap.append(thumbs);
  }
  body.innerHTML = md(m.error ? '⚠ ' + m.error : m.content);
  if (m.pending && !m.content && !m.reasoning) body.textContent = '…';
  wrap.append(body);
  if (m.usage) {
    const u = document.createElement('div');
    u.className = 'usage';
    u.textContent = m.usage;
    wrap.append(u);
  }
  return wrap;
}

function renderMessages() {
  const box = $('messages');
  const near = box.scrollHeight - box.scrollTop - box.clientHeight < 80;
  box.innerHTML = '';
  const c = active();
  if (!c) return;
  for (const m of c.messages) box.append(messageNode(m));
  if (near || c.messages.some((m) => m.pending)) box.scrollTop = box.scrollHeight;
}

function touch(asst) {
  /* Re-render only the streaming message: cheap and keeps scroll position. */
  renderMessages();
}

/* ------------------------------------------------------------------ streaming */
async function readSSE(res, onEvent) {
  const reader = res.body.getReader();
  const dec = new TextDecoder();
  let buf = '';
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += dec.decode(value, { stream: true }).replace(/\r\n/g, '\n');
    let i;
    while ((i = buf.indexOf('\n\n')) >= 0) {
      const chunk = buf.slice(0, i);
      buf = buf.slice(i + 2);
      let name = null; const data = [];
      for (const line of chunk.split('\n')) {
        if (line.startsWith('event:')) name = line.slice(6).trim();
        else if (line.startsWith('data:')) data.push(line.slice(5).trim());
      }
      if (data.length) await onEvent(name, data.join('\n'));
    }
  }
}

function historyFor(convo, dialect) {
  const isResp = dialect === 'responses';
  return convo.messages
    .filter((m) => !m.pending && !m.error && (m.role === 'user' || m.role === 'assistant'))
    .map((m) => {
      if (!m.images || !m.images.length) return { role: m.role, content: m.content || '' };
      /* OpenAI content parts. The image travels as a base64 data: URL, which is the
         only form the server accepts by default -- it will not fetch a URL for us
         (that would be an SSRF primitive on a tunnelled box). */
      const parts = [{ type: isResp ? 'input_text' : 'text', text: m.content || '' }];
      for (const url of m.images) {
        parts.push(isResp ? { type: 'input_image', image_url: url }
                          : { type: 'image_url', image_url: { url } });
      }
      return { role: m.role, content: parts };
    });
}

async function streamTurn(convo, asst, dialect, model, signal) {
  const isResp = dialect === 'responses';
  const history = historyFor(convo, dialect);   // excludes the pending assistant turn
  const url = isResp ? '/v1/responses' : '/v1/chat/completions';
  const body = isResp
    ? { model, input: history, stream: true, max_output_tokens: 2048 }
    : { model, messages: history, stream: true, max_tokens: 2048,
        stream_options: { include_usage: true } };

  const res = await fetch(url, {
    method: 'POST', signal,
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });

  if (!res.ok) {
    let detail = '';
    try { detail = (await res.json()).error?.message || ''; } catch {}
    if (!detail) { try { detail = await res.text(); } catch {} }
    throw new Error('HTTP ' + res.status + (detail ? ' — ' + String(detail).slice(0, 300) : ''));
  }

  const ctype = res.headers.get('content-type') || '';
  if (!ctype.includes('text/event-stream')) {
    /* Not a stream: take the whole JSON answer. */
    const data = await res.json();
    if (isResp) {
      asst.content = data.output_text || '';
      asst.usage = data.usage ? usageLine(data.usage.input_tokens, data.usage.output_tokens) : '';
    } else {
      const msg = data.choices?.[0]?.message || {};
      asst.content = msg.content || '';
      asst.reasoning = msg.reasoning_content || '';
    }
    return;
  }

  await readSSE(res, (name, raw) => {
    if (raw === '[DONE]') return;
    let ev; try { ev = JSON.parse(raw); } catch { return; }
    if (isResp) {
      switch (name) {
        case 'response.reasoning_text.delta':
          asst.reasoning = (asst.reasoning || '') + (ev.delta || ''); touch(asst); break;
        case 'response.output_text.delta':
          asst.content = (asst.content || '') + (ev.delta || ''); touch(asst); break;
        case 'response.completed': {
          const u = ev.response?.usage;
          if (u) asst.usage = usageLine(u.input_tokens, u.output_tokens);
          break;
        }
        case 'response.failed':
          asst.error = (ev.response?.error?.message) || JSON.stringify(ev).slice(0, 300); break;
        default: break;
      }
    } else {
      const ch = (ev.choices || [])[0];
      if (ch?.delta) {
        if (ch.delta.reasoning_content) asst.reasoning = (asst.reasoning || '') + ch.delta.reasoning_content;
        if (ch.delta.content) asst.content = (asst.content || '') + ch.delta.content;
        touch(asst);
      }
      if (ev.usage) asst.usage = usageLine(ev.usage.prompt_tokens, ev.usage.completion_tokens);
    }
  });
}

function usageLine(p, c) {
  const cached = '';
  return `tokens  in ${p ?? '?'}  out ${c ?? '?'}${cached}`;
}

/* ------------------------------------------------------------------ actions */
async function send() {
  const ta = $('prompt');
  const text = ta.value.trim();
  if (!text || controller) return;
  let convo = active() || newConvo();
  if (!convo.messages.length) {
    convo.title = text.slice(0, 48) || 'New conversation';
    renderList();
  }
  const dialect = $('dialect').value;
  const model = $('model').value.trim() || 'qwen3.8-flash-next-exl3';
  const images = pendingImages.slice();
  pendingImages = [];
  renderChips();
  convo.messages.push({ role: 'user', content: text, images, ts: Date.now() });
  const asst = { role: 'assistant', content: '', reasoning: '', pending: true,
                 dialect, model, ts: Date.now() };
  convo.messages.push(asst);
  ta.value = '';
  save(); renderMessages();
  $('stop').disabled = false; $('send').disabled = true;
  controller = new AbortController();
  const t0 = performance.now();
  try {
    await streamTurn(convo, asst, dialect, model, controller.signal);
  } catch (e) {
    if (e.name !== 'AbortError') asst.error = String(e.message || e);
  } finally {
    asst.pending = false;
    asst.ms = Math.round(performance.now() - t0);
    if (asst.usage) asst.usage += `  ·  ${(asst.ms / 1000).toFixed(1)}s`;
    else asst.usage = `${(asst.ms / 1000).toFixed(1)}s`;
    controller = null;
    $('stop').disabled = true; $('send').disabled = false;
    save(); renderMessages();
    ta.focus();
  }
}

function renderChips() {
  const box = $('chips');
  box.innerHTML = '';
  pendingImages.forEach((url, i) => {
    const chip = document.createElement('div');
    chip.className = 'chip';
    const img = document.createElement('img');
    img.src = url;
    const x = document.createElement('button');
    x.textContent = '×';
    x.onclick = () => { pendingImages.splice(i, 1); renderChips(); };
    chip.append(img, x);
    box.append(chip);
  });
}

function addImageFile(file) {
  if (!file || !file.type.startsWith('image/')) return;
  if (file.size > MAX_IMAGE_BYTES) { alert('image is larger than 8 MB'); return; }
  if (pendingImages.length >= MAX_IMAGES) { alert('at most ' + MAX_IMAGES + ' images per message'); return; }
  const reader = new FileReader();
  reader.onload = () => { pendingImages.push(reader.result); renderChips(); };
  reader.readAsDataURL(file);
}

async function boot() {
  const s = load();
  try {
    const res = await fetch('/local/config');
    cfg = await res.json();
  } catch {
    cfg = { endpoint: '?', dialect: 'responses', model: '', has_key: false, dialects: ['responses', 'chat'] };
  }
  $('dialect').value = s.dialect || cfg.dialect || 'responses';
  $('model').value = s.model || cfg.model || '';
  $('endpoint').textContent = (cfg.endpoint || '').replace(/^https?:\/\//, '') +
    (cfg.has_key ? '' : '  ⚠ no key');
  $('endpoint').title = cfg.endpoint || '';

  if (!convos.length) newConvo();
  if (!activeId) activeId = convos[0].id;
  renderList(); renderMessages();

  const health = async () => {
    try {
      const r = await fetch('/health', { cache: 'no-store' });
      const j = await r.json();
      $('status').className = 'dot ' + (j.ok ? 'ok' : 'bad');
      $('status').title = j.ok ? 'proxy up · ' + j.endpoint : 'proxy unhealthy';
    } catch {
      $('status').className = 'dot bad';
      $('status').title = 'proxy unreachable';
    }
  };
  health(); setInterval(health, 15000);

  /* Prefill/decode are measured by the local client while the stream passes; the
     API itself has no field for them. Polled so a long generation shows live
     progress rather than only a final number. */
  const metrics = async () => {
    try {
      const j = await (await fetch('/local/metrics', { cache: 'no-store' })).json();
      const t = j.last, live = j.live, out = [];
      if (live && live.events) out.push(`generating… ${live.chars} chars`);
      if (t) {
        if (t.prefill_tps) out.push(`prefill ~${t.prefill_tps} t/s`);
        if (t.decode_tps) out.push(`decode ~${t.decode_tps} t/s`);
        out.push(`ttft ${t.ttft_s}s`);
        out.push(`in ${t.prompt_tokens} / out ${t.output_tokens}${t.estimated ? ' (est)' : ''}`);
        out.push(`${t.total_s}s`);
      }
      $('metrics').textContent = out.length ? out.join('   ·   ') : 'no turns yet';
    } catch { $('metrics').textContent = ''; }
  };
  metrics(); setInterval(metrics, 1500);

  $('new').onclick = newConvo;
  $('send').onclick = send;
  $('stop').onclick = () => { if (controller) controller.abort(); };
  $('dialect').onchange = save;
  $('model').onchange = save;
  $('prompt').addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(); }
  });
  $('export').onclick = () => {
    const blob = new Blob([JSON.stringify(convos, null, 2)], { type: 'application/json' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = 'collabosm-conversations.json';
    a.click();
  };
  $('import').onclick = () => $('file').click();
  $('file').onchange = async (e) => {
    const f = e.target.files[0];
    if (!f) return;
    try {
      const incoming = JSON.parse(await f.text());
      if (Array.isArray(incoming)) {
        convos = incoming.concat(convos);
        activeId = convos[0].id;
        save(); renderList(); renderMessages();
      }
    } catch (err) { alert('bad file: ' + err); }
  };
  $('attach').onclick = () => $('imgfile').click();
  $('imgfile').onchange = (e) => {
    for (const f of e.target.files) addImageFile(f);
    e.target.value = '';
  };
  $('prompt').addEventListener('paste', (e) => {
    const items = e.clipboardData ? e.clipboardData.items : [];
    let took = 0;
    for (const it of items) {
      if (it.kind === 'file' && it.type.startsWith('image/')) {
        addImageFile(it.getAsFile());
        took++;
      }
    }
    if (took) e.preventDefault();
  });
  const drop = (e) => {
    e.preventDefault();
    for (const f of (e.dataTransfer ? e.dataTransfer.files : [])) addImageFile(f);
  };
  document.addEventListener('dragover', (e) => e.preventDefault());
  document.addEventListener('drop', drop);
  $('prompt').focus();
}

boot();