/* NeuroGraph-HW Studio: one script, no build step. */
const $ = id => document.getElementById(id);
const NS = 'http://www.w3.org/2000/svg';
const S = (tag, attrs, parent) => { const e = document.createElementNS(NS, tag); for (const k in attrs) e.setAttribute(k, attrs[k]); if (parent) parent.appendChild(e); return e; };
const H = (tag, cls, parent, text) => { const e = document.createElement(tag); if (cls) e.className = cls; if (text != null) e.textContent = text; if (parent) parent.appendChild(e); return e; };
const TARGETS = { NeuromorphicCore: ['--nm', 'Neuromorphic core'], PIMCrossbar: ['--pim', 'PIM crossbar'], TensorAccelerator: ['--ta', 'Tensor accelerator'] };
const PLAST = { none: '', local_stdp: 'Local STDP', global_sync: 'Global sync' };
const CATS = { neuromorphic: ['--nm', 'Neuromorphic compute'], plasticity: ['--nm', 'On-core plasticity'], pim_adc: ['--pim', 'Crossbar ADC'], pim_dac: ['--pim', 'Crossbar DAC'], pim_array: ['--pim', 'Crossbar array'], tensor: ['--ta', 'Tensor compute'], memory: ['--ta', 'DRAM traffic'], noc: ['--line', 'Mesh and bridge'], codec: ['--line', 'Transcoding'], sync: ['--line', 'Epoch sync'] };
let presets = [], mode = 'ir', last = null;
const tip = $('tip');

const fmt = v => v == null ? 'n/a' : Math.abs(v) >= 1000 ? Math.round(v).toLocaleString('en-US') : Math.abs(v) >= 10 ? v.toFixed(1) : Math.abs(v) >= 0.1 ? v.toFixed(3) : v === 0 ? '0' : v.toPrecision(3);
function showTip(e, lines) { tip.textContent = ''; lines.forEach((l, i) => H(i ? 'div' : 'strong', '', tip, l)); tip.hidden = false; const r = tip.getBoundingClientRect(); tip.style.left = Math.min(innerWidth - r.width - 8, e.clientX + 12) + 'px'; tip.style.top = Math.max(8, e.clientY - r.height - 10) + 'px'; }
function bindTip(el, lines) { el.addEventListener('pointermove', e => showTip(e, lines())); el.addEventListener('pointerleave', () => { tip.hidden = true; }); el.addEventListener('focus', () => { const r = el.getBoundingClientRect(); showTip({ clientX: r.right, clientY: r.top }, lines()); }); el.addEventListener('blur', () => { tip.hidden = true; }); }

/* ---------- share state ---------- */
const enc = o => btoa(unescape(encodeURIComponent(JSON.stringify(o)))).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '');
const dec = s => JSON.parse(decodeURIComponent(escape(atob(s.replace(/-/g, '+').replace(/_/g, '/')))));
function state() { return { p: $('preset').value, m: mode, src: $('preset').value === 'custom' ? $('editor').value : undefined, steps: +$('steps').value, sync: +$('sync').value, f: $('fusion').checked, hw: $('hw').value.trim() || undefined }; }
function shareUrl() { return location.origin + location.pathname + '#s=' + enc(state()); }

/* ---------- requests ---------- */
async function api(path, body) {
  const r = await fetch(path, body ? { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) } : undefined);
  const j = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(typeof j.detail === 'string' ? j.detail : JSON.stringify(j.detail || j));
  return j;
}
function request() {
  const body = { fusion: $('fusion').checked, steps: +$('steps').value, sync_every: +$('sync').value };
  const hw = $('hw').value.trim();
  if (hw) { try { body.hardware = JSON.parse(hw); } catch (e) { throw new Error('Hardware overrides are not valid JSON: ' + e.message); } }
  const p = presets.find(x => x.id === $('preset').value);
  const text = $('editor').value;
  if (p && text === (mode === 'ir' ? p.ir : JSON.stringify(p.graph, null, 2))) body.preset = p.id;
  else if (mode === 'ir') body.ir_text = text;
  else { try { body.graph = JSON.parse(text); } catch (e) { throw new Error('Graph JSON is not valid: ' + e.message); } }
  return body;
}

/* ---------- drawing ---------- */
function drawDag(c) {
  const svg = $('dag'); svg.textContent = '';
  const L = c.layout, W = 196, Hh = 58, gx = 40, gy = 26, pad = 16;
  const width = pad * 2 + L.cols * W + (L.cols - 1) * gx, height = pad * 2 + L.rows * Hh + (L.rows - 1) * gy + 26;
  svg.setAttribute('viewBox', `0 0 ${width} ${height}`); svg.style.width = width + 'px';
  const defs = S('defs', {}, svg), mk = S('marker', { id: 'arr', viewBox: '0 0 10 10', refX: 9, refY: 5, markerWidth: 6, markerHeight: 6, orient: 'auto' }, defs);
  S('path', { d: 'M0 0L10 5L0 10z', style: 'fill:var(--line)' }, mk);
  const pos = id => { const p = L.nodes[id]; return { x: pad + p.col * (W + gx), y: pad + p.row * (Hh + gy) }; };
  c.graph.edges.forEach(e => {
    const a = pos(e.src), b = pos(e.dst);
    let d;
    if (e.feedback) { const y = height - 12; d = `M${a.x + W / 2} ${a.y + Hh}V${y}H${b.x + W / 2}V${b.y + Hh + 2}`; }
    else if (b.x > a.x) { const mx = (a.x + W + b.x) / 2; d = `M${a.x + W} ${a.y + Hh / 2}C${mx} ${a.y + Hh / 2} ${mx} ${b.y + Hh / 2} ${b.x - 2} ${b.y + Hh / 2}`; }
    else d = `M${a.x + W / 2} ${a.y + Hh}L${b.x + W / 2} ${b.y - 2}`;
    const p = S('path', { class: 'edge', d, 'marker-end': 'url(#arr)' }, svg);
    if (e.kind === 'spikes') p.setAttribute('stroke-dasharray', '2 4');
    if (e.feedback) { const t = S('text', { class: 'node-sub', x: (a.x + b.x + W) / 2, y: height - 16, 'text-anchor': 'middle' }, svg); t.textContent = 'feedback, one epoch late'; }
  });
  c.graph.nodes.forEach(n => {
    const p = pos(n.id), col = `var(${TARGETS[n.target][0]})`;
    const g = S('g', { tabindex: 0, role: 'img', 'aria-label': `${n.id}, ${n.op}, on ${TARGETS[n.target][1]}` }, svg);
    S('rect', { class: 'node-box', x: p.x, y: p.y, width: W, height: Hh, rx: 7, style: `fill:color-mix(in srgb, ${col} 13%, var(--bg));stroke:${col}` }, g);
    S('text', { class: 'node-name', x: p.x + 10, y: p.y + 19 }, g).textContent = '%' + n.id;
    S('text', { class: 'node-sub', x: p.x + 10, y: p.y + 34 }, g).textContent = n.op.replace('ral.', '');
    const pl = PLAST[n.attrs.plasticity || 'none'];
    S('text', { class: 'node-sub', x: p.x + 10, y: p.y + 49 }, g).textContent = TARGETS[n.target][1] + (pl ? ' · ' + pl : '');
    bindTip(g, () => [`%${n.id}: ${n.op}`, n.type, 'Target: ' + n.target, 'Plasticity: ' + (pl || 'none'), ...Object.entries(n.attrs).filter(([k]) => k !== 'plasticity' && k !== 'name').map(([k, v]) => `${k} = ${v}`)]);
  });
  const lg = $('dag-legend'); lg.textContent = '';
  Object.values(TARGETS).forEach(([v, name]) => { const s = H('span', '', lg); H('i', 'box', s).style.background = `var(${v})`; s.appendChild(document.createTextNode(name)); });
  const s1 = H('span', '', lg); H('i', 'line', s1).style.borderTopColor = 'var(--line)'; s1.appendChild(document.createTextNode('tensor edge'));
  const s2 = H('span', '', lg); const i2 = H('i', 'line', s2); i2.style.borderTopColor = 'var(--line)'; i2.style.borderTopStyle = 'dotted'; s2.appendChild(document.createTextNode('spike edge'));
  const r = c.report, f = r.fusion;
  $('compile-summary').textContent = `Fusion: ${f.nodes_before} nodes to ${f.nodes_after} (${f.rewrites.map(x => x.rule).join(', ') || 'no rewrites'}). Cut edges: ${r.cut_edges.with_fusion} with fusion, ${r.cut_edges.without_fusion} without. Cores used: ${c.mapping.usage.NeuromorphicCore}, tiles used: ${c.mapping.usage.PIMCrossbar}.`;
}

function pairBars(id, het, gpu, unit) {
  const host = $(id); host.textContent = '';
  const W = 240, Hh = 150, m = { l: 8, r: 8, t: 18, b: 22 }, ih = Hh - m.t - m.b, max = Math.max(het || 0, gpu || 0) || 1;
  const svg = S('svg', { viewBox: `0 0 ${W} ${Hh}`, role: 'img', 'aria-label': `Heterogeneous ${fmt(het)} ${unit}, GPU baseline ${fmt(gpu)} ${unit}` }, host);
  S('line', { class: 'axis', x1: m.l, x2: W - m.r, y1: m.t + ih, y2: m.t + ih }, svg);
  [['Heterogeneous', het, 'var(--ink)', 70], ['GPU baseline', gpu, 'var(--base)', 170]].forEach(([name, v, col, cx]) => {
    if (v == null) { S('text', { class: 'lab', x: cx, y: m.t + ih - 6, 'text-anchor': 'middle' }, svg).textContent = 'n/a'; }
    else {
      const h = Math.max(1, v / max * ih), y = m.t + ih - h, bw = 24, x = cx - bw / 2;
      const b = S('path', { d: h > 5 ? `M${x} ${y + h}V${y + 4}q0-4 4-4h${bw - 8}q4 0 4 4V${y + h}z` : `M${x} ${y + h}V${y}h${bw}V${y + h}z`, style: `fill:${col}`, tabindex: 0 }, svg);
      S('text', { class: 'val', x: cx, y: y - 5, 'text-anchor': 'middle' }, svg).textContent = fmt(v);
      bindTip(b, () => [fmt(v) + ' ' + unit, name]);
    }
    S('text', { class: 'lab', x: cx, y: Hh - 6, 'text-anchor': 'middle' }, svg).textContent = name;
  });
}

function lines(id, t, series, unit) {
  const host = $(id); host.textContent = '';
  const W = 420, Hh = 190, m = { l: 62, r: 12, t: 10, b: 30 }, iw = W - m.l - m.r, ih = Hh - m.t - m.b;
  const all = series.flatMap(s => s.v.filter(x => x != null));
  if (!all.length) { H('p', 'hint', host, 'This workload has no such metric.'); return; }
  const max = Math.max(...all) * 1.08 || 1, tmax = t[t.length - 1] || 1;
  const sx = v => m.l + v / tmax * iw, sy = v => m.t + ih - v / max * ih;
  const svg = S('svg', { viewBox: `0 0 ${W} ${Hh}`, role: 'img', 'aria-label': series.map(s => `${s.name}: last value ${fmt(s.v[s.v.length - 1])} ${unit}`).join('; ') }, host);
  [0, 0.5, 1].forEach(k => { const y = sy(max * k); S('line', { class: 'gridl', x1: m.l, x2: W - m.r, y1: y, y2: y }, svg); S('text', { class: 'tick', x: m.l - 6, y: y + 4, 'text-anchor': 'end' }, svg).textContent = fmt(max * k); });
  S('line', { class: 'axis', x1: m.l, x2: W - m.r, y1: m.t + ih, y2: m.t + ih }, svg);
  [0, 0.5, 1].forEach(k => S('text', { class: 'tick', x: sx(tmax * k), y: Hh - 12, 'text-anchor': 'middle' }, svg).textContent = Math.round(tmax * k));
  S('text', { class: 'tick', x: W - m.r, y: Hh - 1, 'text-anchor': 'end' }, svg).textContent = 'step';
  series.forEach(s => { const pts = s.v.map((v, i) => v == null ? null : `${sx(t[i]).toFixed(1)} ${sy(v).toFixed(1)}`).filter(Boolean); S('path', { d: 'M' + pts.join('L'), fill: 'none', style: `stroke:${s.color}`, 'stroke-width': 2, 'stroke-linejoin': 'round' }, svg); });
  const xh = S('line', { class: 'axis', y1: m.t, y2: m.t + ih, visibility: 'hidden' }, svg);
  svg.addEventListener('pointermove', e => { const r = svg.getBoundingClientRect(), tv = ((e.clientX - r.left) / r.width * W - m.l) / iw * tmax; let i = 0; t.forEach((x, k) => { if (Math.abs(x - tv) < Math.abs(t[i] - tv)) i = k; }); xh.setAttribute('x1', sx(t[i])); xh.setAttribute('x2', sx(t[i])); xh.setAttribute('visibility', 'visible'); showTip(e, ['step ' + t[i], ...series.map(s => `${fmt(s.v[i])} ${unit}  ${s.name}`)]); });
  svg.addEventListener('pointerleave', () => { xh.setAttribute('visibility', 'hidden'); tip.hidden = true; });
}

function render(b) {
  last = b; const c = b.compiled, cmp = b.comparison, h = b.heterogeneous, g = b.gpu_baseline;
  drawDag(c);
  const tiles = $('tiles'); tiles.textContent = '';
  const e = cmp.energy_ratio_gpu_over_hetero, l = cmp.latency_ratio_gpu_over_hetero;
  [[(e >= 1 ? e : 1 / e).toFixed(2) + '×', e >= 1 ? 'less energy than the GPU baseline' : 'more energy than the GPU baseline'],
   [(l >= 1 ? l : 1 / l).toFixed(2) + '×', l >= 1 ? 'lower latency per step' : 'higher latency per step'],
   [cmp.memory_wall_savings_pct.toFixed(1) + '%', 'fewer weight bytes moved between memory and compute'],
   [h.totals.events_per_step.toFixed(1), 'events per step in the heterogeneous run']].forEach(([v, lab]) => { const t = H('div', 'tile', tiles); H('div', 'v', t, v); H('div', 'l', t, lab); });
  pairBars('c-energy', cmp.energy_mj.heterogeneous, cmp.energy_mj.gpu_baseline, 'mJ');
  pairBars('c-latency', cmp.latency_ms.heterogeneous, cmp.latency_ms.gpu_baseline, 'ms');
  pairBars('c-traffic', cmp.interconnect_bytes_per_step.heterogeneous, cmp.interconnect_bytes_per_step.gpu_baseline, 'bytes per step');
  const sh = cmp.stability.heterogeneous, sg = cmp.stability.gpu_baseline;
  const pick = sh.attractor_phase_error_rad_rms != null ? ['attractor_phase_error_rad_rms', 'Attractor phase error (rad, RMS)', 'rad', 'phase_error', 'Attractor phase error (rad)']
    : sh.koopman_deviation_mean != null ? ['koopman_deviation_mean', 'Koopman state deviation (relative, mean)', '', 'koopman_deviation', 'Koopman state deviation (relative)'] : null;
  $('stab-cap').textContent = pick ? pick[1] + ', lower is better' : 'Stability';
  if (pick) pairBars('c-stab', sh[pick[0]], sg[pick[0]], pick[2]); else $('c-stab').textContent = '';
  const two = [{ name: 'Heterogeneous', color: 'var(--ink)' }, { name: 'GPU baseline', color: 'var(--base)' }];
  lines('c-ts-energy', h.series.t, [{ ...two[0], v: h.series.energy_uj }, { ...two[1], v: g.series.energy_uj }], 'µJ');
  const hasDrift = h.series.sync_drift.some(v => v != null);
  const key = hasDrift ? 'sync_drift' : (pick ? pick[3] : 'koopman_deviation');
  $('ts2-cap').textContent = hasDrift ? 'Synchronization drift: fast weights against the last committed snapshot' : (pick ? pick[4] : 'Stability');
  lines('c-ts-drift', h.series.t, [{ ...two[0], v: h.series[key] }, { ...two[1], v: g.series[key] }], '');
  // energy breakdown, heterogeneous
  const bd = $('breakdown'); bd.textContent = '';
  const parts = Object.entries(h.energy_breakdown_uj).filter(([, v]) => v > 0), tot = parts.reduce((a, [, v]) => a + v, 0) || 1;
  const seg = H('div', 'seg', bd); seg.setAttribute('role', 'img'); seg.setAttribute('aria-label', parts.map(([k, v]) => `${CATS[k][1]} ${(100 * v / tot).toFixed(0)}%`).join(', '));
  const lg = H('div', 'legend', bd); lg.style.marginTop = '8px';
  parts.forEach(([k, v], i) => { const d = H('div', '', seg); d.style.flex = v / tot; d.style.background = `var(${CATS[k][0]})`; d.style.opacity = 1 - 0.22 * (parts.slice(0, i).filter(([kk]) => CATS[kk][0] === CATS[k][0]).length); d.tabIndex = 0; bindTip(d, () => [`${fmt(v)} µJ (${(100 * v / tot).toFixed(1)}%)`, CATS[k][1]]);
    const s = H('span', '', lg); const sw = H('i', 'box', s); sw.style.background = `var(${CATS[k][0]})`; sw.style.opacity = d.style.opacity; s.appendChild(document.createTextNode(`${CATS[k][1]} ${(100 * v / tot).toFixed(0)}%`)); });
  const tb = $('node-table'); tb.textContent = '';
  const hr = H('tr', '', H('thead', '', tb)); ['Node', 'Target', 'Energy (µJ)', 'GPU baseline (µJ)', 'Events', 'Dense MACs'].forEach((x, i) => H('th', i > 1 ? 'num' : '', hr, x));
  const body = H('tbody', '', tb);
  Object.entries(h.per_node).forEach(([id, n]) => { const tr = H('tr', '', body); H('td', '', tr, '%' + id); H('td', '', tr, TARGETS[n.target][1]); [n.energy_uj, (g.per_node[id] || {}).energy_uj, n.events, n.macs].forEach(v => H('td', 'num', tr, fmt(v))); });
  $('ir-out').textContent = c.ir;
  $('dl-ir').disabled = $('dl-map').disabled = false;
}

function download(name, text, type) { const a = document.createElement('a'); a.href = URL.createObjectURL(new Blob([text], { type })); a.download = name; document.body.appendChild(a); a.click(); a.remove(); setTimeout(() => URL.revokeObjectURL(a.href), 1000); }
function status(msg, err) { const s = $('status'); s.textContent = msg; s.className = 'status' + (err ? ' err' : ''); }

async function run() {
  let body; try { body = request(); } catch (e) { status(e.message, true); return; }
  $('run').disabled = true; status('Compiling and simulating…');
  try { const b = await api('/api/benchmark', body); render(b); status(`Done: ${b.steps} steps of ${b.module}.`); history.replaceState(null, '', '#s=' + enc(state())); }
  catch (e) { status(e.message, true); }
  finally { $('run').disabled = false; }
}
function fillEditor() { const p = presets.find(x => x.id === $('preset').value); if (p) { $('editor').value = mode === 'ir' ? p.ir : JSON.stringify(p.graph, null, 2); $('preset-summary').textContent = p.summary; } else $('preset-summary').textContent = 'Edit the source below. It is validated when you run.'; }
function setMode(m) { if (m === mode) return; const custom = $('preset').value === 'custom'; mode = m; $('tab-ir').setAttribute('aria-selected', m === 'ir'); $('tab-json').setAttribute('aria-selected', m === 'json'); $('tab-ir').tabIndex = m === 'ir' ? 0 : -1; $('tab-json').tabIndex = m === 'json' ? 0 : -1; if (!custom) fillEditor(); }

async function init() {
  presets = await api('/api/presets');
  presets.forEach(p => { const o = H('option', '', $('preset'), p.title); o.value = p.id; });
  H('option', '', $('preset'), 'Custom (edit below)').value = 'custom';
  $('preset').addEventListener('change', fillEditor);
  $('editor').addEventListener('input', () => { const p = presets.find(x => x.id === $('preset').value); if (p && $('editor').value !== (mode === 'ir' ? p.ir : JSON.stringify(p.graph, null, 2))) { $('preset').value = 'custom'; $('preset-summary').textContent = 'Edited: this is now a custom workload.'; } });
  $('tab-ir').addEventListener('click', () => setMode('ir')); $('tab-json').addEventListener('click', () => setMode('json'));
  $('run').addEventListener('click', run);
  $('share').addEventListener('click', async () => { const u = shareUrl(); try { await navigator.clipboard.writeText(u); status('Share link copied.'); } catch (_) { history.replaceState(null, '', u); status('Share link is in the address bar.'); } });
  $('dl-ir').addEventListener('click', () => last && download(last.module + '.mlir', last.compiled.ir, 'text/plain'));
  $('dl-map').addEventListener('click', () => last && download(last.module + '.mapping.json', JSON.stringify(last.compiled.mapping, null, 2), 'application/json'));
  let st = null; if (location.hash.startsWith('#s=')) { try { st = dec(location.hash.slice(3)); } catch (_) { st = null; } }
  if (st) { mode = st.m === 'json' ? 'json' : 'ir'; $('tab-ir').setAttribute('aria-selected', mode === 'ir'); $('tab-json').setAttribute('aria-selected', mode === 'json'); $('preset').value = presets.some(p => p.id === st.p) ? st.p : 'custom'; if (st.steps) $('steps').value = st.steps; if (st.sync) $('sync').value = st.sync; $('fusion').checked = st.f !== false; if (st.hw) $('hw').value = st.hw; fillEditor(); if (st.src != null) { $('editor').value = st.src; $('preset').value = 'custom'; } }
  else { $('preset').value = 'hippocampal_ring'; fillEditor(); }
  run();
}
init().catch(e => status('Could not load presets: ' + e.message, true));
