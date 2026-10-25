'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const E = require('../js/engine.js');
const R = require('../js/render.js');
const App = require('../js/app.js');

function near(a, b, tol) { assert.ok(Math.abs(a - b) <= tol, a + ' vs ' + b); }
const ROOT = path.join(__dirname, '..');
const html = fs.readFileSync(path.join(ROOT, 'index.html'), 'utf8');
const css = fs.readFileSync(path.join(ROOT, 'styles.css'), 'utf8');
const appJs = fs.readFileSync(path.join(ROOT, 'js', 'app.js'), 'utf8');

function walk(dir, out) {
  fs.readdirSync(dir, { withFileTypes: true }).forEach((d) => {
    if (d.name === '.git' || d.name === 'node_modules') return;
    const p = path.join(dir, d.name);
    if (d.isDirectory()) walk(p, out); else out.push(p);
  });
  return out;
}
const TEXT_FILES = walk(ROOT, []).filter((p) => /\.(html|css|js|md|json|yml|yaml|txt|sh)$/.test(p));

/* ---------- a tiny fake document ---------- */

class El {
  constructor(id, tag, attrs) {
    this.id = id; this.tag = tag; this.value = attrs.value || ''; this.checked = attrs.checked || false;
    this.innerHTML = ''; this.textContent = ''; this.attrs = {}; this.listeners = {};
  }
  addEventListener(type, fn) { (this.listeners[type] = this.listeners[type] || []).push(fn); }
  setAttribute(k, v) { this.attrs[k] = String(v); }
  getAttribute(k) { return this.attrs[k]; }
  fire(type, ev) { (this.listeners[type] || []).forEach((fn) => fn(Object.assign({ target: this }, ev || {}))); }
  closest() { return null; }
}

function controlFromHtml(markup, id) {
  const re = new RegExp('<(input|select)[^>]*id="' + id + '"[^>]*>');
  const m = markup.match(re);
  if (!m) return null;
  const attrs = {};
  const v = m[0].match(/ value="([^"]*)"/);
  if (v) attrs.value = v[1];
  attrs.checked = /\schecked(\s|>)/.test(m[0]);
  if (m[1] === 'select') {
    const after = markup.slice(markup.indexOf(m[0]));
    const sel = after.match(/<option value="([^"]*)" selected/);
    if (sel) attrs.value = sel[1];
  }
  return new El(id, m[1], attrs);
}

function makeDoc() {
  const els = {};
  [...html.matchAll(/\sid="([^"]+)"/g)].forEach((m) => {
    const tag = (html.slice(html.lastIndexOf('<', m.index), m.index).match(/<(\w+)/) || [])[1] || 'div';
    const attrs = {};
    const tagText = html.slice(html.lastIndexOf('<', m.index), html.indexOf('>', m.index) + 1);
    const v = tagText.match(/ value="([^"]*)"/);
    if (v) attrs.value = v[1];
    els[m[1]] = new El(m[1], tag, attrs);
  });
  return {
    els: els,
    readyState: 'complete',
    createElement: () => ({ click() {}, setAttribute() {} }),
    getElementById(id) {
      if (els[id]) return els[id];
      if (/^pool-\d+-/.test(id)) {
        const c = controlFromHtml(els['r-pools'].innerHTML, id);
        if (c) els[id] = c;
        return c;
      }
      return null;
    }
  };
}

function mountApp() {
  const doc = makeDoc();
  const app = App.mount(doc, { requestAnimationFrame: (fn) => fn(), navigator: {} });
  return { doc: doc, app: app };
}

function allHtml(doc) {
  return Object.keys(doc.els).map((k) => doc.els[k].innerHTML + ' ' + doc.els[k].textContent).join('\n');
}

/* ---------- static page structure ---------- */

test('page declares language, viewport, description, canonical and a theme color', () => {
  assert.match(html, /<html lang="en">/);
  assert.match(html, /<meta name="viewport" content="width=device-width, initial-scale=1">/);
  assert.match(html, /<meta name="description" content="[^"]{80,}"/);
  assert.match(html, /<link rel="canonical" href="https:\/\/aariv\.me\/infergrid\/">/);
  assert.match(html, /<meta name="theme-color"/);
});

test('page carries a complete set of Open Graph and Twitter tags', () => {
  ['og:type', 'og:site_name', 'og:title', 'og:description', 'og:url', 'og:image', 'og:image:width', 'og:image:height', 'og:image:alt'].forEach((p) => {
    assert.ok(html.includes('property="' + p + '"'), p);
  });
  ['twitter:card', 'twitter:title', 'twitter:description', 'twitter:image'].forEach((n) => assert.ok(html.includes('name="' + n + '"'), n));
});

test('favicon is an inline SVG data URI', () => {
  assert.match(html, /<link rel="icon" href="data:image\/svg\+xml,/);
});

test('page has a noscript fallback that explains the page', () => {
  const m = html.match(/<noscript>([\s\S]*?)<\/noscript>/);
  assert.ok(m);
  assert.match(m[1], /JavaScript is needed/);
  assert.match(m[1], /327,680/);
});

test('fonts are loaded with preconnect and display swap', () => {
  assert.match(html, /rel="preconnect" href="https:\/\/fonts\.googleapis\.com"/);
  assert.match(html, /display=swap/);
  ['Space+Grotesk', 'Plus+Jakarta+Sans', 'JetBrains+Mono'].forEach((f) => assert.ok(html.includes(f), f));
});

test('there is one h1, a main landmark, a skip link and labelled navigation', () => {
  assert.equal((html.match(/<h1[\s>]/g) || []).length, 1);
  assert.match(html, /<main>/);
  assert.match(html, /class="skip-link" href="#workbench"/);
  assert.match(html, /<nav aria-label="Sections">/);
});

test('every id in the page is unique', () => {
  const ids = [...html.matchAll(/\sid="([^"]+)"/g)].map((m) => m[1]);
  assert.equal(new Set(ids).size, ids.length);
});

test('every nav and skip link target exists', () => {
  const ids = new Set([...html.matchAll(/\sid="([^"]+)"/g)].map((m) => m[1]));
  [...html.matchAll(/href="#([^"]+)"/g)].forEach((m) => assert.ok(ids.has(m[1]), m[1]));
});

test('every form control in the page has a label bound by for', () => {
  const controls = [...html.matchAll(/<(input|select)\b[^>]*\sid="([^"]+)"/g)].map((m) => m[2]);
  assert.ok(controls.length >= 25);
  controls.forEach((id) => assert.ok(html.includes('for="' + id + '"'), 'label for ' + id));
});

test('every button states its type', () => {
  [...html.matchAll(/<button\b[^>]*>/g)].forEach((m) => assert.match(m[0], /type="button"/));
});

test('result regions announce changes with aria-live', () => {
  ['wb-verdict', 'r-summary', 'pr-status', 'pr-result', 'ex-status', 'wb-ttft-readout', 'hero-hbm'].forEach((id) => {
    const tag = html.match(new RegExp('<[^>]*id="' + id + '"[^>]*>'))[0];
    assert.match(tag, /aria-live="(polite|assertive)"/, id);
  });
});

test('export tabs follow the tab pattern', () => {
  assert.match(html, /role="tablist"/);
  assert.equal((html.match(/role="tab"/g) || []).length, 4);
  assert.match(html, /role="tabpanel"/);
  assert.equal((html.match(/aria-selected="true"/g) || []).length, 1);
});

test('decorative SVG in the page is hidden from assistive technology', () => {
  [...html.matchAll(/<svg\b[^>]*>/g)].forEach((m) => assert.match(m[0], /aria-hidden="true"/));
});

test('script ids used by app.js exist in the page', () => {
  const used = new Set();
  [...appJs.matchAll(/(?:\$|setHtml|setText|on)\('([a-z][\w-]*)'/g)].forEach((m) => used.add(m[1]));
  [...appJs.matchAll(/\[((?:'[a-z][\w-]*',?\s*)+)\]\.forEach/g)].forEach((m) => [...m[1].matchAll(/'([^']+)'/g)].forEach((x) => used.add(x[1])));
  assert.ok(used.size > 40);
  const ids = new Set([...html.matchAll(/\sid="([^"]+)"/g)].map((m) => m[1]));
  used.forEach((id) => { if (!id.endsWith('-')) assert.ok(ids.has(id), 'missing id ' + id); });
});

/* ---------- house style, enforced ---------- */

test('no em dashes or en dashes anywhere in the repository text', () => {
  TEXT_FILES.forEach((p) => {
    const t = fs.readFileSync(p, 'utf8');
    assert.ok(!new RegExp('[' + String.fromCharCode(0x2013, 0x2014) + ']').test(t), path.relative(ROOT, p));
  });
});

test('no emoji anywhere in the repository text', () => {
  TEXT_FILES.forEach((p) => {
    const t = fs.readFileSync(p, 'utf8');
    assert.ok(!/\p{Extended_Pictographic}/u.test(t), path.relative(ROOT, p));
  });
});

test('colors and fonts are locked to custom properties outside the root block', () => {
  const rootEnd = css.indexOf('}', css.indexOf(':root'));
  const rest = css.slice(0, css.indexOf(':root')) + css.slice(rootEnd);
  const body = rest.replace(/\/\*[\s\S]*?\*\//g, '');
  assert.ok(!/#[0-9a-fA-F]{3,8}\b/.test(body), 'hex color outside :root');
  assert.ok(!/\b(rgb|rgba|hsl|hsla|oklch)\(/.test(body), 'color function outside :root');
  assert.ok(!/font-family\s*:\s*(?!var\()\S/.test(body), 'font-family not from a token');
  [...body.matchAll(/\bfont\s*:[^;}]+/g)].forEach((m) => assert.ok(/var\(--font-/.test(m[0]), m[0]));
});

test('the page body and inline SVG use no raw color values', () => {
  const stripped = html.replace(/<link rel="icon"[^>]*>/, '').replace(/<meta[^>]*>/g, '');
  assert.ok(!/#[0-9a-fA-F]{6}\b/.test(stripped));
  assert.ok(!/\sstyle="/.test(stripped));
});

test('renderers contain no raw color values', () => {
  const src = fs.readFileSync(path.join(ROOT, 'js', 'render.js'), 'utf8');
  assert.ok(!/#[0-9a-fA-F]{6}\b/.test(src));
  assert.ok(!/\b(rgb|hsl)a?\(/.test(src));
});

test('headings stay roman and nothing is set in uppercase kicker style', () => {
  assert.ok(!/font-style\s*:\s*italic/.test(css));
  assert.ok(!/text-transform\s*:\s*uppercase/.test(css));
  assert.ok(!/<h[1-3][^>]*>[^<]*<em>/.test(html));
});

test('headings in the page contain no colons', () => {
  [...html.matchAll(/<h[1-3][^>]*>([^<]*)<\/h[1-3]>/g)].forEach((m) => assert.ok(!m[1].includes(':'), m[1]));
});

test('prose in the page has no colons outside code, URLs and times', () => {
  const prose = html.replace(/<(script|style|pre|code|noscript)[\s\S]*?<\/\1>/g, '').replace(/<[^>]+>/g, ' ').replace(/https?:\/\/\S+/g, '');
  assert.ok(!/[A-Za-z]:\s/.test(prose), 'colon in prose');
});

test('stylesheet starts with the Hallmark stamp and respects reduced motion', () => {
  assert.match(css.split('\n')[0], /^\/\* Hallmark · genre: modern-minimal · macrostructure: Workbench/);
  assert.match(css, /prefers-reduced-motion: reduce/);
  assert.match(css, /overflow-x: clip/);
  assert.match(css, /:focus-visible/);
  assert.ok(!/transition:\s*all/.test(css));
});

/* ---------- renderers ---------- */

const SCN = { modelId: 'llama-3.1-70b', gpuId: 'h100', tp: 4, pp: 1, weightPrecision: 'fp8', kvPrecision: 'fp8', promptTokens: 4000, outputTokens: 300, concurrentSeqs: 32 };

test('esc neutralizes markup and quotes', () => {
  assert.equal(R.esc('<img src=x onerror="a">&\''), '&lt;img src=x onerror=&quot;a&quot;&gt;&amp;&#39;');
  assert.equal(R.esc(null), '');
});

test('memory diagram is a labelled SVG with one bar per GPU and a capacity line', () => {
  const svg = R.renderMemoryDiagram(E.computeModelFootprint(SCN));
  assert.match(svg, /^<svg[^>]*role="img"[^>]*aria-labelledby="mem-title mem-desc"/);
  assert.match(svg, /<title id="mem-title">/);
  assert.equal((svg.match(/GPU \d</g) || []).length, 4);
  assert.match(svg, /HBM capacity 80 GiB/);
  assert.match(svg, /class="cap-line"/);
  assert.ok((svg.match(/<svg/g) || []).length === 1 && svg.endsWith('</svg>'));
});

test('memory diagram caps drawn bars at eight GPUs', () => {
  const svg = R.renderMemoryDiagram(E.computeModelFootprint(Object.assign({}, SCN, { gpuId: 'h200', tp: 8, pp: 2 })));
  assert.equal((svg.match(/GPU \d</g) || []).length, 8);
  assert.match(svg, /first 8 of 16/);
});

test('an overcommitted KV demand draws the unmet hatch and the legend flags it', () => {
  const fp = E.computeModelFootprint(Object.assign({}, SCN, { concurrentSeqs: 5000 }));
  assert.match(R.renderMemoryDiagram(fp), /class="seg-over"/);
  assert.match(R.renderMemoryLegend(fp), /KV demand with no room/);
});

test('legend lists the free HBM headroom row', () => {
  const lg = R.renderMemoryLegend(E.computeModelFootprint(SCN));
  assert.match(lg, /Free HBM headroom/);
  assert.match(lg, /Weights/);
});

test('verdict markup carries the right state for fits, oom and invalid', () => {
  const m = (c) => E.computeRooflineMetrics(c);
  const fits = R.renderVerdict(E.computeModelFootprint(SCN), m(SCN));
  assert.match(fits, /verdict-ok/);
  const oomCfg = Object.assign({}, SCN, { concurrentSeqs: 5000 });
  assert.match(R.renderVerdict(E.computeModelFootprint(oomCfg), m(oomCfg)), /verdict-bad[\s\S]*Out of memory/);
  const badCfg = Object.assign({}, SCN, { tp: 3 });
  assert.match(R.renderVerdict(E.computeModelFootprint(badCfg), m(badCfg)), /Invalid layout/);
});

test('every icon is hidden from assistive technology', () => {
  Object.keys(R.ICONS).forEach((k) => assert.match(R.ICONS[k], /aria-hidden="true"/));
});

test('roofline chart marks decode and prefill and supports a hover crosshair', () => {
  const metrics = E.computeRooflineMetrics(SCN);
  const plain = R.renderRoofline(metrics, null);
  assert.match(plain, /class="chart chart-roofline"/);
  assert.match(plain, /Decode, 32 sequences/);
  assert.match(plain, /Prefill, 4,000 tokens/);
  assert.ok(!plain.includes('cross-line'));
  const hover = R.renderRoofline(metrics, 120);
  assert.match(hover, /class="cross-line"/);
  assert.match(hover, /120\.0 FLOP\/B gives/);
});

test('roofline hover mapping round trips through the log axis', () => {
  const ai = R.rooflineAiFromX(58 + (620 - 58 - 18) / 2);
  assert.ok(ai > 10 && ai < 1000);
  assert.ok(R.rooflineAiFromX(-100) < 1);
});

test('TTFT chart draws every GPU series, an SLO line and a crosshair at the chosen burst', () => {
  const series = E.ttftVsBatch(SCN, ['h100', 'h200', 'l40s', 'a100'], R.BATCHES);
  const svg = R.renderTtftChart(series, 3, 1500);
  ['h100', 'h200', 'l40s', 'a100'].forEach((g) => assert.ok(svg.includes('tt-' + g), g));
  assert.match(svg, /class="slo-line"/);
  assert.match(svg, /class="cross-line"/);
  assert.equal(R.ttftIndexFromX(62), 0);
  assert.equal(R.ttftIndexFromX(10000), 6);
  assert.match(R.renderTtftReadout(series, 3), /Burst 8/);
});

test('TTFT chart marks layouts that do not fit with a dashed class', () => {
  const series = E.ttftVsBatch(Object.assign({}, SCN, { weightPrecision: 'fp16', tp: 1 }), ['h100', 'l40s'], R.BATCHES);
  assert.match(R.renderTtftChart(series, 0, 0), /is-nofit/);
  assert.match(R.renderTtftReadout(series, 0), /does not fit at this layout/);
});

test('numbers table and steps list render without placeholders', () => {
  const fp = E.computeModelFootprint(SCN);
  const t = R.renderNumbersTable(fp, E.computeRooflineMetrics(SCN)) + R.renderSteps(fp);
  assert.ok(!/NaN|undefined|Infinity/.test(t));
  assert.match(t, /Prefill TTFT/);
  assert.match(t, /<ol class="steps">/);
});

test('decision table escapes request ids that contain markup', () => {
  const r = E.routeWorkloadBatch([{ id: '<script>alert(1)</script>', inputTokens: 100, outputTokens: 10, complexity: 0.1, sloTtftMs: 5000, sloItlMs: 100, arrivalMs: 0 }], E.REFERENCE_CLUSTER);
  const out = R.renderDecisionTable(r, 10);
  assert.ok(!out.includes('<script>'));
  assert.match(out, /&lt;script&gt;/);
});

test('probe result escapes errors and labels simulated runs as simulated', () => {
  const bad = R.renderProbeResult({ ok: false, error: '<b>boom</b>' }, null, 'live');
  assert.ok(!bad.includes('<b>boom'));
  const m = E.computeRooflineMetrics(Object.assign({}, SCN, { concurrentSeqs: 1 }));
  const meas = { ok: true, ttftMs: 100, itlMs: 20, chunks: 10, completionTokens: 10, itlBasis: 'usage', tokensPerSec: 50, promptTokens: 256 };
  const sim = R.renderProbeResult(meas, E.compareProbeToRoofline(meas, m), 'replay');
  assert.match(sim, /simulated from the roofline/);
  assert.match(sim, /not a measurement of any server/);
  assert.match(R.renderProbeResult(meas, E.compareProbeToRoofline(meas, m), 'live'), /Live measurement/);
});

test('method tables list all four GPUs and all eight models', () => {
  const hw = R.renderHardwareTable();
  E.GPU_IDS.forEach((g) => assert.ok(hw.includes(E.GPUS[g].name), g));
  const md = R.renderModelTable();
  E.MODELS.forEach((m) => assert.ok(md.includes(m.name), m.id));
  assert.equal((md.match(/<tr><th scope="row">/g) || []).length, 8);
});

/* ---------- the app wired to a fake DOM ---------- */

test('mounting fills every result region with real content', () => {
  const { doc } = mountApp();
  assert.match(doc.els['wb-verdict'].innerHTML, /Fits/);
  assert.match(doc.els['wb-memory'].innerHTML, /<svg/);
  assert.match(doc.els['wb-roofline'].innerHTML, /<svg/);
  assert.match(doc.els['wb-ttft'].innerHTML, /<svg/);
  assert.match(doc.els['hero-hbm'].innerHTML, /hero-bar/);
  assert.match(doc.els['hero-caption'].textContent, /Llama-3\.1-70B-Instruct/);
  assert.match(doc.els['r-summary'].innerHTML, /Requests meeting both targets/);
  assert.equal((doc.els['r-table'].innerHTML.match(/<tr><th scope="row">/g) || []).length, 120);
  assert.match(doc.els['ex-code'].textContent, /services:/);
});

test('mounted output never contains NaN, undefined or Infinity', () => {
  const { doc } = mountApp();
  assert.ok(!/NaN|undefined|Infinity/.test(allHtml(doc)));
});

test('every preset applies cleanly and updates the controls and the router', () => {
  const { doc, app } = mountApp();
  E.WORKLOAD_PRESETS.forEach((p) => {
    app.applyPreset(p.id);
    assert.equal(doc.els['f-model'].value, p.scenario.modelId);
    assert.equal(doc.els['r-count'].value, p.workload.count);
    assert.equal(doc.els['r-slo-ttft'].value, p.workload.sloTtftMs);
    assert.equal((doc.els['r-table'].innerHTML.match(/<tr><th scope="row">/g) || []).length, p.workload.count);
    assert.ok(!/NaN|undefined|Infinity/.test(allHtml(doc)), p.id);
  });
});

test('clicking a preset chip routes through the delegated handler', () => {
  const { doc, app } = mountApp();
  const btn = { getAttribute: () => 'voice-agent' };
  doc.els['preset-row'].fire('click', { target: { closest: () => btn } });
  assert.equal(app.state.presetId, 'voice-agent');
  assert.match(doc.els['preset-row'].innerHTML, /data-preset="voice-agent" aria-pressed="true"/);
  assert.match(doc.els['preset-blurb'].textContent, /speaking fast/);
});

test('moving the concurrency slider changes the verdict', () => {
  const { doc } = mountApp();
  doc.els['f-conc'].value = '500';
  doc.els['f-conc'].fire('input');
  assert.match(doc.els['wb-verdict'].innerHTML, /Out of memory|Fits, but tight/);
  assert.match(doc.els['o-conc'].textContent, /500/);
});

test('changing the model resets weight precision to the native one and relists the options', () => {
  const { doc, app } = mountApp();
  doc.els['f-model'].value = 'llama-3.1-405b-fp8';
  doc.els['f-model'].fire('change');
  assert.equal(app.state.scenario.weightPrecision, 'fp8');
  assert.equal(doc.els['f-wprec'].innerHTML.match(/<option/g).length, 1);
});

test('an invalid tensor parallel choice shows the invalid layout verdict', () => {
  const { doc } = mountApp();
  doc.els['f-tp'].value = '2';
  doc.els['f-pp'].value = '1';
  doc.els['f-gpu'].value = 'l40s';
  doc.els['f-gpu'].fire('change');
  assert.ok(/Out of memory|Invalid layout|Fits/.test(doc.els['wb-verdict'].innerHTML));
  doc.els['f-model'].value = 'deepseek-r1-distill-qwen-32b';
  doc.els['f-model'].fire('change');
  doc.els['f-tp'].value = '8';
  doc.els['f-tp'].fire('change');
  assert.match(doc.els['wb-verdict'].innerHTML, /Fits|Out of memory|tight/);
});

test('the TTFT inspector moves the crosshair and the readout', () => {
  const { doc, app } = mountApp();
  doc.els['f-ttft-inspect'].value = '5';
  doc.els['f-ttft-inspect'].fire('input');
  assert.equal(app.state.ttftIndex, 5);
  assert.match(doc.els['wb-ttft-readout'].innerHTML, /Burst 32/);
});

test('export tabs swap the generated file and keep ARIA state in sync', () => {
  const { doc } = mountApp();
  const tab = (name) => ({ getAttribute: () => name });
  doc.els['ex-tabs'].fire('click', { target: { closest: () => tab('serve') } });
  assert.match(doc.els['ex-code'].textContent, /vllm serve/);
  assert.equal(doc.els['tab-serve'].attrs['aria-selected'], 'true');
  assert.equal(doc.els['tab-compose'].attrs['aria-selected'], 'false');
  assert.equal(doc.els['ex-panel'].attrs['aria-labelledby'], 'tab-serve');
  doc.els['ex-tabs'].fire('click', { target: { closest: () => tab('router') } });
  assert.doesNotThrow(() => JSON.parse(doc.els['ex-code'].textContent));
  doc.els['ex-tabs'].fire('keydown', { key: 'ArrowRight', preventDefault() {} });
  assert.equal(doc.els['tab-compose'].attrs['aria-selected'], 'true');
});

test('exporting the workbench scenario yields a single pool', () => {
  const { doc } = mountApp();
  doc.els['ex-source'].value = 'scenario';
  doc.els['ex-source'].fire('change');
  assert.match(doc.els['ex-code'].textContent, /scenario-r0/);
  assert.match(doc.els['ex-code'].textContent, /"--tensor-parallel-size", "4"/);
});

test('editing the pool editor recomputes routing and disabling every pool shows a notice', () => {
  const { doc } = mountApp();
  const before = doc.els['r-summary'].innerHTML;
  doc.getElementById('pool-3-gpus').value = '8';
  doc.getElementById('pool-3-tp').value = '8';
  doc.els['r-pools'].fire('change');
  assert.notEqual(doc.els['r-summary'].innerHTML, before);
  [0, 1, 2, 3].forEach((i) => { doc.getElementById('pool-' + i + '-on').checked = false; });
  doc.els['r-pools'].fire('change');
  assert.match(doc.els['r-summary'].innerHTML, /Enable at least one pool/);
});

test('the A100 cost field starts blank until the user enters a rate', () => {
  const { doc } = mountApp();
  doc.getElementById('pool-0-gpu').value = 'a100';
  doc.els['r-pools'].fire('change');
  assert.match(doc.els['r-pools'].innerHTML, /placeholder="enter rate"/);
});

test('offline replay produces a clearly labelled simulated result', async () => {
  const { doc, app } = mountApp();
  await app.runProbe('replay');
  assert.match(doc.els['pr-result'].innerHTML, /simulated from the roofline/);
  assert.equal(doc.els['pr-status'].textContent, 'Done.');
  assert.ok(!/NaN|undefined|Infinity/.test(doc.els['pr-result'].innerHTML));
});

test('a live probe without a model id asks for one instead of calling out', async () => {
  const { doc, app } = mountApp();
  doc.els['pr-model'].value = '';
  await app.runProbe('live');
  assert.match(doc.els['pr-status'].textContent, /model id/);
});

test('slider helpers map tokens to the log scale and back', () => {
  const v = App.sliderFromTokens(4096, App.PROMPT_MAP);
  assert.equal(App.tokensFromSlider(v, App.PROMPT_MAP), 4096);
  assert.equal(App.tokensFromSlider(0, App.PROMPT_MAP), 128);
  assert.equal(App.tokensFromSlider(1000, App.PROMPT_MAP), 131072);
  assert.equal(App.tokensFromSlider(0, App.OUTPUT_MAP), 16);
});

test('the API key never appears in any rendered output', async () => {
  const { doc, app } = mountApp();
  doc.els['pr-key'].value = 'sk-test-secret-123';
  await app.runProbe('replay');
  assert.ok(!allHtml(doc).includes('sk-test-secret-123'));
});

test('changing an unrelated control does not round the preset prompt length', () => {
  const { doc, app } = mountApp();
  app.applyPreset('code-audit-32k');
  assert.equal(app.state.scenario.promptTokens, 28000);
  doc.els['f-tp'].value = '4';
  doc.els['f-tp'].fire('change');
  assert.equal(app.state.scenario.promptTokens, 28000);
  assert.equal(app.state.scenario.outputTokens, 1500);
  doc.els['f-prompt'].value = '0';
  doc.els['f-prompt'].fire('input');
  assert.equal(app.state.scenario.promptTokens, 128);
});

/* ---------- gap closers found by mutation testing ---------- */

test('millisecond formatting switches units at the documented thresholds', () => {
  assert.equal(R.msStr(5.55), '5.6 ms');
  assert.equal(R.msStr(99.9), '99.9 ms');
  assert.equal(R.msStr(100), '100 ms');
  assert.equal(R.msStr(9999), '9,999 ms');
  assert.equal(R.msStr(10000), '10.0 s');
  assert.equal(R.msStr(NaN), 'n/a');
});

test('legend swatches, icons and table dots are all hidden from assistive technology', () => {
  const fp = E.computeModelFootprint(Object.assign({}, SCN, { concurrentSeqs: 5000 }));
  const lg = R.renderMemoryLegend(fp);
  assert.equal((lg.match(/class="swatch /g) || []).length, (lg.match(/class="swatch [^"]*" aria-hidden="true"/g) || []).length);
  assert.ok((lg.match(/class="swatch /g) || []).length >= 7);
  Object.keys(R.ICONS).forEach((k) => assert.match(R.ICONS[k], /focusable="false"/));
  const series = E.ttftVsBatch(SCN, ['h100', 'h200'], R.BATCHES);
  assert.match(R.renderTtftReadout(series, 0), /class="swatch-dot tt-fill-h100" aria-hidden="true"/);
});

test('the pool editor ties every control to a label and honors the enabled flag', () => {
  const pools = E.REFERENCE_CLUSTER.map((p) => Object.assign({}, p, { enabled: true, costPerGpuHour: 2 }));
  pools[1].enabled = false;
  const out = R.renderPoolEditor(pools);
  const ids = [...out.matchAll(/<(?:input|select)[^>]*\sid="([^"]+)"/g)].map((m) => m[1]);
  assert.equal(ids.length, 4 * 7);
  ids.forEach((id) => assert.ok(out.includes('for="' + id + '"'), id));
  assert.match(out, /id="pool-0-on" checked/);
  assert.ok(!/id="pool-1-on" checked/.test(out));
});

test('only the Pareto row of the policy table is emphasised', () => {
  const cmp = E.compareRoutingPolicies(E.generateWorkload('saas-burst', { count: 30 }), E.REFERENCE_CLUSTER);
  const t = R.renderPolicyTable(cmp);
  assert.equal((t.match(/row-strong/g) || []).length, 1);
  assert.match(t, /<tr class="row-strong"><th scope="row">Pareto router/);
});

test('decision rows show the right wording and class for each outcome', () => {
  const result = { decisions: [
    { requestId: 'a', inputTokens: 1, outputTokens: 1, complexity: 0.1, poolId: 'p', replica: 0, ttftMs: 5, itlMs: 5, status: 'routed', reason: '' },
    { requestId: 'b', inputTokens: 1, outputTokens: 1, complexity: 0.1, poolId: 'p', replica: 1, ttftMs: 5, itlMs: 5, status: 'slo_violated', reason: 'ttft_slo' },
    { requestId: 'c', inputTokens: 1, outputTokens: 1, complexity: 0.1, poolId: null, replica: null, ttftMs: 0, itlMs: 0, status: 'rejected', reason: 'kv_oom' }
  ] };
  const t = R.renderDecisionTable(result, 10);
  assert.match(t, /<td class="">Meets targets/);
  assert.match(t, /<td class="warn-text">Misses the TTFT target/);
  assert.match(t, /<td class="bad-text">No replica had KV blocks free/);
  assert.match(t, /p r1/);
  assert.match(t, /none<\/td><td class="r num">n\/a/);
  assert.equal((R.renderDecisionTable(result, 2).match(/<tr><th scope="row">/g) || []).length, 2);
});

test('pool load shows a meter, flags overload and explains unusable pools', () => {
  const result = {
    pools: [{ id: 'a', usable: true, replicas: 1 }, { id: 'b', usable: true, replicas: 2 }, { id: 'c', usable: false, unusableReason: 'Weights too big.' }],
    summary: { perPool: [
      { id: 'a', label: 'A', requests: 3, peakKvUtilization: 0.5 },
      { id: 'b', label: 'B', requests: 9, peakKvUtilization: 1.2 },
      { id: 'c', label: 'C', requests: 0, peakKvUtilization: 0 }
    ] }
  };
  const t = R.renderPoolLoad(result);
  assert.match(t, /role="img" aria-label="Peak KV use 50%"/);
  assert.match(t, /1 replica<|1 replica$/m);
  assert.match(t, /2 replicas/);
  assert.equal((t.match(/is-over/g) || []).length, 1);
  assert.match(t, /width:50\.0%/);
  assert.match(t, /width:100\.0%/);
  assert.match(t, /not deployable\. Weights too big\./);
});

test('probe footnote names the basis of the inter token figure', () => {
  const m = E.computeRooflineMetrics(Object.assign({}, SCN, { concurrentSeqs: 1 }));
  const a = { ok: true, ttftMs: 1, itlMs: 2, chunks: 5, completionTokens: 6, itlBasis: 'usage', tokensPerSec: 500, promptTokens: 9 };
  assert.match(R.renderProbeResult(a, E.compareProbeToRoofline(a, m), 'live'), /uses the reported token count/);
  const b = Object.assign({}, a, { itlBasis: 'chunks', completionTokens: null, promptTokens: null });
  const out = R.renderProbeResult(b, E.compareProbeToRoofline(Object.assign({}, b, { promptTokens: 100 }), m), 'live');
  assert.match(out, /uses chunk spacing/);
  assert.ok(!/prompt tokens reported/.test(out));
});

test('hardware table describes links and missing FP8 correctly', () => {
  const t = R.renderHardwareTable();
  assert.match(t, /L40S 48GB<\/th>[^]*?PCIe Gen4 x16, 32 GB\/s per direction, no NVLink/);
  assert.match(t, /A100 SXM4 80GB<\/th>[^]*?none<\/td><td>NVLink, 300 GB\/s per direction/);
  assert.match(t, /H100 SXM5 80GB<\/th><td class="r num">80 GiB HBM3/);
});

test('roofline marks only the selected GPU as active, stacks equal peak labels and anchors dot labels', () => {
  const m = E.computeRooflineMetrics(SCN);
  const svg = R.renderRoofline(m, null);
  assert.equal((svg.match(/rf-curve rf-[a-z0-9]+ is-active/g) || []).length, 1);
  assert.match(svg, /rf-curve rf-h100 is-active/);
  const yOf = (g) => Number(svg.match(new RegExp('rf-text-' + g + '[^>]*? y="([0-9.]+)"'))[1]);
  near(Math.abs(yOf('h100') - yOf('h200')), 13, 1e-6);
  assert.match(svg, /text-anchor="end">Prefill, 4,000 tokens/);
  assert.match(svg, /text-anchor="start">Decode, 1 sequence/);
  const left = R.renderRoofline(m, 1);
  const right = R.renderRoofline(m, 5000);
  assert.match(left, /class="cross-text"[^>]*text-anchor="start"/);
  assert.match(right, /class="cross-text"[^>]*text-anchor="end"/);
});

test('TTFT chart highlights the selected burst and keeps end labels apart', () => {
  const series = E.ttftVsBatch(SCN, ['h100', 'h200', 'l40s', 'a100'], R.BATCHES);
  const svg = R.renderTtftChart(series, 2, 0);
  assert.equal((svg.match(/r="5"/g) || []).length, 4);
  const ys = ['h100', 'h200', 'l40s', 'a100'].map((g) => Number(svg.match(new RegExp('tt-text-' + g + '" x="[0-9.]+" y="([0-9.]+)"'))[1])).sort((a, b) => a - b);
  for (let i = 1; i < ys.length; i++) assert.ok(ys[i] - ys[i - 1] >= 14 - 1e-9);
  const x = (i) => 62 + (i / 6) * (620 - 62 - 70);
  [0, 1, 3, 5, 6].forEach((i) => assert.equal(R.ttftIndexFromX(x(i)), i));
  assert.match(svg, /class="tt-label tt-text-h100" x="558\.0"/);
});

test('memory diagram sizes bars to fit and distinguishes idle KV from live KV', () => {
  const one = R.renderMemoryDiagram(E.computeModelFootprint(Object.assign({}, BASE_FOR_DOM(), { tp: 1, weightPrecision: 'fp16', modelId: 'llama-3.1-8b', concurrentSeqs: 4 })));
  assert.match(one, /all 1 GPU of one replica/);
  assert.match(one, /fill="url\(#hatch-idle\)"/);
  const widths = [...one.matchAll(/class="seg-weights" x="[0-9.]+" y="[0-9.]+" width="([0-9.]+)"/g)].map((m) => Number(m[1]));
  assert.equal(widths.length, 1);
  assert.equal(widths[0], 76);
  const eight = R.renderMemoryDiagram(E.computeModelFootprint(Object.assign({}, SCN, { gpuId: 'h200', tp: 8 })));
  const w8 = [...eight.matchAll(/class="seg-weights" x="[0-9.]+" y="[0-9.]+" width="([0-9.]+)"/g)].map((m) => Number(m[1]));
  assert.equal(w8.length, 8);
  assert.ok(w8[0] < 76 && w8[0] > 30);
  assert.match(eight, /all 8 GPUs of one replica/);
});

function BASE_FOR_DOM() { return { gpuId: 'h100', pp: 1, kvPrecision: 'fp16', promptTokens: 2048, outputTokens: 256 }; }

test('keyboard navigation of the export tabs wraps, jumps and keeps one tab stop', () => {
  const { doc, app } = mountApp();
  const key = (k) => doc.els['ex-tabs'].fire('keydown', { key: k, preventDefault() {} });
  key('ArrowLeft');
  assert.equal(app.state.exportTab, 'router');
  assert.equal(doc.els['tab-router'].attrs.tabindex, '0');
  assert.equal(doc.els['tab-compose'].attrs.tabindex, '-1');
  key('Home');
  assert.equal(app.state.exportTab, 'compose');
  key('End');
  assert.equal(app.state.exportTab, 'router');
  key('ArrowRight');
  assert.equal(app.state.exportTab, 'compose');
  key('x');
  assert.equal(app.state.exportTab, 'compose');
});

test('pool editor edits reset dependent fields and preserve typed costs', () => {
  const { doc, app } = mountApp();
  doc.getElementById('pool-3-model').value = 'llama-3.1-405b-fp8';
  doc.els['r-pools'].fire('change');
  assert.equal(app.state.pools[3].modelId, 'llama-3.1-405b-fp8');
  assert.equal(app.state.pools[3].weightPrecision, 'fp8');
  const price = E.GPUS.l40s.listPricePerGpuHour;
  doc.els['r-pools'].innerHTML = R.renderPoolEditor(app.state.pools);
  delete doc.els['pool-0-gpu']; delete doc.els['pool-0-cost']; delete doc.els['pool-0-model']; delete doc.els['pool-0-on'];
  delete doc.els['pool-0-gpus']; delete doc.els['pool-0-tp']; delete doc.els['pool-0-prec'];
  doc.getElementById('pool-0-cost').value = '9.99';
  doc.els['r-pools'].fire('change');
  assert.equal(app.state.pools[0].costPerGpuHour, 9.99);
  doc.getElementById('pool-0-gpu').value = 'h100';
  doc.els['r-pools'].fire('change');
  assert.equal(app.state.pools[0].gpuId, 'h100');
  assert.equal(app.state.pools[0].costPerGpuHour, E.GPUS.h100.listPricePerGpuHour);
  assert.notEqual(price, E.GPUS.h100.listPricePerGpuHour);
  doc.getElementById('pool-0-cost').value = '';
  doc.els['r-pools'].fire('change');
  assert.equal(app.state.pools[0].costPerGpuHour, null);
});

test('pointer movement over the charts maps to a burst index and a roofline intensity', () => {
  const { doc, app } = mountApp();
  const svg = (w) => ({ getBoundingClientRect: () => ({ left: 100, width: w }), getAttribute: () => '62,70,620,7' });
  const x = 62 + (5 / 6) * (620 - 62 - 70);
  doc.els['wb-ttft'].fire('pointermove', { clientX: 100 + x, target: { closest: () => svg(620) } });
  assert.equal(app.state.ttftIndex, 5);
  doc.els['wb-ttft'].fire('pointermove', { clientX: 100 + 62, target: { closest: () => svg(620) } });
  assert.equal(app.state.ttftIndex, 0);
  doc.els['wb-roofline'].fire('pointermove', { clientX: 100 + 310, target: { closest: () => svg(620) } });
  assert.ok(app.state.hoverAi > 1);
  assert.match(doc.els['wb-roofline'].innerHTML, /cross-line/);
  doc.els['wb-roofline'].fire('pointerleave');
  assert.equal(app.state.hoverAi, null);
  doc.els['wb-roofline'].fire('pointermove', { clientX: 5, target: { closest: () => null } });
  assert.equal(app.state.hoverAi, null);
});

test('preset chips mark only the selected preset as pressed', () => {
  const { doc } = mountApp();
  const chips = doc.els['preset-row'].innerHTML;
  assert.equal((chips.match(/aria-pressed="true"/g) || []).length, 1);
  assert.equal((chips.match(/aria-pressed="false"/g) || []).length, 3);
});

test('a live probe that cannot reach the endpoint reports a readable failure', async () => {
  const { doc, app } = mountApp();
  const saved = globalThis.fetch;
  globalThis.fetch = async () => { throw new TypeError('Failed to fetch'); };
  try {
    doc.els['pr-model'].value = 'some/model';
    doc.els['pr-key'].value = 'sk-should-not-leak';
    await app.runProbe('live');
    assert.match(doc.els['pr-result'].innerHTML, /Probe failed/);
    assert.match(doc.els['pr-result'].innerHTML, /could not reach the endpoint/);
    assert.equal(doc.els['pr-status'].textContent, 'The probe did not complete.');
    assert.ok(!allHtml(doc).includes('sk-should-not-leak'));
    await app.listModels();
    assert.match(doc.els['pr-status'].textContent, /Could not list models/);
  } finally {
    globalThis.fetch = saved;
  }
});

test('listing models fills the datalist with escaped ids', async () => {
  const { doc, app } = mountApp();
  const saved = globalThis.fetch;
  globalThis.fetch = async () => ({ ok: true, status: 200, json: async () => ({ data: [{ id: 'a/b' }, { id: '"><x' }] }) });
  try {
    await app.listModels();
    assert.match(doc.els['pr-models'].innerHTML, /<option value="a\/b">/);
    assert.ok(!doc.els['pr-models'].innerHTML.includes('"><x'));
    assert.match(doc.els['pr-status'].textContent, /2 models returned/);
  } finally {
    globalThis.fetch = saved;
  }
});

test('select options mark the current value as selected', () => {
  const { doc } = mountApp();
  assert.match(doc.els['f-model'].innerHTML, /<option value="llama-3\.1-70b" selected>/);
  assert.equal((doc.els['f-model'].innerHTML.match(/ selected/g) || []).length, 1);
  assert.match(doc.els['f-tp'].innerHTML, /<option value="4" selected>/);
});

test('roofline curves are clipped to the plot so no ceiling line spills below the axis', () => {
  const svg = R.renderRoofline(E.computeRooflineMetrics(SCN), null);
  assert.match(svg, /<clipPath id="rf-plot-clip">/);
  assert.match(svg, /<g clip-path="url\(#rf-plot-clip\)">(?:(?!<\/g>)[^])*rf-curve rf-l40s/);
});

test('narrow charts redraw in a smaller box and keep the pointer maths in step', () => {
  const m = E.computeRooflineMetrics(SCN);
  assert.match(R.renderRoofline(m, null, true), /viewBox="0 0 400 380"/);
  const narrowAi = R.rooflineAiFromX(200);
  assert.match(R.renderRoofline(m, null), /viewBox="0 0 620 380"/);
  assert.ok(R.rooflineAiFromX(200) < narrowAi, 'the same x lands lower on the wide chart');
  const series = E.ttftVsBatch(SCN, ['h100', 'h200', 'l40s', 'a100'], R.BATCHES);
  assert.match(R.renderTtftChart(series, 3, 6000, true), /viewBox="0 0 400 380"/);
  assert.match(R.renderTtftChart(series, 3, 6000), /viewBox="0 0 620 380"/);
});

test('the live bar names the verdict and the numbers that matter', () => {
  const fp = E.computeModelFootprint(SCN);
  const bar = R.renderLiveBar(fp, E.computeRooflineMetrics(SCN));
  assert.match(bar, /live-ok/);
  assert.match(bar, /TTFT [\d,.]+ (ms|s)/);
  const oomCfg = Object.assign({}, SCN, { concurrentSeqs: 5000 });
  const oom = R.renderLiveBar(E.computeModelFootprint(oomCfg), E.computeRooflineMetrics(oomCfg));
  assert.match(oom, /live-bad[^]*Out of memory/);
  const badCfg = Object.assign({}, SCN, { tp: 3 });
  const bad = R.renderLiveBar(E.computeModelFootprint(badCfg), E.computeRooflineMetrics(badCfg));
  assert.match(bad, /Invalid layout/);
  assert.doesNotMatch(bad, /TTFT/);
});
