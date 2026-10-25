/*
 * InferGrid page controller. Reads controls, calls the engine, hands results
 * to the renderers. Uses only getElementById, addEventListener, value,
 * innerHTML, textContent and setAttribute so it can be driven by a tiny fake
 * document in node --test.
 */
(function (root, factory) {
  if (typeof module === 'object' && module.exports) module.exports = factory(require('./engine.js'), require('./render.js'));
  else {
    root.InferGridApp = factory(root.InferGridEngine, root.InferGridRender);
    if (typeof document !== 'undefined' && document.getElementById) {
      const start = function () { root.InferGridApp.mount(document, root); };
      if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', start);
      else start();
    }
  }
}(typeof self !== 'undefined' ? self : this, function (E, R) {
  'use strict';

  const GPU_SERIES = ['h100', 'h200', 'l40s', 'a100'];
  const PROMPT_MAP = { lo: 128, octaves: 10 };
  const OUTPUT_MAP = { lo: 16, octaves: 12 };

  function tokensFromSlider(v, map) {
    const raw = map.lo * Math.pow(2, (Number(v) / 1000) * map.octaves);
    const step = raw < 1024 ? 16 : raw < 8192 ? 64 : 256;
    return Math.max(map.lo, Math.round(raw / step) * step);
  }
  function sliderFromTokens(t, map) {
    return Math.round(Math.min(1000, Math.max(0, (Math.log2(t / map.lo) / map.octaves) * 1000)));
  }

  function initialPools() {
    return E.REFERENCE_CLUSTER.map(function (p) {
      return Object.assign({}, p, { enabled: true, costPerGpuHour: E.GPUS[p.gpuId].listPricePerGpuHour });
    });
  }

  function mount(doc, win) {
    const $ = function (id) { return doc.getElementById(id); };
    const first = E.WORKLOAD_PRESETS[0];
    const state = {
      presetId: first.id,
      scenario: Object.assign({
        maxNumSeqs: 256, maxNumBatchedTokens: 8192, efficiency: { memory: 0.8, compute: 0.5, comm: 0.75 }
      }, first.scenario),
      pools: initialPools(),
      router: { count: first.workload.count, seed: 20261030, sloTtftMs: first.workload.sloTtftMs, sloItlMs: first.workload.sloItlMs },
      exportTab: 'compose', exportSource: 'cluster', ttftIndex: 3, hoverAi: null,
      last: {}
    };

    function isNarrow() { return !!(win && win.innerWidth && win.innerWidth <= 620); }
    function setHtml(id, html) { const el = $(id); if (el) el.innerHTML = html; }
    function setText(id, text) { const el = $(id); if (el) el.textContent = text; }
    function options(list, selected) {
      return list.map(function (o) { return '<option value="' + R.esc(o.value) + '"' + (String(o.value) === String(selected) ? ' selected' : '') + '>' + R.esc(o.label) + '</option>'; }).join('');
    }
    function on(id, events, handler) {
      const el = $(id);
      if (!el) return;
      events.forEach(function (ev) { el.addEventListener(ev, handler); });
    }
    function schedule(fn) {
      if (win && win.requestAnimationFrame) win.requestAnimationFrame(fn); else fn();
    }

    /* ---------- scenario ---------- */

    function scenarioConfig() { return Object.assign({}, state.scenario); }

    function fillScenarioOptions() {
      const s = state.scenario;
      const model = E.getModel(s.modelId);
      setHtml('f-model', options(E.MODELS.map(function (m) { return { value: m.id, label: m.name }; }), s.modelId));
      setHtml('f-gpu', options(E.GPU_IDS.map(function (g) { return { value: g, label: E.GPUS[g].name }; }), s.gpuId));
      setHtml('f-tp', options(E.TP_CHOICES.map(function (t) { return { value: t, label: String(t) }; }), s.tp));
      setHtml('f-pp', options(E.PP_CHOICES.map(function (t) { return { value: t, label: String(t) }; }), s.pp));
      setHtml('f-wprec', options(model.allowedWeightPrecisions.map(function (q) { return { value: q, label: E.PRECISIONS[q].label }; }), s.weightPrecision));
      setHtml('f-kvprec', options(E.PRECISION_IDS.map(function (q) { return { value: q, label: E.PRECISIONS[q].label + (E.PRECISIONS[q].vllmKv ? '' : ' (estimate only)') }; }), s.kvPrecision));
    }

    function writeScenarioValues() {
      const s = state.scenario;
      const set = function (id, v) { const el = $(id); if (el) el.value = v; };
      set('f-model', s.modelId); set('f-gpu', s.gpuId); set('f-tp', s.tp); set('f-pp', s.pp);
      set('f-wprec', s.weightPrecision); set('f-kvprec', s.kvPrecision);
      set('f-prompt', sliderFromTokens(s.promptTokens, PROMPT_MAP));
      set('f-output', sliderFromTokens(s.outputTokens, OUTPUT_MAP));
      set('f-conc', s.concurrentSeqs);
      set('f-util', Math.round(s.gpuMemoryUtilization * 100));
      set('f-maxseqs', s.maxNumSeqs); set('f-maxbt', s.maxNumBatchedTokens);
      set('f-effmem', Math.round(s.efficiency.memory * 100));
      set('f-effcomp', Math.round(s.efficiency.compute * 100));
    }

    function readScenario(changedId) {
      const s = state.scenario;
      const val = function (id) { const el = $(id); return el ? el.value : undefined; };
      if (changedId === 'f-model') {
        s.modelId = val('f-model');
        s.weightPrecision = E.getModel(s.modelId).nativeWeightPrecision;
        fillScenarioOptions();
        writeScenarioValues();
      } else {
        s.modelId = val('f-model') || s.modelId;
        s.gpuId = val('f-gpu') || s.gpuId;
        s.tp = Number(val('f-tp')) || s.tp;
        s.pp = Number(val('f-pp')) || s.pp;
        s.weightPrecision = val('f-wprec') || s.weightPrecision;
        s.kvPrecision = val('f-kvprec') || s.kvPrecision;
        if (Number(val('f-prompt')) !== sliderFromTokens(s.promptTokens, PROMPT_MAP)) s.promptTokens = tokensFromSlider(val('f-prompt'), PROMPT_MAP);
        if (Number(val('f-output')) !== sliderFromTokens(s.outputTokens, OUTPUT_MAP)) s.outputTokens = tokensFromSlider(val('f-output'), OUTPUT_MAP);
        s.concurrentSeqs = Number(val('f-conc')) || s.concurrentSeqs;
        s.gpuMemoryUtilization = Number(val('f-util')) / 100 || s.gpuMemoryUtilization;
        s.maxNumSeqs = Number(val('f-maxseqs')) || s.maxNumSeqs;
        s.maxNumBatchedTokens = Number(val('f-maxbt')) || s.maxNumBatchedTokens;
        s.efficiency = Object.assign({}, s.efficiency, {
          memory: (Number(val('f-effmem')) || s.efficiency.memory * 100) / 100,
          compute: (Number(val('f-effcomp')) || s.efficiency.compute * 100) / 100
        });
      }
    }

    function renderOutputs() {
      const s = state.scenario;
      setText('o-prompt', R.num(s.promptTokens) + ' tokens');
      setText('o-output', R.num(s.outputTokens) + ' tokens');
      setText('o-conc', R.num(s.concurrentSeqs));
      setText('o-util', Math.round(s.gpuMemoryUtilization * 100) + '%');
      setText('o-effmem', Math.round(s.efficiency.memory * 100) + '%');
      setText('o-effcomp', Math.round(s.efficiency.compute * 100) + '%');
      setText('o-ttft-inspect', 'Burst of ' + R.BATCHES[state.ttftIndex]);
    }

    function recomputeScenario() {
      const cfg = scenarioConfig();
      const fp = E.computeModelFootprint(cfg);
      const metrics = E.computeRooflineMetrics(cfg);
      const series = E.ttftVsBatch(cfg, GPU_SERIES, R.BATCHES);
      state.last.fp = fp; state.last.metrics = metrics; state.last.series = series;
      renderOutputs();
      setHtml('wb-verdict', R.renderVerdict(fp, metrics));
      setHtml('live-bar', R.renderLiveBar(fp, metrics));
      setHtml('wb-memory', R.renderMemoryDiagram(fp));
      setHtml('wb-legend', R.renderMemoryLegend(fp));
      renderRoofline();
      renderTtft();
      setHtml('wb-numbers', R.renderNumbersTable(fp, metrics));
      setHtml('wb-steps', R.renderSteps(fp));
      renderHero();
      renderExport();
    }

    /* The live bar shows the verdict while the controls are on screen and the full verdict card is not. */
    function watchLiveBar() {
      const bar = $('live-bar'), section = $('workbench'), card = $('wb-verdict');
      if (!bar || !section || !card || !win || typeof win.IntersectionObserver !== 'function') return;
      const seen = { section: false, card: false };
      const apply = function () { bar.setAttribute('data-visible', seen.section && !seen.card ? 'true' : 'false'); };
      const watch = function (el, key) {
        new win.IntersectionObserver(function (entries) {
          entries.forEach(function (e) { seen[key] = e.isIntersecting; });
          apply();
        }).observe(el);
      };
      watch(section, 'section');
      watch(card, 'card');
    }

    function renderRoofline() {
      const m = state.last.metrics;
      setHtml('wb-roofline', R.renderRoofline(m, state.hoverAi, isNarrow()));
      const dBound = m.decode.intensity < m.ridge ? 'below' : 'above';
      setText('rf-note', 'Decode lands at ' + R.num(m.decode.intensity, 1) + ' FLOP per byte, ' + dBound + ' the ridge point of ' + R.num(m.ridge, 0) + ', so it is ' + m.decode.boundBy + ' bound. Prefill lands at ' + R.num(m.prefill.intensity, 0) + ' FLOP per byte and is ' + m.prefill.boundBy + ' bound.');
    }

    function renderTtft() {
      const series = state.last.series;
      setHtml('wb-ttft', R.renderTtftChart(series, state.ttftIndex, state.router.sloTtftMs, isNarrow()));
      setHtml('wb-ttft-readout', R.renderTtftReadout(series, state.ttftIndex));
      setText('o-ttft-inspect', 'Burst of ' + R.BATCHES[state.ttftIndex]);
      const el = $('f-ttft-inspect');
      if (el) el.value = state.ttftIndex;
    }

    function renderHero() {
      const preset = E.getPreset(state.presetId);
      const heroCfg = Object.assign({ maxNumSeqs: 256, maxNumBatchedTokens: 8192 }, preset.scenario);
      const fp = E.computeModelFootprint(heroCfg);
      const model = E.getModel(heroCfg.modelId);
      setHtml('hero-hbm', R.renderHeroBar(fp));
      setText('hero-caption', model.name + ', ' + E.PRECISIONS[heroCfg.weightPrecision].label + ' weights and ' + E.PRECISIONS[heroCfg.kvPrecision].label + ' KV, ' + (heroCfg.tp * heroCfg.pp) + ' x ' + E.GPUS[heroCfg.gpuId].short + ' per replica, ' + heroCfg.concurrentSeqs + ' sequences of ' + R.num(heroCfg.promptTokens + heroCfg.outputTokens) + ' tokens. Per GPU, computed by the engine when this preset is selected.');
    }

    /* ---------- presets ---------- */

    function renderPresets() {
      setHtml('preset-row', E.WORKLOAD_PRESETS.map(function (p) {
        return '<button type="button" class="chip" data-preset="' + p.id + '" aria-pressed="' + (p.id === state.presetId ? 'true' : 'false') + '">' + R.esc(p.name) + '</button>';
      }).join(''));
      setText('preset-blurb', E.getPreset(state.presetId).blurb);
    }

    function applyPreset(id) {
      const preset = E.getPreset(id);
      state.presetId = id;
      state.scenario = Object.assign({}, state.scenario, preset.scenario);
      state.router.count = preset.workload.count;
      state.router.sloTtftMs = preset.workload.sloTtftMs;
      state.router.sloItlMs = preset.workload.sloItlMs;
      fillScenarioOptions();
      writeScenarioValues();
      writeRouterValues();
      renderPresets();
      recomputeScenario();
      recomputeRouter();
    }

    /* ---------- router ---------- */

    function writeRouterValues() {
      const set = function (id, v) { const el = $(id); if (el) el.value = v; };
      set('r-count', state.router.count); set('r-seed', state.router.seed);
      set('r-slo-ttft', state.router.sloTtftMs); set('r-slo-itl', state.router.sloItlMs);
    }

    function readRouter() {
      const val = function (id) { const el = $(id); return el ? Number(el.value) : NaN; };
      const r = state.router;
      r.count = Math.min(300, Math.max(10, Math.round(val('r-count') || r.count)));
      r.seed = Math.max(1, Math.round(val('r-seed') || r.seed));
      r.sloTtftMs = val('r-slo-ttft') || r.sloTtftMs;
      r.sloItlMs = val('r-slo-itl') || r.sloItlMs;
    }

    function readPools() {
      state.pools.forEach(function (p, i) {
        const id = 'pool-' + i;
        const get = function (suffix) { return $(id + '-' + suffix); };
        if (!get('model')) return;
        p.enabled = get('on').checked !== false;
        const nextModel = get('model').value;
        if (nextModel !== p.modelId) {
          p.modelId = nextModel;
          p.weightPrecision = E.getModel(nextModel).nativeWeightPrecision;
        } else {
          p.weightPrecision = get('prec').value;
        }
        const nextGpu = get('gpu').value;
        if (nextGpu !== p.gpuId) {
          p.gpuId = nextGpu;
          p.costPerGpuHour = E.GPUS[nextGpu].listPricePerGpuHour;
        } else {
          const c = get('cost').value;
          p.costPerGpuHour = c === '' ? null : Number(c);
        }
        p.gpus = Math.max(1, Math.round(Number(get('gpus').value) || p.gpus));
        p.tp = Number(get('tp').value) || p.tp;
      });
    }

    function activePools() { return state.pools.filter(function (p) { return p.enabled !== false; }); }

    function renderPoolEditor() { setHtml('r-pools', R.renderPoolEditor(state.pools)); }

    function recomputeRouter() {
      const pools = activePools();
      const preset = E.getPreset(state.presetId);
      const requests = E.generateWorkload(state.presetId, { seed: state.router.seed, count: state.router.count }).map(function (r) {
        return Object.assign({}, r, { sloTtftMs: state.router.sloTtftMs, sloItlMs: state.router.sloItlMs });
      });
      if (!pools.length) {
        setHtml('r-summary', '<p class="bad-text">Enable at least one pool to route requests.</p>');
        setHtml('r-policies', ''); setHtml('r-load', ''); setHtml('r-table', '');
        state.last.cmp = null;
        return;
      }
      const cmp = E.compareRoutingPolicies(requests, pools);
      state.last.cmp = cmp; state.last.requests = requests;
      setHtml('r-summary', R.renderRouterSummary(cmp, preset, requests.length));
      setHtml('r-policies', R.renderPolicyTable(cmp));
      setHtml('r-load', R.renderPoolLoad(cmp.pareto));
      setHtml('r-table', R.renderDecisionTable(cmp.pareto, 300));
      renderTtft();
      renderExport();
    }

    /* ---------- export ---------- */

    function exportPools() {
      if (state.exportSource === 'scenario') {
        const s = state.scenario;
        const cfg = E.normalizeConfig(s);
        return [{
          id: 'scenario', label: 'Workbench scenario', modelId: s.modelId, gpuId: s.gpuId, gpus: s.tp * s.pp, tp: s.tp, pp: s.pp,
          weightPrecision: s.weightPrecision, kvPrecision: s.kvPrecision, gpuMemoryUtilization: s.gpuMemoryUtilization,
          maxNumSeqs: s.maxNumSeqs, maxNumBatchedTokens: s.maxNumBatchedTokens, maxModelLen: cfg.maxModelLen
        }];
      }
      return activePools();
    }

    const FILES = {
      compose: { key: 'composeYaml', name: 'docker-compose.yml', type: 'text/yaml' },
      serve: { key: 'serveScript', name: 'serve.sh', type: 'text/x-shellscript' },
      trtllm: { key: 'trtllmScript', name: 'serve-trtllm.sh', type: 'text/x-shellscript' },
      router: { key: 'routerPolicyJson', name: 'router-policy.json', type: 'application/json' }
    };

    function renderExport() {
      const pools = exportPools();
      if (!pools.length) { setText('ex-code', 'Enable at least one pool, or export the workbench scenario.'); setHtml('ex-warnings', ''); return; }
      const manifest = E.exportClusterManifest(pools);
      state.last.manifest = manifest;
      setText('ex-code', manifest[FILES[state.exportTab].key]);
      setHtml('ex-warnings', manifest.warnings.map(function (w) { return '<li>' + R.esc(w) + '</li>'; }).join(''));
    }

    function selectTab(name) {
      state.exportTab = name;
      Object.keys(FILES).forEach(function (k) {
        const t = $('tab-' + k);
        if (!t) return;
        t.setAttribute('aria-selected', k === name ? 'true' : 'false');
        t.setAttribute('tabindex', k === name ? '0' : '-1');
      });
      const panel = $('ex-panel');
      if (panel) panel.setAttribute('aria-labelledby', 'tab-' + name);
      renderExport();
    }

    function copyExport() {
      const text = state.last.manifest ? state.last.manifest[FILES[state.exportTab].key] : '';
      const done = function () { setText('ex-status', 'Copied ' + FILES[state.exportTab].name + '.'); };
      const fail = function () { setText('ex-status', 'Copy was blocked by the browser. Select the text and copy it by hand.'); };
      if (win && win.navigator && win.navigator.clipboard && win.navigator.clipboard.writeText) win.navigator.clipboard.writeText(text).then(done, fail);
      else fail();
    }

    function downloadExport() {
      if (!state.last.manifest || !win || !win.Blob || !win.URL) return;
      const f = FILES[state.exportTab];
      const blob = new win.Blob([state.last.manifest[f.key]], { type: f.type });
      const url = win.URL.createObjectURL(blob);
      const a = doc.createElement('a');
      a.href = url; a.download = f.name;
      a.click();
      win.URL.revokeObjectURL(url);
      setText('ex-status', 'Downloaded ' + f.name + '.');
    }

    /* ---------- probe ---------- */

    function probeInputs() {
      const val = function (id) { const el = $(id); return el ? el.value : ''; };
      return {
        baseUrl: val('pr-url').trim(), apiKey: val('pr-key'), model: val('pr-model').trim(),
        promptTokens: Number(val('pr-prompt')) || 256, maxTokens: Number(val('pr-max')) || 64
      };
    }

    function probeMetrics(inp) {
      return E.computeRooflineMetrics(Object.assign({}, scenarioConfig(), { promptTokens: inp.promptTokens, outputTokens: inp.maxTokens, concurrentSeqs: 1 }));
    }

    async function listModels() {
      const inp = probeInputs();
      if (!inp.baseUrl) { setText('pr-status', 'Enter a base URL first.'); return; }
      setText('pr-status', 'Asking the endpoint for its models.');
      try {
        const ids = await E.listModels({ baseUrl: inp.baseUrl, apiKey: inp.apiKey });
        setHtml('pr-models', ids.map(function (id) { return '<option value="' + R.esc(id) + '"></option>'; }).join(''));
        setText('pr-status', ids.length + ' models returned. Choose one in the model id field.');
      } catch (err) {
        setText('pr-status', 'Could not list models. ' + describeError(err));
      }
    }

    function describeError(err) {
      const msg = err && err.message ? err.message : 'Unknown error';
      if (/Failed to fetch|NetworkError|Load failed/i.test(msg)) return 'The browser could not reach the endpoint. Check the URL, the key, that the server allows cross origin requests, and that an https page may be blocked from plain http endpoints.';
      return msg;
    }

    async function runProbe(mode) {
      const inp = probeInputs();
      const metrics = probeMetrics(inp);
      setText('pr-status', mode === 'replay' ? 'Replaying the roofline as a simulated stream.' : 'Streaming one completion. This can take several seconds.');
      let measured;
      try {
        if (mode === 'replay') {
          const transport = E.createReplayTransport(Object.assign({}, scenarioConfig(), { promptTokens: inp.promptTokens, outputTokens: inp.maxTokens, concurrentSeqs: 1 }), state.router.seed);
          measured = await E.probeEndpoint({ baseUrl: 'replay://local', model: metrics.model.hfRepo, promptTokens: inp.promptTokens, maxTokens: inp.maxTokens, fetchImpl: transport.fetchImpl, now: transport.now });
        } else {
          if (!inp.baseUrl || !inp.model) { setText('pr-status', 'Enter a base URL and a model id, or run the offline replay.'); return; }
          measured = await E.probeEndpoint({ baseUrl: inp.baseUrl, apiKey: inp.apiKey, model: inp.model, promptTokens: inp.promptTokens, maxTokens: inp.maxTokens });
        }
      } catch (err) {
        measured = { ok: false, error: describeError(err) };
      }
      if (measured.ok && !measured.promptTokens) measured.promptTokens = inp.promptTokens;
      const compare = measured.ok ? E.compareProbeToRoofline(measured, metrics) : null;
      setHtml('pr-result', R.renderProbeResult(measured, compare, mode));
      setText('pr-status', measured.ok ? 'Done.' : 'The probe did not complete.');
    }

    /* ---------- wiring ---------- */

    function bind() {
      ['f-model', 'f-gpu', 'f-tp', 'f-pp', 'f-wprec', 'f-kvprec', 'f-prompt', 'f-output', 'f-conc', 'f-util', 'f-maxseqs', 'f-maxbt', 'f-effmem', 'f-effcomp'].forEach(function (id) {
        on(id, ['input', 'change'], function () { readScenario(id); recomputeScenario(); });
      });
      on('preset-row', ['click'], function (ev) {
        const btn = ev.target && ev.target.closest ? ev.target.closest('[data-preset]') : null;
        if (btn) applyPreset(btn.getAttribute('data-preset'));
      });
      ['r-count', 'r-seed', 'r-slo-ttft', 'r-slo-itl'].forEach(function (id) {
        on(id, ['input', 'change'], function () { readRouter(); recomputeRouter(); });
      });
      on('r-pools', ['input', 'change'], function () { readPools(); recomputeRouter(); });
      on('r-reset', ['click'], function () { state.pools = initialPools(); renderPoolEditor(); recomputeRouter(); });
      on('f-ttft-inspect', ['input', 'change'], function () {
        state.ttftIndex = Number($('f-ttft-inspect').value) || 0;
        renderTtft();
      });
      on('wb-ttft', ['pointermove', 'click'], function (ev) {
        const svg = ev.target && ev.target.closest ? ev.target.closest('svg') : null;
        if (!svg || !svg.getBoundingClientRect) return;
        const rect = svg.getBoundingClientRect();
        const w = Number(svg.getAttribute('data-plot').split(',')[2]);
        const idx = R.ttftIndexFromX((ev.clientX - rect.left) / rect.width * w);
        if (idx !== state.ttftIndex) schedule(function () { state.ttftIndex = idx; renderTtft(); });
      });
      on('wb-roofline', ['pointermove'], function (ev) {
        const svg = ev.target && ev.target.closest ? ev.target.closest('svg') : null;
        if (!svg || !svg.getBoundingClientRect) return;
        const rect = svg.getBoundingClientRect();
        const w = Number(svg.getAttribute('data-plot').split(',')[2]);
        const ai = R.rooflineAiFromX((ev.clientX - rect.left) / rect.width * w);
        schedule(function () { state.hoverAi = ai; renderRoofline(); });
      });
      on('wb-roofline', ['pointerleave'], function () { state.hoverAi = null; renderRoofline(); });
      on('ex-tabs', ['click'], function (ev) {
        const t = ev.target && ev.target.closest ? ev.target.closest('[data-tab]') : null;
        if (t) selectTab(t.getAttribute('data-tab'));
      });
      on('ex-tabs', ['keydown'], function (ev) {
        const names = Object.keys(FILES);
        const i = names.indexOf(state.exportTab);
        let next = null;
        if (ev.key === 'ArrowRight') next = names[(i + 1) % names.length];
        else if (ev.key === 'ArrowLeft') next = names[(i + names.length - 1) % names.length];
        else if (ev.key === 'Home') next = names[0];
        else if (ev.key === 'End') next = names[names.length - 1];
        if (next) {
          if (ev.preventDefault) ev.preventDefault();
          selectTab(next);
          const t = $('tab-' + next);
          if (t && t.focus) t.focus();
        }
      });
      on('ex-source', ['change'], function () { state.exportSource = $('ex-source').value; renderExport(); });
      on('ex-copy', ['click'], copyExport);
      on('ex-download', ['click'], downloadExport);
      on('pr-list', ['click'], listModels);
      on('pr-run', ['click'], function () { return runProbe('live'); });
      on('pr-replay', ['click'], function () { return runProbe('replay'); });
    }

    fillScenarioOptions();
    writeScenarioValues();
    writeRouterValues();
    renderPresets();
    renderPoolEditor();
    setHtml('m-hw', R.renderHardwareTable());
    setHtml('m-models', R.renderModelTable());
    bind();
    recomputeScenario();
    recomputeRouter();
    watchLiveBar();
    if (win && typeof win.addEventListener === 'function') {
      let wasNarrow = isNarrow();
      win.addEventListener('resize', function () {
        if (isNarrow() === wasNarrow) return;
        wasNarrow = isNarrow();
        renderRoofline();
        renderTtft();
      });
    }

    return {
      state: state, applyPreset: applyPreset, recomputeScenario: recomputeScenario, recomputeRouter: recomputeRouter,
      selectTab: selectTab, readScenario: readScenario, runProbe: runProbe, listModels: listModels, renderExport: renderExport
    };
  }

  return { mount: mount, tokensFromSlider: tokensFromSlider, sliderFromTokens: sliderFromTokens, PROMPT_MAP: PROMPT_MAP, OUTPUT_MAP: OUTPUT_MAP };
}));
