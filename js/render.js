/*
 * InferGrid renderers. Pure functions that turn engine results into HTML and
 * SVG strings. No DOM access, so the same code is exercised in node --test.
 * Colors live in styles.css (classes read CSS custom properties), never here.
 */
(function (root, factory) {
  if (typeof module === 'object' && module.exports) module.exports = factory(require('./engine.js'));
  else root.InferGridRender = factory(root.InferGridEngine);
}(typeof self !== 'undefined' ? self : this, function (E) {
  'use strict';

  const GIB = E.GIB;

  function esc(value) {
    return String(value == null ? '' : value)
      .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;').replace(/'/g, '&#39;');
  }

  function num(n, digits) {
    if (!isFinite(n)) return 'n/a';
    return Number(n).toLocaleString('en-US', { minimumFractionDigits: digits || 0, maximumFractionDigits: digits || 0 });
  }
  function gibStr(bytes, digits) { return num(bytes / GIB, digits == null ? 1 : digits) + ' GiB'; }
  function msStr(ms) {
    if (!isFinite(ms)) return 'n/a';
    if (ms >= 10000) return num(ms / 1000, 1) + ' s';
    if (ms >= 100) return num(ms, 0) + ' ms';
    return num(ms, 1) + ' ms';
  }
  function pct(x) { return isFinite(x) ? num(x * 100, 0) + '%' : 'n/a'; }

  const ICONS = {
    ok: '<svg class="icon" viewBox="0 0 20 20" width="20" height="20" aria-hidden="true" focusable="false"><path d="M4 10.5l4 4 8-9" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>',
    warn: '<svg class="icon" viewBox="0 0 20 20" width="20" height="20" aria-hidden="true" focusable="false"><path d="M10 3l8 14H2z" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linejoin="round"/><path d="M10 8v4M10 14.2v.1" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"/></svg>',
    bad: '<svg class="icon" viewBox="0 0 20 20" width="20" height="20" aria-hidden="true" focusable="false"><path d="M5 5l10 10M15 5L5 15" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"/></svg>'
  };

  const SEGMENTS = [
    { key: 'weights', label: 'Weights', cls: 'seg-weights' },
    { key: 'activation', label: 'Activation buffers', cls: 'seg-activation' },
    { key: 'runtime', label: 'Runtime (context, NCCL, graphs)', cls: 'seg-runtime' },
    { key: 'kvLive', label: 'KV cache in use', cls: 'seg-kv' },
    { key: 'idlePool', label: 'KV pool, idle', cls: 'seg-idle' },
    { key: 'reserve', label: 'Left unclaimed by utilization cap', cls: 'seg-reserve' }
  ];

  /* phone charts are drawn this wide; a 309px column then shows 12 unit text at about 11px */
  const NARROW_W = 330;

  function stackModel(fp) {
    const b = fp.bytes;
    const fixed = b.weights + b.activation + b.runtime;
    const unmet = Math.max(0, b.kvDemand - b.kvPool);
    const reserve = Math.max(0, b.capacity - Math.max(b.budget, fixed));
    const values = {
      weights: b.weights, activation: b.activation, runtime: b.runtime,
      kvLive: b.kvLive, idlePool: b.idlePool, reserve: reserve
    };
    const top = Math.max(b.capacity, fixed + b.kvLive + b.idlePool + reserve + unmet);
    return { values: values, unmet: unmet, top: top, fixed: fixed };
  }

  /* narrow draws a 330 unit box so its 12 unit text lands near 11px on a phone */
  function renderMemoryDiagram(fp, narrow) {
    const n = Math.min(fp.gpusPerReplica, 8);
    const W = narrow ? NARROW_W : 440, H = 340, left = 40, right = 10, topPad = 30, bottomPad = 50;
    const barAreaH = H - topPad - bottomPad;
    const gap = 8;
    const avail = W - left - right;
    const barW = Math.min(76, (avail - (n - 1) * gap) / n);
    const used = n * barW + (n - 1) * gap;
    const x0 = left + (avail - used) / 2;
    const m = stackModel(fp);
    const k = barAreaH / m.top;
    const yOf = function (bytes) { return topPad + barAreaH - bytes * k; };
    let out = '';
    out += '<svg class="chart chart-memory" viewBox="0 0 ' + W + ' ' + H + '" role="img" aria-labelledby="mem-title mem-desc">';
    out += '<title id="mem-title">Per GPU memory allocation</title>';
    out += '<desc id="mem-desc">' + esc(memorySummary(fp)) + '</desc>';
    out += '<defs><pattern id="hatch-over" width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)"><rect width="6" height="6" class="hatch-bg"/><line x1="0" y1="0" x2="0" y2="6" class="hatch-line" stroke-width="2.2"/></pattern>';
    out += '<pattern id="hatch-idle" width="5" height="5" patternUnits="userSpaceOnUse" patternTransform="rotate(-45)"><rect width="5" height="5" class="hatch-idle-bg"/><line x1="0" y1="0" x2="0" y2="5" class="hatch-idle-line" stroke-width="1.2"/></pattern></defs>';
    [0, 0.25, 0.5, 0.75, 1].forEach(function (f) {
      const bytes = f * fp.bytes.capacity;
      const y = yOf(bytes);
      out += '<line class="grid-line" x1="' + left + '" x2="' + (W - right) + '" y1="' + y.toFixed(1) + '" y2="' + y.toFixed(1) + '"/>';
      out += '<text class="axis-text" x="' + (left - 6) + '" y="' + (y + 4).toFixed(1) + '" text-anchor="end">' + num(bytes / GIB, 0) + '</text>';
    });
    out += '<text class="axis-text" x="4" y="14">GiB per GPU</text>';
    for (let i = 0; i < n; i++) {
      const x = x0 + i * (barW + gap);
      let cursor = 0;
      SEGMENTS.forEach(function (seg) {
        const v = m.values[seg.key];
        if (!(v > 0)) return;
        const y = yOf(cursor + v);
        const h = v * k;
        const fill = seg.key === 'idlePool' ? ' fill="url(#hatch-idle)"' : '';
        out += '<rect class="' + seg.cls + '" x="' + x.toFixed(1) + '" y="' + y.toFixed(1) + '" width="' + barW.toFixed(1) + '" height="' + Math.max(0.5, h).toFixed(1) + '"' + fill + '/>';
        cursor += v;
      });
      if (m.unmet > 0) {
        const y = yOf(cursor + m.unmet);
        out += '<rect class="seg-over" x="' + x.toFixed(1) + '" y="' + y.toFixed(1) + '" width="' + barW.toFixed(1) + '" height="' + (m.unmet * k).toFixed(1) + '" fill="url(#hatch-over)"/>';
      }
      out += '<text class="axis-text" x="' + (x + barW / 2).toFixed(1) + '" y="' + (H - 28) + '" text-anchor="middle">' + (narrow && n >= 6 ? i : 'GPU ' + i) + '</text>';
    }
    const capY = yOf(fp.bytes.capacity);
    const budY = yOf(fp.bytes.budget);
    out += '<line class="cap-line" x1="' + left + '" x2="' + (W - right) + '" y1="' + capY.toFixed(1) + '" y2="' + capY.toFixed(1) + '"/>';
    out += '<line class="budget-line" x1="' + left + '" x2="' + (W - right) + '" y1="' + budY.toFixed(1) + '" y2="' + budY.toFixed(1) + '"/>';
    out += '<text class="annot-text" x="' + (W - right) + '" y="' + (capY - 5).toFixed(1) + '" text-anchor="end">HBM capacity ' + num(fp.bytes.capacity / GIB, 0) + ' GiB</text>';
    if (Math.abs(capY - budY) > 14) out += '<text class="annot-text" x="' + (W - right) + '" y="' + (budY - 5).toFixed(1) + '" text-anchor="end">budget ' + num(fp.bytes.budget / GIB, 1) + ' GiB</text>';
    const more = narrow
      ? (fp.gpusPerReplica > 8 ? ', first 8 of ' + fp.gpusPerReplica + ' identical GPUs' : n === 1 ? ', one GPU' : ', all ' + n + ' GPUs identical')
      : fp.gpusPerReplica > 8 ? ' (first 8 of ' + fp.gpusPerReplica + ' identical GPUs shown)' : ' (all ' + n + ' GPU' + (n === 1 ? '' : 's') + ' of one replica are identical)';
    out += '<text class="axis-text" x="' + (W / 2) + '" y="' + (H - 8) + '" text-anchor="middle">' + esc('TP ' + fp.cfg.tp + ' x PP ' + fp.cfg.pp + more) + '</text>';
    out += '</svg>';
    return out;
  }

  function memorySummary(fp) {
    const b = fp.bytes;
    return 'Per GPU. Weights ' + gibStr(b.weights, 2) + ', activations ' + gibStr(b.activation, 2) + ', runtime ' + gibStr(b.runtime, 2) +
      ', KV in use ' + gibStr(b.kvLive, 2) + ', free headroom ' + gibStr(b.headroom, 2) + ', capacity ' + gibStr(b.capacity, 0) + '. Verdict ' + fp.verdict + '.';
  }

  function renderMemoryLegend(fp) {
    const m = stackModel(fp);
    const cap = fp.bytes.capacity;
    let out = '<ul class="legend" aria-label="Memory segments per GPU">';
    SEGMENTS.forEach(function (seg) {
      const v = m.values[seg.key];
      out += '<li><span class="swatch ' + seg.cls + '" aria-hidden="true"></span><span class="legend-name">' + esc(seg.label) + '</span><span class="legend-val num">' + gibStr(v, 2) + '</span><span class="legend-pct num">' + pct(v / cap) + '</span></li>';
    });
    if (m.unmet > 0) {
      out += '<li class="legend-danger"><span class="swatch seg-over" aria-hidden="true"></span><span class="legend-name">KV demand with no room</span><span class="legend-val num">' + gibStr(m.unmet, 2) + '</span><span class="legend-pct num">short</span></li>';
    }
    out += '<li class="legend-total"><span class="legend-name">Free HBM headroom</span><span class="legend-val num">' + gibStr(fp.bytes.headroom, 2) + '</span><span class="legend-pct num">' + pct(fp.bytes.headroom / cap) + '</span></li>';
    out += '</ul>';
    return out;
  }

  function renderHeroBar(fp) {
    const m = stackModel(fp);
    let out = '<div class="hero-bar" role="img" aria-label="' + esc(memorySummary(fp)) + '">';
    SEGMENTS.forEach(function (seg) {
      const v = m.values[seg.key];
      if (!(v > 0)) return;
      out += '<span class="hero-seg ' + seg.cls + '" style="flex-grow:' + (v / m.top).toFixed(5) + '"></span>';
    });
    out += '</div>';
    out += '<dl class="hero-stats">';
    out += '<div><dt>Weights</dt><dd class="num">' + gibStr(fp.bytes.weights, 1) + '</dd></div>';
    out += '<div><dt>KV in use</dt><dd class="num">' + gibStr(fp.bytes.kvLive, 1) + '</dd></div>';
    out += '<div><dt>Free headroom</dt><dd class="num">' + gibStr(fp.bytes.headroom, 1) + '</dd></div>';
    out += '</dl>';
    return out;
  }

  function renderVerdict(fp, metrics) {
    const cls = fp.verdict === 'fits' ? 'ok' : fp.verdict === 'tight' ? 'warn' : 'bad';
    const icon = cls === 'ok' ? ICONS.ok : cls === 'warn' ? ICONS.warn : ICONS.bad;
    const word = { fits: 'Fits', tight: 'Fits, but tight', oom: 'Out of memory', invalid: 'Invalid layout' }[fp.verdict];
    let out = '<div class="verdict verdict-' + cls + '">' + icon + '<div><p class="verdict-word">' + esc(word) + '</p>';
    out += '<p class="verdict-why">' + esc(fp.reasons.join(' ')) + '</p>';
    if (metrics && fp.ok) {
      out += '<p class="verdict-metrics num">TTFT ' + msStr(metrics.prefill.ttftMs) + ' for a ' + num(fp.cfg.promptTokens) + ' token prompt. Inter token latency ' + msStr(metrics.decode.itlMs) + ' at ' + fp.cfg.concurrentSeqs + ' concurrent sequences.</p>';
    }
    out += '</div></div>';
    if (fp.warnings.length) {
      out += '<ul class="notes">' + fp.warnings.map(function (w) { return '<li>' + esc(w) + '</li>'; }).join('') + '</ul>';
    }
    return out;
  }

  function renderLiveBar(fp, metrics) {
    const cls = fp.verdict === 'fits' ? 'ok' : fp.verdict === 'tight' ? 'warn' : 'bad';
    const word = { fits: 'Fits', tight: 'Fits, but tight', oom: 'Out of memory', invalid: 'Invalid layout' }[fp.verdict];
    let out = '<a class="live-pill live-' + cls + '" href="#wb-verdict"><strong class="live-word">' + esc(word) + '</strong>';
    if (metrics && fp.ok) {
      out += '<span class="live-stat num">TTFT ' + msStr(metrics.prefill.ttftMs) + '</span><span class="live-stat num">ITL ' + msStr(metrics.decode.itlMs) + '</span><span class="live-stat num">Free ' + gibStr(fp.bytes.headroom, 1) + '</span>';
    }
    return out + '</a>';
  }

  function renderNumbersTable(fp, metrics) {
    const rows = [
      ['KV cache per token', num(fp.kv.bytesPerToken) + ' B', 'whole model, all layers'],
      ['KV cache per token per GPU', num(fp.kv.bytesPerTokenPerGpu) + ' B', 'after TP and PP sharding'],
      ['Weights per GPU', gibStr(fp.bytes.weights, 2), 'total ' + gibStr(fp.bytes.weightsTotal, 1)],
      ['KV pool per GPU', gibStr(fp.bytes.kvPool, 2), num(fp.kv.poolTokens) + ' tokens'],
      ['Full length sequences that fit', num(fp.kv.maxSeqs), num(fp.kv.seqTokens) + ' tokens each'],
      ['KV pool in use at this load', pct(fp.kv.utilization), num(fp.cfg.concurrentSeqs) + ' sequences'],
      ['Prefill TTFT', msStr(metrics.prefill.ttftMs), metrics.prefill.boundBy + ' bound'],
      ['Decode inter token latency', msStr(metrics.decode.itlMs), metrics.decode.boundBy + ' bound'],
      ['Tokens per second per sequence', num(metrics.decode.tokensPerSecPerSeq, 1), 'steady decode'],
      ['Tokens per second per replica', num(metrics.decode.tokensPerSecReplica, 0), fp.gpusPerReplica + ' GPU' + (fp.gpusPerReplica === 1 ? '' : 's')],
      ['Roofline ridge point', num(metrics.ridge, 0) + ' FLOP/B', metrics.rate.dtype.toUpperCase() + ' peak over memory bandwidth']
    ];
    let out = '<table class="data"><caption class="sr-only">Computed capacity and latency numbers</caption><thead><tr><th scope="col">Quantity</th><th scope="col" class="r">Value</th><th scope="col">Basis</th></tr></thead><tbody>';
    rows.forEach(function (r) { out += '<tr><th scope="row">' + esc(r[0]) + '</th><td class="r num">' + esc(r[1]) + '</td><td class="muted">' + esc(r[2]) + '</td></tr>'; });
    out += '</tbody></table>';
    return out;
  }

  function renderSteps(fp) {
    let out = '<ol class="steps">';
    fp.steps.forEach(function (s) {
      out += '<li><span class="step-label">' + esc(s.label) + '</span><code class="step-formula">' + esc(s.formula) + '</code><span class="step-value num">' + esc(s.value) + '</span></li>';
    });
    out += '</ol>';
    return out;
  }

  /* ---------- roofline chart ---------- */

  const RF = { W: 620, H: 380, L: 58, R: 18, T: 18, B: 52, xMin: -0.3, xMax: 4.3, yMin: 0, yMax: 3.6 };

  function rfX(ai) { return RF.L + (Math.log10(ai) - RF.xMin) / (RF.xMax - RF.xMin) * (RF.W - RF.L - RF.R); }
  function rfY(tf) { return RF.T + (1 - (Math.log10(tf) - RF.yMin) / (RF.yMax - RF.yMin)) * (RF.H - RF.T - RF.B); }

  /* narrow draws the same chart in a NARROW_W unit box so phone text stays near 11px */
  function renderRoofline(metrics, hoverAi, narrow) {
    RF.W = narrow ? NARROW_W : 620;
    const cfg = metrics.cfg;
    let out = '<svg class="chart chart-roofline" viewBox="0 0 ' + RF.W + ' ' + RF.H + '" role="img" aria-labelledby="rf-title rf-desc" data-plot="' + [RF.L, RF.R, RF.W, RF.xMin, RF.xMax].join(',') + '">';
    out += '<title id="rf-title">Roofline chart</title>';
    out += '<desc id="rf-desc">Attainable TFLOPS against arithmetic intensity for ' + esc(metrics.gpu.name) + '. Decode sits at ' + num(metrics.decode.intensity, 1) + ' FLOP per byte and prefill at ' + num(metrics.prefill.intensity, 0) + ' FLOP per byte. Ridge point ' + num(metrics.ridge, 0) + '.</desc>';
    [1, 10, 100, 1000].forEach(function (t) {
      const y = rfY(t);
      out += '<line class="grid-line" x1="' + RF.L + '" x2="' + (RF.W - RF.R) + '" y1="' + y.toFixed(1) + '" y2="' + y.toFixed(1) + '"/><text class="axis-text" x="' + (RF.L - 6) + '" y="' + (y + 4).toFixed(1) + '" text-anchor="end">' + t + '</text>';
    });
    [1, 10, 100, 1000].forEach(function (t) {
      const x = rfX(t);
      out += '<line class="grid-line" x1="' + x.toFixed(1) + '" x2="' + x.toFixed(1) + '" y1="' + RF.T + '" y2="' + (RF.H - RF.B) + '"/><text class="axis-text" x="' + x.toFixed(1) + '" y="' + (RF.H - RF.B + 16) + '" text-anchor="middle">' + t + '</text>';
    });
    out += '<text class="axis-text" x="' + ((RF.L + RF.W - RF.R) / 2) + '" y="' + (RF.H - 10) + '" text-anchor="middle">' + (narrow ? 'Arithmetic intensity, FLOP per byte' : 'Arithmetic intensity (FLOP per byte moved)') + '</text>';
    out += '<text class="axis-text" x="12" y="' + (RF.T + 8) + '">TFLOPS</text>';

    out += '<defs><clipPath id="rf-plot-clip"><rect x="' + RF.L + '" y="' + RF.T + '" width="' + (RF.W - RF.L - RF.R) + '" height="' + (RF.H - RF.T - RF.B) + '"/></clipPath></defs>';
    out += '<g clip-path="url(#rf-plot-clip)">';
    const seenPeaks = {};
    const labels = [];
    E.GPU_IDS.forEach(function (gid) {
      const c = E.rooflineCurve(gid, cfg.weightPrecision, cfg.efficiency);
      const peakKey = c.peakTflops.toFixed(3);
      const stack = seenPeaks[peakKey] || 0;
      seenPeaks[peakKey] = stack + 1;
      const ridgeX = Math.min(Math.max(c.ridge, Math.pow(10, RF.xMin)), Math.pow(10, RF.xMax));
      const x0 = Math.pow(10, RF.xMin);
      const y0 = Math.min(c.peakTflops, x0 * c.bandwidthTBs);
      const d = 'M' + rfX(x0).toFixed(1) + ' ' + rfY(y0).toFixed(1) + ' L' + rfX(ridgeX).toFixed(1) + ' ' + rfY(Math.min(c.peakTflops, ridgeX * c.bandwidthTBs)).toFixed(1) + ' L' + rfX(Math.pow(10, RF.xMax)).toFixed(1) + ' ' + rfY(c.peakTflops).toFixed(1);
      const active = gid === cfg.gpuId;
      out += '<path class="rf-curve rf-' + gid + (active ? ' is-active' : '') + '" d="' + d + '"/>';
      const lx = rfX(Math.pow(10, RF.xMax)) - 4;
      labels.push('<text class="rf-label rf-text-' + gid + (active ? ' is-active' : '') + '" x="' + lx.toFixed(1) + '" y="' + (rfY(c.peakTflops) - 5 - stack * 13).toFixed(1) + '" text-anchor="end">' + esc(E.GPUS[gid].short) + '</text>');
    });
    out += '</g>';
    const beforeLabels = out.length;
    out += labels.join('');

    const dots = [
      { key: 'decode1', ai: Math.max(0.5, metrics.decodeSingle.intensity), tf: Math.max(1.05, metrics.decodeSingle.achievedTflops), label: 'Decode, 1 sequence', cls: 'dot-decode1' },
      { key: 'decode', ai: Math.max(0.5, metrics.decode.intensity), tf: Math.max(1.05, metrics.decode.achievedTflops), label: 'Decode, ' + cfg.concurrentSeqs + (cfg.concurrentSeqs === 1 ? ' sequence' : ' sequences'), cls: 'dot-decode' },
      { key: 'prefill', ai: Math.max(0.5, metrics.prefill.intensity), tf: Math.max(1.05, metrics.prefill.achievedTflops), label: 'Prefill, ' + num(cfg.promptTokens) + ' tokens', cls: 'dot-prefill' }
    ];
    const hovering = hoverAi && hoverAi > 0;
    dots.forEach(function (d, i) {
      const x = rfX(Math.min(d.ai, Math.pow(10, RF.xMax))), y = rfY(d.tf);
      const anchor = x > RF.W - 170 ? 'end' : 'start';
      const dx = anchor === 'end' ? -10 : 10;
      out += '<circle class="rf-dot ' + d.cls + '" cx="' + x.toFixed(1) + '" cy="' + y.toFixed(1) + '" r="6"><title>' + esc(d.label + ', ' + num(d.ai, 1) + ' FLOP/B, ' + num(d.tf, 1) + ' TFLOPS') + '</title></circle>';
      if (narrow && d.key === 'prefill' && y < rfY(100)) {
        /* near the ceilings the label would sit on the roof and the GPU names, so it goes above them;
           the crosshair readout uses that band while a finger is on the chart */
        if (hovering) return;
        const ly = RF.T + 12;
        /* the leader goes under the GPU names so their halo masks it where it passes behind them */
        const leader = '<line class="rf-leader" x1="' + x.toFixed(1) + '" y1="' + (y - 7).toFixed(1) + '" x2="' + x.toFixed(1) + '" y2="' + (ly + 4) + '"/>';
        out = out.slice(0, beforeLabels) + leader + out.slice(beforeLabels);
        out += '<text class="rf-dot-label" x="' + (x - 6).toFixed(1) + '" y="' + ly + '" text-anchor="end">' + esc(d.label) + '</text>';
        return;
      }
      const ty = y + (d.key === 'prefill' ? 20 : 4 + (i === 0 ? 14 : 0));
      if (narrow && d.key !== 'prefill') {
        /* about 6.6 units per character at 12 units; flipping left would put the label on the roof line,
           so a label that is too long for the right side wraps onto two lines there instead */
        const room = RF.W - 4 - (x + 10);
        const parts = d.label.split(', ');
        if (d.label.length * 6.6 <= room) {
          out += '<text class="rf-dot-label" x="' + (x + 10).toFixed(1) + '" y="' + ty.toFixed(1) + '">' + esc(d.label) + '</text>';
          return;
        }
        if (parts.length === 2 && Math.max(parts[0].length, parts[1].length) * 6.6 <= room) {
          out += '<text class="rf-dot-label" x="' + (x + 10).toFixed(1) + '" y="' + ty.toFixed(1) + '">' + esc(parts[0] + ',') + '<tspan x="' + (x + 10).toFixed(1) + '" dy="14">' + esc(parts[1]) + '</tspan></text>';
          return;
        }
      }
      out += '<text class="rf-dot-label" x="' + (x + dx).toFixed(1) + '" y="' + ty.toFixed(1) + '" text-anchor="' + anchor + '">' + esc(d.label) + '</text>';
    });

    if (hovering) {
      const c = E.rooflineCurve(cfg.gpuId, cfg.weightPrecision, cfg.efficiency);
      const att = Math.min(c.peakTflops, hoverAi * c.bandwidthTBs);
      const x = rfX(hoverAi), y = rfY(att);
      out += '<line class="cross-line" x1="' + x.toFixed(1) + '" x2="' + x.toFixed(1) + '" y1="' + RF.T + '" y2="' + (RF.H - RF.B) + '"/><circle class="cross-dot" cx="' + x.toFixed(1) + '" cy="' + y.toFixed(1) + '" r="4"/>';
      const side = x > RF.W / 2 ? 'end' : 'start';
      out += '<text class="cross-text" x="' + (x + (side === 'end' ? -8 : 8)).toFixed(1) + '" y="' + (RF.T + 14) + '" text-anchor="' + side + '">' + num(hoverAi, 1) + ' FLOP/B gives ' + num(att, 1) + ' TFLOPS</text>';
    }
    out += '<rect class="hit" x="' + RF.L + '" y="' + RF.T + '" width="' + (RF.W - RF.L - RF.R) + '" height="' + (RF.H - RF.T - RF.B) + '" fill="none" pointer-events="all"/>';
    out += '</svg>';
    return out;
  }

  function rooflineAiFromX(xView) {
    const frac = (xView - RF.L) / (RF.W - RF.L - RF.R);
    const lg = RF.xMin + Math.min(1, Math.max(0, frac)) * (RF.xMax - RF.xMin);
    return Math.pow(10, lg);
  }

  /* ---------- TTFT versus burst size ---------- */

  const TT = { W: 620, H: 380, L: 62, R: 70, T: 18, B: 52 };
  const BATCHES = [1, 2, 4, 8, 16, 32, 64];

  function renderTtftChart(series, selected, sloMs, narrow) {
    TT.W = narrow ? NARROW_W : 620;
    TT.L = narrow ? 66 : 62;
    const all = [];
    series.forEach(function (s) { s.points.forEach(function (p) { all.push(p.ttftMs); }); });
    const lo = Math.pow(10, Math.floor(Math.log10(Math.max(1, Math.min.apply(null, all)))));
    const hi = Math.pow(10, Math.ceil(Math.log10(Math.max.apply(null, all.concat([sloMs || 1])))));
    const lgLo = Math.log10(lo), lgHi = Math.log10(hi === lo ? lo * 10 : hi);
    const x = function (i) { return TT.L + (i / (BATCHES.length - 1)) * (TT.W - TT.L - TT.R); };
    const y = function (ms) { return TT.T + (1 - (Math.log10(ms) - lgLo) / (lgHi - lgLo)) * (TT.H - TT.T - TT.B); };
    let out = '<svg class="chart chart-ttft" viewBox="0 0 ' + TT.W + ' ' + TT.H + '" role="img" aria-labelledby="tt-title tt-desc" data-plot="' + [TT.L, TT.R, TT.W, BATCHES.length].join(',') + '">';
    out += '<title id="tt-title">Time to first token against burst size</title>';
    out += '<desc id="tt-desc">' + esc(series.map(function (s) { return s.gpu.short + ' ' + msStr(s.points[selected].ttftMs) + ' at burst ' + BATCHES[selected] + (s.fits ? '' : ' (does not fit)'); }).join('. ')) + '.</desc>';
    for (let e = Math.round(lgLo); e <= Math.round(lgHi); e++) {
      const v = Math.pow(10, e);
      out += '<line class="grid-line" x1="' + TT.L + '" x2="' + (TT.W - TT.R) + '" y1="' + y(v).toFixed(1) + '" y2="' + y(v).toFixed(1) + '"/><text class="axis-text" x="' + (TT.L - 6) + '" y="' + (y(v) + 4).toFixed(1) + '" text-anchor="end">' + msStr(v) + '</text>';
    }
    BATCHES.forEach(function (b, i) {
      out += '<text class="axis-text" x="' + x(i).toFixed(1) + '" y="' + (TT.H - TT.B + 16) + '" text-anchor="middle">' + b + '</text>';
    });
    out += '<text class="axis-text" x="' + ((TT.L + TT.W - TT.R) / 2) + '" y="' + (TT.H - 10) + '" text-anchor="middle">' + (narrow ? 'Burst size, last prompt served' : 'Prompts arriving together (TTFT shown is the last one served)') + '</text>';
    if (sloMs && sloMs >= lo && sloMs <= hi) {
      out += '<line class="slo-line" x1="' + TT.L + '" x2="' + (TT.W - TT.R) + '" y1="' + y(sloMs).toFixed(1) + '" y2="' + y(sloMs).toFixed(1) + '"/><text class="annot-text" x="' + (TT.L + 4) + '" y="' + (y(sloMs) - 5).toFixed(1) + '">TTFT target ' + msStr(sloMs) + '</text>';
    }
    const labelY = {};
    series.map(function (s) { return { id: s.gpuId, y: y(s.points[s.points.length - 1].ttftMs) + 4 }; })
      .sort(function (a, b) { return a.y - b.y; })
      .forEach(function (l, i, arr) {
        if (i > 0 && l.y - arr[i - 1].y < 14) l.y = arr[i - 1].y + 14;
        labelY[l.id] = l.y;
      });
    series.slice().reverse().forEach(function (s) {
      const d = s.points.map(function (p, i) { return (i ? 'L' : 'M') + x(i).toFixed(1) + ' ' + y(p.ttftMs).toFixed(1); }).join(' ');
      out += '<path class="tt-line tt-' + s.gpuId + (s.fits ? '' : ' is-nofit') + '" d="' + d + '"/>';
      s.points.forEach(function (p, i) {
        out += '<circle class="tt-dot tt-fill-' + s.gpuId + '" cx="' + x(i).toFixed(1) + '" cy="' + y(p.ttftMs).toFixed(1) + '" r="' + (i === selected ? 5 : 2.6) + '"/>';
      });
      const last = s.points[s.points.length - 1];
      out += '<text class="tt-label tt-text-' + s.gpuId + '" x="' + (x(BATCHES.length - 1) + 8).toFixed(1) + '" y="' + labelY[s.gpuId].toFixed(1) + '">' + esc(s.gpu.short) + '</text>';
    });
    out += '<line class="cross-line" x1="' + x(selected).toFixed(1) + '" x2="' + x(selected).toFixed(1) + '" y1="' + TT.T + '" y2="' + (TT.H - TT.B) + '"/>';
    out += '<rect class="hit" x="' + TT.L + '" y="' + TT.T + '" width="' + (TT.W - TT.L - TT.R) + '" height="' + (TT.H - TT.T - TT.B) + '" fill="none" pointer-events="all"/>';
    out += '</svg>';
    return out;
  }

  function ttftIndexFromX(xView) {
    const frac = (xView - TT.L) / (TT.W - TT.L - TT.R);
    return Math.min(BATCHES.length - 1, Math.max(0, Math.round(frac * (BATCHES.length - 1))));
  }

  function renderTtftReadout(series, selected) {
    let out = '<table class="data compact"><caption class="sr-only">TTFT at the selected burst size</caption><thead><tr><th scope="col">GPU</th><th scope="col" class="r">Burst ' + BATCHES[selected] + '</th><th scope="col">Bound by</th><th scope="col">Memory fit</th></tr></thead><tbody>';
    series.forEach(function (s) {
      const p = s.points[selected];
      out += '<tr><th scope="row"><span class="swatch-dot tt-fill-' + s.gpuId + '" aria-hidden="true"></span>' + esc(s.gpu.name) + '</th><td class="r num">' + msStr(p.ttftMs) + '</td><td>' + esc(p.boundBy) + '</td><td>' + (s.fits ? 'fits' : '<span class="bad-text">does not fit at this layout</span>') + '</td></tr>';
    });
    out += '</tbody></table>';
    return out;
  }

  /* ---------- router ---------- */

  function renderPoolEditor(pools) {
    let out = '<table class="data pools"><caption class="sr-only">Cluster pools. Each row is a group of identical replicas.</caption><thead><tr><th scope="col">Use</th><th scope="col">Pool</th><th scope="col">Model</th><th scope="col">GPU</th><th scope="col" class="r">GPUs</th><th scope="col">TP</th><th scope="col">Weights</th><th scope="col" class="r">Dollars per GPU hour</th></tr></thead><tbody>';
    pools.forEach(function (p, i) {
      const id = 'pool-' + i;
      out += '<tr data-pool="' + i + '">';
      out += '<td><input type="checkbox" id="' + id + '-on" ' + (p.enabled === false ? '' : 'checked') + '><label class="sr-only" for="' + id + '-on">Include ' + esc(p.label) + '</label></td>';
      out += '<th scope="row">' + esc(p.label) + '</th>';
      out += '<td><label class="sr-only" for="' + id + '-model">Model for ' + esc(p.label) + '</label><select id="' + id + '-model">' + E.MODELS.map(function (m) { return '<option value="' + m.id + '"' + (m.id === p.modelId ? ' selected' : '') + '>' + esc(m.name) + '</option>'; }).join('') + '</select></td>';
      out += '<td><label class="sr-only" for="' + id + '-gpu">GPU for ' + esc(p.label) + '</label><select id="' + id + '-gpu">' + E.GPU_IDS.map(function (g) { return '<option value="' + g + '"' + (g === p.gpuId ? ' selected' : '') + '>' + esc(E.GPUS[g].short) + '</option>'; }).join('') + '</select></td>';
      out += '<td class="r"><label class="sr-only" for="' + id + '-gpus">GPU count for ' + esc(p.label) + '</label><input type="number" id="' + id + '-gpus" min="1" max="64" step="1" value="' + p.gpus + '"></td>';
      out += '<td><label class="sr-only" for="' + id + '-tp">Tensor parallel size for ' + esc(p.label) + '</label><select id="' + id + '-tp">' + E.TP_CHOICES.map(function (t) { return '<option value="' + t + '"' + (t === p.tp ? ' selected' : '') + '>' + t + '</option>'; }).join('') + '</select></td>';
      out += '<td><label class="sr-only" for="' + id + '-prec">Weight precision for ' + esc(p.label) + '</label><select id="' + id + '-prec">' + E.PRECISION_IDS.map(function (q) { return '<option value="' + q + '"' + (q === p.weightPrecision ? ' selected' : '') + '>' + esc(E.PRECISIONS[q].label) + '</option>'; }).join('') + '</select></td>';
      out += '<td class="r"><label class="sr-only" for="' + id + '-cost">Dollars per GPU hour for ' + esc(p.label) + '</label><input type="number" id="' + id + '-cost" min="0" step="0.05" value="' + (p.costPerGpuHour == null ? '' : p.costPerGpuHour) + '" placeholder="enter rate"></td>';
      out += '</tr>';
    });
    out += '</tbody></table>';
    return out;
  }

  function renderRouterSummary(cmp, preset, count) {
    const s = cmp.pareto.summary;
    const naive = cmp.naive.summary;
    let out = '<dl class="stat-row">';
    out += '<div><dt>Requests meeting both targets</dt><dd class="num">' + s.metSlo + ' of ' + s.total + '</dd></div>';
    out += '<div><dt>p95 TTFT</dt><dd class="num">' + msStr(s.p95TtftMs) + '</dd></div>';
    out += '<div><dt>p95 inter token latency</dt><dd class="num">' + msStr(s.p95ItlMs) + '</dd></div>';
    out += '<div><dt>Cost per million tokens</dt><dd class="num">$' + num(s.costPerMillionTokens, 2) + '</dd></div>';
    out += '<div><dt>KV preemptions avoided</dt><dd class="num">' + naive.kvPreemptions + '</dd></div>';
    out += '</dl>';
    out += '<p class="fine">' + esc(preset.name) + ', ' + count + ' requests. The p95 values are the 95th percentile across this batch of deterministic roofline estimates, not a sample of live traffic. Preemptions avoided counts requests the cheapest first baseline would have placed on a replica with no KV blocks left.</p>';
    return out;
  }

  function renderPolicyTable(cmp) {
    const rows = [
      ['Pareto router', cmp.pareto], ['Largest model only', cmp.largest],
      ['Cheapest capable pool, no KV check', cmp.naive], ['Round robin', cmp.roundRobin]
    ];
    let out = '<table class="data"><caption class="sr-only">Routing policies compared on the same workload</caption><thead><tr><th scope="col">Policy</th><th scope="col" class="r">Targets met</th><th scope="col" class="r">Rejected</th><th scope="col" class="r">KV preemptions</th><th scope="col" class="r">p95 TTFT</th><th scope="col" class="r">Dollars per million tokens</th></tr></thead><tbody>';
    rows.forEach(function (r) {
      const s = r[1].summary;
      out += '<tr' + (r[1].policy === 'pareto' ? ' class="row-strong"' : '') + '><th scope="row">' + esc(r[0]) + '</th><td class="r num">' + s.metSlo + '/' + s.total + '</td><td class="r num">' + s.rejected + '</td><td class="r num">' + s.kvPreemptions + '</td><td class="r num">' + msStr(s.p95TtftMs) + '</td><td class="r num">$' + num(s.costPerMillionTokens, 2) + '</td></tr>';
    });
    out += '</tbody></table>';
    return out;
  }

  function renderPoolLoad(result) {
    let out = '<ul class="pool-load">';
    result.summary.perPool.forEach(function (p) {
      const info = result.pools.find(function (x) { return x.id === p.id; });
      const util = Math.min(1.3, p.peakKvUtilization);
      out += '<li><span class="pool-name">' + esc(p.label) + '</span>';
      if (!info.usable) out += '<span class="bad-text">not deployable. ' + esc(info.unusableReason) + '</span>';
      else out += '<span class="meter" role="img" aria-label="Peak KV use ' + pct(p.peakKvUtilization) + '"><span class="meter-fill' + (p.peakKvUtilization > 1 ? ' is-over' : '') + '" style="width:' + (Math.min(1, util) * 100).toFixed(1) + '%"></span></span><span class="num pool-meta">' + p.requests + ' requests, peak KV ' + pct(p.peakKvUtilization) + ', ' + info.replicas + ' replica' + (info.replicas === 1 ? '' : 's') + '</span>';
      out += '</li>';
    });
    out += '</ul>';
    return out;
  }

  const REASONS = {
    kv_oom: 'No replica had KV blocks free', context_too_long: 'Longer than every capable pool allows',
    no_capable_pool: 'No pool is capable enough', no_usable_pool: 'No pool is deployable', ttft_slo: 'Misses the TTFT target', itl_slo: 'Misses the inter token target'
  };

  function renderDecisionTable(result, limit) {
    const rows = result.decisions.slice(0, limit || 200);
    let out = '<div class="table-scroll" tabindex="0" role="region" aria-label="Routing decisions"><table class="data compact"><caption class="sr-only">Routing decision for each request</caption><thead><tr><th scope="col">Request</th><th scope="col" class="r">In</th><th scope="col" class="r">Out</th><th scope="col" class="r">Needs</th><th scope="col">Routed to</th><th scope="col" class="r">TTFT</th><th scope="col" class="r">ITL</th><th scope="col">Result</th></tr></thead><tbody>';
    rows.forEach(function (d) {
      const status = d.status === 'routed' ? 'Meets targets' : d.status === 'slo_violated' ? (REASONS[d.reason] || 'Misses a target') : (REASONS[d.reason] || 'Rejected');
      const cls = d.status === 'routed' ? '' : d.status === 'slo_violated' ? 'warn-text' : 'bad-text';
      out += '<tr><th scope="row">' + esc(d.requestId) + '</th><td class="r num">' + num(d.inputTokens) + '</td><td class="r num">' + num(d.outputTokens) + '</td><td class="r num">' + num(d.complexity, 2) + '</td><td>' + (d.poolId ? esc(d.poolId) + ' r' + d.replica : 'none') + '</td><td class="r num">' + (d.poolId ? msStr(d.ttftMs) : 'n/a') + '</td><td class="r num">' + (d.poolId ? msStr(d.itlMs) : 'n/a') + '</td><td class="' + cls + '">' + esc(status) + '</td></tr>';
    });
    out += '</tbody></table></div>';
    return out;
  }

  /* ---------- probe ---------- */

  function renderProbeResult(measured, compare, mode) {
    if (!measured || !measured.ok) {
      return '<div class="verdict verdict-bad">' + ICONS.bad + '<div><p class="verdict-word">Probe failed</p><p class="verdict-why">' + esc((measured && measured.error) || 'No result.') + '</p></div></div>';
    }
    const simulated = mode === 'replay';
    let out = '<div class="verdict ' + (simulated ? 'verdict-warn' : 'verdict-ok') + '">' + (simulated ? ICONS.warn : ICONS.ok) + '<div><p class="verdict-word">' + (simulated ? 'Offline replay, simulated from the roofline' : 'Live measurement') + '</p><p class="verdict-why">' +
      (simulated ? 'These numbers are generated by the engine with seeded jitter. They are not a measurement of any server.' : 'Measured from this browser, so TTFT includes your network round trip and queueing on the server.') + '</p></div></div>';
    out += '<table class="data"><caption class="sr-only">Measured values against the roofline prediction</caption><thead><tr><th scope="col">Quantity</th><th scope="col" class="r">' + (simulated ? 'Replay' : 'Measured') + '</th><th scope="col" class="r">Roofline</th><th scope="col" class="r">Ratio</th></tr></thead><tbody>';
    out += '<tr><th scope="row">Time to first token</th><td class="r num">' + msStr(measured.ttftMs) + '</td><td class="r num">' + msStr(compare.predictedTtftMs) + '</td><td class="r num">' + num(compare.ttftRatio, 2) + 'x</td></tr>';
    out += '<tr><th scope="row">Inter token latency</th><td class="r num">' + (measured.itlMs ? msStr(measured.itlMs) : 'n/a') + '</td><td class="r num">' + msStr(compare.predictedItlMs) + '</td><td class="r num">' + (compare.itlRatio ? num(compare.itlRatio, 2) + 'x' : 'n/a') + '</td></tr>';
    out += '<tr><th scope="row">Tokens per second, one stream</th><td class="r num">' + (measured.tokensPerSec ? num(measured.tokensPerSec, 1) : 'n/a') + '</td><td class="r num">' + num(1000 / compare.predictedItlMs, 1) + '</td><td class="r num">n/a</td></tr>';
    out += '</tbody></table>';
    out += '<p class="fine">' + num(measured.chunks) + ' stream chunks' + (measured.completionTokens ? ', ' + num(measured.completionTokens) + ' completion tokens' : '') + (measured.promptTokens ? ', ' + num(measured.promptTokens) + ' prompt tokens reported' : '') + '. Inter token latency uses ' + (measured.itlBasis === 'usage' ? 'the reported token count' : 'chunk spacing, which understates it if the server batches tokens') + '. A ratio above 1 means the real server is slower than the roofline.</p>';
    return out;
  }

  /* ---------- method tables ---------- */

  function renderHardwareTable() {
    let out = '<table class="data"><caption class="sr-only">GPU specifications used by the engine</caption><thead><tr><th scope="col">GPU</th><th scope="col" class="r">Memory</th><th scope="col" class="r">Bandwidth</th><th scope="col" class="r">FP16 dense</th><th scope="col" class="r">FP8 dense</th><th scope="col">Interconnect</th></tr></thead><tbody>';
    E.GPU_IDS.forEach(function (id) {
      const g = E.GPUS[id];
      out += '<tr><th scope="row">' + esc(g.name) + '</th><td class="r num">' + g.memGiB + ' GiB ' + esc(g.memType) + '</td><td class="r num">' + num(g.bwGBs) + ' GB/s</td><td class="r num">' + g.tflops.fp16 + ' TFLOPS</td><td class="r num">' + (g.tflops.fp8 ? g.tflops.fp8 + ' TFLOPS' : 'none') + '</td><td>' + esc(g.link.kind === 'nvlink' ? 'NVLink, ' + g.link.gbsPerDir + ' GB/s per direction' : 'PCIe Gen4 x16, ' + g.link.gbsPerDir + ' GB/s per direction, no NVLink') + '</td></tr>';
    });
    out += '</tbody></table>';
    return out;
  }

  function renderModelTable() {
    let out = '<div class="table-scroll" tabindex="0" role="region" aria-label="Model architectures"><table class="data compact"><caption class="sr-only">Model architectures used by the engine</caption><thead><tr><th scope="col">Model</th><th scope="col" class="r">Layers</th><th scope="col" class="r">Hidden</th><th scope="col" class="r">Query heads</th><th scope="col" class="r">KV heads</th><th scope="col" class="r">Head dim</th><th scope="col" class="r">Params</th><th scope="col" class="r">KV bytes per token, FP16</th></tr></thead><tbody>';
    E.MODELS.forEach(function (m) {
      out += '<tr><th scope="row">' + esc(m.name) + '</th><td class="r num">' + m.layers + '</td><td class="r num">' + num(m.dModel) + '</td><td class="r num">' + m.nQ + '</td><td class="r num">' + m.nKv + '</td><td class="r num">' + m.headDim + '</td><td class="r num">' + num(m.params / 1e9, 1) + 'B</td><td class="r num">' + num(E.kvBytesPerToken(m, 'fp16')) + '</td></tr>';
    });
    out += '</tbody></table></div>';
    return out;
  }

  return {
    esc: esc, num: num, msStr: msStr, gibStr: gibStr, pct: pct, BATCHES: BATCHES, NARROW_W: NARROW_W, SEGMENTS: SEGMENTS, ICONS: ICONS,
    renderMemoryDiagram: renderMemoryDiagram, renderMemoryLegend: renderMemoryLegend, renderHeroBar: renderHeroBar,
    renderVerdict: renderVerdict, renderLiveBar: renderLiveBar, renderNumbersTable: renderNumbersTable, renderSteps: renderSteps,
    renderRoofline: renderRoofline, rooflineAiFromX: rooflineAiFromX,
    renderTtftChart: renderTtftChart, ttftIndexFromX: ttftIndexFromX, renderTtftReadout: renderTtftReadout,
    renderPoolEditor: renderPoolEditor, renderRouterSummary: renderRouterSummary, renderPolicyTable: renderPolicyTable,
    renderPoolLoad: renderPoolLoad, renderDecisionTable: renderDecisionTable, renderProbeResult: renderProbeResult,
    renderHardwareTable: renderHardwareTable, renderModelTable: renderModelTable, memorySummary: memorySummary
  };
}));
