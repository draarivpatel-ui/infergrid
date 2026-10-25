/*
 * InferGrid engine. Deterministic transformer memory and roofline math, a
 * Pareto request router, a vLLM and TensorRT-LLM manifest exporter and an
 * OpenAI compatible endpoint probe. Dual export (browser global
 * InferGridEngine and CommonJS) so the same file runs in the page and under
 * node --test. No dependencies, no network calls unless probeEndpoint is used.
 *
 * Everything here is an analytical model. Hardware numbers come from NVIDIA
 * datasheets (dense Tensor Core throughput, not the sparsity figures) and model
 * shapes come from each model's published config.json. Efficiency factors and
 * overhead constants are labelled assumptions and are exposed so a user can
 * calibrate them against a real endpoint.
 */
(function (root, factory) {
  if (typeof module === 'object' && module.exports) module.exports = factory();
  else root.InferGridEngine = factory();
}(typeof self !== 'undefined' ? self : this, function () {
  'use strict';

  const GIB = 1073741824;
  const DEFAULTS = Object.freeze({
    tp: 1,
    pp: 1,
    weightPrecision: 'fp16',
    kvPrecision: 'fp16',
    gpuMemoryUtilization: 0.9,
    maxNumSeqs: 256,
    maxNumBatchedTokens: 8192,
    blockSize: 16,
    concurrentSeqs: 8,
    promptTokens: 2048,
    outputTokens: 256
  });
  const DEFAULT_EFFICIENCY = Object.freeze({ memory: 0.8, compute: 0.5, comm: 0.75 });
  const INTERNODE_LINK = Object.freeze({ kind: 'infiniband', gbsPerDir: 50, latencyUs: 25 });
  const TP_CHOICES = Object.freeze([1, 2, 4, 8]);
  const PP_CHOICES = Object.freeze([1, 2, 4]);

  /* Weight and KV precisions. Bytes per element. INT4 AWQ carries group scale and
   * zero point overhead (4 bit zero plus 16 bit scale per 128 weights). */
  const PRECISIONS = Object.freeze({
    fp16: { id: 'fp16', label: 'FP16/BF16', weightBytes: 2, kvBytes: 2, compute: 'fp16', vllmQuant: null, vllmKv: null },
    fp8: { id: 'fp8', label: 'FP8-E4M3', weightBytes: 1, kvBytes: 1, compute: 'fp8', vllmQuant: 'fp8', vllmKv: 'fp8' },
    int8: { id: 'int8', label: 'INT8', weightBytes: 1, kvBytes: 1, compute: 'int8', vllmQuant: null, vllmKv: null },
    int4: { id: 'int4', label: 'INT4-AWQ', weightBytes: 0.5 + 2.5 / 128, kvBytes: 0.5, compute: 'fp16', vllmQuant: 'awq', vllmKv: null }
  });
  const PRECISION_IDS = Object.freeze(Object.keys(PRECISIONS));

  /* GPUs. memGiB is the datasheet capacity read as GiB. bwGBs is HBM or GDDR
   * bandwidth in GB/s. tflops is dense Tensor Core throughput (NVIDIA prints
   * the doubled with sparsity figure beside it). link is per direction. */
  const GPUS = Object.freeze({
    h100: {
      id: 'h100', name: 'H100 SXM5 80GB', short: 'H100', memGiB: 80, bwGBs: 3350, memType: 'HBM3',
      tflops: { fp16: 989, fp8: 1979, int8: 1979 },
      link: { kind: 'nvlink', gbsPerDir: 450, latencyUs: 12 }, nodeSize: 8,
      listPricePerGpuHour: 4.5, priceSource: 'nebius.com/prices on 2026-10-25',
      datasheet: 'NVIDIA H100 page lists 1,979 TFLOPS FP16 and 3,958 FP8 with sparsity, so dense is half'
    },
    h200: {
      id: 'h200', name: 'H200 SXM 141GB', short: 'H200', memGiB: 141, bwGBs: 4800, memType: 'HBM3e',
      tflops: { fp16: 989, fp8: 1979, int8: 1979 },
      link: { kind: 'nvlink', gbsPerDir: 450, latencyUs: 12 }, nodeSize: 8,
      listPricePerGpuHour: 5.4, priceSource: 'nebius.com/prices on 2026-10-25',
      datasheet: 'NVIDIA H200 page lists 1,979 TFLOPS FP16 with sparsity, so dense is half'
    },
    l40s: {
      id: 'l40s', name: 'L40S 48GB', short: 'L40S', memGiB: 48, bwGBs: 864, memType: 'GDDR6',
      tflops: { fp16: 362, fp8: 733, int8: 733 },
      link: { kind: 'pcie', gbsPerDir: 32, latencyUs: 30 }, nodeSize: 8,
      listPricePerGpuHour: 1.55, priceSource: 'nebius.com/prices on 2026-10-25, lowest listed (Intel CPU) rate',
      datasheet: 'NVIDIA L40S page, first value of each pair, PCIe Gen4 x16 64 GB/s bidirectional, no NVLink'
    },
    a100: {
      id: 'a100', name: 'A100 SXM4 80GB', short: 'A100', memGiB: 80, bwGBs: 2039, memType: 'HBM2e',
      tflops: { fp16: 312, fp8: null, int8: 624 },
      link: { kind: 'nvlink', gbsPerDir: 300, latencyUs: 12 }, nodeSize: 8,
      listPricePerGpuHour: null, priceSource: 'not listed by Nebius, enter your own rate',
      datasheet: 'NVIDIA A100 80GB SXM, dense throughput, no FP8 Tensor Cores'
    }
  });
  const GPU_IDS = Object.freeze(Object.keys(GPUS));

  /* Model shapes from each model's config.json (see source). Parameter counts
   * are derived from these shapes and checked against paramsNominal. */
  const MODELS = Object.freeze([
    {
      id: 'llama-3.1-8b', name: 'Llama-3.1-8B-Instruct', hfRepo: 'meta-llama/Llama-3.1-8B-Instruct',
      layers: 32, dModel: 4096, nQ: 32, nKv: 8, headDim: 128, ffn: 14336, vocab: 128256,
      tiedEmbeddings: false, gatedMlp: true, contextMax: 131072, paramsNominal: 8.03e9,
      nativeWeightPrecision: 'fp16', allowedWeightPrecisions: ['fp16', 'fp8', 'int8', 'int4'],
      hfRepoInt4: 'hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4', hfRepoInt8: 'RedHatAI/Meta-Llama-3.1-8B-Instruct-quantized.w8a8',
      gated: true, source: 'huggingface.co/unsloth/Meta-Llama-3.1-8B-Instruct/config.json (ungated mirror)'
    },
    {
      id: 'llama-3.1-70b', name: 'Llama-3.1-70B-Instruct', hfRepo: 'meta-llama/Llama-3.1-70B-Instruct',
      layers: 80, dModel: 8192, nQ: 64, nKv: 8, headDim: 128, ffn: 28672, vocab: 128256,
      tiedEmbeddings: false, gatedMlp: true, contextMax: 131072, paramsNominal: 70.6e9,
      nativeWeightPrecision: 'fp16', allowedWeightPrecisions: ['fp16', 'fp8', 'int8', 'int4'],
      hfRepoInt4: 'hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4', hfRepoInt8: 'RedHatAI/Meta-Llama-3.1-70B-Instruct-quantized.w8a8',
      gated: true, source: 'huggingface.co/unsloth/Meta-Llama-3.1-70B-Instruct/config.json (ungated mirror)'
    },
    {
      id: 'llama-3.1-405b-fp8', name: 'Llama-3.1-405B-FP8', hfRepo: 'RedHatAI/Meta-Llama-3.1-405B-Instruct-FP8',
      layers: 126, dModel: 16384, nQ: 128, nKv: 8, headDim: 128, ffn: 53248, vocab: 128256,
      tiedEmbeddings: false, gatedMlp: true, contextMax: 131072, paramsNominal: 405.9e9,
      nativeWeightPrecision: 'fp8', allowedWeightPrecisions: ['fp8'],
      hfRepoInt4: 'hugging-quants/Meta-Llama-3.1-405B-Instruct-AWQ-INT4', hfRepoInt8: null,
      gated: false, source: 'huggingface.co/neuralmagic/Meta-Llama-3.1-405B-Instruct-FP8/config.json (now RedHatAI)'
    },
    {
      id: 'qwen-2.5-72b', name: 'Qwen-2.5-72B-Instruct', hfRepo: 'Qwen/Qwen2.5-72B-Instruct',
      layers: 80, dModel: 8192, nQ: 64, nKv: 8, headDim: 128, ffn: 29568, vocab: 152064,
      tiedEmbeddings: false, gatedMlp: true, contextMax: 32768, paramsNominal: 72.7e9,
      nativeWeightPrecision: 'fp16', allowedWeightPrecisions: ['fp16', 'fp8', 'int8', 'int4'],
      hfRepoInt4: 'Qwen/Qwen2.5-72B-Instruct-AWQ', hfRepoInt8: 'RedHatAI/Qwen2.5-72B-Instruct-quantized.w8a8',
      gated: false, source: 'huggingface.co/Qwen/Qwen2.5-72B-Instruct/config.json'
    },
    {
      id: 'deepseek-r1-distill-qwen-32b', name: 'DeepSeek-R1-Distill-Qwen-32B', hfRepo: 'deepseek-ai/DeepSeek-R1-Distill-Qwen-32B',
      layers: 64, dModel: 5120, nQ: 40, nKv: 8, headDim: 128, ffn: 27648, vocab: 152064,
      tiedEmbeddings: false, gatedMlp: true, contextMax: 131072, paramsNominal: 32.8e9,
      nativeWeightPrecision: 'fp16', allowedWeightPrecisions: ['fp16', 'fp8', 'int8', 'int4'],
      hfRepoInt4: 'casperhansen/deepseek-r1-distill-qwen-32b-awq', hfRepoInt8: 'RedHatAI/DeepSeek-R1-Distill-Qwen-32B-quantized.w8a8',
      gated: false, source: 'huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-32B/config.json'
    },
    {
      id: 'mistral-small-24b', name: 'Mistral-Small-24B-Instruct', hfRepo: 'mistralai/Mistral-Small-24B-Instruct-2501',
      layers: 40, dModel: 5120, nQ: 32, nKv: 8, headDim: 128, ffn: 32768, vocab: 131072,
      tiedEmbeddings: false, gatedMlp: true, contextMax: 32768, paramsNominal: 23.6e9,
      nativeWeightPrecision: 'fp16', allowedWeightPrecisions: ['fp16', 'fp8', 'int8', 'int4'],
      hfRepoInt4: 'stelterlab/Mistral-Small-24B-Instruct-2501-AWQ', hfRepoInt8: 'RedHatAI/Mistral-Small-24B-Instruct-2501-quantized.w8a8',
      gated: false, source: 'huggingface.co/mistralai/Mistral-Small-24B-Instruct-2501/config.json'
    },
    {
      id: 'gemma-2-27b', name: 'Gemma-2-27B-IT', hfRepo: 'google/gemma-2-27b-it',
      layers: 46, dModel: 4608, nQ: 32, nKv: 16, headDim: 128, ffn: 36864, vocab: 256000,
      tiedEmbeddings: true, gatedMlp: true, contextMax: 8192, paramsNominal: 27.2e9,
      nativeWeightPrecision: 'fp16', allowedWeightPrecisions: ['fp16', 'fp8', 'int8', 'int4'],
      hfRepoInt4: null, hfRepoInt8: null,
      gated: true, note: 'Half the layers use a 4096 token sliding window. This engine reserves full KV for every layer, which matches engines without a hybrid allocator and overstates KV for the rest.',
      source: 'huggingface.co/unsloth/gemma-2-27b-it/config.json (ungated mirror)'
    },
    {
      id: 'nemotron-4-340b', name: 'Nemotron-4-340B-Instruct', hfRepo: 'nvidia/Nemotron-4-340B-Instruct',
      layers: 96, dModel: 18432, nQ: 96, nKv: 8, headDim: 192, ffn: 73728, vocab: 256000,
      tiedEmbeddings: false, gatedMlp: false, contextMax: 4096, paramsNominal: 341.0e9,
      nativeWeightPrecision: 'fp16', allowedWeightPrecisions: ['fp16', 'fp8', 'int8', 'int4'],
      hfRepoInt4: null, hfRepoInt8: null,
      gated: false, note: 'Published as a NeMo checkpoint. Serve it with NVIDIA NeMo or TensorRT-LLM, or use a community Hugging Face conversion for vLLM.',
      source: 'arxiv.org/abs/2406.11704 Table 1 (96 layers, 18432 hidden, 96 heads, 8 KV heads, 256000 vocab, squared ReLU MLP)'
    }
  ].map(deriveModel));
  const MODEL_IDS = Object.freeze(MODELS.map(function (m) { return m.id; }));

  function deriveModel(m) {
    const qWidth = m.nQ * m.headDim;
    const kvWidth = m.nKv * m.headDim;
    const attn = m.dModel * qWidth + 2 * m.dModel * kvWidth + qWidth * m.dModel;
    const mlp = (m.gatedMlp ? 3 : 2) * m.dModel * m.ffn;
    const norms = m.layers * 2 * m.dModel + m.dModel;
    const nonEmbedParams = m.layers * (attn + mlp) + norms;
    const embedParams = m.vocab * m.dModel * (m.tiedEmbeddings ? 1 : 2);
    return Object.freeze(Object.assign({}, m, {
      gqaRatio: m.nQ / m.nKv,
      nonEmbedParams: nonEmbedParams,
      embedParams: embedParams,
      params: nonEmbedParams + embedParams
    }));
  }

  function getModel(id) {
    const m = MODELS.find(function (x) { return x.id === id; });
    if (!m) throw new Error('Unknown model id ' + id);
    return m;
  }
  function getGpu(id) {
    const g = GPUS[id];
    if (!g) throw new Error('Unknown GPU id ' + id);
    return g;
  }
  function getPrecision(id) {
    const p = PRECISIONS[id];
    if (!p) throw new Error('Unknown precision id ' + id);
    return p;
  }

  /* ---------- small helpers ---------- */

  function clamp(x, lo, hi) { return Math.min(hi, Math.max(lo, x)); }
  function ceilDiv(a, b) { return Math.ceil(a / b); }
  function gib(bytes) { return bytes / GIB; }
  function round(x, d) { const k = Math.pow(10, d); return Math.round(x * k) / k; }
  function fmtInt(n) { return Math.round(n).toString().replace(/\B(?=(\d{3})+(?!\d))/g, ','); }
  function fmtGiB(bytes) { return gib(bytes).toFixed(2) + ' GiB'; }

  function mulberry32(seed) {
    let a = seed >>> 0;
    return function () {
      a = (a + 0x6D2B79F5) >>> 0;
      let t = a;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  function percentile(values, p) {
    if (!values.length) return 0;
    const sorted = values.slice().sort(function (a, b) { return a - b; });
    const rank = Math.ceil((p / 100) * sorted.length);
    return sorted[clamp(rank, 1, sorted.length) - 1];
  }

  /* ---------- configuration ---------- */

  function normalizeConfig(input) {
    const defined = {};
    Object.keys(input || {}).forEach(function (k) { if (input[k] !== undefined && input[k] !== null) defined[k] = input[k]; });
    const cfg = Object.assign({}, DEFAULTS, defined);
    const model = getModel(cfg.modelId);
    if (!cfg.weightPrecision || !(input && input.weightPrecision)) cfg.weightPrecision = model.nativeWeightPrecision;
    cfg.tp = Math.round(cfg.tp);
    cfg.pp = Math.round(cfg.pp);
    cfg.gpuMemoryUtilization = clamp(Number(cfg.gpuMemoryUtilization), 0.3, 0.98);
    cfg.maxNumSeqs = Math.max(1, Math.round(cfg.maxNumSeqs));
    cfg.maxNumBatchedTokens = Math.max(256, Math.round(cfg.maxNumBatchedTokens));
    cfg.blockSize = Math.max(1, Math.round(cfg.blockSize));
    cfg.concurrentSeqs = Math.max(1, Math.round(cfg.concurrentSeqs));
    cfg.promptTokens = Math.max(1, Math.round(cfg.promptTokens));
    cfg.outputTokens = Math.max(1, Math.round(cfg.outputTokens));
    if (!(cfg.maxModelLen > 0)) cfg.maxModelLen = Math.min(model.contextMax, cfg.promptTokens + cfg.outputTokens);
    cfg.efficiency = Object.assign({}, DEFAULT_EFFICIENCY, cfg.efficiency || {});
    return cfg;
  }

  function validateParallelism(model, gpu, tp, pp) {
    const errors = [];
    if (TP_CHOICES.indexOf(tp) === -1) errors.push('Tensor parallel size must be 1, 2, 4 or 8.');
    if (PP_CHOICES.indexOf(pp) === -1) errors.push('Pipeline parallel size must be 1, 2 or 4.');
    if (model.nQ % tp !== 0) errors.push('Tensor parallel size ' + tp + ' must divide the ' + model.nQ + ' query heads of ' + model.name + '.');
    if (tp <= model.nKv) {
      if (model.nKv % tp !== 0) errors.push('Tensor parallel size ' + tp + ' must divide the ' + model.nKv + ' KV heads.');
    } else if (tp % model.nKv !== 0) {
      errors.push('Tensor parallel size ' + tp + ' must be a multiple of the ' + model.nKv + ' KV heads so KV heads can be replicated.');
    }
    if (pp > model.layers) errors.push('Pipeline parallel size cannot exceed ' + model.layers + ' layers.');
    return errors;
  }

  /* ---------- KV cache and weights ---------- */

  function kvBytesPerToken(model, kvPrecisionId) {
    return 2 * model.layers * model.nKv * model.headDim * getPrecision(kvPrecisionId).kvBytes;
  }

  function kvBytesPerTokenPerGpu(model, kvPrecisionId, tp, pp) {
    const heads = Math.ceil(model.nKv / tp);
    const layers = Math.ceil(model.layers / pp);
    return 2 * layers * heads * model.headDim * getPrecision(kvPrecisionId).kvBytes;
  }

  /* Quantized checkpoints keep embeddings and lm_head at 16 bit. */
  function weightBytes(model, weightPrecisionId) {
    const p = getPrecision(weightPrecisionId);
    if (p.id === 'fp16') return model.params * 2;
    return model.nonEmbedParams * p.weightBytes + model.embedParams * 2;
  }

  /* Bytes streamed from memory per decode step. The input embedding table is a
   * lookup, the lm_head is a full matrix read. */
  function weightReadBytes(model, weightPrecisionId) {
    const p = getPrecision(weightPrecisionId);
    return model.nonEmbedParams * (p.id === 'fp16' ? 2 : p.weightBytes) + model.vocab * model.dModel * 2;
  }

  function activationBytes(model, cfg) {
    const ffnShard = model.ffn / cfg.tp;
    const perToken = 4 * model.dModel * 2 + (model.gatedMlp ? 2 : 1) * ffnShard * 2;
    return cfg.maxNumBatchedTokens * perToken + cfg.maxNumSeqs * model.vocab * 4 * 2;
  }

  function runtimeBytes(cfg) {
    const context = 0.6 * GIB;
    const nccl = cfg.tp * cfg.pp > 1 ? 0.6 * GIB : 0;
    const graphs = (0.4 + 0.8 * Math.min(cfg.maxNumSeqs, 512) / 512) * GIB;
    return context + nccl + graphs;
  }

  function computeRate(gpu, weightPrecisionId) {
    const key = getPrecision(weightPrecisionId).compute;
    const direct = gpu.tflops[key];
    if (direct) return { tflops: direct, dtype: key, fallback: false };
    return { tflops: gpu.tflops.fp16, dtype: 'fp16', fallback: true };
  }

  function computeModelFootprint(input) {
    const cfg = normalizeConfig(input);
    const model = getModel(cfg.modelId);
    const gpu = getGpu(cfg.gpuId);
    const errors = validateParallelism(model, gpu, cfg.tp, cfg.pp);
    const warnings = [];
    if (model.allowedWeightPrecisions.indexOf(cfg.weightPrecision) === -1) {
      errors.push(model.name + ' only ships as ' + getPrecision(model.nativeWeightPrecision).label + ' weights.');
    }
    if (cfg.promptTokens + cfg.outputTokens > cfg.maxModelLen) {
      errors.push('Prompt plus output (' + fmtInt(cfg.promptTokens + cfg.outputTokens) + ' tokens) exceeds max model length ' + fmtInt(cfg.maxModelLen) + '.');
    }
    if (cfg.maxModelLen > model.contextMax) {
      warnings.push('Max model length ' + fmtInt(cfg.maxModelLen) + ' is beyond the ' + fmtInt(model.contextMax) + ' token context published for ' + model.name + '.');
    }
    const rate = computeRate(gpu, cfg.weightPrecision);
    if (rate.fallback) warnings.push(gpu.short + ' has no ' + getPrecision(cfg.weightPrecision).label + ' Tensor Cores, so compute runs at the FP16 rate and the precision only saves memory.');
    if (cfg.kvPrecision === 'int8' || cfg.kvPrecision === 'int4') warnings.push('vLLM has no ' + getPrecision(cfg.kvPrecision).label + ' KV cache dtype. This is an exploratory memory estimate only.');
    if (cfg.kvPrecision === 'fp8' && gpu.id === 'a100') warnings.push('FP8 KV cache on A100 depends on your vLLM attention backend. Check support before relying on it.');
    if (model.note) warnings.push(model.note);

    const capacity = gpu.memGiB * GIB;
    const budget = capacity * cfg.gpuMemoryUtilization;
    const gpusPerReplica = cfg.tp * cfg.pp;
    const weightsTotal = weightBytes(model, cfg.weightPrecision);
    const weights = weightsTotal / gpusPerReplica;
    const activation = activationBytes(model, cfg);
    const runtime = runtimeBytes(cfg);
    const fixed = weights + activation + runtime;
    const kvPool = Math.max(0, budget - fixed);
    const kvTokTotal = kvBytesPerToken(model, cfg.kvPrecision);
    const kvTokGpu = kvBytesPerTokenPerGpu(model, cfg.kvPrecision, cfg.tp, cfg.pp);
    const poolBlocks = Math.floor(kvPool / (kvTokGpu * cfg.blockSize));
    const poolTokens = poolBlocks * cfg.blockSize;
    const seqTokens = cfg.promptTokens + cfg.outputTokens;
    const blocksPerSeq = ceilDiv(seqTokens, cfg.blockSize);
    const maxSeqs = Math.floor(poolBlocks / blocksPerSeq);
    const demandBlocks = blocksPerSeq * cfg.concurrentSeqs;
    const demandBytes = demandBlocks * cfg.blockSize * kvTokGpu;
    const utilization = poolBlocks > 0 ? demandBlocks / poolBlocks : Infinity;
    const live = Math.min(demandBytes, kvPool);
    const headroom = Math.max(0, capacity - fixed - live);

    let verdict = 'fits';
    const reasons = [];
    if (errors.length) { verdict = 'invalid'; reasons.push.apply(reasons, errors); }
    else if (fixed > budget) { verdict = 'oom'; reasons.push('Weights, activations and runtime need ' + fmtGiB(fixed) + ' per GPU but the budget is ' + fmtGiB(budget) + '.'); }
    else if (demandBlocks > poolBlocks) { verdict = 'oom'; reasons.push(cfg.concurrentSeqs + ' sequences need ' + fmtInt(demandBlocks * cfg.blockSize) + ' KV tokens but the pool holds ' + fmtInt(poolTokens) + '. Raise TP, quantize KV, or lower concurrency.'); }
    else if (utilization > 0.85) { verdict = 'tight'; reasons.push('KV pool is ' + Math.round(utilization * 100) + ' percent full at this concurrency, so bursts will preempt sequences.'); }
    else reasons.push('KV pool is ' + Math.round(utilization * 100) + ' percent full at this concurrency.');

    const steps = [
      { label: 'KV bytes per token', formula: '2 x ' + model.layers + ' layers x ' + model.nKv + ' KV heads x ' + model.headDim + ' head dim x ' + getPrecision(cfg.kvPrecision).kvBytes + ' B', value: fmtInt(kvTokTotal) + ' B' },
      { label: 'KV bytes per token per GPU', formula: '2 x ' + Math.ceil(model.layers / cfg.pp) + ' layers x ' + Math.ceil(model.nKv / cfg.tp) + ' heads x ' + model.headDim + ' x ' + getPrecision(cfg.kvPrecision).kvBytes + ' B', value: fmtInt(kvTokGpu) + ' B' },
      { label: 'Weights per GPU', formula: fmtGiB(weightsTotal) + ' / ' + gpusPerReplica + ' GPUs', value: fmtGiB(weights) },
      { label: 'Activations per GPU', formula: fmtInt(cfg.maxNumBatchedTokens) + ' batched tokens x hidden and FFN buffers, plus fp32 logits', value: fmtGiB(activation) },
      { label: 'Runtime per GPU', formula: 'CUDA context, NCCL buffers and CUDA graphs', value: fmtGiB(runtime) },
      { label: 'Budget per GPU', formula: gpu.memGiB + ' GiB x ' + cfg.gpuMemoryUtilization.toFixed(2) + ' utilization', value: fmtGiB(budget) },
      { label: 'KV pool per GPU', formula: fmtGiB(budget) + ' - ' + fmtGiB(fixed), value: fmtGiB(kvPool) },
      { label: 'KV pool in tokens', formula: fmtInt(poolBlocks) + ' blocks x ' + cfg.blockSize + ' tokens', value: fmtInt(poolTokens) + ' tokens' },
      { label: 'Full length sequences that fit', formula: fmtInt(poolBlocks) + ' blocks / ' + fmtInt(blocksPerSeq) + ' blocks per sequence', value: fmtInt(maxSeqs) }
    ];

    return {
      ok: verdict !== 'invalid', verdict: verdict, reasons: reasons, errors: errors, warnings: warnings,
      cfg: cfg, model: model, gpu: gpu, gpusPerReplica: gpusPerReplica,
      bytes: {
        capacity: capacity, budget: budget, weights: weights, weightsTotal: weightsTotal, activation: activation,
        runtime: runtime, kvPool: kvPool, kvDemand: demandBytes, kvLive: live, headroom: headroom,
        unmanaged: capacity - budget, idlePool: Math.max(0, kvPool - live)
      },
      kv: {
        bytesPerToken: kvTokTotal, bytesPerTokenPerGpu: kvTokGpu, poolBlocks: poolBlocks, poolTokens: poolTokens,
        seqTokens: seqTokens, blocksPerSeq: blocksPerSeq, maxSeqs: maxSeqs, demandBlocks: demandBlocks,
        utilization: utilization
      },
      steps: steps
    };
  }

  /* ---------- roofline ---------- */

  function makeContext(input) {
    const cfg = normalizeConfig(input);
    const model = getModel(cfg.modelId);
    const gpu = getGpu(cfg.gpuId);
    const rate = computeRate(gpu, cfg.weightPrecision);
    const intra = gpu.link;
    const link = cfg.tp * cfg.pp <= gpu.nodeSize ? intra : INTERNODE_LINK;
    return {
      cfg: cfg, model: model, gpu: gpu, rate: rate, link: link,
      weightRead: weightReadBytes(model, cfg.weightPrecision),
      kvPerGpu: kvBytesPerTokenPerGpu(model, cfg.kvPrecision, cfg.tp, cfg.pp),
      stepOverheadMs: 0.4 + 0.012 * model.layers
    };
  }

  function allReduceMs(ctx, messageBytes) {
    const tp = ctx.cfg.tp;
    if (tp <= 1) return 0;
    const bw = ctx.gpu.link.gbsPerDir * 1e9 * ctx.cfg.efficiency.comm;
    return ctx.gpu.link.latencyUs / 1000 + (2 * (tp - 1) / tp * messageBytes / bw) * 1000;
  }

  function pipelineHopMs(ctx, messageBytes) {
    const pp = ctx.cfg.pp;
    if (pp <= 1) return 0;
    const bw = ctx.link.gbsPerDir * 1e9 * ctx.cfg.efficiency.comm;
    return (pp - 1) * (ctx.link.latencyUs / 1000 + (messageBytes / bw) * 1000);
  }

  function decodeStep(ctx, batch, contextTokens) {
    const m = ctx.model, cfg = ctx.cfg, eff = cfg.efficiency;
    const flopsLinear = 2 * m.nonEmbedParams * batch + 2 * m.vocab * m.dModel * batch;
    const flopsAttn = 4 * m.layers * m.nQ * m.headDim * contextTokens * batch;
    const flops = (flopsLinear + flopsAttn) / cfg.tp;
    const bytes = ctx.weightRead / cfg.tp + batch * contextTokens * ctx.kvPerGpu * cfg.pp;
    const memoryMs = bytes / (ctx.gpu.bwGBs * 1e9 * eff.memory) * 1000;
    const computeMs = flops / (ctx.rate.tflops * 1e12 * eff.compute) * 1000;
    const message = batch * m.dModel * 2;
    const commMs = m.layers * 2 * allReduceMs(ctx, message) + pipelineHopMs(ctx, message);
    const overheadMs = ctx.stepOverheadMs;
    const ms = Math.max(memoryMs, computeMs) + commMs + overheadMs;
    return {
      ms: ms, memoryMs: memoryMs, computeMs: computeMs, commMs: commMs, overheadMs: overheadMs,
      boundBy: memoryMs >= computeMs ? 'memory' : 'compute',
      flops: flops, bytes: bytes, intensity: flops / bytes,
      achievedTflops: flops / (ms / 1000) / 1e12
    };
  }

  function prefillStep(ctx, promptTokens, nPrompts) {
    const m = ctx.model, cfg = ctx.cfg, eff = cfg.efficiency;
    const n = nPrompts || 1;
    const tokens = promptTokens * n;
    const passes = Math.max(1, ceilDiv(tokens, cfg.maxNumBatchedTokens));
    const flops = (2 * m.nonEmbedParams * tokens + 2 * m.layers * m.nQ * m.headDim * promptTokens * promptTokens * n + 2 * m.vocab * m.dModel * n) / cfg.tp;
    const bytes = passes * ctx.weightRead / cfg.tp + tokens * ctx.kvPerGpu * cfg.pp;
    const memoryMs = bytes / (ctx.gpu.bwGBs * 1e9 * eff.memory) * 1000;
    const computeMs = flops / (ctx.rate.tflops * 1e12 * eff.compute) * 1000;
    const perPass = Math.min(tokens, cfg.maxNumBatchedTokens);
    const message = perPass * m.dModel * 2;
    const commMs = passes * (m.layers * 2 * allReduceMs(ctx, message) + pipelineHopMs(ctx, message));
    const overheadMs = ctx.stepOverheadMs * passes + 1;
    const ms = Math.max(memoryMs, computeMs) + commMs + overheadMs;
    return {
      ms: ms, memoryMs: memoryMs, computeMs: computeMs, commMs: commMs, overheadMs: overheadMs, passes: passes,
      boundBy: memoryMs >= computeMs ? 'memory' : 'compute',
      flops: flops, bytes: bytes, intensity: flops / bytes,
      achievedTflops: flops / (ms / 1000) / 1e12
    };
  }

  function meanDecodeContext(cfg) { return cfg.promptTokens + cfg.outputTokens / 2; }

  function computeRooflineMetrics(input) {
    const ctx = makeContext(input);
    const cfg = ctx.cfg;
    const prefill = prefillStep(ctx, cfg.promptTokens, 1);
    const decode = decodeStep(ctx, cfg.concurrentSeqs, meanDecodeContext(cfg));
    const decodeSingle = decodeStep(ctx, 1, meanDecodeContext(cfg));
    return {
      cfg: cfg, gpu: ctx.gpu, model: ctx.model, rate: ctx.rate,
      peakFlops: ctx.rate.tflops * 1e12,
      peakBandwidth: ctx.gpu.bwGBs * 1e9,
      ridge: (ctx.rate.tflops * 1e12) / (ctx.gpu.bwGBs * 1e9),
      decodeSingle: decodeSingle,
      prefill: Object.assign({ ttftMs: prefill.ms }, prefill),
      decode: Object.assign({
        itlMs: decode.ms,
        tokensPerSecPerSeq: 1000 / decode.ms,
        tokensPerSecReplica: cfg.concurrentSeqs * 1000 / decode.ms,
        singleStreamItlMs: decodeSingle.ms
      }, decode)
    };
  }

  /* TTFT of the last request when n identical prompts arrive together. */
  function ttftVsBatch(input, gpuIds, batchSizes) {
    return gpuIds.map(function (gid) {
      const cfg = Object.assign({}, input, { gpuId: gid });
      const fp = computeModelFootprint(cfg);
      const ctx = makeContext(cfg);
      return {
        gpuId: gid, gpu: getGpu(gid), fits: fp.verdict !== 'oom' && fp.verdict !== 'invalid',
        valid: fp.ok, verdict: fp.verdict,
        points: batchSizes.map(function (n) {
          const p = prefillStep(ctx, ctx.cfg.promptTokens, n);
          return { batch: n, ttftMs: p.ms, boundBy: p.boundBy };
        })
      };
    });
  }

  function rooflineCurve(gpuId, weightPrecisionId, efficiency) {
    const gpu = getGpu(gpuId);
    const rate = computeRate(gpu, weightPrecisionId);
    const eff = Object.assign({}, DEFAULT_EFFICIENCY, efficiency || {});
    const peak = rate.tflops * eff.compute;
    const bw = gpu.bwGBs * eff.memory / 1000;
    const ridge = peak / bw;
    return { peakTflops: peak, bandwidthTBs: bw, ridge: ridge, dtype: rate.dtype };
  }

  /* ---------- capability, cost and the router ---------- */

  function capabilityProxy(model) {
    return clamp(Math.log2(model.params / 1e9) / Math.log2(400), 0, 1);
  }

  function resolvePool(pool) {
    const model = getModel(pool.modelId);
    const cfg = normalizeConfig({
      modelId: pool.modelId, gpuId: pool.gpuId, tp: pool.tp, pp: pool.pp || 1,
      weightPrecision: pool.weightPrecision, kvPrecision: pool.kvPrecision || 'fp16',
      gpuMemoryUtilization: pool.gpuMemoryUtilization, maxNumSeqs: pool.maxNumSeqs,
      maxNumBatchedTokens: pool.maxNumBatchedTokens, blockSize: pool.blockSize,
      maxModelLen: pool.maxModelLen || model.contextMax, promptTokens: 1, outputTokens: 1,
      concurrentSeqs: 1, efficiency: pool.efficiency
    });
    const fp = computeModelFootprint(cfg);
    const gpusPerReplica = cfg.tp * cfg.pp;
    const replicas = Math.floor((pool.gpus || gpusPerReplica) / gpusPerReplica);
    const usable = fp.ok && fp.verdict !== 'oom' && replicas > 0 && fp.kv.poolBlocks > 0;
    const price = pool.costPerGpuHour != null ? pool.costPerGpuHour : (getGpu(pool.gpuId).listPricePerGpuHour || 0);
    return {
      spec: pool, id: pool.id, label: pool.label || pool.id, model: model, cfg: cfg, ctx: makeContext(cfg), footprint: fp,
      gpusPerReplica: gpusPerReplica, replicas: replicas, usable: usable,
      unusableReason: usable ? null : (fp.errors[0] || fp.reasons[0] || 'No GPUs assigned.'),
      capability: pool.capability != null ? pool.capability : capabilityProxy(model),
      replicaCostPerSec: gpusPerReplica * price / 3600,
      poolBlocks: fp.kv.poolBlocks, blockSize: cfg.blockSize
    };
  }

  function paretoFront(candidates) {
    return candidates.filter(function (a, i) {
      return !candidates.some(function (b, j) {
        if (i === j) return false;
        const noWorse = b.cost <= a.cost && b.ttftMs <= a.ttftMs && b.itlMs <= a.itlMs;
        const better = b.cost < a.cost || b.ttftMs < a.ttftMs || b.itlMs < a.itlMs;
        return noWorse && better;
      });
    });
  }

  function evaluateOnReplica(pool, replica, req, now) {
    replica.active = replica.active.filter(function (s) { return s.finishMs > now; });
    const blocks = ceilDiv(req.inputTokens + req.outputTokens, pool.blockSize);
    const usedBlocks = replica.active.reduce(function (a, s) { return a + s.blocks; }, 0);
    const batch = replica.active.length + 1;
    const ctxSum = replica.active.reduce(function (a, s) { return a + s.ctx; }, 0) + req.inputTokens + req.outputTokens / 2;
    const start = Math.max(now, replica.prefillBusyUntil);
    const prefill = prefillStep(pool.ctx, req.inputTokens, 1);
    const decode = decodeStep(pool.ctx, batch, ctxSum / batch);
    const ttftMs = start - now + prefill.ms;
    const itlMs = decode.ms;
    const cost = pool.replicaCostPerSec * (prefill.ms / 1000 + req.outputTokens * itlMs / 1000 / batch);
    return {
      blocks: blocks, usedBlocks: usedBlocks, freeBlocks: pool.poolBlocks - usedBlocks, batch: batch,
      start: start, prefillMs: prefill.ms, ttftMs: ttftMs, itlMs: itlMs, cost: cost
    };
  }

  function summarizeRouting(decisions, pools, requests) {
    const routed = decisions.filter(function (d) { return d.status !== 'rejected'; });
    const ttfts = routed.map(function (d) { return d.ttftMs; });
    const itls = routed.map(function (d) { return d.itlMs; });
    const cost = routed.reduce(function (a, d) { return a + d.cost; }, 0);
    const tokens = routed.reduce(function (a, d) { return a + d.inputTokens + d.outputTokens; }, 0);
    return {
      total: decisions.length,
      routed: routed.length,
      metSlo: decisions.filter(function (d) { return d.status === 'routed'; }).length,
      sloViolated: decisions.filter(function (d) { return d.status === 'slo_violated'; }).length,
      rejected: decisions.filter(function (d) { return d.status === 'rejected'; }).length,
      kvPreemptions: decisions.filter(function (d) { return d.kvOversubscribed; }).length,
      p50TtftMs: percentile(ttfts, 50), p95TtftMs: percentile(ttfts, 95),
      p50ItlMs: percentile(itls, 50), p95ItlMs: percentile(itls, 95),
      totalCost: cost, tokens: tokens,
      costPerMillionTokens: tokens > 0 ? cost / tokens * 1e6 : 0,
      perPool: pools.map(function (p) {
        const mine = decisions.filter(function (d) { return d.poolId === p.id; });
        return {
          id: p.id, label: p.label, requests: mine.length,
          tokens: mine.reduce(function (a, d) { return a + d.inputTokens + d.outputTokens; }, 0),
          cost: mine.reduce(function (a, d) { return a + d.cost; }, 0),
          peakKvUtilization: p.peakKvUtil || 0
        };
      })
    };
  }

  /* Policies are pareto (default), largest, naive-cheapest and round-robin. */
  function routeWorkloadBatch(requests, poolSpecs, options) {
    const opts = Object.assign({ policy: 'pareto' }, options || {});
    const pools = poolSpecs.map(resolvePool).map(function (p) {
      p.replicaState = [];
      for (let i = 0; i < p.replicas; i++) p.replicaState.push({ index: i, active: [], prefillBusyUntil: 0 });
      p.peakKvUtil = 0;
      return p;
    });
    const order = requests.map(function (r, i) { return { r: r, i: i }; }).sort(function (a, b) {
      return (a.r.arrivalMs || 0) - (b.r.arrivalMs || 0) || a.i - b.i;
    });
    const decisions = [];
    let rr = 0;

    order.forEach(function (entry) {
      const req = entry.r;
      const now = req.arrivalMs || 0;
      const totalTokens = req.inputTokens + req.outputTokens;
      const decision = {
        requestId: req.id, inputTokens: req.inputTokens, outputTokens: req.outputTokens, complexity: req.complexity,
        arrivalMs: now, poolId: null, replica: null, status: 'rejected', reason: '', ttftMs: 0, itlMs: 0, e2eMs: 0, cost: 0,
        kvOversubscribed: false, candidates: [], front: []
      };
      const capable = pools.filter(function (p) { return p.usable && p.capability + 1e-9 >= req.complexity; });
      if (!pools.some(function (p) { return p.usable; })) {
        decision.reason = 'no_usable_pool';
        decisions.push(decision);
        return;
      }
      if (!capable.length) {
        decision.reason = 'no_capable_pool';
        decisions.push(decision);
        return;
      }
      const fitsContext = capable.filter(function (p) { return totalTokens <= p.cfg.maxModelLen; });
      if (!fitsContext.length) {
        decision.reason = 'context_too_long';
        decisions.push(decision);
        return;
      }

      let candidates = [];
      let eligible = fitsContext;
      if (opts.policy === 'largest') {
        const top = fitsContext.slice().sort(function (a, b) { return b.capability - a.capability || a.id.localeCompare(b.id); })[0];
        eligible = [top];
      }

      eligible.forEach(function (p) {
        const strict = opts.policy === 'pareto' || opts.policy === 'largest';
        let best = null;
        p.replicaState.forEach(function (rep) {
          const ev = evaluateOnReplica(p, rep, req, now);
          const kvOk = ev.freeBlocks >= ev.blocks;
          const seqOk = rep.active.length < p.cfg.maxNumSeqs;
          if (strict && (!kvOk || !seqOk)) return;
          const key = [ev.batch, rep.index];
          if (!best || key[0] < best.key[0] || (key[0] === best.key[0] && key[1] < best.key[1])) best = { rep: rep, ev: ev, key: key, kvOk: kvOk };
        });
        if (!best) return;
        candidates.push({
          poolId: p.id, pool: p, replica: best.rep, ev: best.ev, kvOk: best.kvOk,
          cost: best.ev.cost, ttftMs: best.ev.ttftMs, itlMs: best.ev.itlMs,
          meetsSlo: best.ev.ttftMs <= req.sloTtftMs && best.ev.itlMs <= req.sloItlMs,
          violation: Math.max(best.ev.ttftMs / req.sloTtftMs, best.ev.itlMs / req.sloItlMs)
        });
      });

      decision.candidates = candidates.map(function (c) {
        return { poolId: c.poolId, cost: c.cost, ttftMs: c.ttftMs, itlMs: c.itlMs, meetsSlo: c.meetsSlo, kvOk: c.kvOk };
      });
      if (!candidates.length) {
        decision.reason = 'kv_oom';
        decisions.push(decision);
        return;
      }

      let chosen;
      if (opts.policy === 'pareto' || opts.policy === 'largest') {
        const feasible = candidates.filter(function (c) { return c.meetsSlo; });
        if (feasible.length) {
          const front = paretoFront(feasible);
          decision.front = front.map(function (c) { return c.poolId; });
          chosen = front.slice().sort(function (a, b) { return a.cost - b.cost || a.ttftMs - b.ttftMs || a.poolId.localeCompare(b.poolId); })[0];
          decision.status = 'routed';
        } else {
          chosen = candidates.slice().sort(function (a, b) { return a.violation - b.violation || a.cost - b.cost || a.poolId.localeCompare(b.poolId); })[0];
          decision.status = 'slo_violated';
          decision.reason = chosen.ttftMs > req.sloTtftMs ? 'ttft_slo' : 'itl_slo';
        }
      } else if (opts.policy === 'round-robin') {
        chosen = candidates[rr % candidates.length];
        rr++;
        decision.status = chosen.meetsSlo ? 'routed' : 'slo_violated';
      } else {
        chosen = candidates.slice().sort(function (a, b) { return a.cost - b.cost || a.poolId.localeCompare(b.poolId); })[0];
        decision.status = chosen.meetsSlo ? 'routed' : 'slo_violated';
      }

      const ev = chosen.ev;
      decision.poolId = chosen.poolId;
      decision.replica = chosen.replica.index;
      decision.ttftMs = ev.ttftMs;
      decision.itlMs = ev.itlMs;
      decision.e2eMs = ev.ttftMs + req.outputTokens * ev.itlMs;
      decision.cost = ev.cost;
      decision.kvOversubscribed = !chosen.kvOk;
      chosen.replica.active.push({ finishMs: now + decision.e2eMs, blocks: ev.blocks, ctx: req.inputTokens + req.outputTokens / 2 });
      chosen.replica.prefillBusyUntil = ev.start + ev.prefillMs;
      const used = ev.usedBlocks + ev.blocks;
      chosen.pool.peakKvUtil = Math.max(chosen.pool.peakKvUtil, used / chosen.pool.poolBlocks);
      decisions.push(decision);
    });

    const original = requests.map(function (r) { return r.id; });
    decisions.sort(function (a, b) { return original.indexOf(a.requestId) - original.indexOf(b.requestId); });
    return {
      policy: opts.policy, decisions: decisions,
      pools: pools.map(function (p) {
        return {
          id: p.id, label: p.label, usable: p.usable, unusableReason: p.unusableReason, replicas: p.replicas,
          capability: p.capability, poolTokens: p.footprint.kv.poolTokens, modelName: p.model.name, gpuName: p.cfg.gpuId,
          gpusPerReplica: p.gpusPerReplica
        };
      }),
      summary: summarizeRouting(decisions, pools, requests)
    };
  }

  function compareRoutingPolicies(requests, poolSpecs) {
    return {
      pareto: routeWorkloadBatch(requests, poolSpecs, { policy: 'pareto' }),
      largest: routeWorkloadBatch(requests, poolSpecs, { policy: 'largest' }),
      naive: routeWorkloadBatch(requests, poolSpecs, { policy: 'naive-cheapest' }),
      roundRobin: routeWorkloadBatch(requests, poolSpecs, { policy: 'round-robin' })
    };
  }

  /* ---------- reference cluster and workload presets ---------- */

  const REFERENCE_CLUSTER = Object.freeze([
    { id: 'small-l40s', label: 'Small, 8B on L40S', modelId: 'llama-3.1-8b', gpuId: 'l40s', gpus: 4, tp: 1, pp: 1, weightPrecision: 'fp8', kvPrecision: 'fp8', gpuMemoryUtilization: 0.9, maxNumSeqs: 128, maxNumBatchedTokens: 8192, maxModelLen: 16384 },
    { id: 'mid-h100', label: 'Mid, 24B on H100', modelId: 'mistral-small-24b', gpuId: 'h100', gpus: 4, tp: 2, pp: 1, weightPrecision: 'fp8', kvPrecision: 'fp8', gpuMemoryUtilization: 0.9, maxNumSeqs: 128, maxNumBatchedTokens: 8192, maxModelLen: 32768 },
    { id: 'long-h200', label: 'Long context, 32B on H200', modelId: 'deepseek-r1-distill-qwen-32b', gpuId: 'h200', gpus: 2, tp: 2, pp: 1, weightPrecision: 'fp16', kvPrecision: 'fp16', gpuMemoryUtilization: 0.9, maxNumSeqs: 64, maxNumBatchedTokens: 8192, maxModelLen: 49152 },
    { id: 'large-h200', label: 'Large, 70B on H200', modelId: 'llama-3.1-70b', gpuId: 'h200', gpus: 4, tp: 4, pp: 1, weightPrecision: 'fp8', kvPrecision: 'fp8', gpuMemoryUtilization: 0.9, maxNumSeqs: 128, maxNumBatchedTokens: 8192, maxModelLen: 32768 }
  ]);

  const WORKLOAD_PRESETS = Object.freeze([
    {
      id: 'rag-batch', name: 'High-Throughput RAG Batch',
      blurb: 'Long retrieved context, short answers, relaxed latency. Cost per token matters most.',
      scenario: { modelId: 'llama-3.1-70b', gpuId: 'h100', tp: 4, pp: 1, weightPrecision: 'fp8', kvPrecision: 'fp8', promptTokens: 4000, outputTokens: 300, concurrentSeqs: 32, gpuMemoryUtilization: 0.9 },
      workload: { count: 120, windowMs: 20000, sloTtftMs: 6000, sloItlMs: 100, classes: [
        { weight: 0.7, input: [2000, 6000], output: [150, 400], complexity: [0.2, 0.55] },
        { weight: 0.3, input: [6000, 12000], output: [200, 500], complexity: [0.4, 0.7] }
      ] }
    },
    {
      id: 'voice-agent', name: 'Low-Latency Voice Agent',
      blurb: 'Short turns that must start speaking fast. Tight TTFT and inter token latency.',
      scenario: { modelId: 'llama-3.1-8b', gpuId: 'h100', tp: 1, pp: 1, weightPrecision: 'fp16', kvPrecision: 'fp8', promptTokens: 600, outputTokens: 80, concurrentSeqs: 16, gpuMemoryUtilization: 0.9 },
      workload: { count: 160, windowMs: 10000, sloTtftMs: 400, sloItlMs: 40, classes: [
        { weight: 0.85, input: [200, 900], output: [40, 120], complexity: [0.05, 0.35] },
        { weight: 0.15, input: [900, 2400], output: [60, 160], complexity: [0.3, 0.55] }
      ] }
    },
    {
      id: 'code-audit-32k', name: 'DeepSeek 32k Long-Context Code Audit',
      blurb: 'A whole repository in the prompt and a long reasoning trace out. KV cache is the constraint.',
      scenario: { modelId: 'deepseek-r1-distill-qwen-32b', gpuId: 'h200', tp: 2, pp: 1, weightPrecision: 'fp16', kvPrecision: 'fp8', promptTokens: 28000, outputTokens: 1500, concurrentSeqs: 8, gpuMemoryUtilization: 0.9 },
      workload: { count: 40, windowMs: 15000, sloTtftMs: 20000, sloItlMs: 80, classes: [
        { weight: 0.8, input: [18000, 30000], output: [800, 2000], complexity: [0.4, 0.58] },
        { weight: 0.2, input: [6000, 18000], output: [400, 1200], complexity: [0.3, 0.7] }
      ] }
    },
    {
      id: 'saas-burst', name: 'Multi-Tenant SaaS Burst',
      blurb: 'Mixed chat, summarization and a few hard questions arriving in a spike.',
      scenario: { modelId: 'mistral-small-24b', gpuId: 'h100', tp: 2, pp: 1, weightPrecision: 'fp8', kvPrecision: 'fp8', promptTokens: 1200, outputTokens: 250, concurrentSeqs: 64, gpuMemoryUtilization: 0.9 },
      workload: { count: 200, windowMs: 5000, sloTtftMs: 1500, sloItlMs: 60, classes: [
        { weight: 0.7, input: [150, 900], output: [60, 250], complexity: [0.05, 0.4] },
        { weight: 0.25, input: [1500, 5000], output: [150, 500], complexity: [0.3, 0.6] },
        { weight: 0.05, input: [800, 3000], output: [300, 900], complexity: [0.7, 0.9] }
      ] }
    }
  ]);

  function getPreset(id) {
    const p = WORKLOAD_PRESETS.find(function (x) { return x.id === id; });
    if (!p) throw new Error('Unknown preset ' + id);
    return p;
  }

  function generateWorkload(presetId, options) {
    const preset = getPreset(presetId);
    const opts = Object.assign({ seed: 20261030, count: preset.workload.count }, options || {});
    const rnd = mulberry32(opts.seed);
    const w = preset.workload;
    const totalWeight = w.classes.reduce(function (a, c) { return a + c.weight; }, 0);
    const out = [];
    for (let i = 0; i < opts.count; i++) {
      let pick = rnd() * totalWeight;
      let cls = w.classes[w.classes.length - 1];
      for (let k = 0; k < w.classes.length; k++) {
        pick -= w.classes[k].weight;
        if (pick <= 0) { cls = w.classes[k]; break; }
      }
      const draw = function (range) { return range[0] + rnd() * (range[1] - range[0]); };
      out.push({
        id: preset.id + '-' + (i + 1),
        inputTokens: Math.round(draw(cls.input)),
        outputTokens: Math.round(draw(cls.output)),
        complexity: round(draw(cls.complexity), 3),
        sloTtftMs: w.sloTtftMs,
        sloItlMs: w.sloItlMs,
        arrivalMs: Math.round(rnd() * w.windowMs)
      });
    }
    return out;
  }

  /* ---------- manifest exporter ---------- */

  function shellQuote(s) { return /^[A-Za-z0-9_\/.:=@%+,-]+$/.test(s) ? s : "'" + String(s).replace(/'/g, "'\\''") + "'"; }
  function yamlQuote(s) { return '"' + String(s).replace(/\\/g, '\\\\').replace(/"/g, '\\"') + '"'; }

  function pickRepo(model, weightPrecisionId, warnings) {
    if (weightPrecisionId === 'int4') {
      if (model.hfRepoInt4) return model.hfRepoInt4;
      warnings.push(model.name + ' has no AWQ checkpoint checked by InferGrid. Replace the model id with your own AWQ checkpoint.');
      return model.hfRepo;
    }
    if (weightPrecisionId === 'int8') {
      if (model.hfRepoInt8) return model.hfRepoInt8;
      warnings.push(model.name + ' has no INT8 W8A8 checkpoint checked by InferGrid. Replace the model id with your own compressed-tensors checkpoint.');
      return model.hfRepo;
    }
    return model.hfRepo;
  }

  function vllmArgs(poolSpec) {
    const rp = resolvePool(poolSpec);
    const cfg = rp.cfg, model = rp.model;
    const warnings = [];
    const repo = pickRepo(model, cfg.weightPrecision, warnings);
    const args = [];
    args.push(['--tensor-parallel-size', cfg.tp]);
    if (cfg.pp > 1) args.push(['--pipeline-parallel-size', cfg.pp]);
    args.push(['--gpu-memory-utilization', cfg.gpuMemoryUtilization.toFixed(2)]);
    args.push(['--max-model-len', cfg.maxModelLen]);
    args.push(['--max-num-seqs', cfg.maxNumSeqs]);
    args.push(['--max-num-batched-tokens', cfg.maxNumBatchedTokens]);
    args.push(['--dtype', 'bfloat16']);
    const prec = getPrecision(cfg.weightPrecision);
    if (prec.vllmQuant && model.nativeWeightPrecision !== cfg.weightPrecision) args.push(['--quantization', prec.vllmQuant]);
    if (cfg.weightPrecision === 'int8') warnings.push('INT8 W8A8 checkpoints use compressed-tensors, which vLLM detects from the checkpoint, so no --quantization flag is emitted.');
    const kv = getPrecision(cfg.kvPrecision);
    if (kv.vllmKv) args.push(['--kv-cache-dtype', kv.vllmKv]);
    else if (cfg.kvPrecision !== 'fp16') warnings.push('KV precision ' + kv.label + ' has no vLLM flag, so the launch command keeps an FP16/BF16 KV cache.');
    if (cfg.tp * cfg.pp > rp.ctx.gpu.nodeSize) warnings.push(rp.label + ' spans more than ' + rp.ctx.gpu.nodeSize + ' GPUs per replica. Multi node serving needs Ray, see the vLLM parallelism docs.');
    if (model.gated) warnings.push(model.name + ' is a gated Hugging Face repo. Accept the licence and set HF_TOKEN.');
    if (model.note) warnings.push(model.note);
    return { repo: repo, args: args, warnings: warnings, resolved: rp };
  }

  function tensorRtArgs(poolSpec) {
    const rp = resolvePool(poolSpec);
    const cfg = rp.cfg, fp = rp.footprint;
    const freeAfterLoad = fp.bytes.capacity - fp.bytes.weights - fp.bytes.runtime;
    const fraction = freeAfterLoad > 0 ? clamp(Math.floor(fp.bytes.kvPool / freeAfterLoad * 100) / 100, 0.1, 0.95) : 0.1;
    return {
      repo: rp.model.hfRepo, tp: cfg.tp, pp: cfg.pp, maxBatchSize: cfg.maxNumSeqs, maxNumTokens: cfg.maxNumBatchedTokens,
      kvFraction: fraction, kvDtype: cfg.kvPrecision === 'fp8' ? 'fp8' : 'auto', resolved: rp
    };
  }

  function exportClusterManifest(poolSpecs, options) {
    const opts = Object.assign({ image: 'vllm/vllm-openai:latest', basePort: 8000 }, options || {});
    const warnings = [];
    const services = [];
    let port = opts.basePort;
    poolSpecs.forEach(function (spec) {
      const built = vllmArgs(spec);
      built.warnings.forEach(function (w) { if (warnings.indexOf(w) === -1) warnings.push(w); });
      if (!built.resolved.usable) warnings.push(built.resolved.label + ' is not deployable as configured. ' + built.resolved.unusableReason);
      for (let i = 0; i < built.resolved.replicas; i++) {
        const start = i * built.resolved.gpusPerReplica;
        const ids = [];
        for (let g = 0; g < built.resolved.gpusPerReplica; g++) ids.push(String((start + g) % built.resolved.ctx.gpu.nodeSize));
        services.push({
          name: spec.id + '-r' + i, poolId: spec.id, repo: built.repo, args: built.args, port: port++,
          host: Math.floor(start / built.resolved.ctx.gpu.nodeSize), deviceIds: ids
        });
      }
    });

    const lines = [];
    lines.push('# Generated by InferGrid. Roofline and memory estimates are analytical; load test before production.');
    lines.push('# Pin the image tag to a vLLM release you have tested. Run each host group on its own node.');
    lines.push('services:');
    services.forEach(function (s) {
      lines.push('  ' + s.name + ':');
      lines.push('    image: ' + opts.image);
      lines.push('    ipc: host');
      lines.push('    ports:');
      lines.push('      - ' + yamlQuote(s.port + ':8000'));
      lines.push('    environment:');
      lines.push('      - HUGGING_FACE_HUB_TOKEN=${HF_TOKEN}');
      lines.push('    volumes:');
      lines.push('      - ' + yamlQuote('${HOME}/.cache/huggingface:/root/.cache/huggingface'));
      lines.push('    deploy:');
      lines.push('      resources:');
      lines.push('        reservations:');
      lines.push('          devices:');
      lines.push('            - driver: nvidia');
      lines.push('              device_ids: [' + s.deviceIds.map(yamlQuote).join(', ') + ']');
      lines.push('              capabilities: [gpu]');
      const cmd = [s.repo].concat(s.args.reduce(function (a, p) { return a.concat([String(p[0]), String(p[1])]); }, []));
      lines.push('    command: [' + cmd.map(yamlQuote).join(', ') + ']');
    });
    const composeYaml = lines.join('\n') + '\n';

    const sh = [];
    sh.push('#!/usr/bin/env bash');
    sh.push('# Generated by InferGrid. One vLLM server per replica, pinned to its GPUs.');
    sh.push('set -euo pipefail');
    sh.push('export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:?set HF_TOKEN first}"');
    services.forEach(function (s) {
      sh.push('');
      sh.push('# ' + s.name + ' on host group ' + s.host);
      sh.push('CUDA_VISIBLE_DEVICES=' + s.deviceIds.join(',') + ' vllm serve ' + shellQuote(s.repo) + ' \\');
      s.args.forEach(function (p) { sh.push('  ' + p[0] + ' ' + shellQuote(String(p[1])) + ' \\'); });
      sh.push('  --port ' + s.port + ' &');
    });
    sh.push('');
    sh.push('wait');
    const serveScript = sh.join('\n') + '\n';

    const trt = [];
    trt.push('#!/usr/bin/env bash');
    trt.push('# Generated by InferGrid for TensorRT-LLM. Flag names follow the trtllm-serve docs, confirm with trtllm-serve --help on your version.');
    trt.push('set -euo pipefail');
    let trtPort = opts.basePort;
    poolSpecs.forEach(function (spec) {
      const t = tensorRtArgs(spec);
      const yml = spec.id + '-extra-llm-api.yml';
      trt.push('');
      trt.push('# ' + spec.id + ' (' + t.resolved.replicas + ' replicas)');
      trt.push("cat > " + yml + " <<'YAML'");
      trt.push('kv_cache_config:');
      trt.push('  free_gpu_memory_fraction: ' + t.kvFraction.toFixed(2));
      trt.push('  dtype: ' + t.kvDtype);
      trt.push('YAML');
      for (let i = 0; i < t.resolved.replicas; i++) {
        const gpr = t.resolved.gpusPerReplica;
        const devs = [];
        for (let g = 0; g < gpr; g++) devs.push((i * gpr + g) % t.resolved.ctx.gpu.nodeSize);
        trt.push('CUDA_VISIBLE_DEVICES=' + devs.join(',') + ' trtllm-serve ' + shellQuote(t.repo) + ' --host 0.0.0.0 --port ' + (trtPort++) +
          ' --tp_size ' + t.tp + ' --pp_size ' + t.pp + ' --max_batch_size ' + t.maxBatchSize + ' --max_num_tokens ' + t.maxNumTokens +
          ' --extra_llm_api_options ' + yml + ' &');
      }
    });
    trt.push('');
    trt.push('wait');
    const trtllmScript = trt.join('\n') + '\n';

    const routerPolicyJson = JSON.stringify({
      note: 'Pool table for a router in front of the replicas. Capability is a size proxy, replace it with your own eval scores.',
      pools: poolSpecs.map(function (spec) {
        const rp = resolvePool(spec);
        return {
          id: spec.id, model: rp.model.hfRepo, capability: round(rp.capability, 3), replicas: rp.replicas,
          maxModelLen: rp.cfg.maxModelLen, kvPoolTokensPerReplica: rp.footprint.kv.poolTokens,
          endpoints: services.filter(function (s) { return s.poolId === spec.id; }).map(function (s) { return 'http://HOST:' + s.port + '/v1'; })
        };
      })
    }, null, 2) + '\n';

    return { composeYaml: composeYaml, serveScript: serveScript, trtllmScript: trtllmScript, routerPolicyJson: routerPolicyJson, services: services, warnings: warnings };
  }

  /* ---------- live probe ---------- */

  function buildProbePrompt(promptTokens) {
    const unit = 'grid cache token ';
    return 'Summarize the following filler text in one sentence. ' + unit.repeat(Math.max(1, Math.round(promptTokens / 3)));
  }

  function joinUrl(baseUrl, path) { return String(baseUrl).replace(/\/+$/, '') + path; }

  function parseSseLines(buffer) {
    const parts = buffer.split('\n');
    const rest = parts.pop();
    const events = [];
    parts.forEach(function (line) {
      const trimmed = line.trim();
      if (trimmed.indexOf('data:') !== 0) return;
      const body = trimmed.slice(5).trim();
      if (body === '[DONE]') { events.push({ done: true }); return; }
      try { events.push({ json: JSON.parse(body) }); } catch (e) { /* ignore partial frames */ }
    });
    return { events: events, rest: rest };
  }

  function chunkText(json) {
    const choice = json && json.choices && json.choices[0];
    if (!choice || !choice.delta) return '';
    return (choice.delta.content || '') + (choice.delta.reasoning_content || '');
  }

  /* chunkTimes are ms offsets from request start for each content chunk. */
  function analyzeStream(chunkTimes, totalMs, completionTokens) {
    if (!chunkTimes.length) return { ok: false, error: 'The stream produced no content chunks.' };
    const ttftMs = chunkTimes[0];
    const lastMs = chunkTimes[chunkTimes.length - 1];
    let itlMs = null;
    let basis = 'chunks';
    if (completionTokens && completionTokens > 1 && lastMs > ttftMs) {
      itlMs = (lastMs - ttftMs) / (completionTokens - 1);
      basis = 'usage';
    } else if (chunkTimes.length > 1) {
      itlMs = (lastMs - ttftMs) / (chunkTimes.length - 1);
    }
    return {
      ok: true, ttftMs: ttftMs, itlMs: itlMs, itlBasis: basis, chunks: chunkTimes.length,
      completionTokens: completionTokens || null, totalMs: totalMs,
      tokensPerSec: itlMs ? 1000 / itlMs : null
    };
  }

  async function listModels(options) {
    const fetchImpl = options.fetchImpl || fetch;
    const res = await fetchImpl(joinUrl(options.baseUrl, '/models'), {
      method: 'GET', headers: options.apiKey ? { Authorization: 'Bearer ' + options.apiKey } : {}
    });
    if (!res.ok) throw new Error('Model list failed with HTTP ' + res.status + '.');
    const body = await res.json();
    return (body.data || []).map(function (m) { return m.id; });
  }

  async function probeEndpoint(options) {
    const fetchImpl = options.fetchImpl || fetch;
    const now = options.now || function () { return performance.now(); };
    const promptTokens = options.promptTokens || 256;
    const maxTokens = options.maxTokens || 64;
    const headers = { 'Content-Type': 'application/json' };
    if (options.apiKey) headers.Authorization = 'Bearer ' + options.apiKey;
    const started = now();
    const res = await fetchImpl(joinUrl(options.baseUrl, '/chat/completions'), {
      method: 'POST', headers: headers, signal: options.signal,
      body: JSON.stringify({
        model: options.model, stream: true, max_tokens: maxTokens, temperature: 0,
        stream_options: { include_usage: true },
        messages: [{ role: 'user', content: buildProbePrompt(promptTokens) }]
      })
    });
    if (!res.ok) {
      return { ok: false, error: 'The endpoint answered HTTP ' + res.status + '.', status: res.status };
    }
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    const times = [];
    let usage = null;
    let idleReads = 0;
    const limitMs = options.timeoutMs || 120000;
    for (;;) {
      const step = await reader.read();
      if (step.done) break;
      idleReads = step.value ? 0 : idleReads + 1;
      if (idleReads > 1000) return { ok: false, error: 'The stream stalled with no data.' };
      if (now() - started > limitMs) return { ok: false, error: 'The stream did not finish within ' + Math.round(limitMs / 1000) + ' seconds.' };
      buffer += typeof step.value === 'string' ? step.value : decoder.decode(step.value, { stream: true });
      const parsed = parseSseLines(buffer);
      buffer = parsed.rest;
      parsed.events.forEach(function (ev) {
        if (!ev.json) return;
        if (ev.json.usage) usage = ev.json.usage;
        if (chunkText(ev.json)) times.push(now() - started);
      });
    }
    const analysis = analyzeStream(times, now() - started, usage && usage.completion_tokens);
    analysis.promptTokens = usage && usage.prompt_tokens ? usage.prompt_tokens : null;
    return analysis;
  }

  function compareProbeToRoofline(measured, metrics) {
    if (!measured.ok) return { ok: false };
    const predictedTtft = prefillStepFromMetrics(metrics, measured.promptTokens);
    const out = { ok: true, predictedTtftMs: predictedTtft, predictedItlMs: metrics.decode.singleStreamItlMs };
    out.ttftRatio = measured.ttftMs / predictedTtft;
    out.itlRatio = measured.itlMs ? measured.itlMs / metrics.decode.singleStreamItlMs : null;
    return out;
  }

  function prefillStepFromMetrics(metrics, promptTokens) {
    const cfg = Object.assign({}, metrics.cfg, { promptTokens: promptTokens || metrics.cfg.promptTokens });
    return prefillStep(makeContext(cfg), cfg.promptTokens, 1).ms;
  }

  /* Deterministic replay transport. It turns the roofline into a streaming
   * response with seeded jitter so the probe panel works offline. The numbers
   * are simulated, never measurements. */
  function createReplayTransport(input, seed) {
    const metrics = computeRooflineMetrics(input);
    const rnd = mulberry32(seed || 7);
    const clock = { t: 0, now: function () { return clock.t; } };
    const modelRepo = metrics.model.hfRepo;
    const fetchImpl = async function (url, init) {
      if (/\/models$/.test(url)) {
        return { ok: true, status: 200, json: async function () { return { data: [{ id: modelRepo }] }; } };
      }
      const body = JSON.parse(init.body);
      const promptTokens = Math.round(body.messages[0].content.length / 4);
      const n = body.max_tokens;
      const first = prefillStepFromMetrics(metrics, promptTokens) * (1 + 0.08 * (rnd() - 0.5));
      const gaps = [];
      for (let i = 0; i < n; i++) gaps.push(i === 0 ? first : metrics.decode.singleStreamItlMs * (1 + 0.12 * (rnd() - 0.5)));
      const frames = gaps.map(function (g, i) {
        return { gap: g, text: 'data: ' + JSON.stringify({ choices: [{ delta: { content: 'tok' + i + ' ' } }] }) + '\n\n' };
      });
      frames.push({ gap: 0, text: 'data: ' + JSON.stringify({ choices: [], usage: { prompt_tokens: promptTokens, completion_tokens: n } }) + '\n\ndata: [DONE]\n\n' });
      let idx = 0;
      return {
        ok: true, status: 200,
        body: { getReader: function () { return { read: async function () {
          if (idx >= frames.length) return { done: true };
          const f = frames[idx++];
          clock.t += f.gap;
          return { done: false, value: f.text };
        } }; } }
      };
    };
    return { fetchImpl: fetchImpl, now: clock.now, clock: clock, metrics: metrics };
  }

  return {
    GIB: GIB, DEFAULTS: DEFAULTS, DEFAULT_EFFICIENCY: DEFAULT_EFFICIENCY, TP_CHOICES: TP_CHOICES, PP_CHOICES: PP_CHOICES,
    PRECISIONS: PRECISIONS, PRECISION_IDS: PRECISION_IDS, GPUS: GPUS, GPU_IDS: GPU_IDS, MODELS: MODELS, MODEL_IDS: MODEL_IDS,
    REFERENCE_CLUSTER: REFERENCE_CLUSTER, WORKLOAD_PRESETS: WORKLOAD_PRESETS,
    getModel: getModel, getGpu: getGpu, getPrecision: getPrecision, getPreset: getPreset,
    normalizeConfig: normalizeConfig, validateParallelism: validateParallelism,
    kvBytesPerToken: kvBytesPerToken, kvBytesPerTokenPerGpu: kvBytesPerTokenPerGpu,
    weightBytes: weightBytes, weightReadBytes: weightReadBytes, activationBytes: activationBytes, runtimeBytes: runtimeBytes,
    computeRate: computeRate, computeModelFootprint: computeModelFootprint,
    makeContext: makeContext, decodeStep: decodeStep, prefillStep: prefillStep, allReduceMs: allReduceMs,
    computeRooflineMetrics: computeRooflineMetrics, ttftVsBatch: ttftVsBatch, rooflineCurve: rooflineCurve,
    capabilityProxy: capabilityProxy, resolvePool: resolvePool, paretoFront: paretoFront,
    routeWorkloadBatch: routeWorkloadBatch, compareRoutingPolicies: compareRoutingPolicies,
    generateWorkload: generateWorkload, vllmArgs: vllmArgs, tensorRtArgs: tensorRtArgs, exportClusterManifest: exportClusterManifest,
    buildProbePrompt: buildProbePrompt, parseSseLines: parseSseLines, analyzeStream: analyzeStream,
    listModels: listModels, probeEndpoint: probeEndpoint, compareProbeToRoofline: compareProbeToRoofline,
    createReplayTransport: createReplayTransport,
    mulberry32: mulberry32, percentile: percentile, fmtInt: fmtInt, fmtGiB: fmtGiB, round: round, clamp: clamp
  };
}));
