'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const E = require('../js/engine.js');

function near(actual, expected, tol, msg) {
  assert.ok(Math.abs(actual - expected) <= tol, (msg || 'value') + ' expected ' + expected + ' within ' + tol + ' but got ' + actual);
}
const GIB = 1073741824;

/* ---------- KV cache and weights, hand computed ---------- */

test('KV bytes per token for Llama-3.1-8B in FP16 is 131072', () => {
  assert.equal(E.kvBytesPerToken(E.getModel('llama-3.1-8b'), 'fp16'), 2 * 32 * 8 * 128 * 2);
  assert.equal(E.kvBytesPerToken(E.getModel('llama-3.1-8b'), 'fp16'), 131072);
});

test('KV bytes per token for every model matches 2 x L x kv heads x head dim x 2', () => {
  const expected = {
    'llama-3.1-8b': 131072, 'llama-3.1-70b': 327680, 'llama-3.1-405b-fp8': 516096, 'qwen-2.5-72b': 327680,
    'deepseek-r1-distill-qwen-32b': 262144, 'mistral-small-24b': 163840, 'gemma-2-27b': 376832, 'nemotron-4-340b': 589824
  };
  Object.keys(expected).forEach((id) => assert.equal(E.kvBytesPerToken(E.getModel(id), 'fp16'), expected[id], id));
});

test('KV bytes scale with the precision byte width', () => {
  const m = E.getModel('llama-3.1-70b');
  assert.equal(E.kvBytesPerToken(m, 'fp8'), 163840);
  assert.equal(E.kvBytesPerToken(m, 'int8'), 163840);
  assert.equal(E.kvBytesPerToken(m, 'int4'), 81920);
});

test('KV bytes per GPU shard heads across TP and layers across PP', () => {
  const m = E.getModel('llama-3.1-70b');
  assert.equal(E.kvBytesPerTokenPerGpu(m, 'fp16', 1, 1), 327680);
  assert.equal(E.kvBytesPerTokenPerGpu(m, 'fp16', 4, 1), 2 * 80 * 2 * 128 * 2);
  assert.equal(E.kvBytesPerTokenPerGpu(m, 'fp16', 8, 1), 2 * 80 * 1 * 128 * 2);
  assert.equal(E.kvBytesPerTokenPerGpu(m, 'fp16', 1, 2), 2 * 40 * 8 * 128 * 2);
  assert.equal(E.kvBytesPerTokenPerGpu(m, 'fp16', 2, 4), 2 * 20 * 4 * 128 * 2);
});

test('Gemma-2 keeps two KV heads per GPU at TP 8 because it has 16 KV heads', () => {
  assert.equal(E.kvBytesPerTokenPerGpu(E.getModel('gemma-2-27b'), 'fp16', 8, 1), 2 * 46 * 2 * 128 * 2);
});

test('parameter count of Llama-3.1-8B from its architecture is exact', () => {
  assert.equal(E.getModel('llama-3.1-8b').params, 8030261248);
});

test('derived parameter counts match published totals within one percent', () => {
  E.MODELS.forEach((m) => {
    const err = Math.abs(m.params / m.paramsNominal - 1);
    assert.ok(err < 0.01, m.id + ' differs by ' + err);
  });
});

test('Nemotron-4 splits into 9.4B embedding and 331.6B non embedding parameters', () => {
  const m = E.getModel('nemotron-4-340b');
  near(m.embedParams / 1e9, 9.44, 0.05);
  near(m.nonEmbedParams / 1e9, 331.6, 0.3);
});

test('weight bytes in FP16 are two bytes per parameter', () => {
  const m = E.getModel('llama-3.1-8b');
  assert.equal(E.weightBytes(m, 'fp16'), 16060522496);
});

test('FP8 weights keep embeddings and the output head at 16 bit', () => {
  const m = E.getModel('llama-3.1-8b');
  assert.equal(E.weightBytes(m, 'fp8'), m.nonEmbedParams + m.embedParams * 2);
  assert.equal(E.weightBytes(m, 'fp8'), 9080934400);
});

test('INT4 AWQ carries group scale overhead above half a byte', () => {
  const m = E.getModel('llama-3.1-70b');
  const b = E.weightBytes(m, 'int4');
  assert.ok(b > m.nonEmbedParams * 0.5 + m.embedParams * 2);
  near(b, m.nonEmbedParams * (0.5 + 2.5 / 128) + m.embedParams * 2, 1);
});

test('Llama-3.1-405B FP8 weighs about 410 GB with 16 bit embeddings', () => {
  const m = E.getModel('llama-3.1-405b-fp8');
  near(E.weightBytes(m, 'fp8') / 1e9, 410.0, 1.0);
});

test('weight read bytes exclude the input embedding table but include the output head', () => {
  const m = E.getModel('llama-3.1-8b');
  assert.equal(E.weightReadBytes(m, 'fp16'), m.nonEmbedParams * 2 + m.vocab * m.dModel * 2);
  assert.equal(E.weightReadBytes(m, 'fp16'), 15009849344);
});

/* ---------- footprint ---------- */

const BASE = { modelId: 'llama-3.1-8b', gpuId: 'h100', tp: 1, pp: 1, weightPrecision: 'fp16', kvPrecision: 'fp16', promptTokens: 2048, outputTokens: 256, concurrentSeqs: 32 };

test('activation estimate for Llama-3.1-8B at default engine settings is exact', () => {
  const fp = E.computeModelFootprint(BASE);
  assert.equal(fp.bytes.activation, 8192 * (4 * 4096 * 2 + 2 * 14336 * 2) + 256 * 128256 * 8);
  assert.equal(fp.bytes.activation, 1000865792);
});

test('runtime overhead is context plus CUDA graphs on one GPU and adds NCCL for TP', () => {
  near(E.runtimeBytes(E.normalizeConfig(BASE)) / GIB, 1.4, 1e-9);
  near(E.runtimeBytes(E.normalizeConfig(Object.assign({}, BASE, { tp: 2 }))) / GIB, 2.0, 1e-9);
});

test('budget is capacity times gpu memory utilization', () => {
  const fp = E.computeModelFootprint(BASE);
  assert.equal(fp.bytes.capacity, 80 * GIB);
  assert.equal(fp.bytes.budget, 72 * GIB);
  assert.equal(fp.bytes.unmanaged, 8 * GIB);
});

test('KV pool equals budget minus weights, activations and runtime', () => {
  const fp = E.computeModelFootprint(BASE);
  near(fp.bytes.kvPool, fp.bytes.budget - fp.bytes.weights - fp.bytes.activation - fp.bytes.runtime, 1);
});

test('pool blocks and sequence capacity follow from the pool size', () => {
  const fp = E.computeModelFootprint(BASE);
  assert.equal(fp.kv.poolBlocks, Math.floor(fp.bytes.kvPool / (131072 * 16)));
  assert.equal(fp.kv.blocksPerSeq, Math.ceil(2304 / 16));
  assert.equal(fp.kv.blocksPerSeq, 144);
  assert.equal(fp.kv.maxSeqs, Math.floor(fp.kv.poolBlocks / 144));
  assert.equal(fp.kv.demandBlocks, 144 * 32);
});

test('an 8B model on one H100 fits 32 sequences', () => {
  const fp = E.computeModelFootprint(BASE);
  assert.equal(fp.verdict, 'fits');
  assert.ok(fp.ok);
});

test('a 70B FP16 model does not fit on a single H100', () => {
  const fp = E.computeModelFootprint(Object.assign({}, BASE, { modelId: 'llama-3.1-70b' }));
  assert.equal(fp.verdict, 'oom');
  assert.match(fp.reasons[0], /budget/);
});

test('too many concurrent sequences produce an out of memory verdict with a fix hint', () => {
  const fp = E.computeModelFootprint(Object.assign({}, BASE, { concurrentSeqs: 2000 }));
  assert.equal(fp.verdict, 'oom');
  assert.match(fp.reasons[0], /Raise TP|quantize KV|lower concurrency/i);
  assert.ok(fp.bytes.kvDemand > fp.bytes.kvPool);
});

test('a pool that is nearly full is reported as tight', () => {
  const probe = E.computeModelFootprint(BASE);
  const conc = Math.floor(0.9 * probe.kv.poolBlocks / probe.kv.blocksPerSeq);
  const fp = E.computeModelFootprint(Object.assign({}, BASE, { concurrentSeqs: conc }));
  assert.equal(fp.verdict, 'tight');
  assert.ok(fp.kv.utilization > 0.85 && fp.kv.utilization <= 1);
});

test('quantizing KV to FP8 doubles the tokens the pool holds', () => {
  const a = E.computeModelFootprint(BASE);
  const b = E.computeModelFootprint(Object.assign({}, BASE, { kvPrecision: 'fp8' }));
  near(b.kv.poolTokens / a.kv.poolTokens, 2, 0.001);
});

test('raising TP shrinks weights per GPU and grows the KV pool', () => {
  const a = E.computeModelFootprint(Object.assign({}, BASE, { modelId: 'llama-3.1-70b', weightPrecision: 'fp8', tp: 2, concurrentSeqs: 1 }));
  const b = E.computeModelFootprint(Object.assign({}, BASE, { modelId: 'llama-3.1-70b', weightPrecision: 'fp8', tp: 4, concurrentSeqs: 1 }));
  assert.ok(b.bytes.weights < a.bytes.weights);
  assert.ok(b.kv.poolTokens > a.kv.poolTokens);
  near(b.bytes.weightsTotal / 4, b.bytes.weights, 1);
});

test('free headroom plus live KV plus fixed memory equals capacity', () => {
  const fp = E.computeModelFootprint(BASE);
  const sum = fp.bytes.weights + fp.bytes.activation + fp.bytes.runtime + fp.bytes.kvLive + fp.bytes.headroom;
  near(sum, fp.bytes.capacity, 2);
});

test('invalid tensor parallel size is rejected with a reason', () => {
  const fp = E.computeModelFootprint(Object.assign({}, BASE, { tp: 3 }));
  assert.equal(fp.verdict, 'invalid');
  assert.equal(fp.ok, false);
  assert.ok(fp.errors.length > 0);
});

test('tensor parallel must divide the query heads', () => {
  const errs = E.validateParallelism(E.getModel('deepseek-r1-distill-qwen-32b'), E.GPUS.h100, 8, 1);
  assert.equal(errs.length, 0);
  const bad = E.validateParallelism(E.getModel('deepseek-r1-distill-qwen-32b'), E.GPUS.h100, 3, 1);
  assert.ok(bad.some((e) => /divide/.test(e)));
});

test('prompt plus output beyond the model context is invalid', () => {
  const fp = E.computeModelFootprint({ modelId: 'gemma-2-27b', gpuId: 'h200', tp: 2, pp: 1, promptTokens: 9000, outputTokens: 1000, concurrentSeqs: 1 });
  assert.equal(fp.verdict, 'invalid');
  assert.match(fp.errors[0], /exceeds max model length/);
});

test('the FP8 405B checkpoint rejects FP16 weights', () => {
  const fp = E.computeModelFootprint({ modelId: 'llama-3.1-405b-fp8', gpuId: 'h200', tp: 8, pp: 1, weightPrecision: 'fp16', promptTokens: 1000, outputTokens: 100 });
  assert.equal(fp.verdict, 'invalid');
  const ok = E.computeModelFootprint({ modelId: 'llama-3.1-405b-fp8', gpuId: 'h200', tp: 8, pp: 1, promptTokens: 1000, outputTokens: 100, concurrentSeqs: 4 });
  assert.equal(ok.cfg.weightPrecision, 'fp8');
  assert.equal(ok.verdict, 'fits');
});

test('A100 with FP8 weights warns that compute falls back to the FP16 rate', () => {
  const fp = E.computeModelFootprint(Object.assign({}, BASE, { gpuId: 'a100', weightPrecision: 'fp8' }));
  assert.ok(fp.warnings.some((w) => /FP16 rate/.test(w)));
});

test('INT8 and INT4 KV caches are flagged as unsupported by vLLM', () => {
  const fp = E.computeModelFootprint(Object.assign({}, BASE, { kvPrecision: 'int8' }));
  assert.ok(fp.warnings.some((w) => /no INT8 KV cache dtype/.test(w)));
});

test('Nemotron-4-340B needs FP8 or more GPUs than eight H100 can hold in BF16', () => {
  const bf16 = E.computeModelFootprint({ modelId: 'nemotron-4-340b', gpuId: 'h100', tp: 8, pp: 1, promptTokens: 1000, outputTokens: 100, concurrentSeqs: 1 });
  assert.equal(bf16.verdict, 'oom');
  const fp8 = E.computeModelFootprint({ modelId: 'nemotron-4-340b', gpuId: 'h100', tp: 8, pp: 1, weightPrecision: 'fp8', promptTokens: 1000, outputTokens: 100, concurrentSeqs: 1 });
  assert.equal(fp8.verdict, 'fits');
});

test('footprint steps explain the arithmetic', () => {
  const fp = E.computeModelFootprint(BASE);
  assert.equal(fp.steps[0].value, '131,072 B');
  assert.match(fp.steps[0].formula, /2 x 32 layers x 8 KV heads x 128 head dim/);
});

/* ---------- GPU specs and roofline ---------- */

test('GPU table carries the datasheet bandwidth and dense throughput', () => {
  assert.equal(E.GPUS.h100.bwGBs, 3350);
  assert.equal(E.GPUS.h200.bwGBs, 4800);
  assert.equal(E.GPUS.l40s.bwGBs, 864);
  assert.equal(E.GPUS.h100.tflops.fp16, 989);
  assert.equal(E.GPUS.h100.tflops.fp8, 1979);
  assert.equal(E.GPUS.l40s.tflops.fp16, 362);
  assert.equal(E.GPUS.a100.tflops.fp16, 312);
  assert.equal(E.GPUS.a100.tflops.fp8, null);
  assert.equal(E.GPUS.h200.memGiB, 141);
});

test('L40S has no NVLink and A100 has no FP8 Tensor Cores', () => {
  assert.equal(E.GPUS.l40s.link.kind, 'pcie');
  assert.equal(E.computeRate(E.GPUS.a100, 'fp8').fallback, true);
  assert.equal(E.computeRate(E.GPUS.a100, 'fp8').tflops, 312);
  assert.equal(E.computeRate(E.GPUS.h100, 'fp8').tflops, 1979);
  assert.equal(E.computeRate(E.GPUS.l40s, 'int4').tflops, 362);
});

test('single stream decode of Llama-3.1-8B on an H100 matches the hand derived roofline', () => {
  const m = E.computeRooflineMetrics(Object.assign({}, BASE, { concurrentSeqs: 1 }));
  const bytes = 15009849344 + 2176 * 131072;
  const memoryMs = bytes / (3.35e12 * 0.8) * 1000;
  near(m.decode.memoryMs, memoryMs, 1e-6);
  near(m.decode.itlMs, memoryMs + 0.4 + 0.012 * 32, 1e-6);
  near(m.decode.itlMs, 6.49, 0.01);
  assert.equal(m.decode.boundBy, 'memory');
});

test('prefill of a 2048 token prompt on an H100 is compute bound near 62 ms', () => {
  const m = E.computeRooflineMetrics(Object.assign({}, BASE, { concurrentSeqs: 1 }));
  const flops = 2 * 6979588096 * 2048 + 2 * 32 * 32 * 128 * 2048 * 2048 + 2 * 128256 * 4096;
  near(m.prefill.computeMs, flops / (989e12 * 0.5) * 1000, 1e-6);
  near(m.prefill.ttftMs, 61.8, 0.5);
  assert.equal(m.prefill.boundBy, 'compute');
});

test('decode memory time scales inversely with HBM bandwidth', () => {
  const a = E.computeRooflineMetrics(Object.assign({}, BASE, { concurrentSeqs: 1, gpuId: 'h100' }));
  const b = E.computeRooflineMetrics(Object.assign({}, BASE, { concurrentSeqs: 1, gpuId: 'h200' }));
  near(a.decode.memoryMs / b.decode.memoryMs, 4800 / 3350, 1e-9);
  assert.ok(b.decode.itlMs < a.decode.itlMs);
});

test('doubling TP roughly halves weight streaming time for a 70B model', () => {
  const cfg = { modelId: 'llama-3.1-70b', gpuId: 'h200', pp: 1, weightPrecision: 'fp8', kvPrecision: 'fp8', concurrentSeqs: 1, promptTokens: 512, outputTokens: 64 };
  const a = E.computeRooflineMetrics(Object.assign({ tp: 2 }, cfg));
  const b = E.computeRooflineMetrics(Object.assign({ tp: 4 }, cfg));
  near(a.decode.memoryMs / b.decode.memoryMs, 2, 0.05);
});

test('no all-reduce cost at TP 1 and a positive cost at TP 2', () => {
  const one = E.computeRooflineMetrics(Object.assign({}, BASE, { tp: 1 }));
  const two = E.computeRooflineMetrics(Object.assign({}, BASE, { tp: 2 }));
  assert.equal(one.decode.commMs, 0);
  assert.ok(two.decode.commMs > 0);
  assert.equal(E.allReduceMs(E.makeContext(BASE), 1e6), 0);
});

test('PCIe all-reduce on L40S costs more than NVLink on H100 at the same TP', () => {
  const cfg = { modelId: 'llama-3.1-8b', tp: 4, pp: 1, weightPrecision: 'fp8', promptTokens: 4000, outputTokens: 100, concurrentSeqs: 1 };
  const l = E.computeRooflineMetrics(Object.assign({ gpuId: 'l40s' }, cfg));
  const h = E.computeRooflineMetrics(Object.assign({ gpuId: 'h100' }, cfg));
  assert.ok(l.prefill.commMs > 3 * h.prefill.commMs);
});

test('larger batches raise decode latency and per replica throughput', () => {
  const a = E.computeRooflineMetrics(Object.assign({}, BASE, { concurrentSeqs: 1 }));
  const b = E.computeRooflineMetrics(Object.assign({}, BASE, { concurrentSeqs: 64 }));
  assert.ok(b.decode.itlMs > a.decode.itlMs);
  assert.ok(b.decode.tokensPerSecReplica > 20 * a.decode.tokensPerSecReplica);
});

test('A100 prefill compute time is the same for FP8 and FP16 weights because it falls back', () => {
  const a = E.computeRooflineMetrics(Object.assign({}, BASE, { gpuId: 'a100', weightPrecision: 'fp16' }));
  const b = E.computeRooflineMetrics(Object.assign({}, BASE, { gpuId: 'a100', weightPrecision: 'fp8' }));
  assert.equal(a.prefill.computeMs, b.prefill.computeMs);
});

test('FP8 on H100 halves prefill compute time against FP16', () => {
  const a = E.computeRooflineMetrics(Object.assign({}, BASE, { weightPrecision: 'fp16' }));
  const b = E.computeRooflineMetrics(Object.assign({}, BASE, { weightPrecision: 'fp8' }));
  near(a.prefill.computeMs / b.prefill.computeMs, 1979 / 989, 1e-9);
});

test('roofline curve ridge is peak over bandwidth after efficiency', () => {
  const full = E.rooflineCurve('h100', 'fp16', { memory: 1, compute: 1 });
  near(full.ridge, 989 / 3.35, 1e-9);
  const def = E.rooflineCurve('h100', 'fp16');
  near(def.peakTflops, 494.5, 1e-9);
  near(def.bandwidthTBs, 2.68, 1e-9);
});

test('TTFT grows with burst size and flags layouts that do not fit', () => {
  const series = E.ttftVsBatch({ modelId: 'llama-3.1-70b', tp: 1, pp: 1, weightPrecision: 'fp16', promptTokens: 1000, outputTokens: 100 }, ['h100', 'l40s'], [1, 2, 4, 8]);
  assert.equal(series.length, 2);
  assert.equal(series[0].points.length, 4);
  for (let i = 1; i < 4; i++) assert.ok(series[0].points[i].ttftMs > series[0].points[i - 1].ttftMs);
  assert.equal(series[0].fits, false);
  assert.equal(series[1].fits, false);
});

test('H100 and H200 have identical compute bound TTFT', () => {
  const s = E.ttftVsBatch(Object.assign({}, BASE, { promptTokens: 4000 }), ['h100', 'h200'], [8]);
  near(s[0].points[0].ttftMs, s[1].points[0].ttftMs, 1e-9);
});

test('chunked prefill adds a weight pass when the batch exceeds max batched tokens', () => {
  const ctx = E.makeContext(Object.assign({}, BASE, { maxNumBatchedTokens: 2048 }));
  assert.equal(E.prefillStep(ctx, 2048, 1).passes, 1);
  assert.equal(E.prefillStep(ctx, 2048, 3).passes, 3);
});

/* ---------- router ---------- */

const POOL_SMALL = { id: 'small', label: 'Small', modelId: 'llama-3.1-8b', gpuId: 'l40s', gpus: 2, tp: 1, pp: 1, weightPrecision: 'fp8', kvPrecision: 'fp8', maxModelLen: 16384, costPerGpuHour: 1.55 };
const POOL_LARGE = { id: 'large', label: 'Large', modelId: 'llama-3.1-70b', gpuId: 'h200', gpus: 4, tp: 4, pp: 1, weightPrecision: 'fp8', kvPrecision: 'fp8', maxModelLen: 32768, costPerGpuHour: 5.4 };
function req(id, extra) {
  return Object.assign({ id: id, inputTokens: 500, outputTokens: 100, complexity: 0.2, sloTtftMs: 5000, sloItlMs: 100, arrivalMs: 0 }, extra || {});
}

test('capability proxy is the log of parameters scaled so 400B is 1.0', () => {
  near(E.capabilityProxy(E.getModel('llama-3.1-70b')), Math.log2(70.55) / Math.log2(400), 0.001);
  assert.equal(E.capabilityProxy({ params: 4e11 }), 1);
  assert.equal(E.capabilityProxy({ params: 1e12 }), 1);
});

test('percentile uses nearest rank', () => {
  const v = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
  assert.equal(E.percentile(v, 95), 10);
  assert.equal(E.percentile(v, 50), 5);
  assert.equal(E.percentile([], 95), 0);
});

test('paretoFront drops dominated candidates only', () => {
  const front = E.paretoFront([
    { id: 'a', cost: 1, ttftMs: 5, itlMs: 5 }, { id: 'b', cost: 2, ttftMs: 1, itlMs: 1 }, { id: 'c', cost: 3, ttftMs: 6, itlMs: 6 }
  ]);
  assert.deepEqual(front.map((c) => c.id), ['a', 'b']);
});

test('an easy request goes to the cheaper small pool', () => {
  const r = E.routeWorkloadBatch([req('r1')], [POOL_LARGE, POOL_SMALL]);
  assert.equal(r.decisions[0].poolId, 'small');
  assert.equal(r.decisions[0].status, 'routed');
});

test('a hard request is only eligible for a capable pool', () => {
  const r = E.routeWorkloadBatch([req('r1', { complexity: 0.6 })], [POOL_SMALL, POOL_LARGE]);
  assert.equal(r.decisions[0].poolId, 'large');
});

test('a request harder than every pool is rejected as no capable pool', () => {
  const r = E.routeWorkloadBatch([req('r1', { complexity: 0.95 })], [POOL_SMALL, POOL_LARGE]);
  assert.equal(r.decisions[0].status, 'rejected');
  assert.equal(r.decisions[0].reason, 'no_capable_pool');
});

test('a request longer than max model length is rejected', () => {
  const r = E.routeWorkloadBatch([req('r1', { inputTokens: 20000 })], [POOL_SMALL]);
  assert.equal(r.decisions[0].reason, 'context_too_long');
});

test('a request that cannot fit in any KV pool is rejected as kv_oom', () => {
  const tiny = Object.assign({}, POOL_SMALL, { weightPrecision: 'fp16', kvPrecision: 'fp16', gpuMemoryUtilization: 0.5, maxModelLen: 131072 });
  const r = E.routeWorkloadBatch([req('r1', { inputTokens: 60000, outputTokens: 100, sloTtftMs: 1e9 })], [tiny]);
  assert.equal(r.decisions[0].reason, 'kv_oom');
});

test('an unreachable SLO is reported as violated on the least bad pool', () => {
  const r = E.routeWorkloadBatch([req('r1', { sloTtftMs: 0.001 })], [POOL_SMALL, POOL_LARGE]);
  assert.equal(r.decisions[0].status, 'slo_violated');
  assert.ok(r.decisions[0].poolId);
});

test('simultaneous requests spread across replicas of a pool', () => {
  const r = E.routeWorkloadBatch([req('r1'), req('r2')], [POOL_SMALL]);
  assert.equal(r.decisions[0].replica, 0);
  assert.equal(r.decisions[1].replica, 1);
});

test('later requests see earlier ones as load and see latency rise', () => {
  const many = [];
  for (let i = 0; i < 40; i++) many.push(req('r' + i));
  const r = E.routeWorkloadBatch(many, [POOL_SMALL]);
  assert.ok(r.decisions[39].itlMs >= r.decisions[0].itlMs);
  assert.ok(r.decisions[39].ttftMs > r.decisions[0].ttftMs);
});

test('requests that arrive after earlier ones finish see an empty pool', () => {
  const r = E.routeWorkloadBatch([req('r1'), req('r2', { arrivalMs: 10 * 60 * 1000 })], [Object.assign({}, POOL_SMALL, { gpus: 1 })]);
  near(r.decisions[1].ttftMs, r.decisions[0].ttftMs, 1e-9);
  near(r.decisions[1].itlMs, r.decisions[0].itlMs, 1e-9);
});

test('the largest policy sends everything to the most capable pool', () => {
  const r = E.routeWorkloadBatch([req('r1'), req('r2')], [POOL_SMALL, POOL_LARGE], { policy: 'largest' });
  assert.deepEqual(r.decisions.map((d) => d.poolId), ['large', 'large']);
});

test('the naive policy counts KV preemptions where pareto routes around them', () => {
  const cramped = Object.assign({}, POOL_SMALL, { gpus: 1, weightPrecision: 'fp16', kvPrecision: 'fp16', gpuMemoryUtilization: 0.6, maxModelLen: 131072 });
  const reqs = [];
  for (let i = 0; i < 12; i++) reqs.push(req('r' + i, { inputTokens: 12000, outputTokens: 500, sloTtftMs: 1e9, sloItlMs: 1e9 }));
  const naive = E.routeWorkloadBatch(reqs, [cramped, POOL_LARGE], { policy: 'naive-cheapest' });
  const pareto = E.routeWorkloadBatch(reqs, [cramped, POOL_LARGE], { policy: 'pareto' });
  assert.ok(naive.summary.kvPreemptions > 0);
  assert.equal(pareto.summary.kvPreemptions, 0);
  assert.equal(pareto.summary.rejected, 0);
});

test('routing is deterministic for the same input', () => {
  const reqs = E.generateWorkload('saas-burst', { seed: 5, count: 60 });
  const a = E.routeWorkloadBatch(reqs, E.REFERENCE_CLUSTER);
  const b = E.routeWorkloadBatch(reqs, E.REFERENCE_CLUSTER);
  assert.equal(JSON.stringify(a), JSON.stringify(b));
});

test('summary cost per million tokens is total cost over total tokens', () => {
  const r = E.routeWorkloadBatch([req('r1'), req('r2')], [POOL_SMALL]);
  const s = r.summary;
  near(s.costPerMillionTokens, s.totalCost / s.tokens * 1e6, 1e-9);
  assert.equal(s.tokens, 1200);
  assert.equal(s.total, 2);
});

test('a pool that cannot hold its weights is reported unusable and skipped', () => {
  const broken = { id: 'broken', label: 'Broken', modelId: 'llama-3.1-70b', gpuId: 'l40s', gpus: 1, tp: 1, pp: 1, weightPrecision: 'fp16' };
  const r = E.routeWorkloadBatch([req('r1')], [broken, POOL_SMALL]);
  assert.equal(r.pools[0].usable, false);
  assert.equal(r.decisions[0].poolId, 'small');
});

test('the reference cluster is deployable and every preset routes without rejection of easy work', () => {
  E.REFERENCE_CLUSTER.forEach((p) => assert.equal(E.resolvePool(p).usable, true, p.id));
  E.WORKLOAD_PRESETS.forEach((p) => {
    const r = E.routeWorkloadBatch(E.generateWorkload(p.id), E.REFERENCE_CLUSTER);
    assert.equal(r.summary.total, p.workload.count);
    assert.ok(r.summary.metSlo > r.summary.total * 0.8, p.id);
  });
});

test('the code audit preset shows the KV pressure story', () => {
  const cmp = E.compareRoutingPolicies(E.generateWorkload('code-audit-32k'), E.REFERENCE_CLUSTER);
  assert.ok(cmp.naive.summary.kvPreemptions > 0);
  assert.equal(cmp.pareto.summary.kvPreemptions, 0);
  assert.ok(cmp.pareto.summary.metSlo > cmp.naive.summary.metSlo);
});

/* ---------- workloads and random numbers ---------- */

test('mulberry32 is deterministic and in range', () => {
  const a = E.mulberry32(42), b = E.mulberry32(42);
  for (let i = 0; i < 50; i++) {
    const x = a();
    assert.equal(x, b());
    assert.ok(x >= 0 && x < 1);
  }
  assert.notEqual(E.mulberry32(1)(), E.mulberry32(2)());
});

test('generateWorkload is reproducible and respects class ranges', () => {
  const a = E.generateWorkload('voice-agent', { seed: 9, count: 80 });
  const b = E.generateWorkload('voice-agent', { seed: 9, count: 80 });
  assert.deepEqual(a, b);
  assert.equal(a.length, 80);
  a.forEach((r) => {
    assert.ok(r.inputTokens >= 200 && r.inputTokens <= 2400);
    assert.ok(r.outputTokens >= 40 && r.outputTokens <= 160);
    assert.ok(r.complexity >= 0.05 && r.complexity <= 0.55);
    assert.ok(r.arrivalMs >= 0 && r.arrivalMs <= 10000);
  });
  assert.notDeepEqual(a, E.generateWorkload('voice-agent', { seed: 10, count: 80 }));
});

test('there are exactly four presets with unique ids and valid scenarios', () => {
  assert.equal(E.WORKLOAD_PRESETS.length, 4);
  assert.equal(new Set(E.WORKLOAD_PRESETS.map((p) => p.id)).size, 4);
  E.WORKLOAD_PRESETS.forEach((p) => {
    const fp = E.computeModelFootprint(p.scenario);
    assert.ok(fp.ok, p.id + ' ' + fp.errors.join(' '));
    assert.notEqual(fp.verdict, 'oom', p.id);
  });
});

/* ---------- exporter ---------- */

const EXPORT_POOL = { id: 'chat', label: 'Chat', modelId: 'llama-3.1-70b', gpuId: 'h100', gpus: 8, tp: 4, pp: 1, weightPrecision: 'fp8', kvPrecision: 'fp8', gpuMemoryUtilization: 0.9, maxNumSeqs: 128, maxNumBatchedTokens: 8192, maxModelLen: 16384 };

test('vLLM arguments carry the engine derived flags', () => {
  const a = E.vllmArgs(EXPORT_POOL);
  const flat = a.args.map((p) => p.join(' '));
  assert.ok(flat.includes('--tensor-parallel-size 4'));
  assert.ok(flat.includes('--gpu-memory-utilization 0.90'));
  assert.ok(flat.includes('--max-model-len 16384'));
  assert.ok(flat.includes('--max-num-seqs 128'));
  assert.ok(flat.includes('--max-num-batched-tokens 8192'));
  assert.ok(flat.includes('--quantization fp8'));
  assert.ok(flat.includes('--kv-cache-dtype fp8'));
  assert.equal(a.repo, 'meta-llama/Llama-3.1-70B-Instruct');
});

test('compose output pins each replica to its own GPUs and ports', () => {
  const m = E.exportClusterManifest([EXPORT_POOL]);
  assert.equal(m.services.length, 2);
  assert.deepEqual(m.services[0].deviceIds, ['0', '1', '2', '3']);
  assert.deepEqual(m.services[1].deviceIds, ['4', '5', '6', '7']);
  assert.equal(m.services[0].port, 8000);
  assert.equal(m.services[1].port, 8001);
  assert.match(m.composeYaml, /device_ids: \["0", "1", "2", "3"\]/);
  assert.match(m.composeYaml, /"--tensor-parallel-size", "4"/);
  assert.match(m.composeYaml, /image: vllm\/vllm-openai:latest/);
  assert.match(m.composeYaml, /HUGGING_FACE_HUB_TOKEN=\$\{HF_TOKEN\}/);
});

test('compose YAML is indented in whole steps of two spaces', () => {
  const m = E.exportClusterManifest(E.REFERENCE_CLUSTER);
  m.composeYaml.split('\n').filter((l) => l && !l.startsWith('#')).forEach((l) => {
    assert.equal((l.length - l.trimStart().length) % 2, 0, l);
  });
});

test('serve script launches one pinned vllm process per replica', () => {
  const m = E.exportClusterManifest([EXPORT_POOL]);
  assert.match(m.serveScript, /CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve meta-llama\/Llama-3.1-70B-Instruct/);
  assert.match(m.serveScript, /CUDA_VISIBLE_DEVICES=4,5,6,7 vllm serve/);
  assert.match(m.serveScript, /--port 8001 &/);
  assert.match(m.serveScript, /^#!\/usr\/bin\/env bash/);
});

test('the native FP8 405B checkpoint gets no quantization flag', () => {
  const pool = { id: 'big', label: 'Big', modelId: 'llama-3.1-405b-fp8', gpuId: 'h200', gpus: 8, tp: 8, pp: 1, kvPrecision: 'fp8', maxModelLen: 32768 };
  const a = E.vllmArgs(pool);
  assert.ok(!a.args.some((p) => p[0] === '--quantization'));
  assert.equal(a.repo, 'RedHatAI/Meta-Llama-3.1-405B-Instruct-FP8');
});

test('INT4 weights use an AWQ checkpoint and the awq flag', () => {
  const a = E.vllmArgs(Object.assign({}, EXPORT_POOL, { weightPrecision: 'int4', kvPrecision: 'fp16' }));
  assert.ok(a.args.some((p) => p[0] === '--quantization' && p[1] === 'awq'));
  assert.equal(a.repo, 'hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4');
  assert.ok(!a.args.some((p) => p[0] === '--kv-cache-dtype'));
});

test('INT8 weights emit no flag but warn about compressed-tensors', () => {
  const a = E.vllmArgs(Object.assign({}, EXPORT_POOL, { weightPrecision: 'int8' }));
  assert.ok(!a.args.some((p) => p[0] === '--quantization'));
  assert.equal(a.repo, 'RedHatAI/Meta-Llama-3.1-70B-Instruct-quantized.w8a8');
  assert.ok(a.warnings.some((w) => /compressed-tensors/.test(w)));
});

test('an INT8 KV cache is dropped from the command with a warning', () => {
  const a = E.vllmArgs(Object.assign({}, EXPORT_POOL, { kvPrecision: 'int8' }));
  assert.ok(!a.args.some((p) => p[0] === '--kv-cache-dtype'));
  assert.ok(a.warnings.some((w) => /no vLLM flag/.test(w)));
});

test('pipeline parallel size appears only when above one', () => {
  const one = E.vllmArgs(EXPORT_POOL);
  assert.ok(!one.args.some((p) => p[0] === '--pipeline-parallel-size'));
  const two = E.vllmArgs(Object.assign({}, EXPORT_POOL, { tp: 2, pp: 2 }));
  assert.ok(two.args.some((p) => p[0] === '--pipeline-parallel-size' && p[1] === 2));
});

test('gated models and NeMo checkpoints produce warnings', () => {
  const m = E.exportClusterManifest([EXPORT_POOL, { id: 'nemo', label: 'Nemo', modelId: 'nemotron-4-340b', gpuId: 'h200', gpus: 8, tp: 8, pp: 1, weightPrecision: 'fp8', maxModelLen: 4096 }]);
  assert.ok(m.warnings.some((w) => /gated/.test(w)));
  assert.ok(m.warnings.some((w) => /NeMo checkpoint/.test(w)));
});

test('TensorRT-LLM script uses the documented flag names and a KV fraction below one', () => {
  const m = E.exportClusterManifest([EXPORT_POOL]);
  assert.match(m.trtllmScript, /--tp_size 4 --pp_size 1 --max_batch_size 128 --max_num_tokens 8192/);
  assert.match(m.trtllmScript, /free_gpu_memory_fraction: 0\.\d\d/);
  assert.match(m.trtllmScript, /dtype: fp8/);
  const t = E.tensorRtArgs(EXPORT_POOL);
  assert.ok(t.kvFraction > 0.1 && t.kvFraction <= 0.95);
});

test('router policy JSON parses and lists endpoints per pool', () => {
  const m = E.exportClusterManifest([EXPORT_POOL]);
  const j = JSON.parse(m.routerPolicyJson);
  assert.equal(j.pools[0].replicas, 2);
  assert.equal(j.pools[0].endpoints.length, 2);
  assert.ok(j.pools[0].kvPoolTokensPerReplica > 0);
});

test('exported files contain no em or en dashes', () => {
  const m = E.exportClusterManifest(E.REFERENCE_CLUSTER);
  [m.composeYaml, m.serveScript, m.trtllmScript, m.routerPolicyJson].concat(m.warnings).forEach((t) => {
    assert.ok(!new RegExp('[' + String.fromCharCode(0x2013, 0x2014) + ']').test(t));
  });
});

/* ---------- probe ---------- */

function sse(obj) { return 'data: ' + JSON.stringify(obj) + '\n\n'; }

function fakeFetch(frames, log) {
  const clock = { t: 0 };
  const impl = async (url, init) => {
    if (log) log.push({ url: url, init: init });
    if (/\/models$/.test(url)) return { ok: true, status: 200, json: async () => ({ data: [{ id: 'm1' }, { id: 'm2' }] }) };
    if (frames === 401) return { ok: false, status: 401 };
    let i = 0;
    return { ok: true, status: 200, body: { getReader: () => ({ read: async () => {
      if (i >= frames.length) return { done: true };
      const f = frames[i++];
      clock.t += f.gap;
      return { done: false, value: f.text };
    } }) } };
  };
  return { impl: impl, now: () => clock.t };
}

test('parseSseLines keeps the incomplete tail and ignores junk', () => {
  const p = E.parseSseLines('data: {"a":1}\n\n: comment\ndata: [DONE]\n\ndata: {"b"');
  assert.equal(p.events.length, 2);
  assert.deepEqual(p.events[0].json, { a: 1 });
  assert.equal(p.events[1].done, true);
  assert.equal(p.rest, 'data: {"b"');
});

test('analyzeStream computes TTFT and ITL from the usage token count', () => {
  const a = E.analyzeStream([100, 110, 120, 130], 140, 4);
  assert.equal(a.ttftMs, 100);
  assert.equal(a.itlMs, 10);
  assert.equal(a.itlBasis, 'usage');
  assert.equal(a.tokensPerSec, 100);
});

test('analyzeStream falls back to chunk spacing and reports empty streams', () => {
  const a = E.analyzeStream([50, 70, 90], 100, null);
  assert.equal(a.itlMs, 20);
  assert.equal(a.itlBasis, 'chunks');
  assert.equal(E.analyzeStream([], 10, null).ok, false);
});

test('probeEndpoint measures time to first chunk and sends the key only as a header', async () => {
  const log = [];
  const f = fakeFetch([
    { gap: 120, text: sse({ choices: [{ delta: { content: 'a' } }] }) },
    { gap: 10, text: sse({ choices: [{ delta: { content: 'b' } }] }) },
    { gap: 10, text: sse({ choices: [{ delta: { content: 'c' } }] }) },
    { gap: 0, text: sse({ choices: [], usage: { prompt_tokens: 77, completion_tokens: 3 } }) + 'data: [DONE]\n\n' }
  ], log);
  const r = await E.probeEndpoint({ baseUrl: 'https://api.example.test/v1/', apiKey: 'sk-secret', model: 'm1', fetchImpl: f.impl, now: f.now });
  assert.equal(r.ok, true);
  assert.equal(r.ttftMs, 120);
  assert.equal(r.itlMs, 10);
  assert.equal(r.promptTokens, 77);
  assert.equal(log[0].url, 'https://api.example.test/v1/chat/completions');
  assert.equal(log[0].init.headers.Authorization, 'Bearer sk-secret');
  assert.ok(!log[0].url.includes('sk-secret'));
  assert.ok(!log[0].init.body.includes('sk-secret'));
  assert.equal(JSON.parse(log[0].init.body).stream, true);
});

test('probeEndpoint omits the Authorization header without a key', async () => {
  const log = [];
  const f = fakeFetch([{ gap: 5, text: sse({ choices: [{ delta: { content: 'a' } }] }) }], log);
  await E.probeEndpoint({ baseUrl: 'http://localhost:8000/v1', model: 'm', fetchImpl: f.impl, now: f.now });
  assert.equal(log[0].init.headers.Authorization, undefined);
});

test('probeEndpoint reports HTTP failures without throwing', async () => {
  const f = fakeFetch(401);
  const r = await E.probeEndpoint({ baseUrl: 'https://x.test/v1', apiKey: 'k', model: 'm', fetchImpl: f.impl, now: f.now });
  assert.equal(r.ok, false);
  assert.equal(r.status, 401);
  assert.ok(!r.error.includes('k'.repeat(5)));
});

test('listModels returns ids from the models endpoint', async () => {
  const f = fakeFetch([]);
  const ids = await E.listModels({ baseUrl: 'https://x.test/v1', apiKey: 'k', fetchImpl: f.impl });
  assert.deepEqual(ids, ['m1', 'm2']);
});

test('offline replay is deterministic and stays near the roofline', async () => {
  const cfg = Object.assign({}, BASE, { promptTokens: 256, outputTokens: 64, concurrentSeqs: 1 });
  const run = async () => {
    const t = E.createReplayTransport(cfg, 11);
    const r = await E.probeEndpoint({ baseUrl: 'replay://local', model: 'm', promptTokens: 256, maxTokens: 64, fetchImpl: t.fetchImpl, now: t.now });
    return { r: r, t: t };
  };
  const a = await run(), b = await run();
  assert.equal(a.r.ttftMs, b.r.ttftMs);
  assert.equal(a.r.itlMs, b.r.itlMs);
  assert.equal(a.r.ok, true);
  near(a.r.itlMs / a.t.metrics.decode.singleStreamItlMs, 1, 0.1);
  assert.equal(a.r.completionTokens, 64);
});

test('compareProbeToRoofline returns measured over predicted ratios', () => {
  const m = E.computeRooflineMetrics(Object.assign({}, BASE, { concurrentSeqs: 1 }));
  const c = E.compareProbeToRoofline({ ok: true, ttftMs: 2 * 61.8, itlMs: m.decode.singleStreamItlMs * 1.5, promptTokens: 2048 }, m);
  near(c.ttftRatio, 2, 0.02);
  near(c.itlRatio, 1.5, 1e-9);
});

test('the Nebius endpoint is the default probe target documented in the page', () => {
  const html = require('node:fs').readFileSync(require('node:path').join(__dirname, '..', 'index.html'), 'utf8');
  assert.ok(html.includes('https://api.studio.nebius.ai/v1'));
});

/* ---------- gap closers found by mutation testing ---------- */

test('FP8 KV on A100 warns but the same setting on H100 does not', () => {
  const a = E.computeModelFootprint(Object.assign({}, BASE, { gpuId: 'a100', kvPrecision: 'fp8' }));
  const h = E.computeModelFootprint(Object.assign({}, BASE, { gpuId: 'h100', kvPrecision: 'fp8' }));
  assert.ok(a.warnings.some((w) => /FP8 KV cache on A100/.test(w)));
  assert.ok(!h.warnings.some((w) => /A100/.test(w)));
});

test('INT4 KV warns and FP8 or FP16 KV do not warn about vLLM dtypes', () => {
  const i4 = E.computeModelFootprint(Object.assign({}, BASE, { kvPrecision: 'int4' }));
  assert.ok(i4.warnings.some((w) => /no INT4-AWQ KV cache dtype/.test(w)));
  ['fp16', 'fp8'].forEach((k) => {
    const fp = E.computeModelFootprint(Object.assign({}, BASE, { kvPrecision: k }));
    assert.ok(!fp.warnings.some((w) => /KV cache dtype/.test(w)), k);
  });
});

test('tensor parallel above the KV head count must be a multiple of it', () => {
  const m = (nKv) => ({ name: 'toy', nQ: 24, nKv: nKv, layers: 24 });
  assert.equal(E.validateParallelism(m(4), E.GPUS.h100, 8, 1).length, 0);
  assert.ok(E.validateParallelism(m(3), E.GPUS.h100, 8, 1).some((e) => /multiple of the 3 KV heads/.test(e)));
  assert.ok(E.validateParallelism(m(8), E.GPUS.h100, 2, 1).length === 0);
});

test('pipeline parallel size must be one, two or four', () => {
  const errs = E.validateParallelism(E.getModel('llama-3.1-8b'), E.GPUS.h100, 1, 3);
  assert.ok(errs.some((e) => /Pipeline parallel size must be 1, 2 or 4/.test(e)));
  assert.equal(E.validateParallelism(E.getModel('llama-3.1-8b'), E.GPUS.h100, 1, 4).length, 0);
});

test('ttftVsBatch marks fitting, out of memory and invalid layouts', () => {
  const ok = E.ttftVsBatch(BASE, ['h100'], [1]);
  assert.equal(ok[0].fits, true);
  assert.equal(ok[0].verdict, 'fits');
  const invalid = E.ttftVsBatch(Object.assign({}, BASE, { tp: 3 }), ['h100'], [1]);
  assert.equal(invalid[0].fits, false);
  assert.equal(invalid[0].valid, false);
});

test('compute rate reports no fallback when the GPU has the Tensor Cores', () => {
  assert.equal(E.computeRate(E.GPUS.h100, 'fp16').fallback, false);
  assert.equal(E.computeRate(E.GPUS.h100, 'fp8').fallback, false);
  assert.equal(E.computeRate(E.GPUS.l40s, 'int8').fallback, false);
});

test('replicas inside one node use the GPU link and larger layouts use InfiniBand', () => {
  assert.equal(E.makeContext(Object.assign({}, BASE, { tp: 4, pp: 2 })).link.kind, 'nvlink');
  assert.equal(E.makeContext(Object.assign({}, BASE, { tp: 8, pp: 2 })).link.kind, 'infiniband');
  assert.equal(E.makeContext(Object.assign({}, BASE, { gpuId: 'l40s', tp: 2, pp: 1 })).link.kind, 'pcie');
});

test('all-reduce time follows the ring formula exactly', () => {
  const ctx = E.makeContext(Object.assign({}, BASE, { tp: 4 }));
  const msg = 8 * 1024 * 1024;
  const expected = 12 / 1000 + (2 * 3 / 4 * msg / (450e9 * 0.75)) * 1000;
  near(E.allReduceMs(ctx, msg), expected, 1e-9);
  const two = E.makeContext(Object.assign({}, BASE, { tp: 2 }));
  near(E.allReduceMs(two, msg), 12 / 1000 + (2 * 1 / 2 * msg / (450e9 * 0.75)) * 1000, 1e-9);
});

test('pipeline hops cost nothing at PP 1 and scale with the stage count', () => {
  const cfg = Object.assign({}, BASE, { modelId: 'llama-3.1-70b', weightPrecision: 'fp8', tp: 2 });
  const one = E.computeRooflineMetrics(Object.assign({}, cfg, { pp: 1 }));
  const two = E.computeRooflineMetrics(Object.assign({}, cfg, { pp: 2 }));
  const four = E.computeRooflineMetrics(Object.assign({}, cfg, { pp: 4 }));
  const hop = (m) => m.decode.commMs - 80 * 2 * E.allReduceMs(E.makeContext(m.cfg), m.cfg.concurrentSeqs * 8192 * 2);
  near(hop(one), 0, 1e-9);
  assert.ok(hop(two) > 0);
  near(hop(four) / hop(two), 3, 1e-9);
});

test('summary counts SLO violations and attributes requests to pools', () => {
  const reqs = [req('a', { sloTtftMs: 0.001 }), req('b', { sloTtftMs: 0.001 }), req('c')];
  const r = E.routeWorkloadBatch(reqs, [POOL_SMALL]);
  assert.equal(r.summary.sloViolated, 2);
  assert.equal(r.summary.metSlo, 1);
  const small = r.summary.perPool.find((p) => p.id === 'small');
  assert.equal(small.requests, 3);
  assert.equal(small.tokens, 1800);
  assert.ok(small.cost > 0);
  assert.ok(small.peakKvUtilization > 0);
});

test('requests are attributed to the right pool when two pools are used', () => {
  const r = E.routeWorkloadBatch([req('a'), req('b', { complexity: 0.6 })], [POOL_SMALL, POOL_LARGE]);
  const by = (id) => r.summary.perPool.find((p) => p.id === id).requests;
  assert.equal(by('small'), 1);
  assert.equal(by('large'), 1);
});

test('placed requests are not marked as KV oversubscribed and carry their Pareto front', () => {
  const r = E.routeWorkloadBatch([req('a')], [POOL_SMALL, POOL_LARGE]);
  assert.equal(r.decisions[0].kvOversubscribed, false);
  assert.ok(r.decisions[0].front.indexOf('small') !== -1);
  assert.equal(r.decisions[0].candidates.length, 2);
});

test('workload ids count up from one', () => {
  const w = E.generateWorkload('saas-burst', { seed: 3, count: 3 });
  assert.deepEqual(w.map((x) => x.id), ['saas-burst-1', 'saas-burst-2', 'saas-burst-3']);
});

test('only the Meta and Google repos are marked gated in the exporter', () => {
  const gated = E.MODELS.filter((m) => m.gated).map((m) => m.id).sort();
  assert.deepEqual(gated, ['gemma-2-27b', 'llama-3.1-70b', 'llama-3.1-8b']);
  const qwen = E.exportClusterManifest([{ id: 'q', label: 'Q', modelId: 'qwen-2.5-72b', gpuId: 'h100', gpus: 4, tp: 4, pp: 1, weightPrecision: 'fp8' }]);
  assert.ok(!qwen.warnings.some((w) => /gated/.test(w)));
  const nemo = E.exportClusterManifest([{ id: 'n', label: 'N', modelId: 'nemotron-4-340b', gpuId: 'h200', gpus: 8, tp: 8, pp: 1, weightPrecision: 'fp8', maxModelLen: 4096 }]);
  assert.ok(!nemo.warnings.some((w) => /gated/.test(w)));
  const gemma = E.vllmArgs({ id: 'g', label: 'G', modelId: 'gemma-2-27b', gpuId: 'h200', gpus: 2, tp: 2, pp: 1, maxModelLen: 8192 });
  assert.ok(gemma.warnings.some((w) => /sliding window/.test(w)));
});

test('the probe asks the server to include usage and decodes byte chunks', async () => {
  const log = [];
  const enc = new TextEncoder();
  const clock = { t: 0 };
  const frames = [
    sse({ choices: [{ delta: { content: 'héllo' } }] }),
    sse({ choices: [], usage: { prompt_tokens: 5, completion_tokens: 2 } })
  ];
  let i = 0;
  const impl = async (url, init) => {
    log.push(JSON.parse(init.body));
    return { ok: true, status: 200, body: { getReader: () => ({ read: async () => {
      if (i >= frames.length) return { done: true };
      clock.t += 25;
      const bytes = enc.encode(frames[i++]);
      return { done: false, value: bytes };
    } }) } };
  };
  const r = await E.probeEndpoint({ baseUrl: 'http://x/v1', model: 'm', fetchImpl: impl, now: () => clock.t });
  assert.equal(log[0].stream_options.include_usage, true);
  assert.equal(r.ok, true);
  assert.equal(r.chunks, 1);
  assert.equal(r.completionTokens, 2);
});

test('compareProbeToRoofline reports a failed probe as not ok', () => {
  const m = E.computeRooflineMetrics(Object.assign({}, BASE, { concurrentSeqs: 1 }));
  assert.deepEqual(E.compareProbeToRoofline({ ok: false }, m), { ok: false });
  assert.equal(E.compareProbeToRoofline({ ok: true, ttftMs: 10, itlMs: 5, promptTokens: 100 }, m).ok, true);
});

test('the replay transport answers a model listing', async () => {
  const t = E.createReplayTransport(BASE, 1);
  const ids = await E.listModels({ baseUrl: 'replay://local', fetchImpl: t.fetchImpl });
  assert.deepEqual(ids, ['meta-llama/Llama-3.1-8B-Instruct']);
});

test('a stream that never ends or never sends data is cut off instead of hanging', async () => {
  const stalled = async () => ({ ok: true, status: 200, body: { getReader: () => ({ read: async () => ({ done: false }) }) } });
  const r = await E.probeEndpoint({ baseUrl: 'http://x/v1', model: 'm', fetchImpl: stalled, now: () => 0 });
  assert.equal(r.ok, false);
  assert.match(r.error, /stalled/);
  let t = 0;
  const slow = async () => ({ ok: true, status: 200, body: { getReader: () => ({ read: async () => { t += 60000; return { done: false, value: ': keepalive\n' }; } }) } });
  const r2 = await E.probeEndpoint({ baseUrl: 'http://x/v1', model: 'm', fetchImpl: slow, now: () => t, timeoutMs: 100000 });
  assert.equal(r2.ok, false);
  assert.match(r2.error, /did not finish within 100 seconds/);
});
