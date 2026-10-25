# InferGrid

Know where your VRAM goes. InferGrid is an open source, self hostable capacity planner, KV cache
calculator and multi GPU routing workbench for open weight LLMs on NVIDIA clusters such as
H100, H200, L40S and A100 on Nebius. It is one static page with zero dependencies, it works
offline with no key, and every number it shows comes with its arithmetic.

Live at https://aariv.me/infergrid/ . Entry for the Nebius x NVIDIA Global AI Hackathon.
Built by Aariv Patel.

## What it does

1. **Memory physics.** Exact KV cache bytes per token for eight open weight models
   (`2 x layers x KV heads x head dim x bytes`), weights, activation buffers and runtime overhead
   per GPU across tensor parallel 1, 2, 4, 8 and pipeline parallel 1, 2, 4, in FP16/BF16,
   FP8-E4M3, INT8 and INT4-AWQ. The result is a verdict (fits, tight, out of memory, invalid
   layout) and a live stacked diagram of weights, KV cache, activations and free HBM per GPU.
2. **Roofline latency.** Prefill time to first token and decode inter token latency from
   datasheet bandwidth and dense Tensor Core throughput, plus NVLink or PCIe all-reduce cost per
   layer. A log log roofline chart shows where prefill and decode land, and a second chart
   compares TTFT against burst size on H100, H200, L40S and A100.
3. **Pareto router.** `routeWorkloadBatch` places each request on the cheapest replica that meets
   its TTFT and inter token targets, has enough free KV blocks and is capable enough, then the
   same requests are replayed through three simpler policies so the difference is visible.
4. **Manifest exporter.** `exportClusterManifest` writes a `docker-compose.yml`, a `serve.sh`
   with one pinned `vllm serve` per replica, a TensorRT-LLM script and a router policy JSON, with
   `--tensor-parallel-size`, `--gpu-memory-utilization`, `--max-model-len`, `--max-num-seqs`,
   `--quantization` and `--kv-cache-dtype` derived from the engine.
5. **Endpoint probe.** Optional. Stream one completion from Nebius AI Studio or a local vLLM
   server and compare measured TTFT and inter token latency with the roofline. An offline replay
   mode simulates the stream from the roofline with seeded jitter and says so on screen.

## Models and GPUs

Llama-3.1-8B-Instruct, Llama-3.1-70B-Instruct, Llama-3.1-405B-FP8, Qwen-2.5-72B-Instruct,
DeepSeek-R1-Distill-Qwen-32B, Mistral-Small-24B-Instruct, Gemma-2-27B-IT and
Nemotron-4-340B-Instruct, on H100 SXM5 80GB, H200 SXM 141GB, L40S 48GB and A100 SXM4 80GB.
Layer counts, hidden sizes, heads and FFN widths come from each model's `config.json` (an
ungated mirror where the original is gated) and, for Nemotron-4, Table 1 of the technical report.
The test suite derives each parameter count from the architecture and checks it against the
published total within one percent.

## Quick start

```
npm test                      # 147 tests, no install needed
python3 -m http.server 8000   # then open http://localhost:8000/
```

Any static host works. GitHub Pages deploys the repository root as is (`.nojekyll` included).
Use the engine from Node with `const E = require('./js/engine.js')`.

## The probe and your key

The probe calls the endpoint straight from your browser. Your key is held in the page for that
tab only, sent only in the `Authorization` header to the base URL you typed, and never stored,
logged or put in a link. The default base URL is `https://api.studio.nebius.ai/v1`. Nebius also
documents `https://api.tokenfactory.nebius.com/v1`. On 2026-10-25 both answered a keyless
cross origin preflight with an open allow-origin header, which is what a browser needs. Some
browsers block plain http endpoints from an https page, so a local vLLM server may need the page
served over http.

## Honest limits

- Everything is an analytical model built from published datasheet figures and model configs.
  Nothing here was benchmarked on a cluster and no benchmark result is claimed.
- Efficiency factors (80 percent of HBM bandwidth, 50 percent of Tensor Core peak, 75 percent of
  link bandwidth), the activation estimate, the runtime overhead and the per step launch overhead
  are labelled assumptions. They are exposed in the page and fit with the probe.
- Not modelled. Mixture of experts, speculative decoding, prefix caching, disaggregated prefill,
  Gemma-2 sliding window savings, allocator fragmentation and real queueing in the workbench.
- The router capability score is a size proxy (log of parameter count, 400B equals 1.0), not a
  quality benchmark. Replace it with your own eval results.
- The p95 values in the router are the 95th percentile across a deterministic batch of
  estimates, not a sample of live traffic.
- Default dollar rates are Nebius list prices read on 2026-10-25 (`$4.50` per H100 hour, `$5.40`
  per H200 hour, L40S from `$1.55`). Nebius does not list A100, so that rate is left blank for
  you. Prices change, so check before budgeting.
- Generated launch files are starting points. Pin the image tag, check flags against your vLLM
  or TensorRT-LLM version and load test before production.

## Layout

```
index.html        the page
styles.css        tokens and styles
js/engine.js      deterministic engine, router, exporter, probe (browser and Node)
js/render.js      pure HTML and SVG renderers
js/app.js         page controller
tests/            node --test suites
video/            Devpost copy and voiceover script
```

## License

MIT. See `LICENSE`.
