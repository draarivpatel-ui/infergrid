# InferGrid Build Notes

Date 2026-10-25. Author Aariv Patel. Built for the Nebius x NVIDIA Global AI Hackathon
(deadline Oct 30).

## What Was Built

A static single page app, vanilla HTML, CSS and JavaScript, zero dependencies, GitHub Pages ready
(`.nojekyll`). Google Fonts is the only network request, as the design brief required.

| File | Role |
| --- | --- |
| `index.html` | page, Open Graph tags, inline SVG favicon, noscript fallback, labelled controls |
| `styles.css` | custom property tokens (colors, type, space, motion) and all styles |
| `js/engine.js` | UMD engine for browser and Node. Footprint, roofline, router, exporter, probe |
| `js/render.js` | pure HTML and SVG renderers, no DOM access |
| `js/app.js` | page controller, drives the renderers |
| `tests/engine.test.js`, `tests/dom-smoke.test.js` | `node --test` suites |
| `img/og.png` | Open Graph card, rendered from the same renderer with headless Chrome |
| `video/SUBMISSION.md` | Devpost copy and the 55 word voiceover script |

The engine implements the three required functions and a fourth.

1. `computeModelFootprint` and `computeRooflineMetrics` for 8 models and 4 GPUs.
2. `routeWorkloadBatch`, a Pareto router with four policies for comparison.
3. `exportClusterManifest`, producing `docker-compose.yml`, `serve.sh`, a TensorRT-LLM script
   and a router policy JSON.
4. `probeEndpoint`, `listModels` and `createReplayTransport` for the optional live probe and the
   offline replay.

## Design Decisions In Place Of The Hallmark Questions

The build ran as an autonomous sub-agent, so the three Hallmark questions were answered by
inference from the brief.

- Audience, engineers and students who plan open weight LLM deployments on NVIDIA GPUs.
- Use case, one job, answer whether a model and layout fit and how fast it will be, then export
  the launch files.
- Tone, technical and austere.
- Genre modern-minimal, macrostructure Workbench (the working tool is the page, no marketing
  screenshots), nav N1b, footer Ft2. Theme is custom because the brief fixed the palette
  (`#F6F4EE` paper, `#EDE9DF` cards, `#141615` ink, `#0F766E` and `#10B981` signal, `#D97706`
  amber, `#DC2626` danger). Display type Space Grotesk, body Plus Jakarta Sans, numbers in
  JetBrains Mono with tabular figures.
- Motion stack. CSS only. One transform animation on the hero bar, guarded by
  `prefers-reduced-motion`. No GSAP, no smooth scroll engine and no Three.js, because the
  charts are data graphics and nothing needed them.
- Hallmark rules honoured. No italic headings, no uppercase kickers, no emoji, no fake browser
  chrome, locked tokens (checked by a test), 44 pixel touch targets, one nav and one footer
  pattern that is not N1a or Ft3.
- Self score against the six Hallmark axes was Philosophy 4, Hierarchy 4, Execution 4,
  Specificity 4, Restraint 4, Variety 3. These are my own scores, not an outside review.

## Sources Read On 2026-10-25

Model shapes (layers, hidden size, heads, KV heads, FFN, vocab, tied embeddings) were read from
these files in the session, not recalled.

- `huggingface.co/unsloth/Meta-Llama-3.1-8B-Instruct/config.json` and the 70B twin (ungated mirrors)
- `huggingface.co/neuralmagic/Meta-Llama-3.1-405B-Instruct-FP8/config.json` (redirects to RedHatAI)
- `huggingface.co/Qwen/Qwen2.5-72B-Instruct/config.json`
- `huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-32B/config.json`
- `huggingface.co/mistralai/Mistral-Small-24B-Instruct-2501/config.json`
- `huggingface.co/unsloth/gemma-2-27b-it/config.json` (ungated mirror)
- `arxiv.org/html/2406.11704v1` Table 1 for Nemotron-4-340B (96 layers, 18432 hidden, 96 heads,
  8 KV heads, 256000 vocab, squared ReLU, 9.4B embedding and 331.6B non embedding parameters).
  The Hugging Face model card does not state the architecture.

GPU figures were read from the NVIDIA datacenter pages for H100, H200, L40S and A100. The H100
and H200 pages print only with sparsity figures (1,979 TFLOPS FP16 and 3,958 TFLOPS FP8), so the
engine halves them for dense throughput. L40S prints pairs with the first value dense (362.05
FP16 and 733 FP8) and the second with sparsity. A100 prints 312 TFLOPS FP16 dense and no FP8.
NVLink is 900 GB/s on Hopper and 600 GB/s on A100 (bidirectional totals, halved to per
direction in the engine), and L40S has no NVLink with PCIe Gen4 x16 at 64 GB/s bidirectional.

Prices came from `nebius.com/prices` (H100 `$4.50`, H200 `$5.40`, L40S from `$1.55` per GPU
hour). A100 is not listed, so its rate is blank in the UI.

Nebius API. `docs.nebius.com/studio/inference/api` redirects to `docs.tokenfactory.nebius.com`,
which documents `https://api.tokenfactory.nebius.com/v1/` and bearer key auth. The brief's
`https://api.studio.nebius.ai/v1/models` still answers (HTTP 401 without a key). A keyless
preflight with `Origin: https://aariv.me` on `api.studio.nebius.ai` and the two other Nebius
hosts returned 204 with `access-control-allow-origin: *`, so a browser can call them. This was
not tested with a real key.

TensorRT-LLM flags. The `trtllm-serve` documentation lists `--tp_size`, `--pp_size`,
`--max_batch_size`, `--max_num_tokens` and `--kv_cache_free_gpu_memory_fraction`. The script uses
the first four as flags and sets the KV fraction and dtype through the `kv_cache_config` block of
an extra options file. That block layout was not verified against a running install.

Hugging Face repo ids used in exports (AWQ and W8A8 checkpoints) were each confirmed to exist with
an API request. Gemma-2 INT4 and Nemotron have no checked alternative, and the exporter says so.

## Assumptions That Are Not Measurements

- Efficiency factors (80 percent of memory bandwidth, 50 percent of Tensor Core peak, 75 percent
  of link bandwidth). These are typical ceilings, adjustable in the page.
- Activation estimate, runtime overhead (CUDA context 0.6 GiB, NCCL 0.6 GiB, graphs 0.4 to
  1.2 GiB), per step launch overhead (`0.4 + 0.012 x layers` ms) and the all-reduce latencies
  (12 microseconds NVLink, 30 microseconds PCIe, 25 microseconds InfiniBand).
- Inter node bandwidth of 50 GB/s per GPU is an assumption, not a Nebius figure.
- The router capability score is a size proxy. Task quality was not evaluated.
- FP8 KV cache on A100 is flagged as backend dependent. It was not checked.

None of these were calibrated against a live endpoint. No run on a real cluster was possible in
this session, and the live probe was exercised only against the offline replay and test doubles.

## Test Results From The Real Run

Command `node --test tests/` on Node v26.8.1 (macOS, arm64).

```
# tests 184
# suites 0
# pass 184
# fail 0
# cancelled 0
# skipped 0
# todo 0
# duration_ms 1002.354625
```

The engine suite pins hand derived values. For example Llama-3.1-8B KV bytes per token (131072),
the exact activation estimate (1,000,865,792 bytes), single stream decode on an H100 (6.49 ms
from `15,295,062,016 bytes / (3.35 TB/s x 0.8) + 0.784 ms`) and 2048 token prefill (about 62 ms
from the FLOP count). Derived parameter counts for all eight models match the published totals
within 0.15 percent.

## Mutation Testing

The station harness mutates the JS files and expects the suite to fail. Its default run only
reached `js/render.js` in this repository, and a hang under mutation raises an uncaught
`TimeoutExpired` that stops the whole run. I did not edit the shared script. I ran an equivalent
sweep with the same eight mutation operators and a per mutant timeout that counts a hang as
caught.

```
engine.js   121 mutants, 116 caught, 5 escaped (96 percent), 1 of the caught was a hang
render.js    57 mutants,  40 caught, 17 escaped (70 percent)
app.js       36 mutants,  33 caught,  3 escaped (92 percent)
total       214 mutants, 189 caught, 25 escaped (88 percent)
```

Escapes were reviewed during the session. Many in `render.js` are class names and label strings that
the tests do not pin, and some are equivalent mutants. These scores are a diagnostic, not a
quality guarantee.

## Verification Performed

- `node --test tests/` passes, and `node --check` loads every module.
- `hark_honesty_and_defect_harness.py --audit` reported 0 violations.
- A scan of all repository text for em dashes, en dashes and emoji came back empty, and the same
  rule is enforced by a test. Colon free headings and prose were checked by a test and a scan.
- Real browser checks with headless Chrome at 1360 and 375 and 320 pixel widths. Document scroll
  width equals the viewport at 375 and 320. Page structure, chart rendering, preset clicks,
  slider and select changes, export tabs, pointer movement over both charts and the offline
  replay were driven through the DevTools protocol and read back.
- Not done. Real keyboard focus traversal was not driven in a browser (the tab pattern is tested
  against a fake DOM), `prefers-reduced-motion` was not emulated, no Lighthouse or axe run, and
  the live probe was not run against Nebius with a key.

## Bugs Found And Fixed During The Build

- The router never stored its decisions (a missing `push`). Found on the first smoke run.
- Undefined pool fields overwrote engine defaults and produced NaN pool sizes. Fixed by dropping
  undefined keys in `normalizeConfig`.
- Screen reader only labels inside scrolling tables widened the page to 994 pixels on mobile.
  Fixed with a positioned scroll container, then confirmed at 375 and 320.
- Reading the sliders back after any change rounded preset values (28,000 became 27,904 prompt
  tokens). Only a moved slider now changes the stored value, with a regression test.
- The roofline prefill dot fell off the right edge of the chart. The axis now runs to 20,000
  FLOP per byte and dots are clamped.
- Overlapping chart labels (H100 and H200 share a peak, L40S and A100 end together) are now
  separated, and the identical H100 and H200 TTFT lines are drawn so both stay visible.
- A stream that never ends could hang the probe, found by mutation testing. The probe now stops
  after 1,000 empty reads or a time limit.
- A hex color in a CSS mask and a literal dash in a test regex tripped the repo's own style tests.

## Open Items For The Submitter

- The Devpost pages for this hackathon returned HTTP 403 to automated fetches, so the official
  rules were not read. A secondary listing says projects must run on Nebius Token Factory or
  Nebius AI Cloud and use at least one NVIDIA open source model. That is unverified. InferGrid
  plans deployments and can probe a Nebius endpoint, but its own code runs in the browser, so
  whether it meets that requirement, and whether the rules restrict AI generated code, needs a
  human read of the rules before submitting. A live probe against a Nebius Nemotron model with a
  real key has not been run and would strengthen the entry.
- Nothing has been submitted to Devpost. `video/SUBMISSION.md` is draft copy only.

## Known Limits

See `README.md`. The main ones are that this is an analytical model, mixture of experts and
speculative decoding are absent, the router capability score is a proxy, and generated launch
files must be checked against your vLLM or TensorRT-LLM version before production use.

## Deployment

Public repository `https://github.com/draarivpatel-ui/infergrid` (checked with `git remote -v`),
GitHub Pages on `main` at `/`. On 2026-10-25 at 17:42 UTC, after the first build finished,
`curl -sI https://aariv.me/infergrid/` returned `HTTP/2 200`, and `img/og.png`, `js/engine.js`,
`js/render.js`, `js/app.js` and `styles.css` each returned 200. The live page was not re-driven in
a browser after deploy, only the same files were exercised locally.
