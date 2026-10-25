# InferGrid | Nebius x NVIDIA Global AI Hackathon

## Name

InferGrid, an open weight LLM capacity planner and GPU routing workbench

## Tagline

Know where your VRAM goes before you launch.

## Description

InferGrid answers the question that decides whether a model deployment works. Where does every
gigabyte of GPU memory go, and how many users fit before the KV cache runs out?

Pick one of eight open weight models, a layout of H100, H200, L40S or A100 GPUs, a precision and
a load. The diagram redraws weights, KV cache, activation buffers and free HBM on every GPU as
you move a slider, and a plain verdict says whether the layout fits, is tight, runs out of
memory or is not a valid tensor parallel split. Underneath is an engine you can read. KV bytes
per token are `2 x layers x KV heads x head dim x bytes`, shown line by line, and each model's
parameter count is rebuilt from its architecture and checked against the published total.

Latency comes from the roofline. Prefill time to first token and decode inter token latency use
published bandwidth and dense Tensor Core throughput, plus NVLink or PCIe all-reduce cost. One
chart plots where prefill and decode land against each GPU's ceiling, and another compares TTFT
against burst size on all four GPUs, with a hover crosshair and a keyboard slider.

Four workload presets drive a Pareto router. It places each request on the cheapest replica that
meets the request's TTFT and inter token targets, has enough free KV blocks and is capable enough
for the task, then replays the same requests through the largest model only, a cheapest first
policy with no KV check and round robin, so the cost, latency and KV preemption differences are
visible. A deployment drawer writes a `docker-compose.yml`, a `serve.sh` with one pinned
`vllm serve` per replica, a TensorRT-LLM script and a router policy JSON from the same numbers.

An optional probe streams one completion from Nebius AI Studio or a local vLLM server and
compares measured TTFT and inter token latency with the roofline. With no key it runs an offline
replay that is labelled as simulated on screen.

Everything is an analytical model. It is not a benchmark and the page says so. Assumptions such
as bandwidth efficiency are exposed as controls and listed in the method section.

## How it was built

Vanilla JavaScript with no dependencies and no build step. One engine file runs in the browser and
in Node, a separate pure renderer produces the HTML and SVG, and a thin controller wires the
controls. The test suite has 184 tests across the engine and the page, including hand derived
golden values for KV bytes, activation estimates, decode and prefill latency, router decisions,
exported flags and probe parsing. Design follows an alabaster paper and carbon ink palette with a
single teal signal color, Space Grotesk, Plus Jakarta Sans and JetBrains Mono.

## Challenges

Getting the facts right mattered more than getting the chart right. Architectures were read from
published configs, GPU throughput was taken as dense figures rather than the doubled sparsity
numbers, and anything unverified is a labelled assumption. Making the router honest was the
second challenge. Its capability score is only a size proxy, and the page says that.

## What is next

Mixture of experts models, prefix caching, speculative decoding, and fitting the efficiency
factors automatically from probe runs.

## Built with

Vanilla JavaScript, SVG, CSS custom properties, node:test, vLLM and TensorRT-LLM flag generation,
Nebius AI Studio OpenAI compatible API for the optional probe.

## Try it

https://aariv.me/infergrid/

## Code

https://github.com/draarivpatel-ui/infergrid

## Voiceover script

Fifty five words, no hyphens, written for the local Chatterbox voice clone. "vLLM" is spelled the
way it should be said.

```
Your model does not fit, and you find out at launch. InferGrid shows where every gigabyte goes. Slide the context length and weights, cache and free memory move on every GPU. It computes latency, routes requests to the cheapest pool that meets your targets, and exports vee L L M files. Open source, no key.
```
