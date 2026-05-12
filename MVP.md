# OInferno — MVP

`htop` for local LLMs. One command, one beautiful dashboard, opinionated comparisons.

---

## 1. Problem

Local-LLM users (Ollama, llama.cpp, LM Studio, MLX) have no good way to:

- Compare two quantizations of the same model on their hardware
- See real-time TTFT / throughput / hardware utilization while running a model
- Decide whether GPU offload is helping
- Share a clean "here's how this model runs on my M3 Max" report

Existing tools fall short:

- **`llama-bench`** — CLI, raw numbers, ugly
- **Ollama** — no benchmark UI
- **LM Studio** — basic perf info, no comparison
- **Open WebUI** — some metrics, not benchmark-shaped

## 2. Positioning

**"`htop` for local LLMs."**

Single binary. One command. Opens a browser tab. Tells you, in 60 seconds, how your model performs on your hardware — and how it compares to the alternatives.

## 3. Target user

Single persona for MVP: **the local-LLM tinkerer.** Has Ollama or llama.cpp installed. Has 3-15 models on disk. Cares which one to use. Posts Reddit threads with `llama-bench` screenshots.

## 4. User stories

- "Show me how fast `llama3.1:8b-instruct-q4_K_M` runs on my MacBook *right now*."
- "Compare Q4_K_M vs Q5_K_M vs Q8 of the same model. Show me the speed/quality tradeoff visually."
- "Sweep context length 4K → 8K → 16K → 32K. Show me throughput drop and memory growth."
- "Am I CPU-bound or GPU-bound on this model?"
- "Export a clean HTML report I can post in r/LocalLLaMA."

## 5. The 90-second loop (install → insight)

```
$ npx oinferno
→ OInferno v0.1 starting…
→ Discovered: 7 Ollama models, 3 GGUFs in ~/models
→ Hardware: M3 Max (40-core GPU, 64GB unified)
→ Dashboard: http://localhost:7878
```

Browser opens. User sees their models. Clicks one. Clicks "Quick Benchmark." 60 seconds later: a report.

## 6. Surfaces

### 6.1 Home / discovery

```
┌──────────────────────────────────────────────────────────┐
│ OInferno                          Apple M3 Max · 64 GB   │
├──────────────────────────────────────────────────────────┤
│  Models on this machine                                  │
│                                                          │
│  Ollama (7)                                              │
│  ┌─ llama3.1:8b-instruct-q4_K_M    4.7 GB    [Bench →]  │
│  ┌─ llama3.1:8b-instruct-q5_K_M    5.7 GB    [Bench →]  │
│  ┌─ llama3.1:8b-instruct-q8_0      8.5 GB    [Bench →]  │
│  ┌─ qwen2.5:7b-instruct-q4_K_M     4.4 GB    [Bench →]  │
│                                                          │
│  GGUF files (3)        + add path                        │
│  ┌─ ~/models/Mistral-Nemo-12B-Q4.gguf       [Bench →]   │
│                                                          │
│  Recent runs                                             │
│  • llama3.1:8b q4  ·  3 min ago  ·  47 tok/s    [view]  │
│  • Compare Q4/Q5/Q8 ·  yesterday  ·  3 runs    [view]   │
└──────────────────────────────────────────────────────────┘
```

### 6.2 Live run

```
┌──────────────────────────────────────────────────────────┐
│ Running: llama3.1:8b-instruct-q4_K_M                     │
│ Workload: chat-typical · req 23/50 · 00:42               │
├──────────────────────────────────────────────────────────┤
│  Throughput (tokens/sec)                                 │
│   80 ┤              ╭─╮                                  │
│   60 ┤       ╭──────╯ ╰────╮ ╭─                          │
│   40 ┤  ╭────╯              ╰─╯                          │
│   20 ┤──╯                                                │
│      └──────────────────────────────                     │
│                                                          │
│  TTFT distribution             TPOT distribution         │
│   ▁▂▃▅█▇▆▃▂▁                   ▁▃▆█▆▃▁                  │
│   72ms p50 · 134ms p95         21ms p50 · 28ms p95       │
│                                                          │
│  Hardware (last 30s)                                     │
│   GPU ▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇  92%                              │
│   CPU ▃▃▃▃▃▃▃▃▃▃         18%                             │
│   RAM ▇▇▇▇▇▇▇▇▇▇         4.7 / 64 GB                     │
│   PWR ▇▇▇▇▇▇▇▇▇▇▇        38 W                            │
└──────────────────────────────────────────────────────────┘
```

### 6.3 Report

```
┌──────────────────────────────────────────────────────────┐
│  llama3.1:8b-instruct-q4_K_M  ·  M3 Max  ·  May 11      │
├──────────────────────────────────────────────────────────┤
│   ⚡ DECODE         🚀 PREFILL          💾 MEMORY        │
│   47 tok/s          380 tok/s          4.7 GB peak       │
│                                                          │
│   🩺 GPU-bound, well-utilized (92% avg). No obvious      │
│      config wins. To go faster: smaller quant or model.  │
└──────────────────────────────────────────────────────────┘
```

### 6.4 Compare

Side-by-side cards (2-4 runs), overlaid TPOT distributions, diff table, verdict.

### 6.5 Sweep

Pick one variable (ctx length, parallelism, GPU layers, quant), specify range. Watt runs each, renders the **Pareto curve**, highlights the dominant point.

---

## 7. Benchmarks — exactly what runs

### Workload A — **Chat** (the everyday case)

- 50 fixed real-style prompts
- Input ~50-100 tokens
- Output cap 150 tokens
- Sequential
- **Reports:** TTFT distribution, TPOT distribution, end-to-end latency

### Workload B — **Long Context** (prefill-heavy)

- 20 prompts, each prefixed with 8K tokens of public-domain text
- Input ~8,000 tokens
- Output cap 200 tokens
- Sequential
- **Reports:** TTFT (= prefill time), prefill tok/s, decode tok/s, peak memory

### Workload C — **Throughput Burst** (batching)

- 100 short prompts
- Input ~100 tokens
- Output cap 100 tokens
- Concurrency: configurable (default 4)
- **Reports:** aggregate tok/s, per-request latency, throughput-vs-concurrency

### Hardware sampling (concurrent with all workloads)

Polled every **500ms** during workload:

- GPU utilization %
- GPU memory in use
- CPU utilization %
- RAM used
- Power draw (W) — Apple Silicon via `powermetrics`, NVIDIA via `nvidia-smi`
- Temperature (where available)

### Quality vibe-check (free, runs once per benchmark)

5 fixed prompts, full outputs captured and surfaced in the report. Lets the user eyeball: *is this output coherent at this quant?* **Not** a quality benchmark — a sanity check.

---

## 8. Metrics — defined precisely

| Metric | Definition | Why it matters |
|---|---|---|
| **TTFT** (Time to First Token) | Wall-clock from request send to first token received | What the user *feels* as "responsiveness." Dominated by prefill. |
| **TPOT** (Time per Output Token) | (end_time − first_token_time) / (output_tokens − 1) | Decode speed. What "tokens/sec" actually means. |
| **Prefill tok/s** | input_tokens / TTFT | How fast the model digests context. Critical for RAG / long-context. |
| **Decode tok/s** | 1 / TPOT | Generation speed. The "feels fast" number. |
| **End-to-end latency** | Total wall-clock for one request | What APIs typically report. |
| **Aggregate throughput** | Sum of output tokens / total wall-clock for the run | Real throughput when many requests overlap. |
| **Peak GPU memory** | Max sampled during workload | The "will this OOM" question. |
| **Avg GPU util %** | Mean over workload | Are you using the silicon you have? |
| **Power draw** | Mean watts during workload | Tokens-per-watt → battery life. |

All distributions reported as **p50 / p95 / p99**. Means are misleading; we don't show them.

---

## 9. Diagnosis — opinionated, evidence-backed

After every run, **one** rule fires from a small engine:

| Condition | Diagnosis |
|---|---|
| GPU util < 60% AND CPU util > 90% | "CPU-bound. Try `--n-gpu-layers all` or check Metal/CUDA is active." |
| GPU util > 90% AND throughput low for hw class | "GPU-compute-bound. A smaller quantization should help." |
| GPU memory > 95% | "Memory-pressure. Reduce ctx length or use a smaller quant." |
| Prefill tok/s < 20% of decode tok/s × seq_len | "Prefill dominates at this context size — consider prompt caching." |
| All metrics healthy | "Well-utilized. No obvious config wins." |

Honest, useful, actionable.

---

## 10. Report export

A single self-contained HTML file:

- Charts inlined as SVG
- Raw data embedded as JSON
- Hardware fingerprint at the top
- Reproduction command at the bottom (`oinferno run --model X --workload all --seed 42`)
- Watermark: OInferno version + URL

Drop on a Gist, GitHub repo, blog post. Renders standalone. No server needed.

---

## 11. Scope — in / out

### In scope (MVP)

- Backends: **Ollama**, **llama.cpp** (`llama-server`), **MLX** (Apple Silicon, optional)
- Auto-discovery of installed models
- 3 workloads (chat, long-context, throughput burst)
- Live dashboard (TTFT, TPOT, throughput, hardware)
- 2-4 run comparison
- 1-dimension sweep
- 1-rule bottleneck diagnosis
- HTML report export
- Single binary distribution
- macOS (Apple Silicon) + Linux (NVIDIA)

### Explicitly out (later versions)

- vLLM / TGI / SGLang / TensorRT-LLM (v0.5)
- Hosted APIs — Anthropic, OpenAI (v1.0)
- Trace replay from prod JSONL (v1.0)
- Cost analysis — no $ for local (v1.0)
- Quality / accuracy evals
- Multi-machine / distributed
- Multi-user / accounts / cloud-saved reports
- Multi-dimensional auto-sweep
- Prefix cache audit (Ollama doesn't expose; v0.5)
- CI mode

---

## 12. Success criteria

- ⏱ **Time to first chart:** < 90 seconds from `npx oinferno`
- ⚡ **One command default:** `oinferno run --model llama3.1:8b` runs all 3 workloads, opens dashboard
- 📊 **Sweep:** 4-quant comparison runs end-to-end and visualizes Pareto in < 15 minutes
- 🌐 **Cross-platform:** macOS Apple Silicon and Linux + NVIDIA out of the box
- 📄 **Shareable artifact:** HTML report exports cleanly and renders on GitHub / Gist

---

## 13. Tech stack

| Layer | Choice | Why |
|---|---|---|
| **Frontend** | Next.js + Tailwind + Recharts | Fastest to ship beautiful charts |
| **Backend** | Rust (`axum`) — load generator + metrics + UI server | Precise timing for TTFT / TPOT; single static binary |
| **Hardware metrics** | `powermetrics` (macOS, sudo helper), `nvidia-smi` (Linux), `sysinfo` crate for CPU / RAM | No new daemon to install |
| **Storage** | SQLite at `~/.oinferno/runs.db` | Zero-config persistence |
| **Distribution** | `cargo install oinferno` + `npx oinferno` wrapper + Homebrew tap | Hits multiple ecosystems |

---

## 14. Risks & mitigations

| Risk | Mitigation |
|---|---|
| `powermetrics` requires `sudo` | First-run prompt explains; degrade to coarser metrics if denied |
| Ollama API doesn't expose internal metrics | Compute everything client-side from streaming response timing |
| Charts feel sluggish under high RPS | Throttle UI updates to 10 Hz; aggregate samples server-side |
| Quant comparison without quality measurement is misleading | Vibe-check (5 prompts side-by-side) lets human judge |

---

## 15. Build order (hackathon weekend)

| When | What |
|---|---|
| Day 1 morning | Rust load generator → hits Ollama → measures TTFT / TPOT / throughput → dumps JSON |
| Day 1 afternoon | Next.js dashboard reads JSON, renders one chart well |
| Day 1 evening | Live mode (SSE generator → dashboard); add hardware metrics sampler |
| Day 2 morning | Comparison view + sweep runner |
| Day 2 afternoon | Bottleneck diagnosis rule + HTML report export |
| Day 2 evening | Polish: auto-discovery, single-command UX, README, demo video |

---

## 16. The killer demo (30 seconds)

1. `npx oinferno` (1s)
2. Browser opens, model list visible (3s)
3. Click "Compare Quants" → check 3 quants of llama3.1:8b → "Run" (5s)
4. Live charts streaming for ~45s (real-time)
5. Comparison report appears (3s)
6. Pan to verdict: **"Q4 is 1.7× faster than Q8 with 45% less memory and indistinguishable output quality on the vibe check."** (5s)

That's the whole pitch. **Pick the right quant in 60 seconds, on your own machine, with your own eyes.**
