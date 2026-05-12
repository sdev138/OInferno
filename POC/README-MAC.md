# Running the NLA Transcript Generator on Apple Silicon

This POC was originally written for CUDA + SGLang. It now also runs on Apple
Silicon Macs via an in-process actor (no SGLang server, no NVIDIA dependencies).
Same JSON output, same viewer.

## What's different from the CUDA path

| Thing | CUDA host | Apple Silicon host |
|---|---|---|
| Actor verbalizer | SGLang server in a second terminal | In-process via `transformers.generate(inputs_embeds=...)` |
| Throughput | Fast (batching, kernel fusion) | ~5–20× slower per decode |
| Install | `pip install -r requirements.txt` (+ matching CUDA torch wheel) | `pip install -r requirements-mac.txt` |
| `--device` | `cuda` | `mps` (auto-detected) |
| `--sglang-url` | Required | Omit it — script auto-picks the local backend |
| Peak memory | ~24–32 GB VRAM | ~14–15 GB unified per stage, loaded sequentially |

The `--actor-backend auto` default means: if you pass `--sglang-url`, the
script uses SGLang. If you don't, it loads the actor in this process. You
do not have to think about it on Mac — just don't pass the URL.

## Hardware target

- **M1 Max / M2 Max / M3 Max / M4 Max / M5 Max** with **≥ 32 GB unified memory.**
  Confirmed-targeted: M5 Max, 36 GB.
- 36 GB is enough to run the full pipeline (base → actor → critic) **only because
  the script now sequences the loads** — none of those 7B models is resident at
  the same time as another. If you see OOMs anyway, add `--skip-critic` (drops
  the critic stage entirely; cos/mse fields are omitted from the output).
- Plan for ~30 GB free disk for the three HuggingFace repos (cached under
  `~/.cache/huggingface`).
- Plan for ~10–30 minutes wall time for a fresh prompt with a ~50-token assistant
  response. The actor decode loop is the slow part.

## Install

```bash
# 1. Fresh virtualenv (recommended)
python3 -m venv .venv
source .venv/bin/activate

# 2. Install the Mac-specific requirements (skips sglang, adds accelerate)
pip install -r requirements-mac.txt
```

That's it. No CUDA toolkit, no SGLang server, no `--index-url` for torch.
PyPI's macOS wheels ship with MPS support.

If the HF Hub asks for authentication on first run (rare — the three repos are
public):

```bash
huggingface-cli login
```

## Run

One terminal, one command:

```bash
python build_transcript.py \
    --prompt "Tell me about the capital of France" \
    --output samples/paris-mac.json
```

Optional but recommended on 36 GB if you hit memory pressure:

```bash
python build_transcript.py \
    --prompt "..." \
    --output samples/out.json \
    --skip-critic
```

The script auto-detects MPS, auto-picks the local actor backend, and runs in
three stages with a free in between each:

```
[setup] device=mps dtype=torch.float16
[setup] actor_backend=local
[setup] loading base model from ...
[gen  ] generating assistant response (max 200 tokens)
[gen  ] done in 12.3s; assistant: 'The capital of France is Paris...'
[hook ] extracting residual-stream activations at layer 20
[hook ] got 47 token activations, boundary at 14
[free ] releasing base model
[av   ] loading local actor model from ...
[av   ] verbalizing 47 activations (backend=local)
[av   ] [  5/ 47] decoded 'The model is thinking about the start of a question...'
...
[free ] releasing local actor
[crit ] loading critic from ...
[crit ] scoring 47 decodes
[crit ] [ 5/ 47] cos=0.71
...
[done ] wrote samples/paris-mac.json (47 tokens)
```

## View the result

```bash
python3 -m http.server 8000
# then open in a browser:
open "http://localhost:8000/viewer.html#file=samples/paris-mac.json"
```

A pre-generated `samples/paris.json` ships in this folder if you want to verify
the viewer end without running the heavy pipeline first.

## Performance notes

- The local actor calls `model.generate()` once per token position. That's `T`
  forward passes through the 7B actor, each generating up to
  `--decode-max-tokens` (default 200) new tokens. A 50-position transcript with
  the default budget is ~50 × ~5 seconds of decode = ~5 minutes on MPS, give or
  take depending on settings and how much your generations hit the EOS early.
- The base-model `.generate()` (one pass, up to 200 tokens) is comparatively
  cheap and is the *fast* part.
- The critic stage is `T` short forward passes — fastest of the three stages.
- Cap things if you're impatient:
  ```
  --max-new-tokens 80     # shorter assistant response → fewer positions
  --decode-max-tokens 80  # smaller per-decode budget → faster per position
  ```

## Memory tuning

If your friend hits "MPS backend out of memory" or sees the OS swap heavily:

1. **`--skip-critic`** — drops the third 7B model entirely. Single biggest win.
2. **Close other GPU-heavy apps** before launching (browsers with hardware
   acceleration, Xcode simulators, video calls). MPS competes with everything
   else for unified memory.
3. **Lower `--decode-max-tokens`** — each generation step allocates KV cache;
   shorter generations = smaller peak.
4. **`PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0`** — disables the MPS allocator's
   memory cap. Use as a last resort; can cause the OS to kill the process if
   it gets too greedy.
   ```bash
   PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 python build_transcript.py ...
   ```

## Troubleshooting

**`No module named accelerate`**
You installed with the old `requirements.txt` instead of `requirements-mac.txt`.
`pip install accelerate>=0.30`.

**`device_map="mps"` raises NotImplementedError**
Your accelerate version is too old. `pip install -U 'accelerate>=0.30'`.

**Process killed with no traceback, RAM pressure visible in Activity Monitor**
The kernel OOM-killed it. Add `--skip-critic`. If that's still not enough,
drop `--max-new-tokens` and `--decode-max-tokens`.

**Decodes are all gibberish / Chinese characters on every position**
Same root causes as the CUDA path — most likely the actor sidecar `nla_meta.yaml`
and the live tokenizer drifted. Re-download:
```bash
rm -rf ~/.cache/huggingface/hub/models--kitft--nla-qwen2.5-7b-L20-av
```
and re-run.

**`RuntimeError: MPS backend doesn't support bfloat16`**
The script defaults to fp16 on MPS to avoid exactly this, but if you forced bf16
somehow, drop the override and let the auto-detection pick fp16.

**Very slow first decode, then fine**
First call compiles MPS kernels and pages weights in. Subsequent calls are the
steady-state speed. This is normal.

**The pre-generated viewer demo works, but a fresh run produces no output**
Check the script's stderr — there's likely a Python exception buried in the
verbalize loop. The catch-all wrapper writes `<verbalize error: ...>` strings
into the JSON for any individual failures so partial output is still inspectable.
