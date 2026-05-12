# OInferno POC — NLA Transcript Generator + Viewer

End-to-end POC for inspecting **NLA (Natural Language Autoencoder)** decodes of a Qwen 2.5-7B transcript. Two pieces:

- **`build_transcript.py`** — runs the base model on a prompt, extracts residual-stream activations at layer 20, verbalizes each via the NLA actor (served by SGLang), scores each via the NLA critic, writes a JSON.
- **`viewer.html`** — single-file local web UI to click through that JSON token-by-token.

The script auto-downloads three HuggingFace repos on first run:
- `Qwen/Qwen2.5-7B-Instruct` (base model)
- `kitft/nla-qwen2.5-7b-L20-av` (verbalizer / actor)
- `kitft/nla-qwen2.5-7b-L20-ar` (reconstructor / critic)

---

## Hardware

- **NVIDIA GPU with ≥ 32 GB VRAM** (e.g., A100 40GB, H100, RTX 6000 Ada, L40S). The base model + critic + SGLang actor add up to ~28–32 GB.
- **≤ 24 GB VRAM** works with `--skip-critic`, which drops the fidelity (cos/mse) scores. Decodes still get generated.
- **CPU only** is supported (`--device cpu`) but is impractically slow — fine for smoke-testing the code paths, not for real runs.
- **~30 GB disk space** for the model weights (downloaded into `$HF_HOME` / `~/.cache/huggingface`).

## Software prerequisites

- Python 3.10 or newer
- NVIDIA driver + CUDA matching your PyTorch build (CUDA 12.1 or 12.4 are the common targets)
- Enough RAM to handle a 7B model on import (~16 GB system RAM is comfortable)

---

## Install

```bash
# 1. (Recommended) fresh virtualenv
python3 -m venv .venv
source .venv/bin/activate

# 2. Install PyTorch matching your CUDA version FIRST
#    (see https://pytorch.org/get-started/locally/ for other versions)
pip install torch --index-url https://download.pytorch.org/whl/cu121   # CUDA 12.1
# pip install torch --index-url https://download.pytorch.org/whl/cu124 # CUDA 12.4
# pip install torch                                                     # CPU only

# 3. Install everything else (transformers, sglang, etc.)
pip install -r requirements.txt
```

If the HF repos ask for authentication (they shouldn't — all three are public, but the Qwen repo can prompt for a Hub login on some accounts):

```bash
huggingface-cli login
```

---

## Run

You need **two terminals**: one for the SGLang actor server (long-running), one for the script.

### Terminal 1 — launch the actor server

```bash
python -m sglang.launch_server \
    --model-path kitft/nla-qwen2.5-7b-L20-av \
    --port 30000 \
    --disable-radix-cache \
    --mem-fraction-static 0.85 \
    --trust-remote-code
```

Leave this running. First launch downloads ~15 GB and warms up over ~1–2 min; subsequent launches start in seconds.

`--disable-radix-cache` is **mandatory** — SGLang's radix cache keys on token IDs, but our requests use `input_embeds`, which would alias unrelated requests in the cache.

### Terminal 2 — generate a transcript

```bash
python build_transcript.py \
    --prompt "Tell me about the capital of France" \
    --sglang-url http://localhost:30000 \
    --output samples/paris-live.json
```

What you'll see:
```
[setup] loading base model from .../Qwen2.5-7B-Instruct
[setup] loading actor sidecar + embed table from .../nla-qwen2.5-7b-L20-av
[gen  ] generating assistant response (max 200 tokens)
[gen  ] done in 8.4s; assistant: 'The capital of France is Paris...'
[hook ] extracting residual-stream activations at layer 20
[hook ] got 47 token activations, boundary at 14
[hook ] median activation norm: 138.2 (high_norm threshold: > 691.0)
[free ] releasing base model from VRAM
[crit ] loading critic from .../nla-qwen2.5-7b-L20-ar
[av   ] verbalizing 47 activations via SGLang at http://localhost:30000
[av   ] [  5/ 47] user      'Tell'                cos=0.71
[av   ] [ 10/ 47] user      'France'              cos=0.93
... etc
[done ] wrote samples/paris-live.json (47 tokens)
```

### Inspect the result

```bash
# Static server (Python builtin; required because viewer.html fetches the JSON)
python3 -m http.server 8000
```

Then open in a browser:

```
http://localhost:8000/viewer.html#file=samples/paris-live.json
```

Click any token to see its decode + fidelity score. Keyboard: `←` `→` step, `F` flag, `C` copy, `N` note, `?` help.

A pre-generated demo fixture `samples/paris.json` ships in this folder so you can verify the viewer works before running the heavy pipeline.

---

## Useful flags

| Flag | Purpose |
|---|---|
| `--prompt "..."` | The user prompt to send. Required. |
| `--output samples/foo.json` | Where to write the JSON. Required. |
| `--sglang-url URL` | Defaults to `http://localhost:30000`. |
| `--skip-critic` | Skip fidelity scoring. Saves ~12 GB VRAM; cos/mse fields omitted. |
| `--max-new-tokens N` | Assistant response length. Default 200. |
| `--decode-max-tokens N` | Per-decode generation budget. Default 200. |
| `--temperature T` | Actor sampling temperature. Default 1.0 (matches training). |
| `--device cuda` / `cpu` | Default `cuda`. CPU is for debugging only. |
| `--base-model REPO_OR_PATH` | Override the base model. Default `Qwen/Qwen2.5-7B-Instruct`. |
| `--actor-model REPO_OR_PATH` | Override the actor. Default `kitft/nla-qwen2.5-7b-L20-av`. |
| `--critic-model REPO_OR_PATH` | Override the critic. Default `kitft/nla-qwen2.5-7b-L20-ar`. |

You can also pass already-downloaded local directories for any of the three — useful on machines without internet egress.

---

## Output JSON schema

```json
{
  "meta": {
    "model": "Qwen2.5-7B",
    "layer": 20,
    "prompt": "...",
    "transcript_hash": "sha256:..."
  },
  "tokens": [
    {
      "position": 0,
      "text": "Tell",
      "role": "user",
      "norm": 142.3,
      "filtered_reason": "early_position",
      "samples": [
        { "decode": "...", "cos": 0.71, "mse": 0.58 }
      ]
    }
  ]
}
```

`filtered_reason` is one of:
- `null` — clean decode, trust the result
- `"early_position"` — first 10 tokens; residual stream hasn't accumulated much signal yet
- `"low_cos"` — `cos < 0.7`; critic couldn't reconstruct the activation faithfully from the decode
- `"high_norm"` — activation norm > 5× the median; out-of-distribution, decode unreliable

---

## Troubleshooting

**"OutOfMemoryError" on the script side**
Add `--skip-critic` (drops the critic model). Total drops to ~16 GB VRAM. You lose fidelity scores but keep decodes.

**Output decodes are all in Chinese**
The injection failed and the actor is reading the literal CJK marker character. Most common causes:
1. Wrong actor for the base model — confirm both repos reference layer 20 of Qwen2.5-7B.
2. SGLang launched without `--disable-radix-cache`.
3. SGLang version too old. Need `>=0.5.6`.

**Occasional Chinese decode on legitimate tokens**
That's expected for Qwen — it's a Chinese model and genuinely decodes Chinese content. Only worry if it happens to *every* decode regardless of input.

**"Could not find a valid injection position"**
The actor's `nla_meta.yaml` and its tokenizer have drifted. Re-download the actor repo: `rm -rf ~/.cache/huggingface/hub/models--kitft--nla-qwen2.5-7b-L20-av`, then re-run.

**`HTTP 422` or `Validation error` from SGLang**
Usually means SGLang is older than 0.5.6. `pip install --upgrade 'sglang[all]>=0.5.6'`.

**Viewer shows the empty state**
Your URL is missing the hash. Use `#file=...` (not `?file=...`), e.g. `http://localhost:8000/viewer.html#file=samples/paris-live.json`. Or drag the JSON file directly onto the page.

---

## What is this for?

Interpretability research. NLA pairs translate model activation vectors into natural-language descriptions ("this activation is about Paris and French geography"). The viewer lets you scrub through a transcript and see what every position of the residual stream "thinks about." Background: see [Tonkin et al., *Natural Language Autoencoders*, Transformer Circuits 2026](https://transformer-circuits.pub/2026/nla/index.html).

This POC ships with the smallest published NLA pair (Qwen 7B). Larger pairs exist for Gemma-12B, Gemma-27B, and Llama 70B at the [`kitft/nla-models` collection](https://huggingface.co/collections/kitft/nla-models) — you can swap them in via `--actor-model` and `--critic-model`, but the per-architecture `injection_scale` and `embed_scale` differ and the script already pulls them from each repo's `nla_meta.yaml` so things should "just work."
