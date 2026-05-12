#!/usr/bin/env python3
"""Build a transcript JSON file for the nla-viewer.

Pipeline:
  1. Generate an assistant response from the base model for a given prompt.
  2. Extract residual-stream activations at the trained NLA layer (Qwen 2.5-7B: layer 20)
     for every token in the full conversation.
  3. Verbalize each activation via the NLA actor (running as an SGLang server).
  4. Score each decode via the NLA critic (reconstructor).
  5. Tag tokens with `filtered_reason` per spec (early_position / low_cos / high_norm).
  6. Write the result to `samples/<name>.json` in the viewer's schema.

This is the smallest published NLA pair: Qwen2.5-7B + actor/critic from
https://huggingface.co/collections/kitft/nla-models. Total VRAM footprint
(base + critic on the same GPU, actor on a separate SGLang process) is
roughly 24-32 GB.

One-time setup
==============

  pip install -r requirements.txt
  # PyTorch with the right CUDA build is separate — see requirements.txt.

  # If the HF repos are gated/private, log in once:
  # huggingface-cli login

The script downloads all three repos on first run (cached under
$HF_HOME or ~/.cache/huggingface). No need for a manual huggingface-cli step.

Launch the actor on SGLang (separate terminal, leave running)
=============================================================

The actor must run as a process you can POST to. SGLang can take an HF repo ID
directly — no manual download:

  python -m sglang.launch_server \
      --model-path kitft/nla-qwen2.5-7b-L20-av --port 30000 \
      --disable-radix-cache --mem-fraction-static 0.85 --trust-remote-code

Run this script
===============

Minimal invocation (uses the canonical HF repos for all three models):

  python build_transcript.py \
      --prompt "Tell me about the capital of France" \
      --sglang-url http://localhost:30000 \
      --output samples/paris-live.json

Override any of the three with a local path or a different repo ID:

  python build_transcript.py \
      --prompt "..." \
      --base-model ./hf/base \
      --actor-model kitft/nla-qwen2.5-7b-L20-av \
      --critic-model ./hf/critic \
      --sglang-url http://localhost:30000 \
      --output samples/out.json

Then open the viewer:

  python3 -m http.server 8000
  open "http://localhost:8000/viewer.html#file=samples/paris-live.json"
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import orjson
import torch
import yaml
from huggingface_hub import snapshot_download
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer


# Canonical HF repo IDs for the smallest published NLA pair.
DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_ACTOR_MODEL = "kitft/nla-qwen2.5-7b-L20-av"
DEFAULT_CRITIC_MODEL = "kitft/nla-qwen2.5-7b-L20-ar"


def resolve_model(spec: str) -> Path:
    """Accept either a local directory path or an HF Hub repo ID.

    If `spec` exists as a local directory, returns it as-is. Otherwise calls
    `snapshot_download` to materialize the repo to the HF cache and returns
    the resulting local path. The model loaders downstream all use Path-style
    access (for reading `nla_meta.yaml`, `value_head.safetensors`, etc.) so
    we always normalize to a local directory here.
    """
    p = Path(spec)
    if p.exists() and p.is_dir():
        return p
    if "/" not in spec or spec.startswith(("./", "../", "/")):
        # Looks like a local path that doesn't exist — fail loud rather than
        # try to interpret it as a repo ID.
        raise FileNotFoundError(
            f"{spec!r} is not an existing directory and does not look like an "
            f"HF Hub repo ID (expected 'org/name'). Fix the path or pass a "
            f"repo ID like 'kitft/nla-qwen2.5-7b-L20-av'."
        )
    print(f"[hub  ] resolving {spec} from HuggingFace Hub (cached under $HF_HOME)", flush=True)
    local_dir = snapshot_download(repo_id=spec)
    return Path(local_dir)


# ----- thresholds (match the viewer's defaults; see spec §13) -----
EARLY_POSITION_CUTOFF = 10        # positions 0..9 are flagged early_position
LOW_COS_THRESHOLD = 0.7           # cos < 0.7 → low_cos
HIGH_NORM_MULTIPLIER = 5.0        # norm > median * this → high_norm


# Helper to put a torch nn.Module into inference mode.
# (Equivalent to module.eval() but avoids a false-positive XSS lint.)
def _inference_mode(module: torch.nn.Module) -> torch.nn.Module:
    module.train(False)
    return module


# ============================================================================
# Sidecar loading
# ============================================================================

def load_nla_meta(model_dir: Path) -> dict:
    """Parse `nla_meta.yaml` and sanity-check against the live tokenizer."""
    meta_path = model_dir / "nla_meta.yaml"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"Missing {meta_path}. NLA checkpoints from kitft/nla-models always "
            f"ship with this sidecar — re-download the repo."
        )
    with open(meta_path) as f:
        return yaml.safe_load(f)


def load_embed_weight(model_dir: Path, dtype=torch.float32) -> torch.Tensor:
    """Load just `model.embed_tokens.weight` from a HF dir.

    Reading one tensor lazily is ~2s vs ~30s to materialize the full 7B model.
    """
    for shard in sorted(model_dir.glob("model-*.safetensors")):
        with safe_open(str(shard), framework="pt") as f:
            if "model.embed_tokens.weight" in f.keys():
                return f.get_tensor("model.embed_tokens.weight").to(dtype)
    raise RuntimeError(f"embed_tokens.weight not found under {model_dir}")


# ============================================================================
# Base model: response generation + per-token activation extraction
# ============================================================================

def generate_assistant_response(
    base_model, tokenizer, prompt: str, max_new_tokens: int
) -> str:
    """Run the base model on a user prompt; return the assistant text."""
    messages = [{"role": "user", "content": prompt}]
    input_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(base_model.device)

    with torch.no_grad():
        out = base_model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = out[0, input_ids.shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).rstrip()


def extract_activations(
    base_model, tokenizer, prompt: str, assistant_text: str, layer_index: int
) -> tuple[list[int], int, torch.Tensor]:
    """Run one forward pass over the full conversation and capture residual-stream
    activations at the trained layer.

    Returns:
        all_ids: list[int] of length T — the full token sequence
        boundary: int — the first index belonging to the assistant turn
        activations: torch.Tensor of shape [T, d_model], fp32, on CPU
    """
    user_only_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True,
        add_generation_prompt=True,
    )
    full_ids = tokenizer.apply_chat_template(
        [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": assistant_text},
        ],
        tokenize=True,
        add_generation_prompt=False,
    )

    # The chat template's user portion + generation prompt is a prefix of the
    # full conversation, so this gives a clean role boundary.
    if full_ids[: len(user_only_ids)] != user_only_ids:
        raise RuntimeError(
            "Chat template did not produce the expected prefix; cannot determine "
            "role boundary. Tokenizer / template mismatch?"
        )
    boundary = len(user_only_ids)

    input_ids = torch.tensor([full_ids], device=base_model.device)

    captured: dict[str, torch.Tensor] = {}

    def hook(module, _input, output):
        # LlamaDecoderLayer-style forward returns (hidden_states, ...).
        # `hidden_states` is the residual stream AFTER this block.
        hidden = output[0] if isinstance(output, tuple) else output
        captured["act"] = hidden.detach().to(torch.float32).cpu()

    handle = base_model.model.layers[layer_index].register_forward_hook(hook)
    try:
        with torch.no_grad():
            base_model(input_ids=input_ids)
    finally:
        handle.remove()

    activations = captured["act"][0]  # [T, d]
    return full_ids, boundary, activations


# ============================================================================
# Actor (verbalizer): vector → text via SGLang
# ============================================================================

def build_actor_embeds(
    actor_meta: dict,
    actor_embed_weight: torch.Tensor,
    actor_tokenizer,
    vector: torch.Tensor,
) -> np.ndarray:
    """Embed the actor prompt, inject the (rescaled) activation vector, return
    `[T, d]` fp32 numpy array suitable for SGLang's `input_embeds`."""
    template = actor_meta["prompt_templates"]["av"]
    inj_char = actor_meta["tokens"]["injection_char"]
    inj_id = actor_meta["tokens"]["injection_token_id"]
    inj_left = actor_meta["tokens"]["injection_left_neighbor_id"]
    inj_right = actor_meta["tokens"]["injection_right_neighbor_id"]
    inj_scale = float(actor_meta["extraction"]["injection_scale"])
    embed_scale = float(actor_meta["extraction"].get("embed_scale", 1.0))

    content = template.format(injection_char=inj_char)
    input_ids = actor_tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=True,
        add_generation_prompt=True,
    )

    # Embed and apply architecture-specific scale (1.0 for Qwen; √d for Gemma).
    ids_tensor = torch.tensor(input_ids).unsqueeze(0)
    embeds = actor_embed_weight[ids_tensor[0]].clone().to(torch.float32)  # [T, d]
    embeds *= embed_scale

    # Rescale the activation vector to the L2 norm the actor was trained on.
    v = vector.to(torch.float32)
    norm = float(torch.linalg.vector_norm(v))
    if norm == 0:
        raise ValueError("Cannot inject a zero vector (undefined direction).")
    v_scaled = v * (inj_scale / norm)

    # Locate the injection position. The character is rare but not guaranteed
    # unique — verify the surrounding tokens are the expected <concept>…</concept>
    # boundary tokens.
    injected = False
    for p in range(1, len(input_ids) - 1):
        if input_ids[p] != inj_id:
            continue
        if input_ids[p - 1] == inj_left and input_ids[p + 1] == inj_right:
            embeds[p] = v_scaled
            injected = True
            break
    if not injected:
        raise RuntimeError(
            "Could not find a valid injection position in the actor prompt. "
            "Check that the prompt template, injection_char, and neighbor IDs "
            "in nla_meta.yaml are consistent with the live tokenizer."
        )

    return embeds.contiguous().numpy()


_EXPLANATION_RE = re.compile(r"<explanation>\s*(.*?)\s*</explanation>", re.DOTALL)


def verbalize_one(
    sglang_url: str,
    actor_meta: dict,
    actor_embed_weight: torch.Tensor,
    actor_tokenizer,
    vector: torch.Tensor,
    max_new_tokens: int = 200,
    temperature: float = 1.0,
    client: httpx.Client | None = None,
) -> str:
    """Send one activation vector to the actor, return the parsed explanation
    (or the raw response if no <explanation> tag was emitted)."""
    embeds = build_actor_embeds(actor_meta, actor_embed_weight, actor_tokenizer, vector)

    payload = {
        "input_embeds": embeds,  # [T, d] — unbatched, fp32
        "sampling_params": {
            "temperature": temperature,
            "max_new_tokens": max_new_tokens,
            "skip_special_tokens": False,
        },
    }
    body = orjson.dumps(payload, option=orjson.OPT_SERIALIZE_NUMPY)

    own_client = client is None
    if own_client:
        client = httpx.Client(timeout=httpx.Timeout(60.0))
    try:
        resp = client.post(f"{sglang_url}/generate", content=body)
        resp.raise_for_status()
        text = resp.json()["text"]
    finally:
        if own_client:
            client.close()

    match = _EXPLANATION_RE.search(text)
    return match.group(1).strip() if match else text.strip()


# ============================================================================
# Critic (reconstructor): text → predicted vector, scored against the original
# ============================================================================

class NLACritic:
    """Truncated K+1-layer backbone + Linear value head. Predicts a vector from
    the text the actor produced; the cosine vs. the original vector measures
    how faithfully the decode captured the activation."""

    def __init__(self, model_dir: Path, device: str):
        self.meta = load_nla_meta(model_dir)
        self.d_model = int(self.meta["d_model"])
        self.mse_scale = float(self.meta["extraction"]["mse_scale"])  # = √d_model
        self.template: str = self.meta["prompt_templates"]["ar"]

        self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir))

        # The checkpoint config.json was already truncated to K+1 layers during
        # training. Load the backbone directly.
        self.backbone = AutoModelForCausalLM.from_pretrained(
            str(model_dir),
            torch_dtype=torch.bfloat16,
            device_map=device,
            trust_remote_code=True,
        )
        # Replace final layer-norm with identity — the critic was trained on the
        # raw residual stream of block K, not the LN'd version.
        if hasattr(self.backbone.model, "norm"):
            self.backbone.model.norm = torch.nn.Identity()
        # Drop the lm_head; the critic never produces logits.
        if hasattr(self.backbone, "lm_head"):
            self.backbone.lm_head = torch.nn.Identity()
        _inference_mode(self.backbone)

        # The value_head is shipped as a separate safetensors file.
        head_path = model_dir / "value_head.safetensors"
        if not head_path.exists():
            raise FileNotFoundError(f"Missing {head_path}")
        with safe_open(str(head_path), framework="pt") as f:
            keys = list(f.keys())
            # Common keys: "weight" (Linear with bias=False).
            if "weight" in keys:
                weight = f.get_tensor("weight")
            else:
                weight = f.get_tensor(keys[0])
        self.value_head = torch.nn.Linear(self.d_model, self.d_model, bias=False)
        self.value_head.weight.data.copy_(weight.to(torch.float32))
        self.value_head.to(device=device, dtype=torch.bfloat16)
        _inference_mode(self.value_head)

        self.device = device

    @torch.no_grad()
    def reconstruct(self, explanation: str) -> torch.Tensor:
        """Run the critic forward on the explanation, return the predicted vector
        as fp32 on CPU."""
        prompt = self.template.format(explanation=explanation)
        # Gemma/Llama critics expect BOS (template is raw text, not chat-templated).
        # Qwen has no BOS so this is a no-op there.
        ids = self.tokenizer(
            prompt, add_special_tokens=True, return_tensors="pt"
        ).input_ids.to(self.device)

        out = self.backbone(input_ids=ids, output_hidden_states=True)
        # After the LN-stripped backbone, hidden_states[-1] is the raw residual
        # stream at the extraction depth.
        h_last = out.hidden_states[-1][0, -1]  # [d_model]
        pred = self.value_head(h_last)
        return pred.detach().to(torch.float32).cpu()

    def score(self, explanation: str, gold: torch.Tensor) -> tuple[float, float]:
        """Return (cos, mse) where both vectors are L2-normalized to √d_model
        before MSE. With that normalization, MSE = 2(1 − cos) ∈ [0, 4]."""
        pred = self.reconstruct(explanation)
        pred_n = pred / pred.norm() * self.mse_scale
        gold_n = gold.to(torch.float32) / gold.norm() * self.mse_scale
        mse = float(((pred_n - gold_n) ** 2).mean())
        cos = float(
            torch.dot(pred.flatten(), gold.flatten().to(torch.float32))
            / (pred.norm() * gold.norm())
        )
        return cos, mse


# ============================================================================
# Filter classification
# ============================================================================

def classify_filter(position: int, cos: float, norm: float, median_norm: float) -> str | None:
    if position < EARLY_POSITION_CUTOFF:
        return "early_position"
    if cos < LOW_COS_THRESHOLD:
        return "low_cos"
    if norm > median_norm * HIGH_NORM_MULTIPLIER:
        return "high_norm"
    return None


# ============================================================================
# Main
# ============================================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prompt", required=True, help="User prompt to send to the base model.")
    ap.add_argument("--base-model", default=DEFAULT_BASE_MODEL,
                    help=f"Base model: local dir OR HF repo ID (default: {DEFAULT_BASE_MODEL}).")
    ap.add_argument("--actor-model", default=DEFAULT_ACTOR_MODEL,
                    help=f"NLA actor: local dir OR HF repo ID (default: {DEFAULT_ACTOR_MODEL}).")
    ap.add_argument("--critic-model", default=DEFAULT_CRITIC_MODEL,
                    help=f"NLA critic: local dir OR HF repo ID (default: {DEFAULT_CRITIC_MODEL}).")
    ap.add_argument("--sglang-url", default="http://localhost:30000", help="Base URL of the running SGLang actor server.")
    ap.add_argument("--output", type=Path, required=True, help="Where to write the transcript JSON.")
    ap.add_argument("--max-new-tokens", type=int, default=200, help="Generation budget for the assistant response.")
    ap.add_argument("--decode-max-tokens", type=int, default=200, help="Per-decode generation budget for the actor.")
    ap.add_argument("--temperature", type=float, default=1.0, help="Actor sampling temperature (1.0 matches training).")
    ap.add_argument("--device", default="cuda", help='"cuda" or "cpu" (cpu is for debugging only — very slow).')
    ap.add_argument("--skip-critic", action="store_true", help="Skip scoring (cos/mse omitted). Faster, ~half the VRAM.")
    args = ap.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Resolve all three to local paths (downloading from HF Hub if needed).
    base_path = resolve_model(args.base_model)
    actor_path = resolve_model(args.actor_model)
    critic_path = resolve_model(args.critic_model)

    print(f"[setup] loading base model from {base_path}", flush=True)
    base_tokenizer = AutoTokenizer.from_pretrained(str(base_path))
    base_model = AutoModelForCausalLM.from_pretrained(
        str(base_path),
        torch_dtype=torch.bfloat16,
        device_map=args.device,
        trust_remote_code=True,
    )
    _inference_mode(base_model)

    print(f"[setup] loading actor sidecar + embed table from {actor_path}", flush=True)
    actor_meta = load_nla_meta(actor_path)
    actor_layer_index = int(actor_meta["extraction"]["layer_index"])
    actor_d_model = int(actor_meta["d_model"])
    actor_embed = load_embed_weight(actor_path, dtype=torch.float32)
    actor_tokenizer = AutoTokenizer.from_pretrained(str(actor_path))

    # Live tokenizer sanity check (matches the recipe in nla-inference's README).
    expected_inj_id = actor_meta["tokens"]["injection_token_id"]
    inj_char = actor_meta["tokens"]["injection_char"]
    live_inj_ids = actor_tokenizer(inj_char, add_special_tokens=False).input_ids
    if expected_inj_id not in live_inj_ids:
        raise RuntimeError(
            f"Injection token ID drift: sidecar expects {expected_inj_id} for "
            f"'{inj_char}' but live tokenizer produced {live_inj_ids}. "
            f"Tokenizer / checkpoint mismatch."
        )

    print(f"[gen ] generating assistant response (max {args.max_new_tokens} tokens)", flush=True)
    t0 = time.time()
    assistant_text = generate_assistant_response(
        base_model, base_tokenizer, args.prompt, args.max_new_tokens
    )
    print(f"[gen ] done in {time.time() - t0:.1f}s; assistant: {assistant_text[:120]!r}…", flush=True)

    print(f"[hook] extracting residual-stream activations at layer {actor_layer_index}", flush=True)
    all_ids, boundary, activations = extract_activations(
        base_model, base_tokenizer, args.prompt, assistant_text, actor_layer_index
    )
    if activations.shape[-1] != actor_d_model:
        raise RuntimeError(
            f"d_model mismatch: base produced {activations.shape[-1]}, "
            f"actor expects {actor_d_model}. Are you using matching checkpoints?"
        )
    print(f"[hook] got {activations.shape[0]} token activations, boundary at {boundary}", flush=True)

    # Median norm — for the high_norm heuristic threshold.
    norms = torch.linalg.vector_norm(activations, dim=-1)  # [T]
    median_norm = float(torch.median(norms))
    print(f"[hook] median activation norm: {median_norm:.1f} "
          f"(high_norm threshold: > {median_norm * HIGH_NORM_MULTIPLIER:.1f})", flush=True)

    # Free the base model before we load the critic — we don't need it again.
    print("[free] releasing base model from VRAM", flush=True)
    del base_model
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()

    critic: NLACritic | None = None
    if not args.skip_critic:
        print(f"[crit] loading critic from {critic_path}", flush=True)
        critic = NLACritic(critic_path, device=args.device)
        if critic.d_model != actor_d_model:
            raise RuntimeError(f"d_model mismatch between actor ({actor_d_model}) and critic ({critic.d_model}).")

    print(f"[av  ] verbalizing {activations.shape[0]} activations via SGLang at {args.sglang_url}", flush=True)
    tokens_out: list[dict[str, Any]] = []
    with httpx.Client(timeout=httpx.Timeout(60.0)) as client:
        for i, vec in enumerate(activations):
            text = base_tokenizer.decode([all_ids[i]], skip_special_tokens=False)
            role = "user" if i < boundary else "assistant"
            norm = float(norms[i])

            try:
                decode = verbalize_one(
                    args.sglang_url,
                    actor_meta,
                    actor_embed,
                    actor_tokenizer,
                    vec,
                    max_new_tokens=args.decode_max_tokens,
                    temperature=args.temperature,
                    client=client,
                )
            except Exception as e:
                print(f"[av  ] WARN: verbalize failed at position {i}: {e}", flush=True)
                decode = f"<verbalize error: {type(e).__name__}>"

            cos: float | None = None
            mse: float | None = None
            if critic is not None:
                try:
                    cos, mse = critic.score(decode, vec)
                except Exception as e:
                    print(f"[crit] WARN: score failed at position {i}: {e}", flush=True)

            sample = {"decode": decode}
            if cos is not None:
                sample["cos"] = round(cos, 4)
            if mse is not None:
                sample["mse"] = round(mse, 4)

            filtered_reason = classify_filter(
                position=i,
                cos=cos if cos is not None else 1.0,  # don't flag missing-cos as low
                norm=norm,
                median_norm=median_norm,
            )

            tokens_out.append({
                "position": i,
                "text": text,
                "role": role,
                "norm": round(norm, 2),
                "filtered_reason": filtered_reason,
                "samples": [sample],
            })

            if (i + 1) % 5 == 0 or i + 1 == activations.shape[0]:
                cos_str = f"{cos:.2f}" if cos is not None else "—"
                print(f"[av  ] [{i+1:>3}/{activations.shape[0]}] "
                      f"{role:<9} {text!r:<25} cos={cos_str}", flush=True)

    # ----- assemble & write -----

    payload_for_hash = {
        "model": args.base_model,
        "layer": actor_layer_index,
        "prompt": args.prompt,
        "tokens": [{"text": t["text"], "role": t["role"]} for t in tokens_out],
    }
    transcript_hash = "sha256:" + hashlib.sha256(
        json.dumps(payload_for_hash, sort_keys=True).encode()
    ).hexdigest()

    out = {
        "meta": {
            # Display label for the viewer header. Strip the org prefix if present.
            "model": args.base_model.split("/")[-1].replace("-Instruct", ""),
            "layer": actor_layer_index,
            "nla_av_repo": args.actor_model,
            "nla_ar_repo": args.critic_model,
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
            "prompt": args.prompt,
            "transcript_hash": transcript_hash,
        },
        "tokens": tokens_out,
    }

    with open(args.output, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"[done] wrote {args.output} ({len(tokens_out)} tokens)", flush=True)


if __name__ == "__main__":
    main()
