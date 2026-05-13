# Running on a RunPod GPU

Scripts and walkthrough for running `build_transcript.py` on a rented cloud GPU.
The fast path: rent an L40S for ~$0.79/hr, SSH in, run two commands.

```
gpu_scripts/
├── README.md            (this file)
├── setup-runpod.sh      run once per pod boot — installs requirements, sets HF cache
├── start-sglang.sh      runs in tmux window 1 — the actor server (long-running)
└── run-transcript.sh    runs in tmux window 2 — generates one transcript
```

---

## One-time RunPod account setup (~10 min)

1. **Make an account** at runpod.io and add ~$10 credit (Settings → Billing). $10
   is comfortably enough for many sessions; you stop paying when the pod stops.

2. **Upload an SSH public key.** Settings → SSH Public Keys. If you don't have
   one yet:
   ```bash
   ssh-keygen -t ed25519 -C "you@example.com"
   cat ~/.ssh/id_ed25519.pub        # paste this into RunPod
   ```

3. **Create a Network Volume.** Storage → Network Volumes → New.
   - **Size**: 50 GB (~$3.50/month — stops billing when the volume is deleted)
   - **Region**: pick one that has L40S availability. US-OR-1 or EU-RO-1 usually
     do. RunPod marks regions with available GPUs in the pod-deploy UI.
   - This volume is the trick that makes second-and-later sessions fast: the
     ~30 GB of HuggingFace model weights cache here and survive pod
     termination. First session pays the download cost; subsequent sessions
     start in seconds.

---

## Per-session workflow

### 1. Deploy a pod

Pods → Deploy. Settings:
- **GPU**: 1× L40S (48 GB VRAM). A100 40GB also works.
- **Cloud type**: Community Cloud (~$0.79/hr) is fine for research.
- **Template**: "RunPod PyTorch 2.4" (or whichever PyTorch template is current).
  CUDA is preinstalled — saves you a torch install.
- **Network Volume**: attach the one you created above. Mount path
  `/workspace` (default).
- **Container disk**: 30 GB default is fine.

Click Deploy. Pod boots in ~30s.

### 2. SSH in

RunPod's UI shows a connection command once the pod is running, like:
```bash
ssh root@<pod-id>-22.proxy.runpod.net -i ~/.ssh/id_ed25519
```

### 3. (First session only) clone the repo

```bash
cd /workspace
git clone https://github.com/sdev138/OInferno.git
```

After this, `/workspace/OInferno/` lives on the network volume — it survives pod
termination. On future sessions, skip this step (or `cd /workspace/OInferno &&
git pull` if you want updates).

### 4. Run the setup script

```bash
cd /workspace/OInferno/POC
bash gpu_scripts/setup-runpod.sh
```

What it does:
- Sets `HF_HOME=/workspace/hf_cache` so model weights cache on the volume.
- Verifies torch + CUDA work.
- `pip install -r requirements.txt` (idempotent; fast on re-run).

Takes ~3-5 min the first time, ~10 seconds on every later run.

### 5. Start tmux + the SGLang server

```bash
tmux new -s nla
bash gpu_scripts/start-sglang.sh
```

First launch downloads ~15 GB of actor weights (one-time, into the network
volume). Subsequent launches start in seconds.

When you see `INFO: Uvicorn running on http://0.0.0.0:30000`, the server is
ready. **Leave this window alone.**

### 6. Open a second tmux window and generate a transcript

Press **`Ctrl+B`** then **`c`** to open a new tmux window. In the new window:

```bash
cd /workspace/OInferno/POC
bash gpu_scripts/run-transcript.sh "Tell me about the capital of France"
```

The script:
- Probes SGLang's `/health` endpoint and fails loud if it's not up.
- Defaults the output to `samples/transcript-<timestamp>.json`.
- Runs `build_transcript.py` with the right `--sglang-url`.

To use a specific output name:
```bash
bash gpu_scripts/run-transcript.sh "Why is the sky blue?" samples/sky.json
```

To pass extra args through to the Python script:
```bash
bash gpu_scripts/run-transcript.sh "Explain RAG" samples/rag.json --skip-critic
bash gpu_scripts/run-transcript.sh "..." samples/out.json --max-new-tokens 120
```

A typical 50-token transcript takes a few minutes on an L40S. You'll see
per-position progress.

### 7. Pull the result down to your laptop

The script tells you the exact `scp` command at the end. Run it locally:

```bash
# on your Mac:
scp root@<pod-id>-22.proxy.runpod.net:/workspace/OInferno/POC/samples/sky.json \
    POC/samples/
```

Then view it:
```bash
cd POC
python3 -m http.server 8000
# open http://localhost:8000/viewer.html#file=samples/sky.json
```

### 8. STOP the pod when done

This is the most important step. RunPod bills per-second.

From the Pods page, click **"Stop"** (not "Terminate"). Stop preserves the pod
config and disk; restarting is fast. Terminate deletes the pod (but keeps the
network volume).

A pod left running overnight by accident: ~$19 on L40S. Stop saves you.

---

## Sequencing every later session

Once you've done the one-time setup and the first deploy, every later session
collapses to:

```
1. Deploy pod (same settings, attach the same network volume)
2. ssh in
3. cd /workspace/OInferno/POC && bash gpu_scripts/setup-runpod.sh
4. tmux new -s nla
5. bash gpu_scripts/start-sglang.sh             # window 1
6. (Ctrl+B c) bash gpu_scripts/run-transcript.sh "..."  # window 2
7. scp result down
8. Stop the pod
```

---

## Cost reference

| What | $ |
|---|---|
| L40S 48 GB, Community Cloud | ~$0.79/hr |
| A100 40 GB, Community Cloud | ~$1.19/hr |
| Network Volume, 50 GB | ~$3.50/month |
| Typical 2-hour research session | $2-4 |

---

## Troubleshooting

**`SGLang isn't responding at http://localhost:30000/health`**
You skipped step 5, or the SGLang window crashed. Check the SGLang window
(`tmux attach -t nla`, then `Ctrl+B 0` to switch to window 0). Common causes:
- Out of VRAM — the pod was sized too small. Need ≥ 24 GB free.
- Model still downloading — wait for `Uvicorn running` line.
- HF_HOME not writable — confirm `/workspace/hf_cache` exists and you have
  permissions.

**`tmux: command not found`**
Rare but happens on minimal templates. `apt-get update && apt-get install -y tmux`.

**SSH disconnected and SGLang/script died**
You didn't run them inside tmux. Reconnect, run `tmux attach -t nla` — if the
session is gone, restart from step 5. Always work inside tmux.

**The pod crashes with no traceback / disappears**
Pre-emption on Community Cloud. Rare for L40S; more common for A100. Redeploy
the pod with the same network volume attached and your cached weights are
still there.

**First decode takes minutes, then the rest are seconds**
Normal — first call hits cold KV cache + JIT compilation. Steady-state is
much faster.

**Pod is running but `nvidia-smi` shows no GPU**
You picked a CPU-only template by accident. Terminate, redeploy with a GPU
template.
