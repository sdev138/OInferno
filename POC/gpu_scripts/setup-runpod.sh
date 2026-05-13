#!/usr/bin/env bash
# Run once after SSHing into a fresh RunPod pod. Idempotent — safe to re-run.
#
# Assumes:
#   - The pod was deployed from "RunPod PyTorch 2.4" (or similar) template.
#     Torch + CUDA are preinstalled; we only add the POC's extras + sglang.
#   - A network volume is mounted at /workspace. The git clone, the HF cache,
#     and this script all live under /workspace so they survive pod termination.

set -euo pipefail

# Resolve POC/ from the script's own location, regardless of cwd.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
POC_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# ----- HF cache on the persistent network volume -----
HF_CACHE_DIR="/workspace/hf_cache"
mkdir -p "$HF_CACHE_DIR"
export HF_HOME="$HF_CACHE_DIR"

# Persist HF_HOME for all future shells in this pod.
if ! grep -q "^export HF_HOME=" ~/.bashrc 2>/dev/null; then
    echo "export HF_HOME=$HF_CACHE_DIR" >> ~/.bashrc
    echo "[setup] persisted HF_HOME=$HF_CACHE_DIR in ~/.bashrc"
fi

# ----- sanity check: torch + CUDA already work -----
python3 - <<'PY'
import sys, torch
if not torch.cuda.is_available():
    print("[setup] FAIL: torch reports no CUDA. Is this a GPU pod?", file=sys.stderr)
    sys.exit(1)
print(f"[setup] ok: torch {torch.__version__} / CUDA {torch.version.cuda} / {torch.cuda.get_device_name(0)}")
PY

# ----- install POC requirements -----
# pip is idempotent — already-satisfied packages are skipped. Torch is already
# in the base image so the constraint in requirements.txt won't trigger a
# reinstall.
cd "$POC_DIR"
echo "[setup] installing requirements (this can take 3-5 min the first time)..."
pip install --quiet -r requirements.txt
echo "[setup] requirements installed."

# ----- done -----
cat <<EOF

==============================================================================
Setup complete.

Next steps (run inside tmux so SSH disconnect doesn't kill SGLang):

  tmux new -s nla

  # tmux window 1: launch the actor server (leave running)
  bash $SCRIPT_DIR/start-sglang.sh

  # press Ctrl+B then 'c' to open a new tmux window

  # tmux window 2: generate a transcript
  bash $SCRIPT_DIR/run-transcript.sh "Tell me about the capital of France"

==============================================================================
EOF
