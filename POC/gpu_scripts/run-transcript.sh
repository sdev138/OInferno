#!/usr/bin/env bash
# Generate a transcript JSON.
#
# Usage:
#   ./run-transcript.sh "<prompt>" [<output-path>] [<extra build_transcript.py args>...]
#
# Examples:
#   ./run-transcript.sh "Tell me about the capital of France"
#     -> writes samples/transcript-YYYYMMDD-HHMMSS.json
#
#   ./run-transcript.sh "Why is the sky blue?" samples/sky.json
#     -> writes samples/sky.json
#
#   ./run-transcript.sh "Explain RAG" samples/rag.json --skip-critic
#     -> passes --skip-critic through to build_transcript.py

set -euo pipefail

if [ "$#" -lt 1 ]; then
    cat >&2 <<EOF
Usage: $0 "<prompt>" [<output-path>] [<extra build_transcript.py args>...]

Examples:
  $0 "Tell me about the capital of France"
  $0 "Why is the sky blue?" samples/sky.json
  $0 "Explain RAG" samples/rag.json --skip-critic
EOF
    exit 2
fi

PROMPT="$1"
shift

# Output path: explicit second arg, or timestamped default.
DEFAULT_OUTPUT="samples/transcript-$(date +%Y%m%d-%H%M%S).json"
OUTPUT="$DEFAULT_OUTPUT"
if [ "$#" -gt 0 ] && [[ "$1" != --* ]]; then
    OUTPUT="$1"
    shift
fi
EXTRA_ARGS=("$@")

export HF_HOME="${HF_HOME:-/workspace/hf_cache}"
SGLANG_URL="${SGLANG_URL:-http://localhost:30000}"

# Resolve POC/ from the script's own location.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
POC_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$POC_DIR"

# Probe SGLang before doing the heavy lifting — fail loud with a clear message.
if ! curl -sf --max-time 3 "$SGLANG_URL/health" > /dev/null; then
    cat >&2 <<EOF
[run] ERROR: SGLang isn't responding at $SGLANG_URL/health

Did you start it in another tmux window? Try:
  bash $SCRIPT_DIR/start-sglang.sh

If you intentionally want the in-process (local) actor instead of SGLang,
call build_transcript.py directly:
  python3 build_transcript.py --prompt "$PROMPT" --output "$OUTPUT" --actor-backend local
EOF
    exit 1
fi

echo "[run] prompt:  $PROMPT"
echo "[run] output:  $OUTPUT"
echo "[run] sglang:  $SGLANG_URL"
if [ "${#EXTRA_ARGS[@]}" -gt 0 ]; then
    echo "[run] extra:   ${EXTRA_ARGS[*]}"
fi
echo ""

python3 build_transcript.py \
    --prompt "$PROMPT" \
    --sglang-url "$SGLANG_URL" \
    --output "$OUTPUT" \
    "${EXTRA_ARGS[@]}"

cat <<EOF

[run] done. Pull the result down to your laptop with:

  scp root@<pod-host>:$POC_DIR/$OUTPUT POC/samples/

(Replace <pod-host> with the host part from the SSH command RunPod shows you,
 e.g. <pod-id>-22.proxy.runpod.net)
EOF
