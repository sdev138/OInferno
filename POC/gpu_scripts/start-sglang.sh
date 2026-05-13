#!/usr/bin/env bash
# Launch the NLA actor as an SGLang server on port 30000.
# Run inside a tmux window so SSH disconnect doesn't kill it.

set -euo pipefail

export HF_HOME="${HF_HOME:-/workspace/hf_cache}"

PORT="${SGLANG_PORT:-30000}"
ACTOR_MODEL="${ACTOR_MODEL:-kitft/nla-qwen2.5-7b-L20-av}"

cat <<EOF
[sglang] starting actor server
  model:   $ACTOR_MODEL
  port:    $PORT
  HF_HOME: $HF_HOME

First launch downloads ~15 GB and warms over ~1-2 min.
Subsequent launches (same network volume) start in seconds.

Leave this terminal running. Ctrl+C to stop.
EOF

# --disable-radix-cache is MANDATORY: SGLang's radix cache keys on token IDs,
# but our requests use input_embeds, which would alias unrelated requests in
# the cache and silently corrupt decodes.
exec python3 -m sglang.launch_server \
    --model-path "$ACTOR_MODEL" \
    --port "$PORT" \
    --disable-radix-cache \
    --mem-fraction-static 0.85 \
    --trust-remote-code
