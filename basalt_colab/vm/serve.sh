#!/bin/bash
# Starts basalt serve detached on $PORT (18081) with native Basalt v2 schema:
#   - slot-full-context: true (CTX is per-slot context, up to 1,048,576)
#   - elastic FP8/FP16 KV, Dynamic Vision, 32 GiB Disk Conversation Cache, optional sampling defaults
set -uo pipefail
S=/content/src/basalt
R=/content/res
BIN="$S/bin/basalt"
[ -x "$BIN" ] || BIN="$S/build/bin/basalt"
export PORT=${PORT:-18081}
export CTX=${CTX:-262144}
export PARALLEL=${PARALLEL:-2}
export KV_TYPE=${KV_TYPE:-fp8}
export VISION_MODE=${VISION_MODE:-dynamic}
export SPEC_MAX=${SPEC_MAX:-6}
export DRAFT_EXPERTS=${DRAFT_EXPERTS:-32}
export VRAM_RESERVE=${VRAM_RESERVE:-700}
export RAM_BUDGET=${RAM_BUDGET:-55}
export ANSWER_ROOM=${ANSWER_ROOM:-0}
export MODEL_PATH=${MODEL_PATH:-/mnt/ram/model.basalt}
export ENGRAM_PATH=${ENGRAM_PATH:-/mnt/ram/engram.bin}
export MODEL_ALIAS=${MODEL_ALIAS:-DeepSeek-V4.1-Flash-1017-UD-IQ1_M}
export CONV_CACHE_MIB=${CONV_CACHE_MIB:-32768}
export CONV_CACHE_DIR=${CONV_CACHE_DIR:-/content/conv_cache}
export BASALT_SAMPLING_JSON=${BASALT_SAMPLING_JSON:-}
mkdir -p "$CONV_CACHE_DIR"

python3 - "$R/basalt_config.json" <<'EOF'
import json, os, sys
ar = os.environ["ANSWER_ROOM"]
spec = int(os.environ["SPEC_MAX"])
model = {
    "model": os.environ["MODEL_PATH"],
    "alias": os.environ["MODEL_ALIAS"],
    "gpu": "0",
    "max-context": int(os.environ["CTX"]),
    "slot-full-context": True,
    "kv": os.environ["KV_TYPE"],
    "parallel": int(os.environ["PARALLEL"]),
    "vision": os.environ["VISION_MODE"],
    "spec": spec,
    "draft-experts": int(os.environ["DRAFT_EXPERTS"]) if spec > 0 else 0,
    "answer-room": int(ar) if ar.isdigit() else ar,
    "idle-unload": 0,
}
sampling_raw = os.environ.get("BASALT_SAMPLING_JSON", "").strip()
if sampling_raw:
    for k, v in json.loads(sampling_raw).items():
        model[k] = v

engine = {
    "vram-reserve-mib": int(os.environ["VRAM_RESERVE"]),
    "resident-budget-gib": int(os.environ["RAM_BUDGET"]),
}
if int(os.environ["CONV_CACHE_MIB"]) > 0:
    engine["conversation-cache-disk-mib"] = int(os.environ["CONV_CACHE_MIB"])
    engine["conversation-cache-dir"] = os.environ["CONV_CACHE_DIR"]

cfg = {
    "version": 2,
    "models": [model],
    "engine": engine,
    "server": {
        "host": "127.0.0.1",
        "port": int(os.environ["PORT"]),
        "allowed-host": ["*"],
    },
}
open(sys.argv[1], "w").write(json.dumps(cfg, indent=2) + "\n")
EOF

export BASALT_ENGRAM_FILE="$ENGRAM_PATH"
if [ -n "${BASALT_PUNCH_FILE:-}" ]; then
  export BASALT_PUNCH_FILE
fi
echo "$BIN serve --config $R/basalt_config.json --api-key-file /content/secrets/key" > $R/serve_cmd.txt
cd "$S"
setsid nohup "$BIN" serve --config "$R/basalt_config.json" --api-key-file /content/secrets/key \
  > "$R/server.log" 2>&1 < /dev/null &
echo $! > "$R/server.pid"
