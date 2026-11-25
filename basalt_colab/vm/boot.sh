#!/bin/bash
# Cold start on Colab G4 using Basalt v0.5.7 (DeepSeek-V4.1-Flash):
#   1. Inspect 64 KiB .basalt header -> auto-plan storage tier (ram+ram / ram+ssd / ssd+ssd, 0 cold experts on SSD)
#   2. Parallel: fetch/build patched Basalt + fetch sparse backbone + fetch 43 KB mrfakename abliteration edit
#   3. In-place abliteration via standalone abliterate (~1.5s, 62/62 tensors, tunable UNCENSOR_SCALE)
#   4. Start basalt serve + POST /load (punch-on-read reclaims backbone storage before Engram stream)
#   5. Stream Engram table into /mnt/ram/engram.bin (ram+ram) or /content/engram.bin (ram+ssd / ssd+ssd)
#   6. Warmup + open gate.py on :18080 + arm VM-side hard-cap backstop
set -uo pipefail
: "${CTX:?}"
export PORT=18081 UNCENSOR=${UNCENSOR:-1} UNCENSOR_SCALE=${UNCENSOR_SCALE:-1.0} WORKERS=${WORKERS:-64}
export PARALLEL=${PARALLEL:-2} KV_TYPE=${KV_TYPE:-fp8} VISION_MODE=${VISION_MODE:-dynamic}
export SPEC_MAX=${SPEC_MAX:-6} DRAFT_EXPERTS=${DRAFT_EXPERTS:-32}
export VRAM_RESERVE=${VRAM_RESERVE:-700} RAM_BUDGET=${RAM_BUDGET:-0}
export CONV_CACHE_MIB=${CONV_CACHE_MIB:-32768}
export MODEL_QUANT=${MODEL_QUANT:-ud-iq1_m}
export MODEL_ALIAS=${MODEL_ALIAS:-DeepSeek-V4.1-Flash-1017-UD-IQ1_M}
export MODEL_URL=${MODEL_URL:-}
[ -s /content/secrets/hf_hdr ] && export HF_HDR=/content/secrets/hf_hdr
R=/content/res; V=/content/vm; D=/content/basalt
mkdir -p $R /mnt/ram /content/patch $D
rm -f $R/READY

T0=$(date +%s)
st() { echo "$(( $(date +%s) - T0 ))s $(date +%T) $*" >> $R/timeline.txt; }
seen() { grep -qE "$1" $R/timeline.txt; }
hdr() { echo "Authorization: Bearer $(cat /content/secrets/key)"; }

# Read pinned URLs & SHA-256 from $V/manifest.json
eval "$(python3 - "$V/manifest.json" "$MODEL_QUANT" <<'EOF'
import json, shlex, sys
m = json.load(open(sys.argv[1]))
q = sys.argv[2].strip().lower()
packs = m.get("packs") or {}
pack_url = (packs.get(q) or {}).get("url") or (m.get("pack_url", "") if q in ("ud-iq1_m", "iq1_m") else "")
for env_k, val in (
    ("ABLIT_URL", m.get("abliteration_url", "")),
    ("ABLIT_SHA", m.get("abliteration_sha256", "")),
    ("MANIFEST_PREBUILT_URL", m.get("prebuilt_url", "")),
    ("MANIFEST_PREBUILT_SHA", m.get("prebuilt_sha256", "")),
    ("MANIFEST_PACK_URL", pack_url),
):
    print(f"export {env_k}={shlex.quote(str(val))}")
EOF
)"

[ -n "$MODEL_URL" ] || MODEL_URL="$MANIFEST_PACK_URL"
if [ -z "$MODEL_URL" ]; then
  st "MODEL_URL missing for quant=$MODEL_QUANT FAILED"
  exit 1
fi

st "boot start engine=basalt quant=$MODEL_QUANT model=$MODEL_ALIAS ref=${BASALT_REF:-pinned} ctx=$CTX parallel=$PARALLEL kv=$KV_TYPE vision=$VISION_MODE uncensored=$UNCENSOR scale=$UNCENSOR_SCALE"

RAM_GIB=$(awk '/MemTotal/ {printf "%.1f", $2/1048576}' /proc/meminfo)
TMPFS_GIB=$(awk '/MemTotal/ {print int($2/1048576) - 6}' /proc/meminfo)
DISK_FREE_GIB=$(df -BG --output=avail /content | tail -1 | tr -dc '0-9')
VRAM_GIB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 | awk '{printf "%.1f", ($1>0?$1:98304)/1024}')
VRAM_GIB=${VRAM_GIB:-96.0}
mountpoint -q /mnt/ram || mount -t tmpfs -o size=${TMPFS_GIB}G tmpfs /mnt/ram

# Inspect remote 64 KiB .basalt header and plan ram+ram / ram+ssd / ssd+ssd (fail-fast if > VM RAM+VRAM)
PLAN_OUT=$(python3 $V/basalt_jobs.py --plan --quant "$MODEL_QUANT" --url "$MODEL_URL" \
  --ram-total-gib "$RAM_GIB" --disk-free-gib "$DISK_FREE_GIB" --vram-gib "$VRAM_GIB" \
  --ram-budget "$RAM_BUDGET" 2> $R/plan_err.txt) || {
  st "storage plan FAILED ($(tr '\n' ' ' < $R/plan_err.txt))"
  exit 1
}
eval "$PLAN_OUT"
st "storage plan mode=$STAGE_MODE bb=${BACKBONE_GIB}GiB eng=${ENGRAM_GIB}GiB ram_budget=${RAM_BUDGET}GiB (vm ram=${RAM_GIB}GiB disk=${DISK_FREE_GIB}GiB vram=${VRAM_GIB}GiB)"

# Probe hole-punching on /content before downloading 200+ GiB in ssd+ssd mode
if [ "$STAGE_MODE" = "ssd+ssd" ]; then
  python3 $V/basalt_jobs.py --test-punch /content > $R/punch_probe.txt 2>&1 || {
    st "GUARD: /content does not support FALLOC_FL_PUNCH_HOLE ($(cat $R/punch_probe.txt 2>/dev/null)) FAILED"
    exit 1
  }
  st "punch probe /content ok"
fi

python3 $V/basalt_jobs.py --quant "$MODEL_QUANT" --url "$MODEL_URL" \
  --backbone-dest "$MODEL_PATH" --backbone-jobs "$R/j_basalt_bb.json" \
  --engram-dest "$ENGRAM_PATH" --engram-jobs "$R/j_basalt_eng.json" \
  --engram-size-out "$R/engram_size.txt" > $R/fetch.log 2>&1 || { st "basalt_jobs FAILED"; exit 1; }
ENGRAM_SIZE=$(cat $R/engram_size.txt)

{ bash $V/build.sh > $R/basalt_build.log 2>&1 && st "basalt built" || st "basalt build FAILED ($(tail -1 $R/build.txt 2>/dev/null))"; } &
if [ "$UNCENSOR" = 1 ]; then
  { curl -sfL --retry 5 ${HF_HDR:+-H @$HF_HDR} -o /content/patch/abliteration-edit.npz "$ABLIT_URL" \
    && sha256sum /content/patch/abliteration-edit.npz | grep -q "^$ABLIT_SHA" \
    && st "abliteration edit fetched" || st "abliteration edit FAILED"; } &
fi
{ python3 $V/pfetch.py $R/j_basalt_bb.json --workers $WORKERS --chunk-mb 64 --tag basalt-bb --log $R/fetch.log \
  && st "backbone fetched ($MODEL_QUANT -> $MODEL_PATH)" || st "backbone FAILED"; } &
wait
seen FAILED && { st "fetch/build FAILED, stopping"; exit 1; }

if [ "$UNCENSOR" = 1 ]; then
  python3 -I $V/abliterate_basalt.py "$MODEL_PATH" /content/patch/abliteration-edit.npz "$D/abliterate" \
    --scale "$UNCENSOR_SCALE" > $R/patch.log 2>&1 \
    && st "abliteration applied (scale=$UNCENSOR_SCALE)" || { st "abliteration FAILED ($(tail -1 $R/patch.log 2>/dev/null))"; exit 1; }
fi

# Create 0-block sparse engram.bin so PleReader::open succeeds before Engram download
python3 -I -c "import os; f=os.open('$ENGRAM_PATH', os.O_WRONLY|os.O_CREAT, 0o644); os.ftruncate(f, int('$ENGRAM_SIZE')); os.close(f)"

bash $V/serve.sh
st "server started"
for k in $(seq 1 120); do
  curl -sf -o /dev/null -H "$(hdr)" http://127.0.0.1:$PORT/health && { st "basalt http ready"; break; }
  [ -d /proc/$(cat $R/server.pid) ] || { st "server died ($(tail -5 $R/server.log 2>/dev/null | tr '\n' ' '))"; exit 1; }
  sleep 0.5
done

# Eager weight load (with punch-on-read reclaiming backbone storage before Engram download)
curl -sf --max-time 900 -X POST -H "$(hdr)" -H "Content-Type: application/json" -d '{}' \
  http://127.0.0.1:$PORT/load > $R/load.json 2>&1 && st "weights loaded" || { st "weights load FAILED ($(tail -5 $R/server.log 2>/dev/null | tr '\n' ' '))"; exit 1; }

VRAM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
if [ "${VRAM:-0}" -lt 60000 ]; then
  st "GUARD: VRAM ${VRAM} MiB after /load FAILED"; exit 1
fi
if [ -n "${BASALT_PUNCH_FILE:-}" ]; then
  python3 $V/basalt_jobs.py --backbone-dest "$MODEL_PATH" --punch-sections dspark/gguf dense > $R/punch.txt 2>&1 || true
fi
st "pin: VRAM ${VRAM} MiB, model $(du -sh "$MODEL_PATH" | awk '{print $1}'), mode=$STAGE_MODE"

# Fill Engram table (into /mnt/ram/engram.bin for ram+ram, or /content/engram.bin for ram+ssd / ssd+ssd)
if [ "$STAGE_MODE" = "ram+ram" ]; then
  ( for k in $(seq 1 600); do [ -f $R/READY ] && exit 0; a=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo)
      [ "$a" -lt 1200 ] && { for p in $(pgrep -f "[p]fetch.py"); do kill $p; done; st "GUARD: ${a} MiB RAM available, engram fetch stopped FAILED"; exit 0; }; sleep 2; done ) &
else
  DISK_NOW_GIB=$(df -BG --output=avail /content | tail -1 | tr -dc '0-9')
  st "disk before engram: /content ${DISK_NOW_GIB}GiB free"
fi
python3 $V/pfetch.py $R/j_basalt_eng.json --workers $WORKERS --chunk-mb 64 --tag basalt-eng --log $R/fetch.log \
  && st "engram done ($ENGRAM_PATH)" || { st "engram FAILED"; exit 1; }
free -b > $R/free_after_engram.txt
touch $R/READY

curl -sf --max-time 180 -H "$(hdr)" http://127.0.0.1:$PORT/v1/chat/completions -H "Content-Type: application/json" \
  -d "{\"model\":\"$MODEL_ALIAS\",\"messages\":[{\"role\":\"user\",\"content\":\"Say OK.\"}],\"max_tokens\":8,\"temperature\":0,\"reasoning_effort\":\"low\"}" \
  > $R/warmup.json && st "first token (warmup done)" || { st "warmup FAILED"; exit 1; }

setsid nohup python3 $V/gate.py 18080 $PORT > $R/gate.log 2>&1 < /dev/null &
for k in $(seq 1 50); do grep -q "gate open" $R/gate.log 2>/dev/null && break; sleep 0.1; done
grep -q "gate open" $R/gate.log && st "gate open 18080" || st "gate FAILED"

# VM-side hard-cap backstop if the local host sleeps or loses power
if [ -n "${MAX_HOURS:-}" ]; then
  CAP_S=$(python3 -c "print(max(600, int(float('$MAX_HOURS') * 3600) + 300))" 2>/dev/null || echo 21900)
  setsid bash -c "sleep $CAP_S; pkill -f '[g]ate.py' || true; kill \$(cat $R/server.pid 2>/dev/null) 2>/dev/null || true; rm -rf /content/secrets" > /dev/null 2>&1 < /dev/null &
fi
