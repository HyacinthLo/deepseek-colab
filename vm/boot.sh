#!/bin/bash
# Cold start on a Colab G4: weights into RAM, llama.cpp built, server up, VRAM copies freed, Engram filled, then the gate opens port 18080.
set -uo pipefail
: "${CTX:?} ${UBATCH:?} ${MOE_CACHE_MIB:?} ${DYN:?} ${MMPROJ:?}"
export WEIGHTS=${WEIGHTS:-ista} RUNG=IQ2_XXS PORT=18081 DRAFT=${DRAFT:-1} UNCENSOR=${UNCENSOR:-1} WORKERS=${WORKERS:-64}
export LLAMA_DSV41_CED=1 LLAMA_DSV41_CED_SPEC=1
export PIN=1 PIN_MAPPED=1 CACHE_RAM=1024 LLAMA_MOE_PAD_ZERO=1 LLAMA_MOE_CACHE_PIN_PREFILL=1 LLAMA_MOE_CACHE_DYN=$DYN
export LLAMA_MOE_CACHE_PROFILE=/content/vm/profile-$RUNG.txt LLAMA_MOE_CACHE_RESIDENT=/content/res/resident.txt
export CPU_RE=$(python3 -c "print(r'blk\.(%s)\.ffn_(up|gate|down)_exps\.weight=CPU' % '|'.join(map(str, range(40))))")
[ -s /content/secrets/hf_hdr ] && export HF_HDR=/content/secrets/hf_hdr
M=/mnt/ram/m; R=/content/res; V=/content/vm
mkdir -p $R /mnt/ram /content/model /content/llama
if [ "$WEIGHTS" = unsloth ]; then
  HF=https://huggingface.co/unsloth/DeepSeek-V4.1-Flash-1017-GGUF/resolve/dfcee0ec8d6b874c6663eca0fc55d90ab32534c8/UD-IQ1_M
  NAME=DeepSeek-V4.1-Flash-1017-UD-IQ1_M
  BACKBONE_SHARDS="1 2 3 4"; NSHARDS=5
  ENGRAM=DeepSeek-V4.1-Flash-1017-UD-IQ1_M-00005-of-00005.gguf
  ENGRAM_SIZE=110595290592
  PUNCH_GLOB="$M/$NAME-0000[2-4]-of-00005.gguf"
  BACKBONE_MSG="backbone fetched (unsloth)"
elif [ "$WEIGHTS" = ista ]; then
  HF=https://huggingface.co/ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF/resolve/a419d74cd28d5b9f8c5e2b9077d11a609c17bc32
  NAME=DeepSeek-V4.1-Flash-1017-GSQ-RCO-$RUNG
  BACKBONE_SHARDS="1 2 3 4 5"; NSHARDS=6
  ENGRAM=DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf
  ENGRAM_SIZE=110595290816
  PUNCH_GLOB="$M/$NAME-0000[2-5]-of-00006.gguf"
  BACKBONE_MSG="backbone fetched"
else
  echo "unknown WEIGHTS=$WEIGHTS FAILED" >> $R/timeline.txt; exit 1
fi
export MODEL_FIRST=/content/model/$NAME-00001-of-0000$NSHARDS.gguf
ABLIT_URL=https://huggingface.co/mrfakename/DeepSeek-V4.1-Flash-1017-abliterated/resolve/4433164758a8045b1b915b7644cce97b9a0e274c/abliteration-edit.npz
ABLIT_SHA=83273cdbe06e65ad2852dd09e8d976a0e8ce3209788938add325989c20b63b02
DRAFT_SHA=6ba6ac1b331c540691a209e7fc97dec2a24e295acb1e755c84c7d24a4373927a
rm -f $R/resident.txt $R/READY
T0=$(date +%s)
st() { echo "$(( $(date +%s) - T0 ))s $(date +%T) $*" >> $R/timeline.txt; }
seen() { grep -qE "$1" $R/timeline.txt; }
hdr() { echo "Authorization: Bearer $(cat /content/secrets/key)"; }
st "boot start weights=$WEIGHTS ctx=$CTX ubatch=$UBATCH cache=$MOE_CACHE_MIB dyn=$DYN images=$MMPROJ draft=$DRAFT uncensored=$UNCENSOR"
mountpoint -q /mnt/ram || mount -t tmpfs -o size=$(( $(awk '/MemTotal/ {print $2}' /proc/meminfo) / 1000000 - 6 ))G tmpfs /mnt/ram
mkdir -p $M

fetch() {  # $1 = backbone|engram|draft
  python3 $V/jobs.py $1 $M $R/j_$1.json >> $R/fetch.log 2>&1 && python3 $V/pfetch.py $R/j_$1.json --workers $WORKERS --chunk-mb 64 --tag $1 --log $R/fetch.log
}

{ bash $V/build.sh > $R/llama.log 2>&1 && st "llama.cpp built" || st "llama.cpp build FAILED"; } &
if [ "$UNCENSOR" = 1 ]; then
  if [ "$WEIGHTS" = unsloth ]; then
    { mkdir -p /content/patch && curl -sfL --retry 5 ${HF_HDR:+-H @$HF_HDR} -o /content/patch/abliteration-edit.npz "$ABLIT_URL" \
      && sha256sum /content/patch/abliteration-edit.npz | grep -q "^$ABLIT_SHA" \
      && st "abliteration edit fetched" || st "abliteration edit FAILED"; } &
  else
    { python3 -I $V/mkpatch.py /content/patch > $R/patch.log 2>&1 && st "uncensor patch fetched" || st "uncensor patch FAILED"; } &
  fi
fi
[ "$MMPROJ" = 1 ] && { MMPROJ_FETCH_ONLY=1 timeout 300 bash $V/serve.sh > $R/mmproj.log 2>&1 && st "image encoder fetched" || st "image encoder FAILED"; } &
{ fetch backbone && st "$BACKBONE_MSG" || st "backbone FAILED"; } &
[ "$DRAFT" = 1 ] && { fetch draft && sha256sum $M/dspark.gguf | grep -q "^$DRAFT_SHA" && st "draft fetched" || st "draft FAILED"; } &
wait
seen FAILED && { st "fetch FAILED, stopping"; exit 1; }
( for k in $(seq 1 900); do [ -f $R/READY ] && exit 0; a=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo)
    [ "$a" -lt 1200 ] && { for p in $(pgrep -f "[p]fetch.py"); do kill $p; done; st "GUARD: ${a} MiB RAM available, fetch stopped FAILED"; exit 0; }; sleep 2; done ) &
for i in $BACKBONE_SHARDS; do ln -sf $M/$NAME-0000$i-of-0000$NSHARDS.gguf /content/model/$NAME-0000$i-of-0000$NSHARDS.gguf; done
ln -sf $M/$ENGRAM /content/model/$NAME-0000$NSHARDS-of-0000$NSHARDS.gguf
python3 -I -c "import os; f=os.open('$M/$ENGRAM', os.O_WRONLY|os.O_CREAT, 0o644); os.ftruncate(f, $ENGRAM_SIZE); os.close(f)"
for k in 1 2 3 4 5; do  # the sparse Engram needs its real header before the server starts; the table itself is filled after the punch
  curl -sfL --retry 3 ${HF_HDR:+-H @$HF_HDR} -r 0-1048575 -o $R/engram_head.bin "$HF/$ENGRAM" && [ "$(head -c 4 $R/engram_head.bin)" = GGUF ] && break
  sleep 3
done
[ "$(head -c 4 $R/engram_head.bin 2>/dev/null)" = GGUF ] || { st "engram header FAILED"; exit 1; }
dd if=$R/engram_head.bin of=$M/$ENGRAM conv=notrunc status=none
python3 -I $V/checksizes.py /content/model > $R/sizes.txt 2>&1 && st "sizes ok" || { st "sizes FAILED"; exit 1; }
if [ "$UNCENSOR" = 1 ]; then
  if [ "$WEIGHTS" = unsloth ]; then
    python3 -I $V/abliterate.py /content/model /content/patch/abliteration-edit.npz /content/llama/bin/abliterate > $R/patch.log 2>&1 && st "abliteration applied" || { st "abliteration FAILED"; exit 1; }
  else
    python3 -I $V/applypatch.py /content/model /content/patch >> $R/patch.log 2>&1 && st "uncensored tensors applied" || { st "patch apply FAILED"; exit 1; }
  fi
fi
bash $V/serve.sh
st "server started"
for k in $(seq 1 900); do
  curl -sf -o /dev/null -H "$(hdr)" http://127.0.0.1:$PORT/health && { st "healthy"; break; }
  [ -d /proc/$(cat $R/server.pid) ] || { st "server died"; exit 1; }
  sleep 1
done
VRAM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
grep -E "model buffer size|KV buffer size|compute buffer size|MoE cache" $R/server.log > $R/buffers.txt
# punching a tensor the server still reads from RAM would zero live weights, so check that the weights went to the GPU first
if [ "$VRAM" -lt 60000 ] || ! grep -q "CUDA0 model buffer size" $R/buffers.txt; then st "GUARD: VRAM ${VRAM} MiB, no CUDA model buffer FAILED"; exit 1; fi
[ -s $R/resident.txt ] && grep -q "prefilled" $R/server.log || { st "GUARD: no resident expert list FAILED"; exit 1; }
PUNCH_FILES="$PUNCH_GLOB"
python3 -I $V/punch.py "$CPU_RE" $PUNCH_FILES > $R/punch.txt 2>&1 && st "punched $(cat $R/punch.txt), VRAM ${VRAM} MiB" || { st "punch FAILED"; exit 1; }
python3 $V/punch.py --experts $R/resident.txt $PUNCH_FILES > $R/punch_res.txt 2>&1 && st "$(cat $R/punch_res.txt)" || { st "resident punch FAILED"; exit 1; }
if [ "$DRAFT" = 1 ] && [ "$(grep -c 'CUDA0 model buffer size' $R/buffers.txt)" -ge 2 ]; then
  python3 -I $V/punch.py 'NO_CPU_TENSOR=CPU' $M/dspark.gguf > $R/punch_draft.txt 2>&1 && st "draft punched $(cat $R/punch_draft.txt)"
fi
python3 $V/punch.py --pin-cold $R/pin_request.tmp $R/resident.txt $PUNCH_FILES > $R/pin.txt 2>&1 && mv $R/pin_request.tmp $R/pin_request
for k in $(seq 1 600); do grep -q DONE $R/pin_result 2>/dev/null && break; sleep 1; done
st "pin: $(tail -1 $R/pin_result 2>/dev/null)"
fetch engram && st "engram done" || { st "engram FAILED"; exit 1; }
free -b > $R/free_after_engram.txt
touch $R/READY
curl -sf -H "$(hdr)" http://127.0.0.1:$PORT/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Say OK."}],"max_tokens":8,"reasoning_effort":"low"}' > $R/warmup.json && st "first token (warmup done)" || { st "warmup FAILED"; exit 1; }
setsid nohup python3 $V/gate.py 18080 $PORT > $R/gate.log 2>&1 < /dev/null &
for k in $(seq 1 50); do grep -q "gate open" $R/gate.log 2>/dev/null && break; sleep 0.1; done
grep -q "gate open" $R/gate.log && st "gate open 18080" || st "gate FAILED"
