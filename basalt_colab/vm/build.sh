#!/bin/bash
# Fetches prebuilt patched Basalt sm_120 tarball (fast path) or builds from source on Colab G4.
# Supports BASALT_REF override (tag / branch / commit SHA) to test upstream Basalt commits.
set -uo pipefail
R=/content/res; V=/content/vm; S=/content/src/basalt; D=/content/basalt
st() { echo "$(date +%T) $*" >> $R/build.txt; }
mkdir -p $R /content/src $D
st "basalt build start"

eval "$(python3 - "$V/manifest.json" <<'EOF'
import json, shlex, sys
m = json.load(open(sys.argv[1]))
for env_k, json_k in (
    ("BASALT_REPO", "basalt_repo"),
    ("PIN_TAG", "basalt_tag"),
    ("PIN_COMMIT", "basalt_commit"),
    ("MANIFEST_PREBUILT_URL", "prebuilt_url"),
    ("MANIFEST_PREBUILT_SHA", "prebuilt_sha256"),
):
    print(f"{env_k}={shlex.quote(str(m.get(json_k, '')))}")
EOF
)"

REF="${BASALT_REF:-$PIN_TAG}"
PREBUILT_URL="${BASALT_PREBUILT_URL:-}"
PREBUILT_SHA="${BASALT_PREBUILT_SHA256:-}"
# Use manifest prebuilt tarball only when running pinned version (no BASALT_REF override)
if [ -z "${BASALT_REF:-}" ] && [ -z "$PREBUILT_URL" ] && [ -n "$MANIFEST_PREBUILT_URL" ]; then
  PREBUILT_URL="$MANIFEST_PREBUILT_URL"
  PREBUILT_SHA="$MANIFEST_PREBUILT_SHA"
fi

if [ ! -d "$S/.git" ]; then
  if git clone -q --depth 1 --branch "$REF" "$BASALT_REPO" "$S" 2>/dev/null; then
    :
  else
    git clone -q --filter=blob:none --no-checkout "$BASALT_REPO" "$S" \
      && git -C "$S" fetch -q --depth 1 origin "$REF" \
      && git -C "$S" checkout -q FETCH_HEAD \
      || { st "basalt clone FAILED (ref=$REF)"; exit 1; }
  fi
fi

if [ -z "${BASALT_REF:-}" ]; then
  [ "$(git -C "$S" rev-parse HEAD)" = "$PIN_COMMIT" ] || { st "basalt commit mismatch $(git -C $S rev-parse HEAD) FAILED"; exit 1; }
else
  st "custom BASALT_REF=$BASALT_REF -> $(git -C "$S" rev-parse --short HEAD)"
fi

python3 $V/patch_basalt.py "$S" >> $R/build.txt 2>&1 || { st "basalt patch FAILED ($(tail -1 $R/build.txt 2>/dev/null))"; exit 1; }

if [ -n "$PREBUILT_URL" ] && [ -z "${BASALT_REF:-}" ]; then
  mkdir -p "$S/bin"
  ARCHIVE="$R/prebuilt.pkg"
  if curl -sfL --retry 2 --max-time 60 -o "$ARCHIVE" "$PREBUILT_URL" \
    && { [ -z "$PREBUILT_SHA" ] || sha256sum "$ARCHIVE" | grep -q "^$PREBUILT_SHA"; }; then
    if tar -tzf "$ARCHIVE" >/dev/null 2>&1; then
      tar -xzf "$ARCHIVE" -C "$D"
      [ -f "$D/basalt" ] && mv -f "$D/basalt" "$S/bin/basalt"
      [ -f "$D/bin/basalt" ] && mv -f "$D/bin/basalt" "$S/bin/basalt"
    else
      mv -f "$ARCHIVE" "$S/bin/basalt"
    fi
    rm -f "$ARCHIVE"
    if [ -f "$S/bin/basalt" ] && chmod +x "$S/bin/basalt" && "$S/bin/basalt" --help >/dev/null 2>&1; then
      st "basalt prebuilt binary ready"
    else
      rm -f "$S/bin/basalt"
      st "basalt prebuilt binary unusable, falling back to source build"
    fi
  else
    rm -f "$ARCHIVE"
    st "basalt prebuilt unavailable, falling back to source build"
  fi
fi

if [ ! -x "$S/bin/basalt" ]; then
  command -v cmake > /dev/null || pip install -q cmake > $R/build_cmake.log 2>&1
  for cand in /usr/local/cuda-13.2/bin /usr/local/cuda-13.1/bin /usr/local/cuda-13.0/bin /usr/local/cuda/bin; do
    [ -x "$cand/nvcc" ] && { export PATH="$cand:$PATH"; break; }
  done
  cmake -S "$S" -B "$S/build" -DCMAKE_BUILD_TYPE=Release -DBASALT_MODELS=deepseek41 -DBUILD_TESTING=OFF \
    > $R/build_cmake.log 2>&1 || { st "basalt cmake FAILED: $(tail -3 $R/build_cmake.log | tr '\n' ' ')"; exit 1; }
  JOBS=$(nproc); [ "$JOBS" -gt 16 ] && JOBS=16
  nice -n 5 cmake --build "$S/build" -j "$JOBS" --target basalt \
    > $R/build_make.log 2>&1 || { st "basalt make FAILED: $(grep -m3 -i error $R/build_make.log | tr '\n' ' ')"; exit 1; }
fi

pip install -q -r "$S/requirements.txt" > $R/build_pip.log 2>&1 || { st "basalt pip FAILED"; exit 1; }

if [ ! -x "$D/abliterate" ] && [ -f "$V/abliterate.c" ]; then
  gcc -O3 -fopenmp -DABLITERATE_STANDALONE "$V/abliterate.c" -lm -o "$D/abliterate" \
    >> $R/build.txt 2>&1 || { st "standalone abliterate build FAILED"; exit 1; }
fi
st "basalt build done"
