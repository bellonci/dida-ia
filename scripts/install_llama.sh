#!/usr/bin/env bash
set -euo pipefail
dida_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$dida_root"
llama_revision=aa94f2086147cd4bfbcff818ce998dffce5fcb39
llama_source="$dida_root/.tools/llama.cpp"
if [[ ! -d "$llama_source/.git" ]]; then
  mkdir -p "$dida_root/.tools"
  git clone --depth 1 https://github.com/ggml-org/llama.cpp.git "$llama_source"
fi
if [[ "$(git -C "$llama_source" rev-parse HEAD)" != "$llama_revision" ]]; then
  if [[ -n "$(git -C "$llama_source" status --porcelain)" ]]; then
    echo 'Cache de llama.cpp tem alterações; preserve-as antes de atualizar.' >&2
    exit 1
  fi
  git -C "$llama_source" fetch --depth 1 origin "$llama_revision"
  git -C "$llama_source" checkout --detach "$llama_revision"
fi
.venv/bin/cmake -S "$llama_source" -B "$llama_source/build" -G Ninja \
  -DCMAKE_MAKE_PROGRAM="$dida_root/.venv/bin/ninja" \
  -DCMAKE_C_COMPILER=/usr/bin/gcc -DCMAKE_CXX_COMPILER=/usr/bin/g++ \
  -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=OFF -DLLAMA_BUILD_TESTS=OFF \
  -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF -DLLAMA_BUILD_SERVER=ON -DBUILD_SHARED_LIBS=OFF
.venv/bin/cmake --build "$llama_source/build" --target llama-server -j 4
"$llama_source/build/bin/llama-server" --version
