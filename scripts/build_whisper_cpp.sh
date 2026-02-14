#!/usr/bin/env bash
set -euo pipefail
work_dir=${1:?usage: build_whisper_cpp.sh EXTERNAL_WORK_DIRECTORY}
mkdir -p "$work_dir"
checkout_pinned() {
  local target=$1 remote=$2 revision=$3
  if [ -e "$target" ] && [ ! -d "$target/.git" ]; then
    echo "Use a fresh directory or the matching clean git checkout" >&2
    return 1
  fi
  if [ ! -d "$target/.git" ]; then
    git init "$target"
    git -C "$target" remote add origin "$remote"
  fi
  test "$(git -C "$target" remote get-url origin)" = "$remote"
  test -z "$(git -C "$target" status --porcelain)"
  git -C "$target" fetch --depth 1 origin "$revision"
  git -C "$target" checkout --detach FETCH_HEAD
}
checkout_pinned "$work_dir/whisper.cpp" https://github.com/ggml-org/whisper.cpp.git 231638a0717b7822784297e5980965ffcec05810
checkout_pinned "$work_dir/whisper" https://github.com/openai/whisper.git 86098128c0b4f24f0e2aa2994de830614b474227
cmake -S "$work_dir/whisper.cpp" -B "$work_dir/whisper.cpp/build" \
  -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=OFF -DWHISPER_CURL=OFF -DWHISPER_BUILD_SERVER=ON
cmake --build "$work_dir/whisper.cpp/build" --target whisper-cli whisper-server whisper-quantize -j2
test -f "$work_dir/whisper/whisper/assets/mel_filters.npz"
