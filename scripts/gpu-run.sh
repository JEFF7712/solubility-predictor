#!/usr/bin/env bash
# Run a project module on the GPU venv (.venv-gpu, CUDA torch installed via uv).
#
# Two NixOS quirks this works around:
#   1. The flake devshell's PYTHONPATH points at nix-store site-packages and
#      would shadow the venv's CUDA torch, so we clear it (env -u PYTHONPATH).
#   2. The uv-installed numpy/torch wheels need libz (zlib) and libstdc++ at
#      runtime, which the bare environment may not provide, so we add them to
#      LD_LIBRARY_PATH along with the GPU driver libs.
#
# Usage (from repo root): scripts/gpu-run.sh -m src.train
set -euo pipefail

repo_root="$(cd "$(dirname "$0")/.." && pwd)"
zlib_dir="$(dirname "$(ls /nix/store/*-zlib-*/lib/libz.so.1 2>/dev/null | head -1)")"
gcc_dir="$(dirname "$(ls /nix/store/*-gcc-*-lib/lib/libstdc++.so.6 2>/dev/null | head -1)")"

cd "$repo_root"
exec env -u PYTHONPATH \
  LD_LIBRARY_PATH="${zlib_dir}:${gcc_dir}:/run/opengl-driver/lib" \
  "$repo_root/.venv-gpu/bin/python" "$@"
