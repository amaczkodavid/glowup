#!/usr/bin/env bash
# Build the Futhark kernels into a C-ABI shared library consumed by the Zig
# runtime (`runtime/futhark_bindings.zig`).
#
#   ./build.sh              # auto-detect the best available backend
#   ./build.sh cuda         # force a specific backend (c|multicore|cuda|opencl)
#   FUTHARK=/path/to/futhark ./build.sh
#
# Pinned toolchain versions (reproducible builds):
#   futhark 0.25.27
#   zig     0.15.1
#   clang   >= 16

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

FUTHARK="${FUTHARK:-futhark}"
CC="${CC:-cc}"
BACKEND="${1:-auto}"

if ! command -v "$FUTHARK" >/dev/null 2>&1; then
  echo "futhark compiler not found; the Zig runtime will fall back to its native kernels." >&2
  exit 0
fi

detect_backend() {
  if command -v nvcc >/dev/null 2>&1; then echo cuda; return; fi
  if [ -f /usr/lib/x86_64-linux-gnu/libOpenCL.so ] || [ -f /usr/lib/libOpenCL.so ]; then echo opencl; return; fi
  echo multicore
}

if [ "$BACKEND" = "auto" ]; then
  BACKEND="$(detect_backend)"
fi

echo ">> futhark version: $("$FUTHARK" --version | head -n1)"
echo ">> backend: $BACKEND"

"$FUTHARK" "$BACKEND" --library kernels.fut

EXTRA_LIBS=""
case "$BACKEND" in
  cuda)   EXTRA_LIBS="-lcuda -lcudart -lnvrtc" ;;
  opencl) EXTRA_LIBS="-lOpenCL" ;;
  multicore) EXTRA_LIBS="-lpthread" ;;
  c)      EXTRA_LIBS="" ;;
esac

echo ">> compiling libfutkernels.so"
"$CC" -O3 -fPIC -shared -std=c11 kernels.c -o libfutkernels.so -lm ${EXTRA_LIBS}

echo ">> running futhark's own test suite"
"$FUTHARK" test --backend="$BACKEND" kernels.fut

echo ">> done: $(pwd)/libfutkernels.so"
echo "   header: $(pwd)/kernels.h"
