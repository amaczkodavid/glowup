#!/usr/bin/env bash
# Full continuous-integration pipeline: control plane, optimizer self-test,
# Zig runtime and Futhark kernels. Every stage is fail-fast.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "== 1/6 typegen + typecheck =="
npx next typegen
npm exec tsc -- --noEmit --pretty false

echo "== 2/6 lint =="
npm run lint

echo "== 3/6 production build =="
npm run build

echo "== 4/6 optimizer self-test =="
node ci/selftest.mjs

echo "== 5/6 zig runtime =="
if command -v zig >/dev/null 2>&1; then
  ( cd runtime && zig fmt --check . && zig build test )
else
  echo "zig not installed - skipping runtime tests"
fi

echo "== 6/6 futhark kernels =="
if command -v futhark >/dev/null 2>&1; then
  ( cd futhark_kernels && ./build.sh c )
else
  echo "futhark not installed - skipping kernel build"
fi

echo "CI OK"
