#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
npx eslint . --fix || true
if command -v zig >/dev/null 2>&1; then (cd runtime && zig fmt .); fi
if command -v futhark >/dev/null 2>&1; then futhark fmt futhark_kernels/kernels.fut; fi
bash ci/export_txt.sh
