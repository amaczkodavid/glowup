#!/usr/bin/env bash
# Keep the *_zig.txt deliverables byte-identical to the compiled Zig modules.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT/runtime"
for pair in "tensor.zig:tensor_zig.txt" "gpu_coordinator.zig:gpu_coordinator_zig.txt" "futhark_bindings.zig:futhark_bindings_zig.txt"; do
  src="${pair%%:*}"; dst="${pair##*:}"
  if [ -f "$src" ]; then cp "$src" "$dst"; else cp "$dst" "$src"; fi
  echo "synced $src <-> $dst"
done
