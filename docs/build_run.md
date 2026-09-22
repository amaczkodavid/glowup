# Build & run

## Pinned toolchain

| Component | Version |
|-----------|---------|
| Node.js   | 22.x    |
| PostgreSQL| 16.x    |
| Zig       | 0.15.1  |
| Futhark   | 0.25.27 |
| CUDA      | 12.x (optional) |
| clang     | ≥ 16 (Futhark C output) |

All versions are pinned so builds are reproducible; `package.json` pins exact
npm dependency versions (no ranges).

## 1. Optimizer + control plane (Next.js / PostgreSQL)

```bash
cp .env.example .env             # DATABASE_URL=postgresql://postgres:postgres@127.0.0.1:5432/app_db
npm install
npx drizzle-kit push             # create runs, candidates, counterexamples, search_events, eqsat_runs, verifications
npm run build
npm run start                    # http://localhost:3000
```

Health check: `curl -s localhost:3000/api/health` → `{"ok":true}`.

Kick off a synthesis run from the CLI:

```bash
curl -sX POST localhost:3000/api/runs \
  -H 'content-type: application/json' \
  -d '{"taskId":"isolate_lsb","config":{"rounds":3,"mcmcMs":1500,"isa":"aarch64"}}' | jq '.report.best'
```

Run equality saturation + ILP extraction:

```bash
curl -sX POST localhost:3000/api/eqsat -H 'content-type: application/json' \
  -d '{"programId":"matchain"}' | jq '{before:.before.cost, after:.after.cost, expr:.after.text}'
```

Verify a hand-written rewrite:

```bash
curl -sX POST localhost:3000/api/verify -H 'content-type: application/json' \
  -d '{"taskId":"abs","program":"abs r0, r0","width":8}' | jq '.result.status'
```

## 2. Zig runtime

```bash
cd runtime
zig build                     # static library + demo
zig build test                # allocators, lock-free, tensors, GPU, bindings
zig build demo                # end-to-end demonstration
zig build bench -Doptimize=ReleaseFast
```

Cross compilation (the runtime is portable across x86-64 and AArch64):

```bash
zig build -Dtarget=aarch64-linux-gnu -Doptimize=ReleaseFast
zig build -Dtarget=x86_64-linux-gnu  -Doptimize=ReleaseSafe
```

`runtime/tensor.zig`, `runtime/gpu_coordinator.zig` and
`runtime/futhark_bindings.zig` are also shipped verbatim as
`tensor_zig.txt`, `gpu_coordinator_zig.txt` and `futhark_bindings_zig.txt`;
`ci/export_txt.sh` keeps the two copies byte-identical.

### GPU setup

* CUDA 12 with `libcudart.so` on the loader path enables the `.cuda` backend;
  the coordinator resolves symbols lazily with `std.DynLib`, so a CPU-only
  machine transparently uses the deterministic `.host` backend.
* For multi-GPU peer-to-peer reductions ensure `nvidia-smi topo -m` reports
  `NV#`/`PIX` links and that `cudaDeviceEnablePeerAccess` succeeds (the
  coordinator records this in `peer_enabled`).
* Force the host backend for reproducible CI runs: `CUDA_VISIBLE_DEVICES= zig build test`.

## 3. Futhark kernels

```bash
cd futhark_kernels
./build.sh            # auto-detect cuda → opencl → multicore
./build.sh c          # portable, no accelerator required
```

Outputs `kernels.h`, `kernels.c` and `libfutkernels.so`, then runs
`futhark test kernels.fut`. Point the runtime at it with

```bash
LD_LIBRARY_PATH=$PWD zig build demo
```

## 4. Solver setup

The SMT layer is self-contained: symbolic execution bit-blasts to CNF and the
bundled DPLL engine (two-watched literals, chronological backtracking) decides
the miter. There is **no external solver dependency**, so verification works in
a hermetic build. Budgets are configurable per request
(`verifyMs`, verification width); when a budget is exhausted the layer reports
`budget-exhausted` and the orchestrator falls back to exhaustive bounded
enumeration, which is complete at the verification width.

## 5. CI

```bash
ci/ci.sh          # typecheck → lint → build → optimizer self-test → zig test → futhark test
ci/fmt.sh         # prettier/eslint --fix + zig fmt + futhark fmt
ci/export_txt.sh  # regenerate the *_zig.txt deliverables
```
