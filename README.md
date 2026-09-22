# Superoptimising Synthesis Stack

An end-to-end system that **synthesises, verifies and executes** ultra-optimised
code by combining stochastic superoptimisation, reinforcement-learning assembly
synthesis, equality saturation with ILP extraction, formal verification and an
ultra-low-level Zig runtime with GPU collectives and Futhark integrations.

```
optimizer (TypeScript, src/lib/superopt)
  isa.ts          retargetable ISA abstraction (x86-64, AArch64, pluggable)
  machine.ts      CPU/memory model, faults, dependence + liveness analysis
  perf.ts         port/latency cost model + online-learned runtime predictor
  cost.ts         cost(R;T) = we·eq(R;T) + wp·perf(R) with ULP/bit metrics
  testcases.ts    dynamic test database (random, adversarial, counterexamples)
  mcmc.ts         Metropolis–Hastings superoptimiser (5 proposal kernels)
  mcts.ts         PUCT/MCTS synthesis with online policy + value models
  enumerative.ts  bidirectional meet-in-the-middle enumeration
  smt.ts          symbolic execution → CNF bit-blasting → DPLL equivalence
  egraph.ts       e-graph, equality saturation, exact ILP extraction
  hybrid.ts       parallel orchestrator with a shared knowledge store
  tasks.ts        benchmark suite (11 kernels, integer + floating point)

runtime (Zig)      allocators, lock-free structures, SIMD/COW tensors,
                   cache-blocked matmul, CUDA-stream GPU collectives,
                   Futhark C-ABI bindings, synthesised-kernel dispatch
futhark_kernels    dot/saxpy/matmul/softmax/conv1d/stencil/FFT/linear-relu
docs               architecture.md, build_run.md, apis.md
ci                 ci.sh, fmt.sh, export_txt.sh, selftest.mjs
```

The Next.js app is the control plane: launch hybrid searches, watch the engines
work, inspect the Pareto front, prove rewrites with the bundled SMT solver, run
equality saturation with the ILP model rendered in full, browse the persisted
counterexample corpus and read every source artefact.

Quick start:

```bash
cp .env.example .env
npm install && npx drizzle-kit push && npm run build && npm run start
node ci/selftest.mjs          # optimizer self-test (search + proofs + eqsat)
cd runtime && zig build test  # runtime unit tests
cd futhark_kernels && ./build.sh
```

See [docs/build_run.md](docs/build_run.md), [docs/architecture.md](docs/architecture.md)
and [docs/apis.md](docs/apis.md).
