# Architecture

The system is a closed loop between **search**, **verification** and
**execution**. Code is represented at three levels, each with its own optimiser,
and all three feed a single runtime.

```
        high-level tensor graph            core kernels / algorithms
                  │                                   │
        ┌─────────▼──────────┐              ┌─────────▼───────────┐
        │  eqsat_ilp         │              │  hybrid_search      │
        │  e-graph +         │              │  ├─ assembly_superopt (MCMC)
        │  equality          │              │  ├─ rl_asm_synth      (MCTS/RL)
        │  saturation +      │              │  ├─ enumerative MITM  │
        │  ILP extraction    │              │  └─ smt (symbolic exec + SAT)
        └─────────┬──────────┘              └─────────┬───────────┘
                  │  optimal DAG                      │  proven rewrite
                  └───────────────┬───────────────────┘
                                  ▼
                    ┌─────────────────────────────┐
                    │  runtime (Zig)              │
                    │  allocators · lock-free ·   │
                    │  SIMD tensors · GPU         │
                    │  collectives · Futhark FFI  │
                    └─────────────────────────────┘
```

## Subsystems

### `optimizer/isa` — retargetable ISA abstraction (`src/lib/superopt/isa.ts`)

* A **virtual core** opcode table carries semantics-independent metadata:
  operand roles, commutativity, flag effects, immediate ranges, latency,
  reciprocal throughput, µops, execution ports and faulting behaviour.
* Concrete backends implement `IsaBackend`: register naming, native printing
  (x86-64 AT&T, AArch64), instruction size, per-opcode latency overrides and
  issue width. `registerIsa()` adds a new target without touching the search.
* `parseProgram` / `formatProgram` round-trip the textual virtual-core form;
  `disassemble` emits native assembly for either backend.
* `validate()` enforces operand-range and register-file legality — the search
  can never emit a malformed instruction.

### `optimizer/common` — machine model (`machine.ts`, `types.ts`)

* CPU state: 8 GPRs, 8 FPRs, 4×4-lane vector registers, byte-addressable
  memory with bounds *and* alignment checks, x86-style flags (ZF/SF/CF/OF),
  fault state (`segfault`, `unaligned`, `divide-by-zero`, `trap`, `timeout`)
  and a retired-instruction counter.
* The datapath width is a parameter: 32 bits for execution and testing, 8 bits
  (or any width ≤ 16) for bounded verification. All opcodes are implemented for
  every width, including width-relative `clz`/`ctz`/`popcnt`/`rol`/`bswap`.
* `effects()` derives read/write sets per instruction; `commutesInOrder()`
  answers reordering legality (data, flag and memory dependences);
  `deadCodeEliminate()` prunes instructions that cannot reach the live-out set.

### `optimizer/perf_model` — cost models (`perf.ts`)

* `staticPerf` schedules the program on an out-of-order abstraction: critical
  path over the dependence graph, port pressure over the issue ports and a
  front-end µop limit; `cycles = max(criticalPath, portPressure, µops/width)`.
* `LearnedPerfPredictor` is an online ridge-regularised linear model over the
  opcode-class histogram plus structural features, trained by SGD against real
  measurements (`measureProgram` times repeated interpretation). `PerfModel`
  blends the analytic and learned estimates so a cold model is never worse than
  the analytic one.

### `optimizer/assembly_superopt` — stochastic superoptimisation (`mcmc.ts`)

Metropolis–Hastings over loop-free sequences with π(x) ∝ exp(−cost(x)/T):

* **Proposals**: opcode mutation (signature-preserving), operand mutation,
  instruction swap (dependence-checked; speculative swaps are allowed with a
  cost penalty), insertion and deletion. Insert/delete are asymmetric, so the
  exact proposal ratio `q(x*→x)/q(x→x*)` is computed from the position, opcode
  and operand-space cardinalities and folded into the acceptance test.
* **Acceptance**: `α = min(1, exp(−Δcost/T) · q-ratio)`; downhill always
  accepted, uphill accepted with probability α.
* **Engine**: independent chains with distinct seeds and temperatures, geometric
  annealing, replica exchange between neighbouring chains, diversified starts
  (target / empty / random) and a Pareto front over (correctness, performance).

### `optimizer/smt` — symbolic execution and SMT (`smt.ts`)

* Each register becomes a bit-vector of Boolean literals; every opcode expands
  into gates (ripple-carry adders, shift-add multipliers, barrel shifters,
  priority encoders for `clz`/`ctz`, popcount adder trees, signed/unsigned
  comparators, flag logic) that are Tseitin-encoded into CNF on the fly.
* The miter `⋁_liveouts (R_out ⊕ T_out)` is solved by a bundled DPLL engine:
  two-watched-literal unit propagation, input-first decision ordering,
  chronological backtracking, decision/conflict/time budgets.
* **UNSAT** ⇒ the rewrite is proven equivalent for the modelled semantics at
  the verification width. **SAT** ⇒ the model is decoded into a concrete
  counterexample which is added to the dynamic test database, invalidating all
  cost caches so every engine immediately sees the new constraint.
* Floating-point, vector, memory and division opcodes fall back to exhaustive
  bounded enumeration on the concrete interpreter (complete at the reduced
  width, sampled above it).

### `optimizer/rl_asm_synth` — RL + MCTS synthesis (`mcts.ts`)

* Single-player game: state = program prefix, action = append instruction or
  halt. Actions are generated from the legal operand space restricted to
  already-defined registers plus one fresh register.
* PUCT selection with priors from a **policy** model
  `p(a|s) = softmax(θ_p·φ(s,a))`, trained by cross-entropy against the MCTS
  visit distribution, and a **value** model `V(s) = θ_v·ψ(s)` trained by
  regression on rollout returns (AlphaZero-style policy iteration).
* Reward `exp(−cost/scale)` reuses the shared cost function, so RL, MCMC and
  enumeration optimise exactly the same objective.

### `optimizer/hybrid_search` — parallel orchestrator (`hybrid.ts`, `enumerative.ts`)

* Round-robin time slices over MCMC, MCTS, enumerative MITM and the SMT layer,
  all sharing: the dynamic test database, the candidate table (deduplicated by
  program hash), the Pareto front and the event log.
* The enumerative engine performs bidirectional search: forward BFS over
  prefixes keyed by the *signature* of the machine states they produce, and
  backward BFS over invertible suffixes applied in reverse to the target's
  output states; a signature hit composes prefix ⧺ suffix.
* After each round the enlarged test set is re-broadcast: caches are
  invalidated and every stored candidate is re-scored, so a counterexample
  found by the SMT layer instantly demotes wrong candidates in all engines.

### `optimizer/eqsat_ilp` — equality saturation + ILP (`egraph.ts`)

* Hash-consed e-graph with union-find, congruence closure (`rebuild`) and an
  e-class **shape analysis** that propagates tensor dimensions.
* 22 rewrite rules: commutativity/associativity, unit and zero elimination,
  distribute/factor, matmul associativity (both directions), transpose
  involutivity, transpose–matmul, ReLU idempotence, exp/log laws, sum
  linearity, FMA fusion.
* Extraction is emitted as a real 0/1 ILP (selection, child-implication,
  root and big-M topological-order constraints) and solved exactly by branch
  and bound with a bottom-up fixpoint incumbent, DAG-sharing objective, cycle
  rejection and a node/time limit that reports whether optimality was proved.

### `runtime/` — Zig execution layer

* `memory.zig`: arena (bump, page-aligned, LIFO free fast path), slab (intrusive
  free list), typed pool, buddy (power-of-two split/merge) — all exposing
  `std.mem.Allocator`, all with optional secure zeroization.
* `lockfree.zig`: Vyukov bounded MPMC queue, ABA-resistant tagged Treiber
  stack, wait-free SPSC ring; cache-line padded to avoid false sharing.
* `tensor.zig`: reference-counted copy-on-write storage, 64-byte aligned and
  vector-width padded buffers, `@Vector` element-wise kernels, FMA `axpb`,
  vector reductions, cache-blocked matmul (L1/L2/L3 panels + SIMD micro-kernel)
  and an atomically load-balanced parallel matmul.
* `gpu_coordinator.zig`: lazy `libcudart` binding, per-device streams and
  events, `allReduce` (ring / tree / direct), `reduceScatter`, `allGather`,
  `broadcast`, `reduce`, chained completions, and a deterministic host backend
  used when no GPU is present.
* `futhark_bindings.zig`: RAII wrappers over the Futhark C ABI (context,
  1-D/2-D arrays, eight entry points) with graceful degradation.
* `kernels.zig`: registry of the synthesised kernels, W^X executable mapping
  for machine-code blobs and a differential tester against the reference
  implementations.

### `futhark_kernels/`

`kernels.fut` implements dot, saxpy, matmul, reduce, softmax, 1-D convolution,
Jacobi stencil, radix-2 FFT, fused linear+ReLU and population count, with
inline `-- ==` test blocks. `build.sh` picks the best backend (cuda → opencl →
multicore), emits `kernels.h`/`kernels.c`, links `libfutkernels.so` and runs
`futhark test`.

## Data flow of one synthesis run

1. The dynamic test database is seeded with adversarial and random inputs.
2. The perf model is calibrated with real timing samples of the target.
3. Each round: MCMC → MCTS → enumerative MITM → SMT verification.
4. SMT counterexamples enlarge the test set; all caches are invalidated and
   candidates are re-scored.
5. The best verified (or best test-equivalent) rewrite is dead-code eliminated,
   re-verified, disassembled for the selected ISA and persisted in PostgreSQL
   together with the candidate table, the event log and the counterexamples.
