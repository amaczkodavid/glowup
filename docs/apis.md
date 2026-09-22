# Public APIs

## HTTP API

### `GET /api/health`
`{ "ok": true }` when the database is reachable.

### `GET /api/tasks`
Returns the benchmark suite, ISA list, orchestrator defaults, tensor programs
and rewrite-rule names.

```jsonc
{
  "isas": ["x86-64", "aarch64"],
  "defaults": { "rounds": 3, "mcmcMs": 1200, ... },
  "tasks": [{ "id": "popcount", "target": "shri r1, r0, #1\n...", "asmX86": "...",
              "asmArm": "...", "cycles": 21.5, "pool": ["add", ...],
              "knownOptimum": "popcnt r0, r0", "verifyWidth": 8 }],
  "tensorPrograms": [{ "id": "matchain", "env": { "A": [64,1024] }, "expr": "matmul(matmul(A, B), C)" }],
  "rules": ["add-comm", "matmul-assoc-r", ...]
}
```

### `POST /api/runs`
Body:

```jsonc
{
  "taskId": "abs",
  "config": {
    "isa": "x86-64" | "aarch64",
    "rounds": 1..6,
    "seed": 1..2^31-1,
    "mcmcMs": 100..8000, "mctsMs": 0..6000, "enumerativeMs": 0..6000, "verifyMs": 100..8000,
    "mcmcChains": 1..8, "mcmcIterations": 1000..400000, "mctsSimulations": 100..40000,
    "randomTests": 8..256,
    "weightsWe": 1.0, "weightsWp": 1.0, "temperature": 12, "annealing": 0.35,
    "engines": { "mcmc": true, "mcts": true, "enumerative": true, "symbolic": true }
  }
}
```

Response: `{ id, report, run }` where `report` is the `HybridReport`:
target/best programs (virtual core + native asm), speedup, size reduction,
verification verdict and history, Pareto front, candidate table,
counterexamples, per-engine summaries, event log, learned perf-model weights.

### `GET /api/runs` / `GET /api/runs/{id}`
List of runs, or one run with its candidates and event stream.

### `POST /api/eqsat`
`{ "programId": "matchain", "rules": ["matmul-assoc-r", ...] }` (rules optional).
Returns before/after costs, the extracted expression, saturation statistics and
the ILP model (objective, variables, constraints, binary-variable count).

### `POST /api/verify`
`{ "taskId": "abs", "program": "abs r0, r0", "width": 8, "method": "smt"|"enumeration" }`
Returns the `VerificationResult`: status (`equivalent` / `counterexample` /
`budget-exhausted`), method, width, CNF size, decisions, propagations,
conflicts, elapsed time, counterexample and human-readable detail.

### `GET /api/corpus?taskId=…`
The persisted counterexample corpus (dynamic test database).

### `GET /api/source` / `GET /api/source?file=runtime/memory.zig`
Manifest of, and read access to, every source artefact (`runtime/`,
`futhark_kernels/`, `docs/`, `ci/`, `src/lib/superopt/`).

## Optimizer library (TypeScript)

```ts
import { getTask, TASKS } from "@/lib/superopt/tasks";
import { runHybrid } from "@/lib/superopt/hybrid";
import { verifyEquivalence, boundedEnumeration } from "@/lib/superopt/smt";
import { runMcmc, DEFAULT_IMMEDIATES } from "@/lib/superopt/mcmc";
import { runMcts } from "@/lib/superopt/mcts";
import { runEnumerative } from "@/lib/superopt/enumerative";
import { EGraph, saturate, extractIlp, runEqsat, TENSOR_RULES } from "@/lib/superopt/egraph";
import { CostEvaluator } from "@/lib/superopt/cost";
import { PerfModel, staticPerf, measureProgram } from "@/lib/superopt/perf";
import { disassemble, parseProgram, formatProgram, registerIsa } from "@/lib/superopt/isa";
```

| Function | Signature | Notes |
|---|---|---|
| `runHybrid` | `(task: Task, cfg?: Partial<HybridConfig>) => HybridReport` | full orchestrated search |
| `runMcmc` | `(ev: CostEvaluator, start: Program, opts: McmcOptions, onEvent?) => McmcResult` | MH superoptimiser |
| `runMcts` | `(ev: CostEvaluator, opts: MctsOptions, onEvent?) => MctsResult` | PUCT synthesis |
| `runEnumerative` | `(ev, target, tests, opts, onEvent?) => EnumResult` | bidirectional MITM |
| `verifyEquivalence` | `(target, candidate, opts: VerifyOptions) => VerificationResult` | bit-blast + DPLL |
| `boundedEnumeration` | same | exhaustive at the verification width |
| `saturate` | `(g: EGraph, rules: Rule[], limits) => SaturationStats` | equality saturation |
| `extractIlp` | `(g, root, limits) => ExtractionResult` | exact ILP extraction |
| `staticPerf` | `(prog, isa) => StaticPerf` | ports, critical path, µops, bytes |
| `registerIsa` | `(backend: IsaBackend) => void` | add a new target ISA |

### Extending the ISA

```ts
registerIsa({
  name: "riscv64",
  gprNames: ["a0", "a1", ...], fprNames: [...], vregNames: [...],
  unsupported: new Set(["blsi", "blsr", "blsmsk"]),
  latency: { mul: 3, divu: 34, load: 3 },
  issueWidth: 2,
  print: (ins) => `${ins.op} x${ins.rd}, x${ins.rs1}, x${ins.rs2}`,
  encodedSize: () => 4,
});
```

## Zig runtime API

```zig
const rt = @import("runtime");

// Allocators
var arena = try rt.memory.ArenaAllocator.init(base, 1 << 20, .secure);
var slab  = try rt.memory.SlabAllocator.init(base, 128, 16, 1024, .secure);
var pool  = try rt.memory.PoolAllocator(Particle).init(base, 4096, .none);
var buddy = try rt.memory.BuddyAllocator.init(base, 1 << 24, 64, .secure);

// Lock-free
var q = try rt.lockfree.MpmcQueue(Job).init(base, 1024);
var s = rt.lockfree.TreiberStack(*Node).init(base);
var r = try rt.lockfree.SpscRing(f32).init(base, 4096);

// Tensors
var t = try rt.tensor.Tensor.init(base, &.{ 256, 256 });
try rt.tensor.elementwise(&out, &a, &b, .mul);
try rt.tensor.axpb(&out, &a, 2.0, 1.0);
try rt.tensor.matmulParallel(base, &c, &a, &b, .{ .threads = 8, .mc = 96, .kc = 256 });

// GPU collectives
var coord = try rt.gpu.Coordinator.init(base, 8);
var done  = try coord.allReduce(buffers, .sum, .ring);
try done.then(1);           // chain onto device 1's stream
try done.wait();

// Futhark
var ctx = try rt.futhark.Context.init(base, "libfutkernels.so");
const d = try ctx.dot(&xs, &ys);
var spectrum = try ctx.fft(&re, &im);

// Synthesised kernels
const k = try rt.kernels.lookup("isolate_lsb");
const code = try rt.kernels.mapExecutable(base, blob);
try rt.kernels.differentialTest(k, @ptrCast(code.ptr), 100_000, 0xC0FFEE);
```

## Futhark entry points

| Entry | Type |
|---|---|
| `dot` | `[n]f32 -> [n]f32 -> f32` |
| `saxpy` | `f32 -> [n]f32 -> [n]f32 -> [n]f32` |
| `matmul` | `[m][n]f32 -> [n][p]f32 -> [m][p]f32` |
| `reduce_sum` | `[n]f32 -> f32` |
| `softmax` | `[n]f32 -> [n]f32` |
| `conv1d` | `[n]f32 -> [k]f32 -> [n]f32` |
| `stencil` | `[n]f32 -> i32 -> [n]f32` |
| `fft` | `[n]f32 -> [n]f32 -> ([n]f32, [n]f32)` |
| `linear_relu` | `[b][m]f32 -> [m][k]f32 -> [k]f32 -> [b][k]f32` |
| `popcount_u32` | `[n]u32 -> [n]i32` |
