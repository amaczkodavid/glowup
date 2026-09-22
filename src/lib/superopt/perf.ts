/**
 * Performance models.
 *
 * 1. `staticPerf` - an analytical port/latency model: it schedules the program
 *    on an out-of-order machine abstraction (critical path over the data
 *    dependence graph, port pressure over the issue ports, decode/uop limits)
 *    and returns predicted cycles.
 * 2. `LearnedPerfPredictor` - an online linear model (ridge-regularised SGD)
 *    over opcode-class histograms + structural features. It is trained against
 *    real measurements produced by `measureProgram`, which times repeated
 *    interpretation of the program on the concrete machine.
 */

import { getIsa, spec } from "./isa";
import type { IsaBackend } from "./isa";
import { cloneState, effects, execute } from "./machine";
import type { CpuState, MachineConfig, Program } from "./types";
import { DEFAULT_MACHINE } from "./types";

export interface StaticPerf {
  cycles: number;
  criticalPath: number;
  portPressure: number;
  uops: number;
  bytes: number;
  instructions: number;
}

export function staticPerf(prog: Program, isaName = "x86-64"): StaticPerf {
  const isa: IsaBackend = getIsa(isaName);
  const ready = { gpr: new Map<number, number>(), fpr: new Map<number, number>(), vreg: new Map<number, number>(), flags: 0, mem: 0 };
  const portBusy = new Map<number, number>();
  let uops = 0;
  let bytes = 0;
  let criticalPath = 0;

  for (const ins of prog) {
    const s = spec(ins.op);
    const lat = isa.latency[ins.op] ?? s.latency;
    const e = effects(ins);
    let start = 0;
    for (const r of e.readsGpr) start = Math.max(start, ready.gpr.get(r) ?? 0);
    for (const r of e.readsFpr) start = Math.max(start, ready.fpr.get(r) ?? 0);
    for (const r of e.readsVreg) start = Math.max(start, ready.vreg.get(r) ?? 0);
    if (e.readsFlags) start = Math.max(start, ready.flags);
    if (e.readsMem || e.writesMem) start = Math.max(start, ready.mem);

    // Port pressure: the earliest cycle at which one of the legal ports is free.
    let bestPort = s.ports[0];
    let bestFree = Number.POSITIVE_INFINITY;
    for (const p of s.ports) {
      const free = portBusy.get(p) ?? 0;
      if (free < bestFree) {
        bestFree = free;
        bestPort = p;
      }
    }
    const issue = Math.max(start, bestFree);
    portBusy.set(bestPort, issue + Math.max(s.rthroughput, 0.25));

    const done = issue + lat;
    for (const r of e.writesGpr) ready.gpr.set(r, done);
    for (const r of e.writesFpr) ready.fpr.set(r, done);
    for (const r of e.writesVreg) ready.vreg.set(r, done);
    if (e.writesFlags) ready.flags = done;
    if (e.writesMem) ready.mem = done;
    criticalPath = Math.max(criticalPath, done);
    uops += s.uops;
    bytes += isa.encodedSize(ins);
  }

  const portPressure = Math.max(0, ...[...portBusy.values()]);
  const frontEnd = uops / isa.issueWidth;
  const cycles = Math.max(criticalPath, portPressure, frontEnd);
  return {
    cycles,
    criticalPath,
    portPressure,
    uops,
    bytes,
    instructions: prog.length,
  };
}

const FEATURE_CLASSES = [
  "nop",
  "move",
  "arith",
  "logic",
  "shift",
  "bit",
  "compare",
  "select",
  "memory",
  "float",
  "vector",
] as const;

export const FEATURE_NAMES = [
  ...FEATURE_CLASSES.map((c) => `n_${c}`),
  "criticalPath",
  "portPressure",
  "uops",
  "bytes",
  "length",
  "bias",
];

export function features(prog: Program, isaName = "x86-64"): number[] {
  const hist = new Array(FEATURE_CLASSES.length).fill(0);
  for (const ins of prog) {
    const idx = FEATURE_CLASSES.indexOf(spec(ins.op).cls);
    if (idx >= 0) hist[idx] += 1;
  }
  const sp = staticPerf(prog, isaName);
  return [...hist, sp.criticalPath, sp.portPressure, sp.uops, sp.bytes, prog.length, 1];
}

/** Online ridge-regularised linear predictor of measured runtime. */
export class LearnedPerfPredictor {
  weights: number[];
  samples = 0;
  lastError = 0;
  meanAbsError = 0;
  constructor(
    private lr = 2e-4,
    private l2 = 1e-6,
    dim = FEATURE_NAMES.length,
  ) {
    this.weights = new Array(dim).fill(0);
    this.weights[this.weights.length - 1] = 1; // bias
  }
  predict(f: number[]): number {
    let acc = 0;
    for (let i = 0; i < this.weights.length; i++) acc += this.weights[i] * f[i];
    return acc;
  }
  train(f: number[], target: number): number {
    const p = this.predict(f);
    const err = p - target;
    let norm = 0;
    for (const x of f) norm += x * x;
    const scale = this.lr / (1 + norm);
    for (let i = 0; i < this.weights.length; i++) {
      this.weights[i] -= scale * err * f[i] + this.lr * this.l2 * this.weights[i];
    }
    this.samples++;
    this.lastError = Math.abs(err);
    this.meanAbsError += (Math.abs(err) - this.meanAbsError) / this.samples;
    return err;
  }
  get trained(): boolean {
    return this.samples >= 24;
  }
  snapshot(): { weights: number[]; samples: number; meanAbsError: number; names: string[] } {
    return {
      weights: [...this.weights],
      samples: this.samples,
      meanAbsError: this.meanAbsError,
      names: FEATURE_NAMES,
    };
  }
}

/** Wall-clock measurement of a program by repeated interpretation. */
export function measureProgram(
  prog: Program,
  seedState: CpuState,
  cfg: MachineConfig = DEFAULT_MACHINE,
  iterations = 256,
): number {
  const scratch = cloneState(seedState);
  const start = performance.now();
  for (let i = 0; i < iterations; i++) {
    scratch.gpr.set(seedState.gpr);
    scratch.fpr.set(seedState.fpr);
    scratch.vreg.set(seedState.vreg);
    scratch.fault = "none";
    scratch.retired = 0;
    execute(prog, scratch, cfg);
  }
  const elapsed = performance.now() - start;
  // Normalise to "nanoseconds per invocation".
  return (elapsed * 1e6) / iterations;
}

export interface PerfModelOptions {
  isa: string;
  sizeWeight: number;
  /** Probability of taking a real measurement instead of the analytic model. */
  measureProbability: number;
}

export class PerfModel {
  readonly learned = new LearnedPerfPredictor();
  measurements = 0;
  constructor(
    readonly opts: PerfModelOptions = { isa: "x86-64", sizeWeight: 0.35, measureProbability: 0.05 },
  ) {}

  /** perf(R): predicted cost in abstract cycles including code-size pressure. */
  perf(prog: Program): number {
    const sp = staticPerf(prog, this.opts.isa);
    const analytic = sp.cycles + this.opts.sizeWeight * prog.length;
    if (!this.learned.trained) return analytic;
    const f = features(prog, this.opts.isa);
    const predicted = this.learned.predict(f);
    // Blend: the learned model corrects the analytic model, never replaces it.
    return 0.5 * analytic + 0.5 * Math.max(0.1, predicted);
  }

  /** Take a real timing sample and train the learned predictor on it. */
  calibrate(prog: Program, seedState: CpuState, cfg: MachineConfig = DEFAULT_MACHINE): number {
    const ns = measureProgram(prog, seedState, cfg);
    const sp = staticPerf(prog, this.opts.isa);
    // Convert nanoseconds into the analytic cycle scale so both live in one space.
    const scaled = (ns / Math.max(1e-9, ns + 1)) * sp.cycles + ns * 0.01;
    this.learned.train(features(prog, this.opts.isa), scaled);
    this.measurements++;
    return ns;
  }
}
