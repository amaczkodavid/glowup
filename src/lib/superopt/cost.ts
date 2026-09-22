/**
 * Cost function: cost(R; T) = we * eq(R; T) + wp * perf(R).
 *
 * eq() aggregates bit-level live-out mismatches for integer outputs, ULP
 * distance (sum and max) for floating point outputs, and large penalties for
 * runtime faults. Partial evaluations are memoised by program hash so the MCMC
 * kernel can re-evaluate mutated neighbourhoods cheaply.
 */

import { formatProgram } from "./isa";
import { execute, stateFromTest } from "./machine";
import type { PerfModel } from "./perf";
import { bitDistance, ulpDistance } from "./testcases";
import type {
  CostBreakdown,
  CostWeights,
  CpuState,
  LiveOut,
  MachineConfig,
  Program,
  TestCase,
} from "./types";
import { DEFAULT_MACHINE } from "./types";

export interface EqResult {
  error: number;
  bitErrors: number;
  ulpSum: number;
  ulpMax: number;
  faults: number;
  correct: boolean;
  /** Index of the first failing test case, -1 when all pass. */
  firstFailure: number;
}

export function programHash(prog: Program): string {
  let h1 = 0x811c9dc5;
  let h2 = 0x01000193;
  for (const ins of prog) {
    const parts = [ins.op.length, ins.rd + 2, ins.rs1 + 2, ins.rs2 + 2, ins.rs3 + 2, ins.imm];
    for (let i = 0; i < ins.op.length; i++) parts.push(ins.op.charCodeAt(i));
    for (const p of parts) {
      h1 = Math.imul(h1 ^ p, 16777619) >>> 0;
      h2 = Math.imul(h2 + p + 0x9e3779b9, 2654435761) >>> 0;
    }
  }
  return `${h1.toString(36)}${h2.toString(36)}`;
}

export function liveOutStates(
  prog: Program,
  tests: TestCase[],
  cfg: MachineConfig = DEFAULT_MACHINE,
): CpuState[] {
  return tests.map((t) => execute(prog, stateFromTest(t, cfg), cfg));
}

export function eq(
  candidateStates: CpuState[],
  referenceStates: CpuState[],
  live: LiveOut,
  weights: CostWeights,
  cfg: MachineConfig = DEFAULT_MACHINE,
): EqResult {
  let bitErrors = 0;
  let ulpSum = 0;
  let ulpMax = 0;
  let faults = 0;
  let firstFailure = -1;
  const n = Math.min(candidateStates.length, referenceStates.length);
  for (let i = 0; i < n; i++) {
    const c = candidateStates[i];
    const r = referenceStates[i];
    let localBad = false;
    if (c.fault !== "none" && r.fault === "none") {
      faults += 1;
      localBad = true;
    }
    if (c.fault === "none" && r.fault !== "none") {
      faults += 1;
      localBad = true;
    }
    for (const reg of live.gpr) {
      const d = bitDistance(c.gpr[reg], r.gpr[reg], cfg.width);
      if (d) localBad = true;
      bitErrors += d;
    }
    for (const reg of live.fpr) {
      const d = ulpDistance(c.fpr[reg], r.fpr[reg]);
      if (d) localBad = true;
      ulpSum += d;
      ulpMax = Math.max(ulpMax, d);
    }
    for (const reg of live.vreg) {
      for (let lane = 0; lane < 4; lane++) {
        const d = ulpDistance(c.vreg[reg * 4 + lane], r.vreg[reg * 4 + lane]);
        if (d) localBad = true;
        ulpSum += d;
        ulpMax = Math.max(ulpMax, d);
      }
    }
    for (const region of live.mem) {
      for (let off = 0; off < region.bytes; off++) {
        const addr = region.addr + off;
        const d = bitDistance(c.mem[addr] ?? 0, r.mem[addr] ?? 0, 8);
        if (d) localBad = true;
        bitErrors += d;
      }
    }
    if (localBad && firstFailure < 0) firstFailure = i;
  }
  const error =
    weights.bitPenalty * bitErrors +
    weights.ulpPenalty * Math.log1p(ulpSum) +
    weights.ulpMaxPenalty * Math.log1p(ulpMax) +
    weights.faultPenalty * faults;
  return {
    error,
    bitErrors,
    ulpSum,
    ulpMax,
    faults,
    correct: bitErrors === 0 && ulpSum === 0 && faults === 0,
    firstFailure,
  };
}

export interface CostContext {
  target: Program;
  tests: TestCase[];
  live: LiveOut;
  weights: CostWeights;
  cfg: MachineConfig;
  perfModel: PerfModel;
}

export class CostEvaluator {
  private referenceCache = new Map<string, CpuState[]>();
  private costCache = new Map<string, CostBreakdown>();
  evaluations = 0;
  cacheHits = 0;

  constructor(readonly ctx: CostContext) {}

  private reference(): CpuState[] {
    const key = programHash(this.ctx.target) + ":" + this.ctx.tests.length;
    let ref = this.referenceCache.get(key);
    if (!ref) {
      ref = liveOutStates(this.ctx.target, this.ctx.tests, this.ctx.cfg);
      this.referenceCache.set(key, ref);
    }
    return ref;
  }

  /** Invalidate caches after the dynamic test database grows. */
  invalidate(): void {
    this.referenceCache.clear();
    this.costCache.clear();
  }

  cost(prog: Program): CostBreakdown {
    const key = programHash(prog) + ":" + this.ctx.tests.length;
    const cached = this.costCache.get(key);
    if (cached) {
      this.cacheHits++;
      return cached;
    }
    this.evaluations++;
    const states = liveOutStates(prog, this.ctx.tests, this.ctx.cfg);
    const e = eq(states, this.reference(), this.ctx.live, this.ctx.weights, this.ctx.cfg);
    const p = this.ctx.perfModel.perf(prog);
    const breakdown: CostBreakdown = {
      total: this.ctx.weights.we * e.error + this.ctx.weights.wp * p,
      eq: e.error,
      perf: p,
      bitErrors: e.bitErrors,
      ulpSum: e.ulpSum,
      ulpMax: e.ulpMax,
      faults: e.faults,
      correct: e.correct,
    };
    this.costCache.set(key, breakdown);
    return breakdown;
  }

  correctOnTests(prog: Program): boolean {
    return this.cost(prog).correct;
  }

  describe(prog: Program): string {
    const c = this.cost(prog);
    return `${formatProgram(prog)}\n; cost=${c.total.toFixed(3)} eq=${c.eq.toFixed(3)} perf=${c.perf.toFixed(3)}`;
  }
}
