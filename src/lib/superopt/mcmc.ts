/**
 * Stochastic superoptimisation by Metropolis–Hastings MCMC over loop-free
 * assembly sequences.
 *
 *   π(x) ∝ exp(-cost(x) / T)
 *   α(x → x*) = min(1, [π(x*) q(x* → x)] / [π(x) q(x → x*)])
 *
 * Five proposal kernels are implemented (opcode, operand, swap, insert,
 * delete); insert/delete are asymmetric so their proposal ratio is computed
 * explicitly and folded into the acceptance test. Multiple chains run with
 * independent seeds and periodically exchange states (replica exchange).
 */

import { OPCODES_BY_SIGNATURE, registerFileCap, signatureOf, spec } from "./isa";
import { commutesInOrder } from "./machine";
import type { CostEvaluator } from "./cost";
import { programHash } from "./cost";
import { Rng } from "./testcases";
import type { CostBreakdown, Instruction, MachineConfig, Program, SearchEvent } from "./types";

export interface McmcOptions {
  pool: string[];
  maxLength: number;
  cfg: MachineConfig;
  /** Initial temperature and its geometric decay per 1000 iterations. */
  temperature: number;
  annealing: number;
  minTemperature: number;
  iterations: number;
  chains: number;
  seed: number;
  /** Iterations between replica-exchange attempts. */
  exchangeInterval: number;
  /** Immediate values the proposal kernel may sample. */
  immediates: number[];
  timeBudgetMs: number;
}

export interface ChainStats {
  chain: number;
  proposals: number;
  accepted: number;
  rejected: number;
  uphillAccepted: number;
  exchanges: number;
  finalTemperature: number;
  bestCost: number;
}

export interface McmcResult {
  best: Program;
  bestCost: CostBreakdown;
  bestCorrect: Program | null;
  bestCorrectCost: CostBreakdown | null;
  pareto: Array<{ program: Program; cost: CostBreakdown }>;
  stats: ChainStats[];
  events: SearchEvent[];
  iterations: number;
  elapsedMs: number;
}

export const DEFAULT_IMMEDIATES = [
  -2147483648, -65536, -256, -32, -16, -8, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 15, 16,
  24, 31, 32, 63, 64, 127, 128, 255, 256, 65535, 65536, 0x0f0f0f0f, 0x33333333, 0x55555555,
  0x01010101, 0x7fffffff,
];

export function randomInstruction(
  rng: Rng,
  pool: string[],
  cfg: MachineConfig,
  immediates: number[],
): Instruction {
  const op = rng.pick(pool);
  return randomOperands(rng, op, cfg, immediates);
}

export function randomOperands(
  rng: Rng,
  op: string,
  cfg: MachineConfig,
  immediates: number[],
): Instruction {
  const s = spec(op);
  const ins: Instruction = { op, rd: -1, rs1: -1, rs2: -1, rs3: -1, imm: 0 };
  for (const role of s.roles) {
    if (role === "imm") {
      const candidates = immediates.filter((v) => v >= s.immMin && v <= s.immMax);
      ins.imm = candidates.length ? rng.pick(candidates) : 0;
      continue;
    }
    const cap = registerFileCap(s, role, cfg.gprCount, cfg.fprCount, cfg.vregCount);
    const v = rng.int(cap);
    if (role === "rd" || role === "vd") ins.rd = v;
    else if (role === "rs1" || role === "vs1") ins.rs1 = v;
    else if (role === "rs2" || role === "vs2") ins.rs2 = v;
    else ins.rs3 = v;
  }
  return ins;
}

/** Number of distinct operand assignments for an opcode (for q-ratios). */
function operandSpace(op: string, cfg: MachineConfig, immediates: number[]): number {
  const s = spec(op);
  let n = 1;
  for (const role of s.roles) {
    if (role === "imm") {
      n *= Math.max(1, immediates.filter((v) => v >= s.immMin && v <= s.immMax).length);
    } else {
      n *= Math.max(1, registerFileCap(s, role, cfg.gprCount, cfg.fprCount, cfg.vregCount));
    }
  }
  return n;
}

export type MoveKind = "opcode" | "operand" | "swap" | "insert" | "delete";

export interface Proposal {
  program: Program;
  kind: MoveKind;
  /** log q(x* → x) − log q(x → x*). */
  logQRatio: number;
  /** Extra cost penalty for speculative (dependence-violating) swaps. */
  penalty: number;
}

const MOVE_WEIGHTS: Record<MoveKind, number> = {
  opcode: 0.28,
  operand: 0.34,
  swap: 0.12,
  insert: 0.13,
  delete: 0.13,
};

function pickMove(rng: Rng, len: number, maxLength: number): MoveKind {
  const kinds: MoveKind[] = ["opcode", "operand", "swap", "insert", "delete"];
  const w = kinds.map((k) => {
    if (len === 0 && (k === "opcode" || k === "operand" || k === "swap" || k === "delete")) return 0;
    if (len < 2 && k === "swap") return 0;
    if (len >= maxLength && k === "insert") return 0;
    return MOVE_WEIGHTS[k];
  });
  const total = w.reduce((a, b) => a + b, 0);
  let r = rng.next() * total;
  for (let i = 0; i < kinds.length; i++) {
    r -= w[i];
    if (r <= 0) return kinds[i];
  }
  return "operand";
}

export function propose(
  rng: Rng,
  prog: Program,
  opts: McmcOptions,
): Proposal {
  const kind = pickMove(rng, prog.length, opts.maxLength);
  const next = prog.slice();
  const { cfg, immediates } = opts;
  switch (kind) {
    case "opcode": {
      const i = rng.int(next.length);
      const sig = signatureOf(next[i].op);
      const alternatives = (OPCODES_BY_SIGNATURE[sig] ?? []).filter(
        (o) => opts.pool.includes(o) && o !== next[i].op,
      );
      if (alternatives.length === 0) return { program: next, kind, logQRatio: 0, penalty: 0 };
      const op = rng.pick(alternatives);
      const keep = next[i];
      const s = spec(op);
      const ins: Instruction = { ...keep, op };
      if (s.roles.includes("imm") && (ins.imm < s.immMin || ins.imm > s.immMax)) {
        ins.imm = Math.max(s.immMin, Math.min(s.immMax, ins.imm));
      }
      next[i] = ins;
      // Symmetric: the reverse move picks the same index from the same class.
      return { program: next, kind, logQRatio: 0, penalty: 0 };
    }
    case "operand": {
      const i = rng.int(next.length);
      const mutated = randomOperands(rng, next[i].op, cfg, immediates);
      next[i] = mutated;
      return { program: next, kind, logQRatio: 0, penalty: 0 };
    }
    case "swap": {
      const i = rng.int(next.length);
      let j = rng.int(next.length);
      if (i === j) j = (j + 1) % next.length;
      const [a, b] = [Math.min(i, j), Math.max(i, j)];
      let penalty = 0;
      for (let k = a; k < b; k++) {
        if (!commutesInOrder(next[a], next[k + 1])) {
          penalty = 6; // speculative reorder: allowed but penalised
          break;
        }
      }
      const tmp = next[a];
      next[a] = next[b];
      next[b] = tmp;
      return { program: next, kind, logQRatio: 0, penalty };
    }
    case "insert": {
      const pos = rng.int(next.length + 1);
      const ins = randomInstruction(rng, opts.pool, cfg, immediates);
      next.splice(pos, 0, ins);
      // q(x→x*) = P(insert) * 1/(n+1) * 1/|pool| * 1/|operands(op)|
      // q(x*→x) = P(delete) * 1/(n+1)
      const qForward =
        MOVE_WEIGHTS.insert / ((prog.length + 1) * opts.pool.length * operandSpace(ins.op, cfg, immediates));
      const qBackward = MOVE_WEIGHTS.delete / next.length;
      return { program: next, kind, logQRatio: Math.log(qBackward / qForward), penalty: 0 };
    }
    case "delete": {
      const pos = rng.int(next.length);
      const removed = next[pos];
      next.splice(pos, 1);
      const qForward = MOVE_WEIGHTS.delete / prog.length;
      const qBackward =
        MOVE_WEIGHTS.insert / ((next.length + 1) * opts.pool.length * operandSpace(removed.op, cfg, immediates));
      return { program: next, kind, logQRatio: Math.log(qBackward / qForward), penalty: 0 };
    }
  }
}

interface ChainState {
  rng: Rng;
  current: Program;
  currentCost: number;
  temperature: number;
  stats: ChainStats;
}

/** Maintain a Pareto front over (eq-error, perf). */
export class ParetoFront {
  private items: Array<{ program: Program; cost: CostBreakdown; hash: string }> = [];
  offer(program: Program, cost: CostBreakdown): boolean {
    const hash = programHash(program);
    if (this.items.some((i) => i.hash === hash)) return false;
    const dominated = this.items.some((i) => i.cost.eq <= cost.eq && i.cost.perf <= cost.perf);
    if (dominated) return false;
    this.items = this.items.filter((i) => !(cost.eq <= i.cost.eq && cost.perf <= i.cost.perf));
    this.items.push({ program: program.slice(), cost, hash });
    this.items.sort((a, b) => a.cost.perf - b.cost.perf);
    return true;
  }
  list(): Array<{ program: Program; cost: CostBreakdown }> {
    return this.items
      .slice()
      .sort((a, b) => a.cost.eq - b.cost.eq || a.cost.perf - b.cost.perf)
      .map(({ program, cost }) => ({ program, cost }));
  }
}

export function runMcmc(
  evaluator: CostEvaluator,
  start: Program,
  opts: McmcOptions,
  onEvent?: (e: SearchEvent) => void,
): McmcResult {
  const t0 = Date.now();
  const events: SearchEvent[] = [];
  const emit = (e: SearchEvent) => {
    events.push(e);
    onEvent?.(e);
  };
  const pareto = new ParetoFront();
  const chains: ChainState[] = [];
  for (let c = 0; c < opts.chains; c++) {
    const rng = new Rng(opts.seed + c * 7919 + 13);
    // Chain 0 anneals from the target; the remaining chains start from
    // diversified points (empty and short random programs) so that short
    // rewrites are reachable without a long sequence of deletions.
    let current: Program;
    if (c === 0) current = start.slice();
    else if (c === 1) current = [];
    else {
      const len = 1 + rng.int(Math.max(1, Math.min(3, opts.maxLength)));
      current = Array.from({ length: len }, () =>
        randomInstruction(rng, opts.pool, opts.cfg, opts.immediates),
      );
    }
    chains.push({
      rng,
      current,
      currentCost: evaluator.cost(current).total,
      temperature: opts.temperature * (1 + 0.35 * c),
      stats: {
        chain: c,
        proposals: 0,
        accepted: 0,
        rejected: 0,
        uphillAccepted: 0,
        exchanges: 0,
        finalTemperature: opts.temperature,
        bestCost: Number.POSITIVE_INFINITY,
      },
    });
  }

  let best = start.slice();
  let bestCost = evaluator.cost(best);
  let bestCorrect: Program | null = bestCost.correct ? start.slice() : null;
  let bestCorrectCost: CostBreakdown | null = bestCost.correct ? bestCost : null;
  pareto.offer(best, bestCost);

  let iterations = 0;
  outer: for (let it = 0; it < opts.iterations; it++) {
    for (const chain of chains) {
      iterations++;
      const p = propose(chain.rng, chain.current, opts);
      chain.stats.proposals++;
      const cost = evaluator.cost(p.program);
      const total = cost.total + p.penalty;
      const delta = total - chain.currentCost;
      const logAlpha = -delta / Math.max(chain.temperature, 1e-6) + p.logQRatio;
      const accept = delta <= 0 || Math.log(Math.max(chain.rng.next(), 1e-12)) < logAlpha;
      if (accept) {
        chain.stats.accepted++;
        if (delta > 0) chain.stats.uphillAccepted++;
        chain.current = p.program;
        chain.currentCost = total;
        pareto.offer(p.program, cost);
        if (cost.total < bestCost.total) {
          best = p.program.slice();
          bestCost = cost;
          chain.stats.bestCost = cost.total;
          emit({
            t: Date.now() - t0,
            engine: `mcmc/chain${chain.stats.chain}`,
            kind: "improve",
            message: `${p.kind} move → cost ${cost.total.toFixed(3)} (len ${p.program.length})`,
            cost: cost.total,
          });
        }
        if (cost.correct && (!bestCorrectCost || cost.perf < bestCorrectCost.perf)) {
          bestCorrect = p.program.slice();
          bestCorrectCost = cost;
          emit({
            t: Date.now() - t0,
            engine: `mcmc/chain${chain.stats.chain}`,
            kind: "correct",
            message: `test-equivalent rewrite, perf ${cost.perf.toFixed(3)}, len ${p.program.length}`,
            cost: cost.total,
          });
        }
      } else {
        chain.stats.rejected++;
      }
      // Geometric annealing schedule.
      chain.temperature = Math.max(
        opts.minTemperature,
        chain.temperature * Math.pow(opts.annealing, 1 / 1000),
      );
      chain.stats.finalTemperature = chain.temperature;
    }

    if (opts.exchangeInterval > 0 && it % opts.exchangeInterval === 0 && chains.length > 1) {
      for (let c = 0; c + 1 < chains.length; c++) {
        const a = chains[c];
        const b = chains[c + 1];
        const logSwap =
          (a.currentCost - b.currentCost) * (1 / Math.max(a.temperature, 1e-6) - 1 / Math.max(b.temperature, 1e-6));
        if (Math.log(Math.max(a.rng.next(), 1e-12)) < logSwap) {
          const tmpProg = a.current;
          const tmpCost = a.currentCost;
          a.current = b.current;
          a.currentCost = b.currentCost;
          b.current = tmpProg;
          b.currentCost = tmpCost;
          a.stats.exchanges++;
          b.stats.exchanges++;
        }
      }
    }

    if ((it & 63) === 0 && Date.now() - t0 > opts.timeBudgetMs) {
      emit({ t: Date.now() - t0, engine: "mcmc", kind: "budget", message: "time budget exhausted" });
      break outer;
    }
  }

  return {
    best,
    bestCost,
    bestCorrect,
    bestCorrectCost,
    pareto: pareto.list(),
    stats: chains.map((c) => c.stats),
    events,
    iterations,
    elapsedMs: Date.now() - t0,
  };
}
