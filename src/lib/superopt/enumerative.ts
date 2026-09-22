/**
 * Bidirectional (meet-in-the-middle) enumerative search.
 *
 * Forward direction: breadth-first enumeration of prefixes, each summarised by
 * the *signature* of the machine states it produces on the shared test vector
 * (all architectural registers over all tests). Signatures deduplicate the
 * frontier, which is what makes the enumeration tractable.
 *
 * Backward direction: enumeration of invertible suffixes. An instruction is
 * invertible when it is of the read-modify-write form `d = g(d, s)` with an
 * inverse `g⁻¹` (add/sub/xor/not/neg/rol/addi/xori). Applying the inverse of a
 * suffix to the *target's* output states yields the intermediate signature the
 * prefix must reach; a hit in the forward table composes a full program.
 *
 * A program is emitted when prefix ⧺ suffix passes the whole dynamic test set;
 * it is then handed to the SMT layer for a real proof.
 */

import { spec } from "./isa";
import { execute, stateFromTest, toUnsigned } from "./machine";
import type { CostEvaluator } from "./cost";
import { Rng } from "./testcases";
import type { CostBreakdown, CpuState, Instruction, MachineConfig, Program, SearchEvent, TestCase } from "./types";

export interface EnumOptions {
  pool: string[];
  cfg: MachineConfig;
  maxForwardDepth: number;
  maxBackwardDepth: number;
  frontierLimit: number;
  timeBudgetMs: number;
  immediates: number[];
  seed: number;
}

export interface EnumResult {
  found: Array<{ program: Program; cost: CostBreakdown }>;
  best: Program | null;
  bestCost: CostBreakdown | null;
  forwardStates: number;
  backwardStates: number;
  joins: number;
  events: SearchEvent[];
  elapsedMs: number;
}

const INVERTIBLE_OPS = ["add", "sub", "xor", "not", "neg", "rol", "addi", "xori"];

function signature(states: CpuState[], width: number, gprCount: number): string {
  const parts: string[] = [];
  for (const s of states) {
    for (let r = 0; r < gprCount; r++) parts.push(toUnsigned(s.gpr[r], width).toString(36));
    parts.push(s.fault === "none" ? "" : s.fault);
  }
  return parts.join(",");
}

function candidateInstructions(opts: EnumOptions, rng: Rng, limit: number): Instruction[] {
  const out: Instruction[] = [];
  const regs = Array.from({ length: Math.min(opts.cfg.gprCount, 3) }, (_, i) => i);
  for (const op of opts.pool) {
    const s = spec(op);
    if (s.cls === "memory" || s.cls === "vector" || s.cls === "float") continue;
    const dsts = s.roles.includes("rd") ? regs : [-1];
    for (const rd of dsts) {
      const s1s = s.roles.includes("rs1") ? regs : [-1];
      for (const rs1 of s1s) {
        const s2s = s.roles.includes("rs2") ? regs : [-1];
        for (const rs2 of s2s) {
          if (s.roles.includes("imm")) {
            const imms = opts.immediates.filter((x) => x >= s.immMin && x <= s.immMax).slice(0, 8);
            for (const imm of imms) out.push({ op, rd, rs1, rs2, rs3: -1, imm });
          } else {
            out.push({ op, rd, rs1, rs2, rs3: -1, imm: 0 });
          }
        }
      }
    }
  }
  if (out.length <= limit) return out;
  const sampled: Instruction[] = [];
  const seen = new Set<number>();
  while (sampled.length < limit) {
    const i = rng.int(out.length);
    if (seen.has(i)) continue;
    seen.add(i);
    sampled.push(out[i]);
  }
  return sampled;
}

/** Inverse of an invertible read-modify-write instruction, or null. */
export function invert(ins: Instruction): Instruction | null {
  if (!INVERTIBLE_OPS.includes(ins.op)) return null;
  switch (ins.op) {
    case "add":
      if (ins.rd !== ins.rs1 || ins.rd === ins.rs2) return null;
      return { ...ins, op: "sub" };
    case "sub":
      if (ins.rd !== ins.rs1 || ins.rd === ins.rs2) return null;
      return { ...ins, op: "add" };
    case "xor":
      if (ins.rd !== ins.rs1 || ins.rd === ins.rs2) return null;
      return { ...ins };
    case "not":
      if (ins.rd !== ins.rs1) return null;
      return { ...ins };
    case "neg":
      if (ins.rd !== ins.rs1) return null;
      return { ...ins };
    case "rol":
      if (ins.rd !== ins.rs1) return null;
      return { ...ins, imm: (32 - (ins.imm % 32)) % 32 };
    case "addi":
      if (ins.rd !== ins.rs1) return null;
      return { ...ins, imm: -ins.imm };
    case "xori":
      if (ins.rd !== ins.rs1) return null;
      return { ...ins };
    default:
      return null;
  }
}

export function runEnumerative(
  evaluator: CostEvaluator,
  target: Program,
  tests: TestCase[],
  opts: EnumOptions,
  onEvent?: (e: SearchEvent) => void,
): EnumResult {
  const t0 = Date.now();
  const events: SearchEvent[] = [];
  const emit = (e: SearchEvent) => {
    events.push(e);
    onEvent?.(e);
  };
  const rng = new Rng(opts.seed);
  const cfg = opts.cfg;
  const initial = tests.map((t) => stateFromTest(t, cfg));
  const targetStates = tests.map((t) => execute(target, stateFromTest(t, cfg), cfg));

  // ---- forward enumeration -------------------------------------------------
  const forward = new Map<string, Program>();
  let frontier: Array<{ prog: Program; states: CpuState[] }> = [
    { prog: [], states: initial.map((s) => ({ ...s, gpr: Uint32Array.from(s.gpr), fpr: Float64Array.from(s.fpr), vreg: Float64Array.from(s.vreg), mem: Uint8Array.from(s.mem), flags: { ...s.flags } })) },
  ];
  forward.set(signature(frontier[0].states, cfg.width, cfg.gprCount), []);
  let forwardStates = 1;
  const instrs = candidateInstructions(opts, rng, 4000);

  for (let depth = 0; depth < opts.maxForwardDepth; depth++) {
    const next: Array<{ prog: Program; states: CpuState[] }> = [];
    for (const item of frontier) {
      for (const ins of instrs) {
        const states = item.states.map((s) => {
          const copy: CpuState = {
            gpr: Uint32Array.from(s.gpr),
            fpr: Float64Array.from(s.fpr),
            vreg: Float64Array.from(s.vreg),
            mem: Uint8Array.from(s.mem),
            flags: { ...s.flags },
            fault: s.fault,
            faultDetail: s.faultDetail,
            retired: s.retired,
          };
          return execute([ins], copy, cfg);
        });
        const sig = signature(states, cfg.width, cfg.gprCount);
        if (forward.has(sig)) continue;
        const prog = [...item.prog, ins];
        forward.set(sig, prog);
        forwardStates++;
        next.push({ prog, states });
        if (forward.size >= opts.frontierLimit) break;
      }
      if (forward.size >= opts.frontierLimit || Date.now() - t0 > opts.timeBudgetMs * 0.6) break;
    }
    frontier = next;
    emit({
      t: Date.now() - t0,
      engine: "enumerative",
      kind: "frontier",
      message: `forward depth ${depth + 1}: ${forward.size} distinct signatures`,
    });
    if (forward.size >= opts.frontierLimit || Date.now() - t0 > opts.timeBudgetMs * 0.6) break;
  }

  // ---- backward enumeration ------------------------------------------------
  const found: Array<{ program: Program; cost: CostBreakdown }> = [];
  let joins = 0;
  let backwardStates = 0;
  const invertible = instrs.filter((i) => invert(i) !== null);

  interface BackItem {
    suffix: Program;
    states: CpuState[];
  }
  let backFrontier: BackItem[] = [{ suffix: [], states: targetStates }];
  const checkJoin = (item: BackItem) => {
    const sig = signature(item.states, cfg.width, cfg.gprCount);
    const prefix = forward.get(sig);
    if (!prefix) return;
    joins++;
    const program = [...prefix, ...item.suffix];
    const cost = evaluator.cost(program);
    if (cost.correct) {
      found.push({ program, cost });
      emit({
        t: Date.now() - t0,
        engine: "enumerative",
        kind: "join",
        message: `meet-in-the-middle join: |prefix|=${prefix.length}, |suffix|=${item.suffix.length}, perf=${cost.perf.toFixed(2)}`,
        cost: cost.total,
      });
    }
  };
  checkJoin(backFrontier[0]);

  for (let depth = 0; depth < opts.maxBackwardDepth; depth++) {
    const next: BackItem[] = [];
    for (const item of backFrontier) {
      for (const ins of invertible) {
        const inv = invert(ins);
        if (!inv) continue;
        const states = item.states.map((s) => {
          const copy: CpuState = {
            gpr: Uint32Array.from(s.gpr),
            fpr: Float64Array.from(s.fpr),
            vreg: Float64Array.from(s.vreg),
            mem: Uint8Array.from(s.mem),
            flags: { ...s.flags },
            fault: s.fault,
            faultDetail: s.faultDetail,
            retired: s.retired,
          };
          return execute([inv], copy, cfg);
        });
        const back: BackItem = { suffix: [ins, ...item.suffix], states };
        backwardStates++;
        next.push(back);
        checkJoin(back);
        if (Date.now() - t0 > opts.timeBudgetMs) break;
      }
      if (Date.now() - t0 > opts.timeBudgetMs) break;
    }
    backFrontier = next.slice(0, Math.max(1, Math.floor(opts.frontierLimit / 8)));
    if (Date.now() - t0 > opts.timeBudgetMs) break;
  }

  found.sort((a, b) => a.cost.perf - b.cost.perf || a.program.length - b.program.length);
  return {
    found,
    best: found.length ? found[0].program : null,
    bestCost: found.length ? found[0].cost : null,
    forwardStates,
    backwardStates,
    joins,
    events,
    elapsedMs: Date.now() - t0,
  };
}
