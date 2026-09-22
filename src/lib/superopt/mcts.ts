/**
 * Reinforcement-learning assembly synthesis.
 *
 * The synthesis problem is a single-player game: a state is a program prefix,
 * an action appends one instruction (or halts). Search is UCT/PUCT Monte-Carlo
 * tree search guided by two online-trained linear models:
 *
 *   • policy  p(a | s) = softmax(θ_p · φ(s, a))  trained by cross-entropy
 *     against the MCTS visit distribution (AlphaZero-style policy iteration);
 *   • value   V(s)     = θ_v · ψ(s)              trained by regression against
 *     the observed returns of the rollouts under s.
 *
 * The reward is derived from the same cost function used by the stochastic
 * superoptimiser, so all engines share a single objective.
 */

import { spec } from "./isa";
import type { CostEvaluator } from "./cost";
import { Rng } from "./testcases";
import type { CostBreakdown, Instruction, MachineConfig, Program, SearchEvent } from "./types";
import { OPSPEC } from "./isa";

export interface MctsOptions {
  pool: string[];
  maxLength: number;
  cfg: MachineConfig;
  simulations: number;
  explorationC: number;
  seed: number;
  timeBudgetMs: number;
  /** Maximum actions expanded per node. */
  branching: number;
  immediates: number[];
  /** Learning rates. */
  policyLr: number;
  valueLr: number;
  /** Reward temperature: r = exp(-cost / rewardScale). */
  rewardScale: number;
}

interface Node {
  prefix: Program;
  actions: Instruction[];
  priors: number[];
  visits: number[];
  values: number[];
  children: Array<Node | null>;
  totalVisits: number;
  terminal: boolean;
  valueEstimate: number;
}

const HALT: Instruction = { op: "__halt__", rd: -1, rs1: -1, rs2: -1, rs3: -1, imm: 0 };

function actionFeatures(prefix: Program, action: Instruction, opts: MctsOptions): number[] {
  const isHalt = action.op === "__halt__";
  const s = isHalt ? null : spec(action.op);
  const lengthFrac = prefix.length / Math.max(1, opts.maxLength);
  const usesFreshDst = isHalt ? 0 : prefix.some((i) => i.rd === action.rd) ? 0 : 1;
  const readsRecent = isHalt
    ? 0
    : prefix.length > 0 && (prefix[prefix.length - 1].rd === action.rs1 || prefix[prefix.length - 1].rd === action.rs2)
      ? 1
      : 0;
  return [
    isHalt ? 1 : 0,
    s ? s.latency / 10 : 0,
    s ? s.uops / 4 : 0,
    s && s.cls === "arith" ? 1 : 0,
    s && s.cls === "logic" ? 1 : 0,
    s && s.cls === "shift" ? 1 : 0,
    s && s.cls === "bit" ? 1 : 0,
    s && s.cls === "select" ? 1 : 0,
    s && s.cls === "float" ? 1 : 0,
    usesFreshDst,
    readsRecent,
    lengthFrac,
    1,
  ];
}

function stateFeatures(prefix: Program, opts: MctsOptions): number[] {
  const counts = { arith: 0, logic: 0, shift: 0, bit: 0, select: 0, other: 0 };
  for (const ins of prefix) {
    const cls = OPSPEC[ins.op]?.cls ?? "other";
    if (cls in counts) (counts as Record<string, number>)[cls] += 1;
    else counts.other += 1;
  }
  return [
    prefix.length / Math.max(1, opts.maxLength),
    counts.arith,
    counts.logic,
    counts.shift,
    counts.bit,
    counts.select,
    counts.other,
    1,
  ];
}

class LinearModel {
  w: number[];
  updates = 0;
  loss = 0;
  constructor(dim: number, readonly lr: number) {
    this.w = new Array(dim).fill(0);
  }
  dot(f: number[]): number {
    let acc = 0;
    for (let i = 0; i < this.w.length; i++) acc += this.w[i] * f[i];
    return acc;
  }
  sgd(f: number[], grad: number): void {
    for (let i = 0; i < this.w.length; i++) this.w[i] -= this.lr * grad * f[i];
    this.updates++;
  }
}

export interface MctsResult {
  best: Program;
  bestCost: CostBreakdown;
  bestCorrect: Program | null;
  bestCorrectCost: CostBreakdown | null;
  simulations: number;
  expandedNodes: number;
  policyUpdates: number;
  valueUpdates: number;
  meanReward: number;
  events: SearchEvent[];
  elapsedMs: number;
}

export function runMcts(
  evaluator: CostEvaluator,
  opts: MctsOptions,
  onEvent?: (e: SearchEvent) => void,
): MctsResult {
  const t0 = Date.now();
  const rng = new Rng(opts.seed);
  const events: SearchEvent[] = [];
  const emit = (e: SearchEvent) => {
    events.push(e);
    onEvent?.(e);
  };
  const policy = new LinearModel(actionFeatures([], HALT, opts).length, opts.policyLr);
  const value = new LinearModel(stateFeatures([], opts).length, opts.valueLr);

  const enumerateActions = (prefix: Program): Instruction[] => {
    const out: Instruction[] = [HALT];
    const defined = new Set<number>([0, 1]);
    for (const ins of prefix) if (ins.rd >= 0) defined.add(ins.rd);
    const dstChoices = [...new Set([...defined, Math.min(prefix.length + 2, opts.cfg.gprCount - 1)])];
    for (const op of opts.pool) {
      const s = spec(op);
      if (s.cls === "memory" || s.cls === "vector") continue;
      const cap = s.cls === "float" ? opts.cfg.fprCount : opts.cfg.gprCount;
      const srcPool = s.cls === "float" ? Array.from({ length: cap }, (_, i) => i) : [...defined];
      const dsts = s.cls === "float" ? Array.from({ length: cap }, (_, i) => i) : dstChoices;
      for (const rd of dsts) {
        const roles = s.roles;
        const needs1 = roles.includes("rs1");
        const needs2 = roles.includes("rs2");
        const needs3 = roles.includes("rs3");
        const needsImm = roles.includes("imm");
        const s1s = needs1 ? srcPool : [-1];
        for (const rs1 of s1s) {
          const s2s = needs2 ? srcPool : [-1];
          for (const rs2 of s2s) {
            const s3s = needs3 ? srcPool : [-1];
            for (const rs3 of s3s) {
              if (needsImm) {
                const imms = opts.immediates.filter((x) => x >= s.immMin && x <= s.immMax);
                const chosen = imms.length > 6 ? [0, 1, 2, 3, 4, 5].map((k) => imms[(k * 7 + prefix.length) % imms.length]) : imms;
                for (const imm of chosen) out.push({ op, rd: roles.includes("rd") || roles.includes("vd") ? rd : -1, rs1, rs2, rs3, imm });
              } else {
                out.push({ op, rd: roles.includes("rd") || roles.includes("vd") ? rd : -1, rs1, rs2, rs3, imm: 0 });
              }
            }
          }
        }
      }
    }
    // Subsample to the branching factor, always keeping HALT.
    if (out.length <= opts.branching) return out;
    const sampled: Instruction[] = [HALT];
    while (sampled.length < opts.branching) sampled.push(out[1 + rng.int(out.length - 1)]);
    return sampled;
  };

  const makeNode = (prefix: Program): Node => {
    const terminal = prefix.length >= opts.maxLength;
    const actions = terminal ? [HALT] : enumerateActions(prefix);
    const logits = actions.map((a) => policy.dot(actionFeatures(prefix, a, opts)));
    const maxLogit = Math.max(...logits);
    const exps = logits.map((l) => Math.exp(l - maxLogit));
    const sum = exps.reduce((a, b) => a + b, 0) || 1;
    return {
      prefix,
      actions,
      priors: exps.map((e) => e / sum),
      visits: new Array(actions.length).fill(0),
      values: new Array(actions.length).fill(0),
      children: new Array(actions.length).fill(null),
      totalVisits: 0,
      terminal,
      valueEstimate: value.dot(stateFeatures(prefix, opts)),
    };
  };

  const reward = (prog: Program): { r: number; cost: CostBreakdown } => {
    const cost = evaluator.cost(prog);
    return { r: Math.exp(-cost.total / opts.rewardScale), cost };
  };

  const rollout = (prefix: Program): { r: number; prog: Program; cost: CostBreakdown } => {
    const prog = prefix.slice();
    while (prog.length < opts.maxLength) {
      if (rng.next() < 0.25) break;
      const actions = enumerateActions(prog);
      const a = actions[rng.int(actions.length)];
      if (a.op === "__halt__") break;
      prog.push(a);
    }
    const { r, cost } = reward(prog);
    return { r, prog, cost };
  };

  const root = makeNode([]);
  let expandedNodes = 1;
  let best: Program = [];
  let bestCost = evaluator.cost([]);
  let bestCorrect: Program | null = bestCost.correct ? [] : null;
  let bestCorrectCost: CostBreakdown | null = bestCost.correct ? bestCost : null;
  let rewardSum = 0;
  let simulations = 0;

  for (let sim = 0; sim < opts.simulations; sim++) {
    if ((sim & 15) === 0 && Date.now() - t0 > opts.timeBudgetMs) {
      emit({ t: Date.now() - t0, engine: "mcts", kind: "budget", message: "time budget exhausted" });
      break;
    }
    simulations++;
    const path: Array<{ node: Node; action: number }> = [];
    let node = root;
    for (;;) {
      // PUCT selection.
      let bestIdx = 0;
      let bestScore = -Infinity;
      for (let i = 0; i < node.actions.length; i++) {
        const q = node.visits[i] > 0 ? node.values[i] / node.visits[i] : node.valueEstimate;
        const u =
          opts.explorationC * node.priors[i] * Math.sqrt(node.totalVisits + 1) / (1 + node.visits[i]);
        const score = q + u;
        if (score > bestScore) {
          bestScore = score;
          bestIdx = i;
        }
      }
      path.push({ node, action: bestIdx });
      const action = node.actions[bestIdx];
      if (action.op === "__halt__" || node.terminal) break;
      let child = node.children[bestIdx];
      if (!child) {
        child = makeNode([...node.prefix, action]);
        node.children[bestIdx] = child;
        expandedNodes++;
        node = child;
        break;
      }
      node = child;
    }

    const leafPrefix = node.prefix;
    const { r, prog, cost } = rollout(leafPrefix);
    rewardSum += r;
    if (cost.total < bestCost.total) {
      best = prog.slice();
      bestCost = cost;
      emit({
        t: Date.now() - t0,
        engine: "mcts",
        kind: "improve",
        message: `rollout cost ${cost.total.toFixed(3)} (len ${prog.length})`,
        cost: cost.total,
      });
    }
    if (cost.correct && (!bestCorrectCost || cost.perf < bestCorrectCost.perf)) {
      bestCorrect = prog.slice();
      bestCorrectCost = cost;
      emit({
        t: Date.now() - t0,
        engine: "mcts",
        kind: "correct",
        message: `synthesised test-equivalent program, perf ${cost.perf.toFixed(3)}`,
        cost: cost.total,
      });
    }

    // Backup.
    for (const step of path) {
      step.node.visits[step.action] += 1;
      step.node.values[step.action] += r;
      step.node.totalVisits += 1;
    }
    // Value regression toward the observed return.
    for (const step of path) {
      const f = stateFeatures(step.node.prefix, opts);
      const pred = value.dot(f);
      value.sgd(f, pred - r);
      step.node.valueEstimate = value.dot(f);
    }
    // Policy improvement: cross-entropy against normalised visit counts.
    for (const step of path) {
      const nd = step.node;
      if (nd.totalVisits < 4) continue;
      const feats = nd.actions.map((a) => actionFeatures(nd.prefix, a, opts));
      const logits = feats.map((f) => policy.dot(f));
      const maxLogit = Math.max(...logits);
      const exps = logits.map((l) => Math.exp(l - maxLogit));
      const z = exps.reduce((a, b) => a + b, 0) || 1;
      for (let i = 0; i < nd.actions.length; i++) {
        const target = nd.visits[i] / nd.totalVisits;
        const p = exps[i] / z;
        policy.sgd(feats[i], p - target);
      }
      nd.priors = exps.map((e) => e / z);
    }
  }

  return {
    best,
    bestCost,
    bestCorrect,
    bestCorrectCost,
    simulations,
    expandedNodes,
    policyUpdates: policy.updates,
    valueUpdates: value.updates,
    meanReward: simulations ? rewardSum / simulations : 0,
    events,
    elapsedMs: Date.now() - t0,
  };
}
