/**
 * Hybrid parallel orchestrator.
 *
 * Runs the stochastic (MCMC), symbolic (SMT-guided verification/refutation),
 * enumerative (bidirectional meet-in-the-middle) and RL (MCTS) engines against
 * a shared knowledge store: the dynamic test database, the incumbent Pareto
 * front, and the verified-candidate table. Engines execute in interleaved time
 * slices (cooperative round-robin); every counterexample discovered by the SMT
 * layer immediately invalidates the cost caches of all other engines, and every
 * incumbent found by one engine becomes the restart point of the others.
 */

import { disassemble, formatProgram } from "./isa";
import { CostEvaluator, programHash } from "./cost";
import { deadCodeEliminate } from "./machine";
import { PerfModel, staticPerf } from "./perf";
import { DEFAULT_IMMEDIATES, runMcmc } from "./mcmc";
import type { McmcOptions } from "./mcmc";
import { runMcts } from "./mcts";
import { runEnumerative } from "./enumerative";
import { verifyEquivalence } from "./smt";
import { Rng, TestDatabase, adversarialTests, randomTest } from "./testcases";
import { stateFromTest } from "./machine";
import type { Task } from "./tasks";
import type {
  Candidate,
  CostBreakdown,
  MachineConfig,
  Program,
  SearchEvent,
  TestCase,
  VerificationResult,
} from "./types";
import { DEFAULT_MACHINE, DEFAULT_WEIGHTS } from "./types";

export interface HybridConfig {
  isa: string;
  rounds: number;
  seed: number;
  /** Per-round time slices in milliseconds. */
  mcmcMs: number;
  mctsMs: number;
  enumerativeMs: number;
  verifyMs: number;
  mcmcChains: number;
  mcmcIterations: number;
  mctsSimulations: number;
  randomTests: number;
  weightsWe: number;
  weightsWp: number;
  temperature: number;
  annealing: number;
  engines: { mcmc: boolean; mcts: boolean; enumerative: boolean; symbolic: boolean };
}

export const DEFAULT_HYBRID: HybridConfig = {
  isa: "x86-64",
  rounds: 3,
  seed: 20260401,
  mcmcMs: 1200,
  mctsMs: 700,
  enumerativeMs: 500,
  verifyMs: 1500,
  mcmcChains: 4,
  mcmcIterations: 40000,
  mctsSimulations: 3000,
  randomTests: 48,
  weightsWe: 1,
  weightsWp: 1,
  temperature: 12,
  annealing: 0.35,
  engines: { mcmc: true, mcts: true, enumerative: true, symbolic: true },
};

export interface EngineSummary {
  engine: string;
  invocations: number;
  elapsedMs: number;
  bestCost: number | null;
  detail: Record<string, number | string | boolean>;
}

export interface HybridReport {
  taskId: string;
  isa: string;
  config: HybridConfig;
  target: {
    text: string;
    asm: string;
    perf: number;
    length: number;
    cycles: number;
  };
  best: {
    text: string;
    asm: string;
    perf: number;
    length: number;
    cycles: number;
    source: string;
    verified: boolean;
  } | null;
  speedup: number | null;
  sizeReduction: number | null;
  verification: VerificationResult | null;
  verificationHistory: VerificationResult[];
  pareto: Array<{ text: string; eq: number; perf: number; length: number; correct: boolean }>;
  candidates: Array<{
    text: string;
    asm: string;
    source: string;
    cost: CostBreakdown;
    verified: boolean;
    length: number;
  }>;
  counterexamples: TestCase[];
  engines: EngineSummary[];
  events: SearchEvent[];
  testCount: number;
  costEvaluations: number;
  cacheHits: number;
  perfModel: { samples: number; meanAbsError: number; weights: number[]; names: string[] };
  elapsedMs: number;
}

class SharedStore {
  readonly candidates = new Map<string, Candidate>();
  readonly events: SearchEvent[] = [];
  readonly counterexamples: TestCase[] = [];
  readonly verifications: VerificationResult[] = [];
  bestVerified: Candidate | null = null;
  offer(program: Program, cost: CostBreakdown, source: Candidate["source"]): Candidate {
    const key = programHash(program);
    const existing = this.candidates.get(key);
    if (existing) return existing;
    const c: Candidate = {
      program: program.slice(),
      cost,
      source,
      discoveredAtMs: Date.now(),
      verified: false,
    };
    this.candidates.set(key, c);
    return c;
  }
  correctCandidates(): Candidate[] {
    return [...this.candidates.values()]
      .filter((c) => c.cost.correct)
      .sort((a, b) => a.cost.perf - b.cost.perf || a.program.length - b.program.length);
  }
}

export function runHybrid(task: Task, cfgIn: Partial<HybridConfig> = {}): HybridReport {
  const config: HybridConfig = { ...DEFAULT_HYBRID, ...cfgIn, engines: { ...DEFAULT_HYBRID.engines, ...(cfgIn.engines ?? {}) } };
  const t0 = Date.now();
  const machine: MachineConfig = { ...DEFAULT_MACHINE };
  const rng = new Rng(config.seed);
  const tests = new TestDatabase(adversarialTests(machine));
  for (let i = 0; i < config.randomTests; i++) tests.add(randomTest(rng, machine));

  const perfModel = new PerfModel({ isa: config.isa, sizeWeight: DEFAULT_WEIGHTS.sizeWeight, measureProbability: 0.05 });
  const weights = { ...DEFAULT_WEIGHTS, we: config.weightsWe, wp: config.weightsWp };
  let testList = tests.all();
  const evaluator = new CostEvaluator({
    target: task.target,
    tests: testList,
    live: task.live,
    weights,
    cfg: machine,
    perfModel,
  });

  // Real timing samples train the learned latency predictor.
  const calibrationState = stateFromTest(testList[0], machine);
  perfModel.calibrate(task.target, calibrationState, machine);
  for (let i = 1; i <= 8; i++) {
    perfModel.calibrate(task.target.slice(0, Math.max(1, task.target.length - i)), calibrationState, machine);
  }

  const store = new SharedStore();
  const emit = (e: SearchEvent) => store.events.push(e);
  const targetCost = evaluator.cost(task.target);
  store.offer(task.target, targetCost, "target");

  const engineSummaries: EngineSummary[] = [
    { engine: "mcmc", invocations: 0, elapsedMs: 0, bestCost: null, detail: {} },
    { engine: "mcts", invocations: 0, elapsedMs: 0, bestCost: null, detail: {} },
    { engine: "enumerative", invocations: 0, elapsedMs: 0, bestCost: null, detail: {} },
    { engine: "symbolic", invocations: 0, elapsedMs: 0, bestCost: null, detail: {} },
  ];
  const summary = (name: string) => engineSummaries.find((e) => e.engine === name)!;

  const paretoAll = new Map<string, { program: Program; cost: CostBreakdown }>();
  let incumbent: Program = task.target.slice();

  for (let round = 0; round < config.rounds; round++) {
    emit({ t: Date.now() - t0, engine: "orchestrator", kind: "round", message: `round ${round + 1}/${config.rounds} starting with ${testList.length} tests` });

    if (config.engines.mcmc) {
      const opts: McmcOptions = {
        pool: task.pool,
        maxLength: Math.max(task.maxLength, task.target.length),
        cfg: machine,
        temperature: config.temperature,
        annealing: config.annealing,
        minTemperature: 0.25,
        iterations: config.mcmcIterations,
        chains: config.mcmcChains,
        seed: config.seed + round * 977,
        exchangeInterval: 64,
        immediates: DEFAULT_IMMEDIATES,
        timeBudgetMs: config.mcmcMs,
      };
      const res = runMcmc(evaluator, incumbent, opts, emit);
      const s = summary("mcmc");
      s.invocations++;
      s.elapsedMs += res.elapsedMs;
      s.bestCost = res.bestCost.total;
      s.detail = {
        iterations: res.iterations,
        chains: opts.chains,
        accepted: res.stats.reduce((a, b) => a + b.accepted, 0),
        uphillAccepted: res.stats.reduce((a, b) => a + b.uphillAccepted, 0),
        exchanges: res.stats.reduce((a, b) => a + b.exchanges, 0),
        finalTemperature: Number(res.stats[0]?.finalTemperature.toFixed(3) ?? 0),
      };
      for (const p of res.pareto) {
        paretoAll.set(programHash(p.program), p);
        store.offer(p.program, p.cost, "mcmc");
      }
      if (res.bestCorrect) {
        store.offer(res.bestCorrect, res.bestCorrectCost!, "mcmc");
        incumbent = res.bestCorrect;
      }
    }

    if (config.engines.mcts) {
      const res = runMcts(
        evaluator,
        {
          pool: task.pool,
          maxLength: Math.min(task.maxLength, 6),
          cfg: machine,
          simulations: config.mctsSimulations,
          explorationC: 1.6,
          seed: config.seed + round * 311 + 7,
          timeBudgetMs: config.mctsMs,
          branching: 48,
          immediates: DEFAULT_IMMEDIATES,
          policyLr: 0.05,
          valueLr: 0.05,
          rewardScale: Math.max(4, targetCost.total),
        },
        emit,
      );
      const s = summary("mcts");
      s.invocations++;
      s.elapsedMs += res.elapsedMs;
      s.bestCost = res.bestCost.total;
      s.detail = {
        simulations: res.simulations,
        expandedNodes: res.expandedNodes,
        policyUpdates: res.policyUpdates,
        valueUpdates: res.valueUpdates,
        meanReward: Number(res.meanReward.toFixed(4)),
      };
      if (res.bestCorrect) store.offer(res.bestCorrect, res.bestCorrectCost!, "mcts");
      store.offer(res.best, res.bestCost, "mcts");
    }

    if (config.engines.enumerative) {
      const res = runEnumerative(
        evaluator,
        task.target,
        testList.slice(0, 8),
        {
          pool: task.pool,
          cfg: machine,
          maxForwardDepth: 2,
          maxBackwardDepth: 2,
          frontierLimit: 6000,
          timeBudgetMs: config.enumerativeMs,
          immediates: DEFAULT_IMMEDIATES,
          seed: config.seed + round * 131,
        },
        emit,
      );
      const s = summary("enumerative");
      s.invocations++;
      s.elapsedMs += res.elapsedMs;
      s.bestCost = res.bestCost?.total ?? null;
      s.detail = {
        forwardStates: res.forwardStates,
        backwardStates: res.backwardStates,
        joins: res.joins,
        found: res.found.length,
      };
      for (const f of res.found) store.offer(f.program, f.cost, "enumerative");
    }

    if (config.engines.symbolic) {
      const s = summary("symbolic");
      const verifyStart = Date.now();
      const pending = store
        .correctCandidates()
        .filter((c) => !c.verified && programHash(c.program) !== programHash(task.target))
        .slice(0, 4);
      for (const cand of pending) {
        const verdict = verifyEquivalence(task.target, cand.program, {
          width: task.verifyWidth,
          inputGpr: task.inputGpr,
          live: task.live,
          cfg: machine,
          timeMs: config.verifyMs,
        });
        s.invocations++;
        store.verifications.push(verdict);
        cand.verdict = verdict;
        if (verdict.status === "equivalent") {
          cand.verified = true;
          if (!store.bestVerified || cand.cost.perf < store.bestVerified.cost.perf) store.bestVerified = cand;
          emit({
            t: Date.now() - t0,
            engine: "symbolic",
            kind: "proved",
            message: `${verdict.method}: UNSAT (${verdict.clauses} clauses, ${verdict.variables} vars) — rewrite proven equivalent`,
          });
        } else if (verdict.status === "counterexample" && verdict.counterexample) {
          store.counterexamples.push(verdict.counterexample);
          tests.add(verdict.counterexample);
          emit({
            t: Date.now() - t0,
            engine: "symbolic",
            kind: "counterexample",
            message: `${verdict.detail} — added to dynamic test database`,
          });
        } else {
          emit({ t: Date.now() - t0, engine: "symbolic", kind: "budget", message: verdict.detail });
        }
        if (Date.now() - verifyStart > config.verifyMs * 2) break;
      }
      s.elapsedMs += Date.now() - verifyStart;
    }

    // Share knowledge: refresh the test vector and invalidate cost caches.
    const newList = tests.all();
    if (newList.length !== testList.length) {
      testList = newList;
      evaluator.ctx.tests.length = 0;
      evaluator.ctx.tests.push(...testList);
      evaluator.invalidate();
      // Re-score every candidate against the enlarged test set.
      for (const cand of store.candidates.values()) cand.cost = evaluator.cost(cand.program);
    }
  }

  // Final selection: prefer verified, then correct-on-tests with lowest perf.
  const correct = store.correctCandidates();
  const chosen =
    store.bestVerified ??
    correct.find((c) => programHash(c.program) !== programHash(task.target)) ??
    correct[0] ??
    null;

  let best: HybridReport["best"] = null;
  let finalVerdict: VerificationResult | null = chosen?.verdict ?? null;
  if (chosen) {
    const cleaned = deadCodeEliminate(chosen.program, task.live.gpr, task.live.fpr);
    const cleanedCost = evaluator.cost(cleaned);
    const program = cleanedCost.correct && cleanedCost.perf <= chosen.cost.perf ? cleaned : chosen.program;
    if (program !== chosen.program) {
      finalVerdict = verifyEquivalence(task.target, program, {
        width: task.verifyWidth,
        inputGpr: task.inputGpr,
        live: task.live,
        cfg: machine,
        timeMs: config.verifyMs,
      });
      store.verifications.push(finalVerdict);
    }
    const sp = staticPerf(program, config.isa);
    best = {
      text: formatProgram(program),
      asm: disassemble(program, config.isa),
      perf: perfModel.perf(program),
      length: program.length,
      cycles: sp.cycles,
      source: chosen.source,
      verified: finalVerdict?.status === "equivalent" || chosen.verified,
    };
  }

  const targetPerf = perfModel.perf(task.target);
  const targetStatic = staticPerf(task.target, config.isa);
  const pareto = [...paretoAll.values()]
    .sort((a, b) => a.cost.eq - b.cost.eq || a.cost.perf - b.cost.perf)
    .slice(0, 32)
    .map((p) => ({
      text: formatProgram(p.program),
      eq: p.cost.eq,
      perf: p.cost.perf,
      length: p.program.length,
      correct: p.cost.correct,
    }));

  return {
    taskId: task.id,
    isa: config.isa,
    config,
    target: {
      text: formatProgram(task.target),
      asm: disassemble(task.target, config.isa),
      perf: targetPerf,
      length: task.target.length,
      cycles: targetStatic.cycles,
    },
    best,
    speedup: best ? targetPerf / Math.max(best.perf, 1e-6) : null,
    sizeReduction: best ? 1 - best.length / task.target.length : null,
    verification: finalVerdict,
    verificationHistory: store.verifications,
    pareto,
    candidates: [...store.candidates.values()]
      .sort((a, b) => a.cost.total - b.cost.total)
      .slice(0, 24)
      .map((c) => ({
        text: formatProgram(c.program),
        asm: disassemble(c.program, config.isa),
        source: c.source,
        cost: c.cost,
        verified: c.verified,
        length: c.program.length,
      })),
    counterexamples: store.counterexamples,
    engines: engineSummaries,
    events: store.events.slice(-400),
    testCount: testList.length,
    costEvaluations: evaluator.evaluations,
    cacheHits: evaluator.cacheHits,
    perfModel: {
      samples: perfModel.learned.samples,
      meanAbsError: perfModel.learned.meanAbsError,
      weights: perfModel.learned.weights,
      names: perfModel.learned.snapshot().names,
    },
    elapsedMs: Date.now() - t0,
  };
}
