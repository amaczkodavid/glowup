"use client";

import { useCallback, useEffect, useMemo, useState } from "react";

type TaskInfo = {
  id: string;
  name: string;
  category: string;
  description: string;
  target: string;
  asmX86: string;
  asmArm: string;
  length: number;
  cycles: number;
  pool: string[];
  maxLength: number;
  knownOptimum: string | null;
  verifyWidth: number;
};

type TensorProgramInfo = {
  id: string;
  name: string;
  description: string;
  env: Record<string, [number, number]>;
  expr: string;
};

type CostBreakdown = {
  total: number;
  eq: number;
  perf: number;
  bitErrors: number;
  ulpSum: number;
  ulpMax: number;
  faults: number;
  correct: boolean;
};

type Verification = {
  status: string;
  method: string;
  width: number;
  variables: number;
  clauses: number;
  decisions: number;
  propagations: number;
  conflicts: number;
  elapsedMs: number;
  detail: string;
};

type Report = {
  taskId: string;
  isa: string;
  target: { text: string; asm: string; perf: number; length: number; cycles: number };
  best: { text: string; asm: string; perf: number; length: number; cycles: number; source: string; verified: boolean } | null;
  speedup: number | null;
  sizeReduction: number | null;
  verification: Verification | null;
  verificationHistory: Verification[];
  pareto: Array<{ text: string; eq: number; perf: number; length: number; correct: boolean }>;
  candidates: Array<{ text: string; asm: string; source: string; cost: CostBreakdown; verified: boolean; length: number }>;
  counterexamples: Array<{ id: string; gpr: number[]; origin: string }>;
  engines: Array<{ engine: string; invocations: number; elapsedMs: number; bestCost: number | null; detail: Record<string, string | number | boolean> }>;
  events: Array<{ t: number; engine: string; kind: string; message: string; cost?: number }>;
  testCount: number;
  costEvaluations: number;
  cacheHits: number;
  perfModel: { samples: number; meanAbsError: number; weights: number[]; names: string[] };
  elapsedMs: number;
};

type RunRow = {
  id: number;
  taskId: string;
  isa: string;
  bestText: string | null;
  verified: boolean;
  speedup: number | null;
  targetLength: number;
  bestLength: number | null;
  elapsedMs: number;
  createdAt: string;
};

type EqsatResponse = {
  id: number;
  program: { id: string; name: string; env: Record<string, [number, number]>; description: string };
  before: { cost: number; text: string };
  after: { cost: number; text: string; optimal: boolean; exploredNodes: number; greedyCost: number; elapsedMs: number };
  speedup: number;
  stats: { iterations: number; applied: number; classes: number; nodes: number; saturated: boolean; perRule: Record<string, number>; elapsedMs: number };
  ilp: {
    objective: string;
    binaryCount: number;
    variables: Array<{ name: string; kind: string; cost: number }>;
    constraints: Array<{ name: string; expr: string; sense: string; rhs: number }>;
    totalVariables: number;
    totalConstraints: number;
  };
};

type SourceFile = { path: string; bytes: number; language: string; root: string };

const TABS = ["synthesis", "eqsat", "verifier", "corpus", "source"] as const;
type Tab = (typeof TABS)[number];

const number = (v: number | null | undefined, digits = 2) =>
  v === null || v === undefined || !Number.isFinite(v) ? "—" : v.toFixed(digits);

function Panel({ title, children, right }: { title: string; children: React.ReactNode; right?: React.ReactNode }) {
  return (
    <section className="rounded-xl border border-slate-800 bg-slate-900/60 shadow-lg shadow-black/30">
      <header className="flex items-center justify-between border-b border-slate-800 px-4 py-2">
        <h2 className="text-xs font-semibold uppercase tracking-widest text-cyan-300">{title}</h2>
        {right}
      </header>
      <div className="p-4">{children}</div>
    </section>
  );
}

function Code({ text, className = "" }: { text: string; className?: string }) {
  return (
    <pre className={`overflow-auto rounded-lg bg-black/60 p-3 font-mono text-[11px] leading-relaxed text-emerald-200 ${className}`}>
      {text}
    </pre>
  );
}

function Stat({ label, value, accent = "text-slate-100" }: { label: string; value: string; accent?: string }) {
  return (
    <div className="rounded-lg border border-slate-800 bg-slate-950/60 px-3 py-2">
      <div className="text-[10px] uppercase tracking-wider text-slate-500">{label}</div>
      <div className={`font-mono text-sm ${accent}`}>{value}</div>
    </div>
  );
}

export default function Dashboard() {
  const [tab, setTab] = useState<Tab>("synthesis");
  const [tasks, setTasks] = useState<TaskInfo[]>([]);
  const [tensorPrograms, setTensorPrograms] = useState<TensorProgramInfo[]>([]);
  const [isas, setIsas] = useState<string[]>([]);
  const [taskId, setTaskId] = useState("abs");
  const [isa, setIsa] = useState("x86-64");
  const [rounds, setRounds] = useState(3);
  const [mcmcMs, setMcmcMs] = useState(1200);
  const [mctsMs, setMctsMs] = useState(700);
  const [enumMs, setEnumMs] = useState(500);
  const [engines, setEngines] = useState({ mcmc: true, mcts: true, enumerative: true, symbolic: true });
  const [running, setRunning] = useState(false);
  const [report, setReport] = useState<Report | null>(null);
  const [runs, setRuns] = useState<RunRow[]>([]);
  const [error, setError] = useState<string | null>(null);

  const [tensorId, setTensorId] = useState("matchain");
  const [eqsat, setEqsat] = useState<EqsatResponse | null>(null);
  const [eqsatBusy, setEqsatBusy] = useState(false);

  const [verifyTask, setVerifyTask] = useState("abs");
  const [verifyProgram, setVerifyProgram] = useState("abs r0, r0");
  const [verifyWidth, setVerifyWidth] = useState(8);
  const [verifyMethod, setVerifyMethod] = useState<"smt" | "enumeration">("smt");
  const [verifyResult, setVerifyResult] = useState<Verification | null>(null);
  const [verifyBusy, setVerifyBusy] = useState(false);

  const [corpus, setCorpus] = useState<Array<{ id: number; taskId: string; gpr: number[]; origin: string; detail: string; createdAt: string }>>([]);
  const [files, setFiles] = useState<SourceFile[]>([]);
  const [activeFile, setActiveFile] = useState<string | null>(null);
  const [fileContent, setFileContent] = useState("");

  const task = useMemo(() => tasks.find((t) => t.id === taskId) ?? null, [tasks, taskId]);

  const loadRuns = useCallback(async () => {
    const res = await fetch("/api/runs");
    const json = (await res.json()) as { runs: RunRow[] };
    setRuns(json.runs);
  }, []);

  useEffect(() => {
    void (async () => {
      const res = await fetch("/api/tasks");
      const json = (await res.json()) as { tasks: TaskInfo[]; isas: string[]; tensorPrograms: TensorProgramInfo[] };
      setTasks(json.tasks);
      setIsas(json.isas);
      setTensorPrograms(json.tensorPrograms);
      await loadRuns();
    })();
  }, [loadRuns]);

  useEffect(() => {
    if (tab !== "corpus") return;
    void (async () => {
      const res = await fetch("/api/corpus");
      const json = (await res.json()) as { counterexamples: typeof corpus };
      setCorpus(json.counterexamples);
    })();
  }, [tab]);

  useEffect(() => {
    if (tab !== "source" || files.length > 0) return;
    void (async () => {
      const res = await fetch("/api/source");
      const json = (await res.json()) as { files: SourceFile[] };
      setFiles(json.files);
    })();
  }, [tab, files.length]);

  const openFile = useCallback(async (p: string) => {
    setActiveFile(p);
    setFileContent("loading …");
    const res = await fetch(`/api/source?file=${encodeURIComponent(p)}`);
    const json = (await res.json()) as { content?: string; error?: string };
    setFileContent(json.content ?? json.error ?? "");
  }, []);

  const startRun = useCallback(async () => {
    setRunning(true);
    setError(null);
    try {
      const res = await fetch("/api/runs", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({
          taskId,
          config: { isa, rounds, mcmcMs, mctsMs, enumerativeMs: enumMs, engines },
        }),
      });
      const json = (await res.json()) as { report?: Report; error?: string };
      if (json.error) setError(json.error);
      if (json.report) setReport(json.report);
      await loadRuns();
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setRunning(false);
    }
  }, [taskId, isa, rounds, mcmcMs, mctsMs, enumMs, engines, loadRuns]);

  const runEqsat = useCallback(async () => {
    setEqsatBusy(true);
    try {
      const res = await fetch("/api/eqsat", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ programId: tensorId }),
      });
      setEqsat((await res.json()) as EqsatResponse);
    } finally {
      setEqsatBusy(false);
    }
  }, [tensorId]);

  const runVerify = useCallback(async () => {
    setVerifyBusy(true);
    try {
      const res = await fetch("/api/verify", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ taskId: verifyTask, program: verifyProgram, width: verifyWidth, method: verifyMethod }),
      });
      const json = (await res.json()) as { result?: Verification; error?: string };
      if (json.result) setVerifyResult(json.result);
      else setVerifyResult({ status: json.error ?? "error", method: "-", width: verifyWidth, variables: 0, clauses: 0, decisions: 0, propagations: 0, conflicts: 0, elapsedMs: 0, detail: json.error ?? "" });
    } finally {
      setVerifyBusy(false);
    }
  }, [verifyTask, verifyProgram, verifyWidth, verifyMethod]);

  const loadRun = useCallback(async (id: number) => {
    const res = await fetch(`/api/runs/${id}`);
    const json = (await res.json()) as { run: { report: Report } };
    setReport(json.run.report);
    setTab("synthesis");
  }, []);

  return (
    <main className="min-h-screen bg-slate-950 px-6 py-6 text-slate-200">
      <header className="mb-6 flex flex-wrap items-end justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold tracking-tight text-white">
            Superoptimising Synthesis Stack
          </h1>
          <p className="mt-1 max-w-3xl text-sm text-slate-400">
            Stochastic MCMC superoptimisation · RL/MCTS assembly synthesis · bidirectional
            meet-in-the-middle enumeration · symbolic execution with a bit-blasting DPLL solver ·
            equality saturation with exact ILP extraction · Zig runtime with GPU collectives and
            Futhark kernels.
          </p>
        </div>
        <nav className="flex gap-1 rounded-lg border border-slate-800 bg-slate-900 p-1">
          {TABS.map((t) => (
            <button
              key={t}
              onClick={() => setTab(t)}
              className={`rounded-md px-3 py-1.5 text-xs font-semibold uppercase tracking-wider transition ${
                tab === t ? "bg-cyan-500 text-slate-950" : "text-slate-400 hover:text-cyan-300"
              }`}
            >
              {t}
            </button>
          ))}
        </nav>
      </header>

      {tab === "synthesis" && (
        <div className="grid gap-4 xl:grid-cols-[380px_1fr]">
          <div className="flex flex-col gap-4">
            <Panel title="search configuration">
              <div className="flex flex-col gap-3 text-xs">
                <label className="flex flex-col gap-1">
                  <span className="text-slate-400">benchmark kernel</span>
                  <select
                    value={taskId}
                    onChange={(e) => setTaskId(e.target.value)}
                    className="rounded-md border border-slate-700 bg-slate-950 px-2 py-1.5 font-mono text-xs"
                  >
                    {tasks.map((t) => (
                      <option key={t.id} value={t.id}>
                        {t.id} — {t.name}
                      </option>
                    ))}
                  </select>
                </label>
                {task && <p className="text-[11px] leading-relaxed text-slate-400">{task.description}</p>}
                <label className="flex flex-col gap-1">
                  <span className="text-slate-400">target ISA</span>
                  <select
                    value={isa}
                    onChange={(e) => setIsa(e.target.value)}
                    className="rounded-md border border-slate-700 bg-slate-950 px-2 py-1.5 font-mono text-xs"
                  >
                    {isas.map((i) => (
                      <option key={i} value={i}>
                        {i}
                      </option>
                    ))}
                  </select>
                </label>
                <div className="grid grid-cols-2 gap-2">
                  <label className="flex flex-col gap-1">
                    <span className="text-slate-400">rounds</span>
                    <input type="number" min={1} max={6} value={rounds} onChange={(e) => setRounds(Number(e.target.value))} className="rounded-md border border-slate-700 bg-slate-950 px-2 py-1.5 font-mono" />
                  </label>
                  <label className="flex flex-col gap-1">
                    <span className="text-slate-400">MCMC ms/round</span>
                    <input type="number" min={100} max={8000} step={100} value={mcmcMs} onChange={(e) => setMcmcMs(Number(e.target.value))} className="rounded-md border border-slate-700 bg-slate-950 px-2 py-1.5 font-mono" />
                  </label>
                  <label className="flex flex-col gap-1">
                    <span className="text-slate-400">MCTS ms/round</span>
                    <input type="number" min={0} max={6000} step={100} value={mctsMs} onChange={(e) => setMctsMs(Number(e.target.value))} className="rounded-md border border-slate-700 bg-slate-950 px-2 py-1.5 font-mono" />
                  </label>
                  <label className="flex flex-col gap-1">
                    <span className="text-slate-400">enum ms/round</span>
                    <input type="number" min={0} max={6000} step={100} value={enumMs} onChange={(e) => setEnumMs(Number(e.target.value))} className="rounded-md border border-slate-700 bg-slate-950 px-2 py-1.5 font-mono" />
                  </label>
                </div>
                <div className="flex flex-wrap gap-2">
                  {(Object.keys(engines) as Array<keyof typeof engines>).map((k) => (
                    <button
                      key={k}
                      onClick={() => setEngines({ ...engines, [k]: !engines[k] })}
                      className={`rounded-md border px-2 py-1 font-mono text-[11px] ${
                        engines[k] ? "border-cyan-500 bg-cyan-500/10 text-cyan-300" : "border-slate-700 text-slate-500"
                      }`}
                    >
                      {k}
                    </button>
                  ))}
                </div>
                <button
                  onClick={startRun}
                  disabled={running}
                  className="mt-1 rounded-md bg-cyan-500 px-3 py-2 text-sm font-semibold text-slate-950 transition hover:bg-cyan-400 disabled:cursor-not-allowed disabled:bg-slate-700 disabled:text-slate-400"
                >
                  {running ? "searching + verifying …" : "run hybrid search"}
                </button>
                {error && <p className="text-[11px] text-rose-400">{error}</p>}
              </div>
            </Panel>

            {task && (
              <Panel title="reference implementation">
                <Code text={task.target} />
                <div className="mt-2 text-[11px] text-slate-400">
                  {task.length} instructions · {number(task.cycles, 1)} modelled cycles
                  {task.knownOptimum ? ` · known optimum: ${task.knownOptimum}` : ""}
                </div>
              </Panel>
            )}

            <Panel title="recent runs">
              <div className="max-h-72 overflow-auto text-[11px]">
                <table className="w-full">
                  <thead className="text-slate-500">
                    <tr>
                      <th className="text-left">#</th>
                      <th className="text-left">task</th>
                      <th className="text-left">isa</th>
                      <th className="text-right">speedup</th>
                      <th className="text-right">len</th>
                      <th className="text-center">proof</th>
                    </tr>
                  </thead>
                  <tbody className="font-mono">
                    {runs.map((r) => (
                      <tr key={r.id} onClick={() => void loadRun(r.id)} className="cursor-pointer hover:bg-slate-800/60">
                        <td>{r.id}</td>
                        <td>{r.taskId}</td>
                        <td>{r.isa}</td>
                        <td className="text-right">{number(r.speedup)}×</td>
                        <td className="text-right">{r.bestLength ?? "—"}/{r.targetLength}</td>
                        <td className="text-center">{r.verified ? "✓" : "·"}</td>
                      </tr>
                    ))}
                    {runs.length === 0 && (
                      <tr>
                        <td colSpan={6} className="py-3 text-center text-slate-600">no runs yet</td>
                      </tr>
                    )}
                  </tbody>
                </table>
              </div>
            </Panel>
          </div>

          <div className="flex flex-col gap-4">
            {!report && (
              <Panel title="result">
                <p className="text-sm text-slate-500">
                  Launch a run to synthesise, verify and disassemble an optimised rewrite.
                </p>
              </Panel>
            )}
            {report && (
              <>
                <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
                  <Stat label="speedup (modelled)" value={`${number(report.speedup)}×`} accent="text-cyan-300" />
                  <Stat label="size" value={`${report.best?.length ?? "—"} / ${report.target.length} instr`} />
                  <Stat label="proof" value={report.verification?.status ?? "—"} accent={report.best?.verified ? "text-emerald-300" : "text-amber-300"} />
                  <Stat label="wall clock" value={`${(report.elapsedMs / 1000).toFixed(2)} s`} />
                  <Stat label="dynamic tests" value={String(report.testCount)} />
                  <Stat label="cost evaluations" value={String(report.costEvaluations)} />
                  <Stat label="cost-cache hits" value={String(report.cacheHits)} />
                  <Stat label="perf-model samples" value={`${report.perfModel.samples} (MAE ${number(report.perfModel.meanAbsError, 3)})`} />
                </div>

                <div className="grid gap-4 lg:grid-cols-2">
                  <Panel title={`target — ${report.isa}`}>
                    <Code text={report.target.text} />
                    <Code text={report.target.asm} className="mt-2 text-sky-200" />
                  </Panel>
                  <Panel title={`synthesised rewrite — ${report.best?.source ?? "none"}`}>
                    {report.best ? (
                      <>
                        <Code text={report.best.text} />
                        <Code text={report.best.asm} className="mt-2 text-sky-200" />
                      </>
                    ) : (
                      <p className="text-sm text-slate-500">no improvement found within the budget</p>
                    )}
                  </Panel>
                </div>

                {report.verification && (
                  <Panel title="formal verification">
                    <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
                      <Stat label="status" value={report.verification.status} accent={report.verification.status === "equivalent" ? "text-emerald-300" : "text-amber-300"} />
                      <Stat label="method" value={report.verification.method} />
                      <Stat label="datapath width" value={`${report.verification.width} bit`} />
                      <Stat label="elapsed" value={`${report.verification.elapsedMs} ms`} />
                      <Stat label="CNF variables" value={String(report.verification.variables)} />
                      <Stat label="CNF clauses" value={String(report.verification.clauses)} />
                      <Stat label="decisions" value={String(report.verification.decisions)} />
                      <Stat label="propagations" value={String(report.verification.propagations)} />
                    </div>
                    <p className="mt-3 font-mono text-[11px] text-slate-400">{report.verification.detail}</p>
                  </Panel>
                )}

                <Panel title="engines">
                  <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-4">
                    {report.engines.map((e) => (
                      <div key={e.engine} className="rounded-lg border border-slate-800 bg-slate-950/60 p-3">
                        <div className="mb-1 font-mono text-xs uppercase tracking-wider text-cyan-300">{e.engine}</div>
                        <div className="text-[11px] text-slate-400">
                          {e.invocations} invocations · {e.elapsedMs} ms · best cost {number(e.bestCost)}
                        </div>
                        <dl className="mt-2 space-y-0.5 font-mono text-[10px] text-slate-500">
                          {Object.entries(e.detail).map(([k, v]) => (
                            <div key={k} className="flex justify-between gap-2">
                              <dt>{k}</dt>
                              <dd className="text-slate-300">{String(v)}</dd>
                            </div>
                          ))}
                        </dl>
                      </div>
                    ))}
                  </div>
                </Panel>

                <div className="grid gap-4 lg:grid-cols-2">
                  <Panel title="pareto front (correctness × performance)">
                    <div className="max-h-64 overflow-auto">
                      <table className="w-full text-[11px]">
                        <thead className="text-slate-500">
                          <tr>
                            <th className="text-left">program</th>
                            <th className="text-right">eq error</th>
                            <th className="text-right">perf</th>
                            <th className="text-right">len</th>
                          </tr>
                        </thead>
                        <tbody className="font-mono">
                          {report.pareto.map((p, i) => (
                            <tr key={i} className={p.correct ? "text-emerald-300" : "text-slate-300"}>
                              <td className="max-w-[280px] truncate">{p.text.replace(/\n/g, " ; ")}</td>
                              <td className="text-right">{number(p.eq)}</td>
                              <td className="text-right">{number(p.perf)}</td>
                              <td className="text-right">{p.length}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </Panel>
                  <Panel title="search telemetry">
                    <div className="max-h-64 overflow-auto font-mono text-[11px]">
                      {report.events.map((e, i) => (
                        <div key={i} className="flex gap-2 border-b border-slate-800/60 py-0.5">
                          <span className="w-14 shrink-0 text-slate-600">{e.t} ms</span>
                          <span className="w-32 shrink-0 text-cyan-400">{e.engine}</span>
                          <span className="w-24 shrink-0 text-amber-300">{e.kind}</span>
                          <span className="text-slate-300">{e.message}</span>
                        </div>
                      ))}
                    </div>
                  </Panel>
                </div>

                <Panel title="candidate table">
                  <div className="max-h-72 overflow-auto">
                    <table className="w-full text-[11px]">
                      <thead className="text-slate-500">
                        <tr>
                          <th className="text-left">source</th>
                          <th className="text-left">program</th>
                          <th className="text-right">cost</th>
                          <th className="text-right">eq</th>
                          <th className="text-right">perf</th>
                          <th className="text-center">ok</th>
                          <th className="text-center">proved</th>
                        </tr>
                      </thead>
                      <tbody className="font-mono">
                        {report.candidates.map((c, i) => (
                          <tr key={i} className="border-b border-slate-800/60">
                            <td className="text-cyan-400">{c.source}</td>
                            <td className="max-w-[420px] truncate">{c.text.replace(/\n/g, " ; ")}</td>
                            <td className="text-right">{number(c.cost.total)}</td>
                            <td className="text-right">{number(c.cost.eq)}</td>
                            <td className="text-right">{number(c.cost.perf)}</td>
                            <td className="text-center">{c.cost.correct ? "✓" : "·"}</td>
                            <td className="text-center">{c.verified ? "✓" : "·"}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </Panel>
              </>
            )}
          </div>
        </div>
      )}

      {tab === "eqsat" && (
        <div className="grid gap-4 xl:grid-cols-[360px_1fr]">
          <Panel title="tensor program">
            <div className="flex flex-col gap-3 text-xs">
              <select
                value={tensorId}
                onChange={(e) => setTensorId(e.target.value)}
                className="rounded-md border border-slate-700 bg-slate-950 px-2 py-1.5 font-mono text-xs"
              >
                {tensorPrograms.map((p) => (
                  <option key={p.id} value={p.id}>
                    {p.id} — {p.name}
                  </option>
                ))}
              </select>
              {tensorPrograms
                .filter((p) => p.id === tensorId)
                .map((p) => (
                  <div key={p.id} className="space-y-2">
                    <p className="text-[11px] leading-relaxed text-slate-400">{p.description}</p>
                    <Code text={p.expr} />
                    <Code text={Object.entries(p.env).map(([k, v]) => `${k}: ${v[0]}×${v[1]}`).join("\n")} className="text-amber-200" />
                  </div>
                ))}
              <button
                onClick={runEqsat}
                disabled={eqsatBusy}
                className="rounded-md bg-cyan-500 px-3 py-2 text-sm font-semibold text-slate-950 hover:bg-cyan-400 disabled:bg-slate-700 disabled:text-slate-400"
              >
                {eqsatBusy ? "saturating …" : "saturate + extract (ILP)"}
              </button>
            </div>
          </Panel>
          <div className="flex flex-col gap-4">
            {eqsat && (
              <>
                <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
                  <Stat label="cost before" value={eqsat.before.cost.toLocaleString()} />
                  <Stat label="cost after" value={eqsat.after.cost.toLocaleString()} accent="text-emerald-300" />
                  <Stat label="speedup" value={`${number(eqsat.speedup)}×`} accent="text-cyan-300" />
                  <Stat label="ILP optimal" value={eqsat.after.optimal ? "proved" : "budget"} />
                  <Stat label="e-classes" value={String(eqsat.stats.classes)} />
                  <Stat label="e-nodes" value={String(eqsat.stats.nodes)} />
                  <Stat label="rules applied" value={String(eqsat.stats.applied)} />
                  <Stat label="saturated" value={eqsat.stats.saturated ? "yes" : "iteration limit"} />
                </div>
                <div className="grid gap-4 lg:grid-cols-2">
                  <Panel title="input expression"><Code text={eqsat.before.text} /></Panel>
                  <Panel title="extracted expression"><Code text={eqsat.after.text} className="text-emerald-200" /></Panel>
                </div>
                <Panel title="ILP model" right={<span className="font-mono text-[10px] text-slate-500">{eqsat.ilp.totalVariables} vars · {eqsat.ilp.totalConstraints} rows · {eqsat.ilp.binaryCount} binary</span>}>
                  <Code text={eqsat.ilp.objective} className="text-amber-200" />
                  <Code
                    className="mt-2 max-h-64"
                    text={eqsat.ilp.constraints.map((c) => `${c.name}: ${c.expr} ${c.sense} ${c.rhs}`).join("\n")}
                  />
                </Panel>
                <Panel title="rewrite applications">
                  <div className="grid grid-cols-2 gap-2 font-mono text-[11px] md:grid-cols-4">
                    {Object.entries(eqsat.stats.perRule).map(([rule, n]) => (
                      <div key={rule} className="flex justify-between rounded border border-slate-800 px-2 py-1">
                        <span className="text-slate-400">{rule}</span>
                        <span className="text-cyan-300">{n}</span>
                      </div>
                    ))}
                  </div>
                </Panel>
              </>
            )}
            {!eqsat && (
              <Panel title="result">
                <p className="text-sm text-slate-500">Run saturation to see the e-graph statistics, the extracted DAG and the ILP model.</p>
              </Panel>
            )}
          </div>
        </div>
      )}

      {tab === "verifier" && (
        <div className="grid gap-4 xl:grid-cols-[420px_1fr]">
          <Panel title="candidate rewrite">
            <div className="flex flex-col gap-3 text-xs">
              <select value={verifyTask} onChange={(e) => setVerifyTask(e.target.value)} className="rounded-md border border-slate-700 bg-slate-950 px-2 py-1.5 font-mono text-xs">
                {tasks.map((t) => (
                  <option key={t.id} value={t.id}>{t.id}</option>
                ))}
              </select>
              <textarea
                value={verifyProgram}
                onChange={(e) => setVerifyProgram(e.target.value)}
                rows={8}
                spellCheck={false}
                className="rounded-md border border-slate-700 bg-black/60 p-2 font-mono text-[11px] text-emerald-200"
              />
              <div className="grid grid-cols-2 gap-2">
                <label className="flex flex-col gap-1">
                  <span className="text-slate-400">width (bits)</span>
                  <input type="number" min={2} max={16} value={verifyWidth} onChange={(e) => setVerifyWidth(Number(e.target.value))} className="rounded-md border border-slate-700 bg-slate-950 px-2 py-1.5 font-mono" />
                </label>
                <label className="flex flex-col gap-1">
                  <span className="text-slate-400">method</span>
                  <select value={verifyMethod} onChange={(e) => setVerifyMethod(e.target.value as "smt" | "enumeration")} className="rounded-md border border-slate-700 bg-slate-950 px-2 py-1.5 font-mono">
                    <option value="smt">bit-blast + DPLL</option>
                    <option value="enumeration">bounded enumeration</option>
                  </select>
                </label>
              </div>
              <button onClick={runVerify} disabled={verifyBusy} className="rounded-md bg-cyan-500 px-3 py-2 text-sm font-semibold text-slate-950 hover:bg-cyan-400 disabled:bg-slate-700 disabled:text-slate-400">
                {verifyBusy ? "solving …" : "verify equivalence"}
              </button>
              <p className="text-[11px] text-slate-500">
                Syntax: one instruction per line, e.g. <span className="font-mono text-slate-300">blsi r0, r0</span>,
                <span className="font-mono text-slate-300"> addi r1, r0, #-1</span>,
                <span className="font-mono text-slate-300"> fma f4, f1, f0, f2</span>.
              </p>
            </div>
          </Panel>
          <Panel title="solver verdict">
            {verifyResult ? (
              <>
                <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
                  <Stat label="status" value={verifyResult.status} accent={verifyResult.status === "equivalent" ? "text-emerald-300" : "text-rose-300"} />
                  <Stat label="method" value={verifyResult.method} />
                  <Stat label="width" value={`${verifyResult.width} bit`} />
                  <Stat label="elapsed" value={`${verifyResult.elapsedMs} ms`} />
                  <Stat label="variables" value={String(verifyResult.variables)} />
                  <Stat label="clauses" value={String(verifyResult.clauses)} />
                  <Stat label="decisions" value={String(verifyResult.decisions)} />
                  <Stat label="conflicts" value={String(verifyResult.conflicts)} />
                </div>
                <Code className="mt-3" text={verifyResult.detail} />
              </>
            ) : (
              <p className="text-sm text-slate-500">Submit a rewrite to bit-blast it against the reference implementation.</p>
            )}
          </Panel>
        </div>
      )}

      {tab === "corpus" && (
        <Panel title="dynamic test database (counterexample corpus)">
          <div className="max-h-[70vh] overflow-auto">
            <table className="w-full text-[11px]">
              <thead className="text-slate-500">
                <tr>
                  <th className="text-left">#</th>
                  <th className="text-left">task</th>
                  <th className="text-left">registers</th>
                  <th className="text-left">origin</th>
                  <th className="text-left">detail</th>
                </tr>
              </thead>
              <tbody className="font-mono">
                {corpus.map((c) => (
                  <tr key={c.id} className="border-b border-slate-800/60">
                    <td>{c.id}</td>
                    <td className="text-cyan-300">{c.taskId}</td>
                    <td>{(c.gpr ?? []).slice(0, 4).map((v, i) => `r${i}=${v}`).join(" ")}</td>
                    <td className="text-amber-300">{c.origin}</td>
                    <td className="max-w-[520px] truncate text-slate-400">{c.detail}</td>
                  </tr>
                ))}
                {corpus.length === 0 && (
                  <tr><td colSpan={5} className="py-4 text-center text-slate-600">no counterexamples recorded yet</td></tr>
                )}
              </tbody>
            </table>
          </div>
        </Panel>
      )}

      {tab === "source" && (
        <div className="grid gap-4 xl:grid-cols-[360px_1fr]">
          <Panel title="repository artefacts">
            <div className="max-h-[70vh] overflow-auto font-mono text-[11px]">
              {files.map((f) => (
                <button
                  key={f.path}
                  onClick={() => void openFile(f.path)}
                  className={`flex w-full justify-between gap-2 rounded px-2 py-1 text-left ${
                    activeFile === f.path ? "bg-cyan-500/10 text-cyan-300" : "text-slate-400 hover:bg-slate-800/60"
                  }`}
                >
                  <span className="truncate">{f.path}</span>
                  <span className="shrink-0 text-slate-600">{(f.bytes / 1024).toFixed(1)} KiB</span>
                </button>
              ))}
            </div>
          </Panel>
          <Panel title={activeFile ?? "select a file"}>
            <Code text={fileContent || "—"} className="max-h-[70vh]" />
          </Panel>
        </div>
      )}
    </main>
  );
}
