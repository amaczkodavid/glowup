import { desc, eq } from "drizzle-orm";
import { db } from "@/db";
import { candidates, counterexamples, runs, searchEvents } from "@/db/schema";
import { getTask } from "@/lib/superopt/tasks";
import { runHybrid } from "@/lib/superopt/hybrid";
import type { HybridConfig } from "@/lib/superopt/hybrid";
import { fingerprint } from "@/lib/superopt/testcases";

export const dynamic = "force-dynamic";
export const maxDuration = 300;

export async function GET() {
  const rows = await db
    .select({
      id: runs.id,
      taskId: runs.taskId,
      isa: runs.isa,
      status: runs.status,
      bestText: runs.bestText,
      verified: runs.verified,
      speedup: runs.speedup,
      targetLength: runs.targetLength,
      bestLength: runs.bestLength,
      elapsedMs: runs.elapsedMs,
      createdAt: runs.createdAt,
    })
    .from(runs)
    .orderBy(desc(runs.id))
    .limit(40);
  return Response.json({ runs: rows });
}

interface RunRequest {
  taskId?: string;
  config?: Partial<HybridConfig>;
}

const clamp = (v: number | undefined, lo: number, hi: number, dflt: number): number =>
  typeof v === "number" && Number.isFinite(v) ? Math.min(hi, Math.max(lo, Math.round(v))) : dflt;

export async function POST(request: Request) {
  let body: RunRequest;
  try {
    body = (await request.json()) as RunRequest;
  } catch {
    return Response.json({ error: "invalid JSON body" }, { status: 400 });
  }
  const taskId = body.taskId ?? "abs";
  let task;
  try {
    task = getTask(taskId);
  } catch {
    return Response.json({ error: `unknown task '${taskId}'` }, { status: 404 });
  }

  const inbound = body.config ?? {};
  const config: Partial<HybridConfig> = {
    isa: inbound.isa === "aarch64" ? "aarch64" : "x86-64",
    rounds: clamp(inbound.rounds, 1, 6, 3),
    seed: clamp(inbound.seed, 1, 2 ** 31 - 1, 20260401),
    mcmcMs: clamp(inbound.mcmcMs, 100, 8000, 1200),
    mctsMs: clamp(inbound.mctsMs, 0, 6000, 700),
    enumerativeMs: clamp(inbound.enumerativeMs, 0, 6000, 500),
    verifyMs: clamp(inbound.verifyMs, 100, 8000, 1500),
    mcmcChains: clamp(inbound.mcmcChains, 1, 8, 4),
    mcmcIterations: clamp(inbound.mcmcIterations, 1000, 400000, 40000),
    mctsSimulations: clamp(inbound.mctsSimulations, 100, 40000, 3000),
    randomTests: clamp(inbound.randomTests, 8, 256, 48),
    weightsWe: typeof inbound.weightsWe === "number" ? inbound.weightsWe : 1,
    weightsWp: typeof inbound.weightsWp === "number" ? inbound.weightsWp : 1,
    temperature: typeof inbound.temperature === "number" ? inbound.temperature : 12,
    annealing: typeof inbound.annealing === "number" ? inbound.annealing : 0.35,
    engines: {
      mcmc: inbound.engines?.mcmc ?? true,
      mcts: inbound.engines?.mcts ?? true,
      enumerative: inbound.engines?.enumerative ?? true,
      symbolic: inbound.engines?.symbolic ?? true,
    },
  };

  const report = runHybrid(task, config);

  const [inserted] = await db
    .insert(runs)
    .values({
      taskId: report.taskId,
      isa: report.isa,
      status: "completed",
      config: report.config,
      report,
      bestText: report.best?.text ?? null,
      bestAsm: report.best?.asm ?? null,
      verified: report.best?.verified ?? false,
      speedup: report.speedup ?? null,
      targetLength: report.target.length,
      bestLength: report.best?.length ?? null,
      elapsedMs: report.elapsedMs,
    })
    .returning({ id: runs.id });

  const runId = inserted.id;

  if (report.candidates.length > 0) {
    await db.insert(candidates).values(
      report.candidates.map((c) => ({
        runId,
        source: c.source,
        text: c.text,
        asm: c.asm,
        costTotal: c.cost.total,
        costEq: c.cost.eq,
        costPerf: c.cost.perf,
        length: c.length,
        correct: c.cost.correct,
        verified: c.verified,
      })),
    );
  }

  if (report.events.length > 0) {
    await db.insert(searchEvents).values(
      report.events.slice(-200).map((e) => ({
        runId,
        tMs: Math.round(e.t),
        engine: e.engine,
        kind: e.kind,
        message: e.message,
        cost: e.cost ?? null,
      })),
    );
  }

  for (const cex of report.counterexamples) {
    await db
      .insert(counterexamples)
      .values({
        taskId: report.taskId,
        runId,
        fingerprint: fingerprint(cex),
        gpr: cex.gpr,
        origin: cex.origin,
        detail: `discovered during run ${runId}`,
        weight: cex.weight,
      })
      .onConflictDoNothing();
  }

  const persisted = await db.select().from(runs).where(eq(runs.id, runId)).limit(1);
  return Response.json({ id: runId, report, run: persisted[0] ?? null });
}
