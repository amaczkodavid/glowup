import { desc } from "drizzle-orm";
import { db } from "@/db";
import { counterexamples, verifications } from "@/db/schema";
import { parseProgram } from "@/lib/superopt/isa";
import { getTask } from "@/lib/superopt/tasks";
import { boundedEnumeration, verifyEquivalence } from "@/lib/superopt/smt";
import { fingerprint } from "@/lib/superopt/testcases";
import { DEFAULT_MACHINE } from "@/lib/superopt/types";

export const dynamic = "force-dynamic";
export const maxDuration = 120;

export async function GET() {
  const rows = await db.select().from(verifications).orderBy(desc(verifications.id)).limit(25);
  return Response.json({ verifications: rows });
}

export async function POST(request: Request) {
  let body: { taskId?: string; program?: string; width?: number; method?: "smt" | "enumeration" };
  try {
    body = (await request.json()) as typeof body;
  } catch {
    return Response.json({ error: "invalid JSON body" }, { status: 400 });
  }
  let task;
  try {
    task = getTask(body.taskId ?? "abs");
  } catch {
    return Response.json({ error: `unknown task '${body.taskId}'` }, { status: 404 });
  }
  let program;
  try {
    program = parseProgram(body.program ?? "");
  } catch (err) {
    return Response.json({ error: (err as Error).message }, { status: 400 });
  }
  if (program.length === 0) {
    return Response.json({ error: "empty candidate program" }, { status: 400 });
  }
  const width = Math.min(16, Math.max(2, Math.round(body.width ?? task.verifyWidth)));
  const opts = {
    width,
    inputGpr: task.inputGpr,
    live: task.live,
    cfg: DEFAULT_MACHINE,
    timeMs: 8000,
  };
  const result =
    body.method === "enumeration"
      ? boundedEnumeration(task.target, program, opts)
      : verifyEquivalence(task.target, program, opts);

  await db.insert(verifications).values({
    taskId: task.id,
    candidateText: body.program ?? "",
    status: result.status,
    method: result.method,
    width: result.width,
    variables: result.variables,
    clauses: result.clauses,
    decisions: result.decisions,
    propagations: result.propagations,
    conflicts: result.conflicts,
    detail: result.detail,
    elapsedMs: result.elapsedMs,
  });

  if (result.counterexample) {
    await db
      .insert(counterexamples)
      .values({
        taskId: task.id,
        fingerprint: fingerprint(result.counterexample),
        gpr: result.counterexample.gpr,
        origin: result.counterexample.origin,
        detail: result.detail,
        weight: result.counterexample.weight,
      })
      .onConflictDoNothing();
  }

  return Response.json({ task: task.id, result });
}
