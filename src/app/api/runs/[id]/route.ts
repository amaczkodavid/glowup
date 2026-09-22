import { asc, eq } from "drizzle-orm";
import { db } from "@/db";
import { candidates, runs, searchEvents } from "@/db/schema";

export const dynamic = "force-dynamic";

export async function GET(
  _request: Request,
  context: { params: Promise<{ id: string }> },
) {
  const { id } = await context.params;
  const runId = Number.parseInt(id, 10);
  if (!Number.isFinite(runId)) {
    return Response.json({ error: "invalid run id" }, { status: 400 });
  }
  const [run] = await db.select().from(runs).where(eq(runs.id, runId)).limit(1);
  if (!run) return Response.json({ error: "run not found" }, { status: 404 });
  const cands = await db
    .select()
    .from(candidates)
    .where(eq(candidates.runId, runId))
    .orderBy(asc(candidates.costTotal))
    .limit(64);
  const events = await db
    .select()
    .from(searchEvents)
    .where(eq(searchEvents.runId, runId))
    .orderBy(asc(searchEvents.tMs))
    .limit(400);
  return Response.json({ run, candidates: cands, events });
}
