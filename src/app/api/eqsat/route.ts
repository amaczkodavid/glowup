import { desc } from "drizzle-orm";
import { db } from "@/db";
import { eqsatRuns } from "@/db/schema";
import { TENSOR_PROGRAMS, runEqsat } from "@/lib/superopt/egraph";

export const dynamic = "force-dynamic";
export const maxDuration = 120;

export async function GET() {
  const rows = await db.select().from(eqsatRuns).orderBy(desc(eqsatRuns.id)).limit(25);
  return Response.json({ runs: rows });
}

export async function POST(request: Request) {
  let body: { programId?: string; rules?: string[] };
  try {
    body = (await request.json()) as { programId?: string; rules?: string[] };
  } catch {
    return Response.json({ error: "invalid JSON body" }, { status: 400 });
  }
  const program = TENSOR_PROGRAMS.find((p) => p.id === body.programId) ?? TENSOR_PROGRAMS[0];
  const result = runEqsat(program, body.rules);

  const [row] = await db
    .insert(eqsatRuns)
    .values({
      programId: program.id,
      beforeCost: result.before.cost,
      afterCost: result.after.cost,
      beforeExpr: result.before.text,
      afterExpr: result.after.text,
      optimal: result.after.optimal,
      eclasses: result.stats.classes,
      enodes: result.stats.nodes,
      ilpVariables: result.after.model.variables.length,
      ilpConstraints: result.after.model.constraints.length,
      stats: {
        saturation: result.stats,
        exploredNodes: result.after.exploredNodes,
        greedyCost: result.after.greedyCost,
        objective: result.after.model.objective,
      },
      elapsedMs: result.stats.elapsedMs + result.after.elapsedMs,
    })
    .returning({ id: eqsatRuns.id });

  return Response.json({
    id: row.id,
    program: { id: program.id, name: program.name, env: program.env, description: program.description },
    before: result.before,
    after: {
      cost: result.after.cost,
      text: result.after.text,
      optimal: result.after.optimal,
      exploredNodes: result.after.exploredNodes,
      greedyCost: result.after.greedyCost,
      elapsedMs: result.after.elapsedMs,
    },
    speedup: result.speedup,
    stats: result.stats,
    ilp: {
      objective: result.after.model.objective,
      binaryCount: result.after.model.binaryCount,
      variables: result.after.model.variables.slice(0, 60),
      constraints: result.after.model.constraints.slice(0, 60),
      totalVariables: result.after.model.variables.length,
      totalConstraints: result.after.model.constraints.length,
    },
  });
}
