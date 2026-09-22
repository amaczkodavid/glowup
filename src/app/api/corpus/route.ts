import { desc, eq } from "drizzle-orm";
import { db } from "@/db";
import { counterexamples } from "@/db/schema";

export const dynamic = "force-dynamic";

export async function GET(request: Request) {
  const url = new URL(request.url);
  const taskId = url.searchParams.get("taskId");
  const base = db.select().from(counterexamples);
  const rows = taskId
    ? await base.where(eq(counterexamples.taskId, taskId)).orderBy(desc(counterexamples.id)).limit(100)
    : await base.orderBy(desc(counterexamples.id)).limit(100);
  return Response.json({ counterexamples: rows });
}
