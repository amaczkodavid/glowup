import { TASKS } from "@/lib/superopt/tasks";
import { disassemble, formatProgram, listIsas } from "@/lib/superopt/isa";
import { staticPerf } from "@/lib/superopt/perf";
import { TENSOR_PROGRAMS, TENSOR_RULES, exprToString } from "@/lib/superopt/egraph";
import { DEFAULT_HYBRID } from "@/lib/superopt/hybrid";

export const dynamic = "force-dynamic";

export async function GET() {
  return Response.json({
    isas: listIsas(),
    defaults: DEFAULT_HYBRID,
    tasks: TASKS.map((t) => ({
      id: t.id,
      name: t.name,
      category: t.category,
      description: t.description,
      target: formatProgram(t.target),
      asmX86: disassemble(t.target, "x86-64"),
      asmArm: disassemble(t.target, "aarch64"),
      length: t.target.length,
      cycles: staticPerf(t.target, "x86-64").cycles,
      pool: t.pool,
      maxLength: t.maxLength,
      knownOptimum: t.knownOptimum ?? null,
      verifyWidth: t.verifyWidth,
      live: t.live,
      inputGpr: t.inputGpr,
    })),
    tensorPrograms: TENSOR_PROGRAMS.map((p) => ({
      id: p.id,
      name: p.name,
      description: p.description,
      env: p.env,
      expr: exprToString(p.expr),
    })),
    rules: TENSOR_RULES.map((r) => r.name),
  });
}
