import {
  boolean,
  doublePrecision,
  index,
  integer,
  jsonb,
  pgTable,
  serial,
  text,
  timestamp,
  uniqueIndex,
} from "drizzle-orm/pg-core";

/** One orchestrated synthesis run (hybrid search over a benchmark task). */
export const runs = pgTable(
  "runs",
  {
    id: serial("id").primaryKey(),
    taskId: text("task_id").notNull(),
    isa: text("isa").notNull(),
    status: text("status").notNull().default("completed"),
    config: jsonb("config").notNull(),
    report: jsonb("report").notNull(),
    bestText: text("best_text"),
    bestAsm: text("best_asm"),
    verified: boolean("verified").notNull().default(false),
    speedup: doublePrecision("speedup"),
    targetLength: integer("target_length").notNull(),
    bestLength: integer("best_length"),
    elapsedMs: integer("elapsed_ms").notNull(),
    createdAt: timestamp("created_at", { withTimezone: true }).notNull().defaultNow(),
  },
  (t) => [index("runs_task_idx").on(t.taskId), index("runs_created_idx").on(t.createdAt)],
);

/** Candidate rewrites discovered during a run. */
export const candidates = pgTable(
  "candidates",
  {
    id: serial("id").primaryKey(),
    runId: integer("run_id")
      .notNull()
      .references(() => runs.id, { onDelete: "cascade" }),
    source: text("source").notNull(),
    text: text("text").notNull(),
    asm: text("asm").notNull(),
    costTotal: doublePrecision("cost_total").notNull(),
    costEq: doublePrecision("cost_eq").notNull(),
    costPerf: doublePrecision("cost_perf").notNull(),
    length: integer("length").notNull(),
    correct: boolean("correct").notNull(),
    verified: boolean("verified").notNull(),
    createdAt: timestamp("created_at", { withTimezone: true }).notNull().defaultNow(),
  },
  (t) => [index("candidates_run_idx").on(t.runId)],
);

/** Dynamic test database: counterexamples produced by the SMT layer. */
export const counterexamples = pgTable(
  "counterexamples",
  {
    id: serial("id").primaryKey(),
    taskId: text("task_id").notNull(),
    runId: integer("run_id").references(() => runs.id, { onDelete: "set null" }),
    fingerprint: text("fingerprint").notNull(),
    gpr: jsonb("gpr").notNull(),
    origin: text("origin").notNull(),
    detail: text("detail").notNull().default(""),
    weight: doublePrecision("weight").notNull().default(10),
    hits: integer("hits").notNull().default(1),
    createdAt: timestamp("created_at", { withTimezone: true }).notNull().defaultNow(),
  },
  (t) => [uniqueIndex("counterexamples_unique").on(t.taskId, t.fingerprint)],
);

/** Search telemetry emitted by every engine. */
export const searchEvents = pgTable(
  "search_events",
  {
    id: serial("id").primaryKey(),
    runId: integer("run_id")
      .notNull()
      .references(() => runs.id, { onDelete: "cascade" }),
    tMs: integer("t_ms").notNull(),
    engine: text("engine").notNull(),
    kind: text("kind").notNull(),
    message: text("message").notNull(),
    cost: doublePrecision("cost"),
  },
  (t) => [index("events_run_idx").on(t.runId)],
);

/** Equality saturation + ILP extraction runs on tensor graphs. */
export const eqsatRuns = pgTable(
  "eqsat_runs",
  {
    id: serial("id").primaryKey(),
    programId: text("program_id").notNull(),
    beforeCost: doublePrecision("before_cost").notNull(),
    afterCost: doublePrecision("after_cost").notNull(),
    beforeExpr: text("before_expr").notNull(),
    afterExpr: text("after_expr").notNull(),
    optimal: boolean("optimal").notNull(),
    eclasses: integer("eclasses").notNull(),
    enodes: integer("enodes").notNull(),
    ilpVariables: integer("ilp_variables").notNull(),
    ilpConstraints: integer("ilp_constraints").notNull(),
    stats: jsonb("stats").notNull(),
    elapsedMs: integer("elapsed_ms").notNull(),
    createdAt: timestamp("created_at", { withTimezone: true }).notNull().defaultNow(),
  },
  (t) => [index("eqsat_program_idx").on(t.programId)],
);

/** Standalone verification queries issued from the UI/API. */
export const verifications = pgTable("verifications", {
  id: serial("id").primaryKey(),
  taskId: text("task_id").notNull(),
  candidateText: text("candidate_text").notNull(),
  status: text("status").notNull(),
  method: text("method").notNull(),
  width: integer("width").notNull(),
  variables: integer("variables").notNull(),
  clauses: integer("clauses").notNull(),
  decisions: integer("decisions").notNull(),
  propagations: integer("propagations").notNull(),
  conflicts: integer("conflicts").notNull(),
  detail: text("detail").notNull(),
  elapsedMs: integer("elapsed_ms").notNull(),
  createdAt: timestamp("created_at", { withTimezone: true }).notNull().defaultNow(),
});
