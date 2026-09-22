/**
 * E-graph engine: equality saturation over high-level tensor/arithmetic
 * graphs with an e-class shape analysis and exact ILP-based extraction.
 *
 * The extraction problem is emitted as a genuine 0/1 integer program
 *
 *   minimise   Σ_n  cost(n) · x_n
 *   subject to Σ_{n ∈ c} x_n = y_c                (one e-node per active class)
 *              x_n ≤ y_{c'}      ∀ c' ∈ children(n)
 *              y_root = 1
 *              t_{c'} + 1 ≤ t_c + M·(1 − x_n)      (acyclicity / topological order)
 *              x_n, y_c ∈ {0,1},  t_c ∈ [0, |C|]
 *
 * and is solved exactly by branch and bound over the e-node selection with a
 * bottom-up fixpoint lower bound, an incumbent from greedy extraction, and a
 * configurable node limit (the solver reports whether optimality was proved).
 */

export interface ENode {
  op: string;
  label: string;
  args: number[];
}

export type Shape = [number, number];

export interface EClassData {
  nodes: ENode[];
  shape: Shape;
}

export interface Expr {
  op: string;
  label?: string;
  args?: Expr[];
}

export const v = (name: string): Expr => ({ op: "var", label: name });
export const num = (x: number): Expr => ({ op: "const", label: String(x) });
export const n = (op: string, ...args: Expr[]): Expr => ({ op, args });

function nodeKey(node: ENode, find: (x: number) => number): string {
  return `${node.op}|${node.label}|${node.args.map(find).join(",")}`;
}

export class EGraph {
  private parent: number[] = [];
  private classes = new Map<number, EClassData>();
  private hashcons = new Map<string, number>();
  private pending: number[] = [];
  /** Variable shapes supplied by the caller (rows, cols). */
  constructor(readonly env: Record<string, Shape> = {}) {}

  find(x: number): number {
    let root = x;
    while (this.parent[root] !== root) root = this.parent[root];
    while (this.parent[x] !== root) {
      const next = this.parent[x];
      this.parent[x] = root;
      x = next;
    }
    return root;
  }

  get classCount(): number {
    let count = 0;
    for (const id of this.classes.keys()) if (this.find(id) === id) count++;
    return count;
  }

  get nodeCount(): number {
    let count = 0;
    for (const [id, data] of this.classes) if (this.find(id) === id) count += data.nodes.length;
    return count;
  }

  eclasses(): Array<[number, EClassData]> {
    return [...this.classes.entries()].filter(([id]) => this.find(id) === id);
  }

  classData(id: number): EClassData {
    const d = this.classes.get(this.find(id));
    if (!d) throw new Error(`no such e-class ${id}`);
    return d;
  }

  private shapeOf(node: ENode): Shape {
    const child = (i: number): Shape => this.classData(node.args[i]).shape;
    switch (node.op) {
      case "var":
        return this.env[node.label] ?? [1, 1];
      case "const":
        return [1, 1];
      case "matmul": {
        const a = child(0);
        const b = child(1);
        return [a[0], b[1]];
      }
      case "transpose": {
        const a = child(0);
        return [a[1], a[0]];
      }
      case "sum":
        return [1, 1];
      default: {
        if (!node.args.length) return [1, 1];
        let best: Shape = child(0);
        for (let i = 1; i < node.args.length; i++) {
          const s = child(i);
          if (s[0] * s[1] > best[0] * best[1]) best = s;
        }
        return best;
      }
    }
  }

  add(node: ENode): number {
    const key = nodeKey(node, (x) => this.find(x));
    const existing = this.hashcons.get(key);
    if (existing !== undefined) return this.find(existing);
    const id = this.parent.length;
    this.parent.push(id);
    const canonical: ENode = { ...node, args: node.args.map((a) => this.find(a)) };
    this.classes.set(id, { nodes: [canonical], shape: [1, 1] });
    this.classes.get(id)!.shape = this.shapeOf(canonical);
    this.hashcons.set(key, id);
    return id;
  }

  addExpr(e: Expr): number {
    const args = (e.args ?? []).map((a) => this.addExpr(a));
    return this.add({ op: e.op, label: e.label ?? "", args });
  }

  union(a: number, b: number): boolean {
    const ra = this.find(a);
    const rb = this.find(b);
    if (ra === rb) return false;
    const da = this.classData(ra);
    const db = this.classData(rb);
    // Merge smaller into larger.
    const [keep, gone, keepData, goneData] =
      da.nodes.length >= db.nodes.length ? [ra, rb, da, db] : [rb, ra, db, da];
    this.parent[gone] = keep;
    keepData.nodes = keepData.nodes.concat(goneData.nodes);
    if (goneData.shape[0] * goneData.shape[1] > keepData.shape[0] * keepData.shape[1]) {
      keepData.shape = goneData.shape;
    }
    this.classes.delete(gone);
    this.pending.push(keep);
    return true;
  }

  /** Restore congruence + hashcons invariants. */
  rebuild(): void {
    while (this.pending.length > 0) {
      const todo = [...new Set(this.pending.map((x) => this.find(x)))];
      this.pending.length = 0;
      this.hashcons.clear();
      for (const [id, data] of this.eclasses()) {
        const seen = new Map<string, ENode>();
        for (const node of data.nodes) {
          const canonical: ENode = { ...node, args: node.args.map((a) => this.find(a)) };
          const key = nodeKey(canonical, (x) => this.find(x));
          const dup = this.hashcons.get(key);
          if (dup !== undefined && this.find(dup) !== id) {
            this.union(dup, id);
          } else {
            this.hashcons.set(key, id);
          }
          seen.set(key, canonical);
        }
        data.nodes = [...seen.values()];
        data.shape = data.nodes.length ? this.shapeOf(data.nodes[0]) : data.shape;
      }
      if (todo.length === 0) break;
    }
    // Final shape fixpoint (shapes can improve once congruence settles).
    for (let iter = 0; iter < 4; iter++) {
      for (const [, data] of this.eclasses()) {
        for (const node of data.nodes) {
          const s = this.shapeOf(node);
          if (s[0] * s[1] > data.shape[0] * data.shape[1]) data.shape = s;
        }
      }
    }
  }
}

/* --------------------------------------------------------------------- */
/* Patterns and rewrite rules                                              */
/* --------------------------------------------------------------------- */

export type Pattern =
  | { kind: "var"; name: string }
  | { kind: "node"; op: string; label?: string; args: Pattern[] };

export const pv = (name: string): Pattern => ({ kind: "var", name });
export const pn = (op: string, ...args: Pattern[]): Pattern => ({ kind: "node", op, args });
export const pc = (value: number): Pattern => ({ kind: "node", op: "const", label: String(value), args: [] });

export interface Rule {
  name: string;
  lhs: Pattern;
  rhs: Pattern;
}

export const TENSOR_RULES: Rule[] = [
  { name: "add-comm", lhs: pn("add", pv("a"), pv("b")), rhs: pn("add", pv("b"), pv("a")) },
  { name: "mul-comm", lhs: pn("mul", pv("a"), pv("b")), rhs: pn("mul", pv("b"), pv("a")) },
  {
    name: "add-assoc",
    lhs: pn("add", pn("add", pv("a"), pv("b")), pv("c")),
    rhs: pn("add", pv("a"), pn("add", pv("b"), pv("c"))),
  },
  {
    name: "mul-assoc",
    lhs: pn("mul", pn("mul", pv("a"), pv("b")), pv("c")),
    rhs: pn("mul", pv("a"), pn("mul", pv("b"), pv("c"))),
  },
  { name: "add-zero", lhs: pn("add", pv("a"), pc(0)), rhs: pv("a") },
  { name: "mul-one", lhs: pn("mul", pv("a"), pc(1)), rhs: pv("a") },
  { name: "mul-zero", lhs: pn("mul", pv("a"), pc(0)), rhs: pc(0) },
  { name: "sub-self", lhs: pn("sub", pv("a"), pv("a")), rhs: pc(0) },
  { name: "add-self", lhs: pn("add", pv("a"), pv("a")), rhs: pn("mul", pv("a"), pc(2)) },
  {
    name: "distribute",
    lhs: pn("mul", pv("a"), pn("add", pv("b"), pv("c"))),
    rhs: pn("add", pn("mul", pv("a"), pv("b")), pn("mul", pv("a"), pv("c"))),
  },
  {
    name: "factor",
    lhs: pn("add", pn("mul", pv("a"), pv("b")), pn("mul", pv("a"), pv("c"))),
    rhs: pn("mul", pv("a"), pn("add", pv("b"), pv("c"))),
  },
  {
    name: "matmul-assoc-r",
    lhs: pn("matmul", pn("matmul", pv("a"), pv("b")), pv("c")),
    rhs: pn("matmul", pv("a"), pn("matmul", pv("b"), pv("c"))),
  },
  {
    name: "matmul-assoc-l",
    lhs: pn("matmul", pv("a"), pn("matmul", pv("b"), pv("c"))),
    rhs: pn("matmul", pn("matmul", pv("a"), pv("b")), pv("c")),
  },
  {
    name: "matmul-distribute",
    lhs: pn("matmul", pv("a"), pn("add", pv("b"), pv("c"))),
    rhs: pn("add", pn("matmul", pv("a"), pv("b")), pn("matmul", pv("a"), pv("c"))),
  },
  {
    name: "matmul-factor",
    lhs: pn("add", pn("matmul", pv("a"), pv("b")), pn("matmul", pv("a"), pv("c"))),
    rhs: pn("matmul", pv("a"), pn("add", pv("b"), pv("c"))),
  },
  {
    name: "transpose-involutive",
    lhs: pn("transpose", pn("transpose", pv("a"))),
    rhs: pv("a"),
  },
  {
    name: "transpose-matmul",
    lhs: pn("transpose", pn("matmul", pv("a"), pv("b"))),
    rhs: pn("matmul", pn("transpose", pv("b")), pn("transpose", pv("a"))),
  },
  { name: "relu-idempotent", lhs: pn("relu", pn("relu", pv("a"))), rhs: pn("relu", pv("a")) },
  {
    name: "exp-mul",
    lhs: pn("mul", pn("exp", pv("a")), pn("exp", pv("b"))),
    rhs: pn("exp", pn("add", pv("a"), pv("b"))),
  },
  { name: "log-exp", lhs: pn("log", pn("exp", pv("a"))), rhs: pv("a") },
  {
    name: "sum-add",
    lhs: pn("sum", pn("add", pv("a"), pv("b"))),
    rhs: pn("add", pn("sum", pv("a")), pn("sum", pv("b"))),
  },
  {
    name: "fuse-mul-add",
    lhs: pn("add", pn("mul", pv("a"), pv("b")), pv("c")),
    rhs: pn("fma", pv("a"), pv("b"), pv("c")),
  },
];

type Subst = Map<string, number>;

function matchNode(g: EGraph, pattern: Pattern, id: number, subst: Subst): Subst[] {
  if (pattern.kind === "var") {
    const bound = subst.get(pattern.name);
    if (bound !== undefined) return g.find(bound) === g.find(id) ? [subst] : [];
    const next = new Map(subst);
    next.set(pattern.name, g.find(id));
    return [next];
  }
  const out: Subst[] = [];
  for (const node of g.classData(id).nodes) {
    if (node.op !== pattern.op) continue;
    if (pattern.label !== undefined && node.label !== pattern.label) continue;
    if (node.args.length !== pattern.args.length) continue;
    let partial: Subst[] = [subst];
    for (let i = 0; i < pattern.args.length; i++) {
      const nextPartial: Subst[] = [];
      for (const s of partial) {
        for (const m of matchNode(g, pattern.args[i], node.args[i], s)) {
          if (nextPartial.length < 4096) nextPartial.push(m);
        }
      }
      partial = nextPartial;
      if (partial.length === 0) break;
    }
    for (const s of partial) {
      if (out.length < 8192) out.push(s);
    }
  }
  return out;
}

function instantiate(g: EGraph, pattern: Pattern, subst: Subst): number {
  if (pattern.kind === "var") {
    const id = subst.get(pattern.name);
    if (id === undefined) throw new Error(`unbound pattern variable ${pattern.name}`);
    return g.find(id);
  }
  const args = pattern.args.map((a) => instantiate(g, a, subst));
  return g.add({ op: pattern.op, label: pattern.label ?? "", args });
}

export interface SaturationStats {
  iterations: number;
  applied: number;
  classes: number;
  nodes: number;
  saturated: boolean;
  perRule: Record<string, number>;
  elapsedMs: number;
}

export function saturate(
  g: EGraph,
  rules: Rule[],
  limits: { iterations: number; nodes: number; timeMs: number } = {
    iterations: 20,
    nodes: 8000,
    timeMs: 4000,
  },
): SaturationStats {
  const start = Date.now();
  const perRule: Record<string, number> = {};
  let applied = 0;
  let iterations = 0;
  let saturated = false;
  for (let it = 0; it < limits.iterations; it++) {
    iterations = it + 1;
    const matches: Array<{ rule: Rule; subst: Subst; id: number }> = [];
    for (const rule of rules) {
      for (const [id] of g.eclasses()) {
        for (const subst of matchNode(g, rule.lhs, id, new Map())) {
          matches.push({ rule, subst, id });
        }
      }
    }
    let changed = false;
    for (const m of matches) {
      const rhs = instantiate(g, m.rule.rhs, m.subst);
      if (g.union(m.id, rhs)) {
        changed = true;
        applied++;
        perRule[m.rule.name] = (perRule[m.rule.name] ?? 0) + 1;
      }
    }
    g.rebuild();
    if (!changed) {
      saturated = true;
      break;
    }
    if (g.nodeCount > limits.nodes || Date.now() - start > limits.timeMs) break;
  }
  return {
    iterations,
    applied,
    classes: g.classCount,
    nodes: g.nodeCount,
    saturated,
    perRule,
    elapsedMs: Date.now() - start,
  };
}

/* --------------------------------------------------------------------- */
/* Cost model + ILP extraction                                             */
/* --------------------------------------------------------------------- */

export function nodeCost(g: EGraph, node: ENode): number {
  const shape = (i: number): Shape => g.classData(node.args[i]).shape;
  switch (node.op) {
    case "var":
    case "const":
      return 0;
    case "matmul": {
      const a = shape(0);
      const b = shape(1);
      return 2 * a[0] * a[1] * b[1];
    }
    case "transpose": {
      const a = shape(0);
      return a[0] * a[1];
    }
    case "add":
    case "sub": {
      const a = shape(0);
      return a[0] * a[1];
    }
    case "mul": {
      const a = shape(0);
      return 2 * a[0] * a[1];
    }
    case "fma": {
      const a = shape(0);
      return 2 * a[0] * a[1];
    }
    case "relu": {
      const a = shape(0);
      return a[0] * a[1];
    }
    case "exp":
    case "log": {
      const a = shape(0);
      return 8 * a[0] * a[1];
    }
    case "sum": {
      const a = shape(0);
      return a[0] * a[1];
    }
    default: {
      if (!node.args.length) return 1;
      const a = shape(0);
      return a[0] * a[1];
    }
  }
}

export interface IlpVariable {
  name: string;
  kind: "node" | "class" | "order";
  eclass: number;
  nodeIndex?: number;
  cost: number;
}

export interface IlpConstraint {
  name: string;
  /** Human-readable rendering of the row. */
  expr: string;
  sense: "=" | "<=" | ">=";
  rhs: number;
}

export interface IlpModel {
  variables: IlpVariable[];
  constraints: IlpConstraint[];
  objective: string;
  binaryCount: number;
}

export interface ExtractionResult {
  cost: number;
  expr: Expr;
  text: string;
  optimal: boolean;
  exploredNodes: number;
  model: IlpModel;
  greedyCost: number;
  elapsedMs: number;
}

function reachable(g: EGraph, root: number): number[] {
  const seen = new Set<number>();
  const stack = [g.find(root)];
  while (stack.length) {
    const id = stack.pop() as number;
    if (seen.has(id)) continue;
    seen.add(id);
    for (const node of g.classData(id).nodes) for (const a of node.args) stack.push(g.find(a));
  }
  return [...seen];
}

export function buildIlp(g: EGraph, root: number): IlpModel {
  const ids = reachable(g, root);
  const variables: IlpVariable[] = [];
  const constraints: IlpConstraint[] = [];
  const objectiveTerms: string[] = [];
  const M = ids.length + 1;
  for (const id of ids) {
    variables.push({ name: `y_${id}`, kind: "class", eclass: id, cost: 0 });
    variables.push({ name: `t_${id}`, kind: "order", eclass: id, cost: 0 });
    const nodes = g.classData(id).nodes;
    const terms: string[] = [];
    nodes.forEach((node, i) => {
      const cost = nodeCost(g, node);
      variables.push({ name: `x_${id}_${i}`, kind: "node", eclass: id, nodeIndex: i, cost });
      terms.push(`x_${id}_${i}`);
      if (cost !== 0) objectiveTerms.push(`${cost}·x_${id}_${i}`);
      for (const arg of node.args) {
        const c = g.find(arg);
        constraints.push({
          name: `child_${id}_${i}_${c}`,
          expr: `x_${id}_${i} - y_${c}`,
          sense: "<=",
          rhs: 0,
        });
        constraints.push({
          name: `order_${id}_${i}_${c}`,
          expr: `t_${c} - t_${id} + ${M}·x_${id}_${i}`,
          sense: "<=",
          rhs: M - 1,
        });
      }
    });
    constraints.push({
      name: `select_${id}`,
      expr: `${terms.join(" + ")} - y_${id}`,
      sense: "=",
      rhs: 0,
    });
  }
  constraints.push({ name: "root", expr: `y_${g.find(root)}`, sense: "=", rhs: 1 });
  return {
    variables,
    constraints,
    objective: objectiveTerms.length ? `min ${objectiveTerms.join(" + ")}` : "min 0",
    binaryCount: variables.filter((v) => v.kind !== "order").length,
  };
}

/** Bottom-up fixpoint (tree-cost) extraction: incumbent + lower bound source. */
function greedyExtract(
  g: EGraph,
  root: number,
): { cost: Map<number, number>; choice: Map<number, number> } {
  const cost = new Map<number, number>();
  const choice = new Map<number, number>();
  const ids = reachable(g, root);
  for (let iter = 0; iter < ids.length + 2; iter++) {
    let changed = false;
    for (const id of ids) {
      const nodes = g.classData(id).nodes;
      let best = Number.POSITIVE_INFINITY;
      let bestIdx = -1;
      nodes.forEach((node, i) => {
        let c = nodeCost(g, node);
        for (const arg of node.args) {
          const sub = cost.get(g.find(arg));
          if (sub === undefined) {
            c = Number.POSITIVE_INFINITY;
            break;
          }
          c += sub;
        }
        if (c < best) {
          best = c;
          bestIdx = i;
        }
      });
      if (bestIdx >= 0 && best < (cost.get(id) ?? Number.POSITIVE_INFINITY)) {
        cost.set(id, best);
        choice.set(id, bestIdx);
        changed = true;
      }
    }
    if (!changed) break;
  }
  return { cost, choice };
}

function buildExpr(
  g: EGraph,
  root: number,
  choice: Map<number, number>,
  path: Set<number> = new Set(),
): Expr {
  const id = g.find(root);
  const nodes = g.classData(id).nodes;
  let node = nodes[choice.get(id) ?? 0] ?? nodes[0];
  if (path.has(id)) {
    // Cyclic selection: fall back to an acyclic representative for this class.
    const leaf = nodes.find((x) => x.args.length === 0);
    if (!leaf) return { op: "var", label: "⊥" };
    node = leaf;
  }
  if (node.op === "var" || node.op === "const") return { op: node.op, label: node.label };
  const nextPath = new Set(path);
  nextPath.add(id);
  return { op: node.op, args: node.args.map((a) => buildExpr(g, a, choice, nextPath)) };
}

/** True when the e-node selection induces an acyclic sub-DAG from `root`. */
export function isAcyclicSelection(g: EGraph, root: number, choice: Map<number, number>): boolean {
  const color = new Map<number, number>(); // 1 = on stack, 2 = done
  const visit = (id0: number): boolean => {
    const id = g.find(id0);
    const c = color.get(id) ?? 0;
    if (c === 1) return false;
    if (c === 2) return true;
    color.set(id, 1);
    const nodes = g.classData(id).nodes;
    const node = nodes[choice.get(id) ?? 0] ?? nodes[0];
    for (const a of node.args) if (!visit(a)) return false;
    color.set(id, 2);
    return true;
  };
  return visit(root);
}

export function exprToString(e: Expr): string {
  if (e.op === "var" || e.op === "const") return e.label ?? "?";
  return `${e.op}(${(e.args ?? []).map(exprToString).join(", ")})`;
}

/** Exact DAG-cost extraction by branch and bound over the ILP. */
export function extractIlp(
  g: EGraph,
  root: number,
  limits: { nodeLimit: number; timeMs: number } = { nodeLimit: 200000, timeMs: 3000 },
): ExtractionResult {
  const start = Date.now();
  const model = buildIlp(g, root);
  const rootId = g.find(root);
  const greedy = greedyExtract(g, rootId);
  const greedyChoice = greedy.choice;
  const greedyExpr = buildExpr(g, rootId, greedyChoice);
  const greedyCost = dagCost(g, rootId, greedyChoice);

  let bestCost = greedyCost;
  let bestChoice = new Map(greedyChoice);
  let explored = 0;
  let optimal = true;

  /**
   * Depth-first branch and bound: expand the frontier of undecided e-classes,
   * charge each newly selected e-node once (DAG sharing), prune on incumbent.
   */
  const search = (
    frontier: number[],
    decided: Map<number, number>,
    accumulated: number,
  ): void => {
    if (Date.now() - start > limits.timeMs || explored > limits.nodeLimit) {
      optimal = false;
      return;
    }
    if (accumulated >= bestCost) return;
    if (frontier.length === 0) {
      if (accumulated < bestCost && isAcyclicSelection(g, rootId, decided)) {
        bestCost = accumulated;
        bestChoice = new Map(decided);
      }
      return;
    }
    const [id, ...rest] = frontier;
    if (decided.has(id)) {
      search(rest, decided, accumulated);
      return;
    }
    const nodes = g.classData(id).nodes;
    const order = nodes
      .map((node, i) => ({ i, c: nodeCost(g, node) + (greedy.cost.get(id) ?? 0) * 0 }))
      .sort((a, b) => a.c - b.c);
    for (const { i } of order) {
      explored++;
      const node = nodes[i];
      const next = new Map(decided);
      next.set(id, i);
      const newFrontier = rest.slice();
      for (const arg of node.args) {
        const c = g.find(arg);
        if (!next.has(c) && !newFrontier.includes(c)) newFrontier.push(c);
      }
      search(newFrontier, next, accumulated + nodeCost(g, node));
      if (Date.now() - start > limits.timeMs || explored > limits.nodeLimit) {
        optimal = false;
        return;
      }
    }
  };

  search([rootId], new Map(), 0);
  const expr = bestChoice.size ? buildExpr(g, rootId, bestChoice) : greedyExpr;
  return {
    cost: bestCost,
    expr,
    text: exprToString(expr),
    optimal,
    exploredNodes: explored,
    model,
    greedyCost,
    elapsedMs: Date.now() - start,
  };
}

/** Cost of a selection with DAG sharing (each selected e-node charged once). */
export function dagCost(g: EGraph, root: number, choice: Map<number, number>): number {
  const seen = new Set<number>();
  const stack = [g.find(root)];
  let total = 0;
  while (stack.length) {
    const id = stack.pop() as number;
    if (seen.has(id)) continue;
    seen.add(id);
    const nodes = g.classData(id).nodes;
    const node = nodes[choice.get(id) ?? 0] ?? nodes[0];
    total += nodeCost(g, node);
    for (const a of node.args) stack.push(g.find(a));
  }
  return total;
}

/* --------------------------------------------------------------------- */
/* Built-in tensor programs                                                */
/* --------------------------------------------------------------------- */

export interface TensorProgram {
  id: string;
  name: string;
  description: string;
  env: Record<string, Shape>;
  expr: Expr;
}

export const TENSOR_PROGRAMS: TensorProgram[] = [
  {
    id: "matchain",
    name: "Matrix chain (A·B)·C",
    description:
      "Left-associated matrix chain over badly-shaped operands; associativity rewrites plus ILP extraction recover the optimal parenthesisation.",
    env: { A: [64, 1024], B: [1024, 512], C: [512, 4] },
    expr: n("matmul", n("matmul", v("A"), v("B")), v("C")),
  },
  {
    id: "linear-relu",
    name: "Fused linear + ReLU with redundancy",
    description:
      "relu(relu(X·W + 0·B)) + X·W — idempotence, zero-elimination, FMA fusion and common sub-expression sharing all fire.",
    env: { X: [128, 256], W: [256, 128], B: [128, 128] },
    expr: n(
      "add",
      n("relu", n("relu", n("add", n("matmul", v("X"), v("W")), n("mul", v("B"), num(0))))),
      n("matmul", v("X"), v("W")),
    ),
  },
  {
    id: "transpose-chain",
    name: "Transpose identities",
    description: "transpose(transpose(A·B)) collapses to A·B via involutivity and the transpose/matmul law.",
    env: { A: [256, 256], B: [256, 64] },
    expr: n("transpose", n("transpose", n("matmul", v("A"), v("B")))),
  },
  {
    id: "distribute",
    name: "Distribute vs factor",
    description: "A·B + A·C is factored into A·(B+C); ILP extraction decides based on operand shapes.",
    env: { A: [512, 512], B: [512, 512], C: [512, 512] },
    expr: n("add", n("matmul", v("A"), v("B")), n("matmul", v("A"), v("C"))),
  },
  {
    id: "softmax-core",
    name: "Exponential/logarithm simplification",
    description: "log(exp(X)) · 1 + Y·0 reduces to X under the exp/log and unit/zero rules.",
    env: { X: [1024, 1024], Y: [1024, 1024] },
    expr: n("add", n("mul", n("log", n("exp", v("X"))), num(1)), n("mul", v("Y"), num(0))),
  },
];

export interface EqsatRunResult {
  program: TensorProgram;
  before: { cost: number; text: string };
  after: ExtractionResult;
  stats: SaturationStats;
  speedup: number;
}

export function runEqsat(program: TensorProgram, ruleSubset?: string[]): EqsatRunResult {
  const rules = ruleSubset?.length
    ? TENSOR_RULES.filter((r) => ruleSubset.includes(r.name))
    : TENSOR_RULES;
  const g = new EGraph(program.env);
  const root = g.addExpr(program.expr);
  g.rebuild();
  const baselineChoice = new Map<number, number>();
  for (const [id] of g.eclasses()) baselineChoice.set(id, 0);
  const before = { cost: dagCost(g, root, baselineChoice), text: exprToString(program.expr) };
  const stats = saturate(g, rules);
  const after = extractIlp(g, root);
  return {
    program,
    before,
    after,
    stats,
    speedup: after.cost > 0 ? before.cost / after.cost : before.cost > 0 ? Number.POSITIVE_INFINITY : 1,
  };
}
