#!/usr/bin/env node
/**
 * Optimizer self-test: compiles the TypeScript optimizer to a scratch
 * directory and exercises every subsystem end-to-end (ISA round-trip,
 * interpreter, cost model, MCMC, MCTS, meet-in-the-middle, DPLL verification,
 * equality saturation + ILP extraction). Exits non-zero on any failure.
 */
import { execFileSync } from "node:child_process";
import { mkdtempSync, writeFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import assert from "node:assert/strict";

const root = path.resolve(path.dirname(new URL(import.meta.url).pathname), "..");
const out = mkdtempSync(path.join(tmpdir(), "superopt-selftest-"));
const tsconfig = path.join(out, "tsconfig.json");
writeFileSync(
  tsconfig,
  JSON.stringify({
    compilerOptions: {
      target: "ES2020",
      module: "commonjs",
      moduleResolution: "node",
      esModuleInterop: true,
      strict: true,
      skipLibCheck: true,
      outDir: path.join(out, "build"),
      rootDir: root,
    },
    include: [path.join(root, "src/lib/superopt/*.ts")],
  }),
);

console.log("compiling optimizer …");
execFileSync("npx", ["tsc", "-p", tsconfig], { cwd: root, stdio: "inherit" });

const base = path.join(out, "build", "src", "lib", "superopt");
const isa = await import(path.join(base, "isa.js"));
const machine = await import(path.join(base, "machine.js"));
const tasks = await import(path.join(base, "tasks.js"));
const smt = await import(path.join(base, "smt.js"));
const hybrid = await import(path.join(base, "hybrid.js"));
const egraph = await import(path.join(base, "egraph.js"));
const types = await import(path.join(base, "types.js"));

let failures = 0;
const check = (name, fn) => {
  try {
    fn();
    console.log(`  ✓ ${name}`);
  } catch (err) {
    failures += 1;
    console.error(`  ✗ ${name}: ${err.message}`);
  }
};

console.log("ISA layer");
check("program round-trips through the textual form", () => {
  const text = "blsi r0, r0\nadd r1, r0, r0\naddi r2, r1, #-7";
  assert.equal(isa.formatProgram(isa.parseProgram(text)), text);
});
check("both backends disassemble every opcode", () => {
  for (const name of isa.listIsas()) {
    for (const spec of isa.OPCODES) {
      const ins = { op: spec.op, rd: 0, rs1: 1, rs2: 2, rs3: 3, imm: 1 };
      const text = isa.getIsa(name).print(ins);
      assert.ok(!text.includes("unsupported"), `${name}/${spec.op}`);
    }
  }
});

console.log("interpreter");
check("popcount reference matches the SWAR target", () => {
  const task = tasks.getTask("popcount");
  for (const value of [0, 1, 0xffffffff, 0x0f0f0f0f, 123456789]) {
    const st = machine.stateFromTest(
      { id: "t", gpr: [value, 0, 0, 0, 0, 0, 0, 0], fpr: [], vreg: [], mem: [], origin: "seed", weight: 1 },
      types.DEFAULT_MACHINE,
    );
    machine.execute(task.target, st, types.DEFAULT_MACHINE);
    let expected = 0;
    for (let i = 0; i < 32; i++) expected += (value >>> i) & 1;
    assert.equal(st.gpr[0], expected);
  }
});
check("memory faults are recorded, not thrown", () => {
  const prog = isa.parseProgram("movi r0, #100000\nload r1, r0, #0");
  const st = machine.createState(types.DEFAULT_MACHINE);
  machine.execute(prog, st, types.DEFAULT_MACHINE);
  assert.equal(st.fault, "segfault");
});

console.log("SMT layer");
check("equivalent rewrite is proven UNSAT", () => {
  const task = tasks.getTask("isolate_lsb");
  const r = smt.verifyEquivalence(task.target, isa.parseProgram("blsi r0, r0"), {
    width: 8,
    inputGpr: task.inputGpr,
    live: task.live,
    cfg: types.DEFAULT_MACHINE,
    timeMs: 20000,
  });
  assert.equal(r.status, "equivalent");
  assert.equal(r.method, "bit-blast-dpll");
});
check("wrong rewrite yields a counterexample", () => {
  const task = tasks.getTask("isolate_lsb");
  const r = smt.verifyEquivalence(task.target, isa.parseProgram("blsr r0, r0"), {
    width: 8,
    inputGpr: task.inputGpr,
    live: task.live,
    cfg: types.DEFAULT_MACHINE,
    timeMs: 20000,
  });
  assert.equal(r.status, "counterexample");
  assert.ok(r.counterexample);
});
check("DPLL and exhaustive enumeration agree", () => {
  const task = tasks.getTask("clear_lsb");
  const cand = isa.parseProgram("blsr r0, r0");
  const opts = { width: 8, inputGpr: task.inputGpr, live: task.live, cfg: types.DEFAULT_MACHINE, timeMs: 20000 };
  assert.equal(smt.verifyEquivalence(task.target, cand, opts).status, smt.boundedEnumeration(task.target, cand, opts).status);
});

console.log("hybrid search");
for (const id of ["abs", "isolate_lsb", "clear_lsb"]) {
  check(`${id}: discovers and proves a shorter rewrite`, () => {
    const report = hybrid.runHybrid(tasks.getTask(id), {
      rounds: 2,
      mcmcMs: 900,
      mctsMs: 300,
      enumerativeMs: 300,
      verifyMs: 2000,
    });
    assert.ok(report.best, "no candidate found");
    assert.ok(report.best.verified, "candidate not verified");
    assert.ok(report.best.length <= tasks.getTask(id).target.length, "candidate is longer than the target");
  });
}

console.log("equality saturation");
check("matrix chain is re-associated and ILP extracted", () => {
  const program = egraph.TENSOR_PROGRAMS.find((p) => p.id === "matchain");
  const r = egraph.runEqsat(program);
  assert.ok(r.after.cost < r.before.cost, "no cost improvement");
  assert.ok(r.after.model.constraints.length > 0, "empty ILP model");
});
check("transpose identities collapse", () => {
  const program = egraph.TENSOR_PROGRAMS.find((p) => p.id === "transpose-chain");
  const r = egraph.runEqsat(program);
  assert.equal(r.after.text, "matmul(A, B)");
});

rmSync(out, { recursive: true, force: true });
if (failures > 0) {
  console.error(`${failures} self-test failure(s)`);
  process.exit(1);
}
console.log("optimizer self-test OK");
