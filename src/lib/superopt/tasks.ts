/**
 * Benchmark suite: each task provides a reference ("target") implementation,
 * its live-out specification, the opcode pool the search may draw from, and a
 * length budget. All programs are loop-free straight-line code.
 */

import { parseProgram } from "./isa";
import type { LiveOut, Program } from "./types";

export interface Task {
  id: string;
  name: string;
  category: "bitwise" | "arithmetic" | "select" | "float" | "memory";
  description: string;
  /** Reference implementation the rewrite must match. */
  target: Program;
  live: LiveOut;
  /** Registers that carry meaningful inputs (used by symbolic execution). */
  inputGpr: number[];
  inputFpr: number[];
  /** Maximum instruction count the searchers may emit. */
  maxLength: number;
  /** Legal opcodes for synthesis. */
  pool: string[];
  /** Known-good hand optimum (documentation / regression baseline). */
  knownOptimum?: string;
  /** Verification datapath width used by the SMT layer. */
  verifyWidth: number;
}

const INT_POOL = [
  "nop", "mov", "movi", "add", "sub", "addi", "and", "or", "xor", "andn", "andi", "ori",
  "xori", "not", "neg", "shl", "shr", "sar", "shli", "shri", "sari", "rol", "popcnt",
  "clz", "ctz", "blsi", "blsr", "blsmsk", "lea", "mul", "muli", "smin", "smax", "umin",
  "umax", "abs", "cmp", "cmpi", "setl", "setb", "sete", "cmovl", "cmove", "cmovb",
];

const FLOAT_POOL = ["nop", "fmov", "fadd", "fsub", "fmul", "fdiv", "fsqrt", "fma", "fabs", "fneg", "fmin", "fmax"];

const liveGpr = (regs: number[]): LiveOut => ({ gpr: regs, fpr: [], vreg: [], mem: [] });
const liveFpr = (regs: number[]): LiveOut => ({ gpr: [], fpr: regs, vreg: [], mem: [] });

export const TASKS: Task[] = [
  {
    id: "popcount",
    name: "Population count (SWAR → popcnt)",
    category: "bitwise",
    description:
      "Twelve-instruction SWAR population count. The optimal rewrite is a single popcnt on machines that provide it.",
    target: parseProgram(`
      shri r1, r0, #1
      andi r1, r1, #1431655765
      sub  r0, r0, r1
      andi r1, r0, #858993459
      shri r0, r0, #2
      andi r0, r0, #858993459
      add  r0, r0, r1
      shri r1, r0, #4
      add  r0, r0, r1
      andi r0, r0, #252645135
      muli r0, r0, #16843009
      shri r0, r0, #24
    `),
    live: liveGpr([0]),
    inputGpr: [0],
    inputFpr: [],
    maxLength: 12,
    pool: INT_POOL,
    knownOptimum: "popcnt r0, r0",
    verifyWidth: 8,
  },
  {
    id: "abs",
    name: "Branchless absolute value",
    category: "arithmetic",
    description: "Classic sar/xor/sub absolute value; optimum is a single abs (or cneg on AArch64).",
    target: parseProgram(`
      sari r1, r0, #31
      xor  r0, r0, r1
      sub  r0, r0, r1
    `),
    live: liveGpr([0]),
    inputGpr: [0],
    inputFpr: [],
    maxLength: 4,
    pool: INT_POOL,
    knownOptimum: "abs r0, r0",
    verifyWidth: 8,
  },
  {
    id: "isolate_lsb",
    name: "Isolate lowest set bit",
    category: "bitwise",
    description: "x & -x, expressed as neg+and; optimum is blsi (BMI1).",
    target: parseProgram(`
      neg r1, r0
      and r0, r0, r1
    `),
    live: liveGpr([0]),
    inputGpr: [0],
    inputFpr: [],
    maxLength: 3,
    pool: INT_POOL,
    knownOptimum: "blsi r0, r0",
    verifyWidth: 8,
  },
  {
    id: "clear_lsb",
    name: "Clear lowest set bit",
    category: "bitwise",
    description: "x & (x-1); optimum is blsr (BMI1).",
    target: parseProgram(`
      addi r1, r0, #-1
      and  r0, r0, r1
    `),
    live: liveGpr([0]),
    inputGpr: [0],
    inputFpr: [],
    maxLength: 3,
    pool: INT_POOL,
    knownOptimum: "blsr r0, r0",
    verifyWidth: 8,
  },
  {
    id: "mask_upto_lsb",
    name: "Mask up to lowest set bit",
    category: "bitwise",
    description: "x ^ (x-1); optimum is blsmsk (BMI1).",
    target: parseProgram(`
      addi r1, r0, #-1
      xor  r0, r0, r1
    `),
    live: liveGpr([0]),
    inputGpr: [0],
    inputFpr: [],
    maxLength: 3,
    pool: INT_POOL,
    knownOptimum: "blsmsk r0, r0",
    verifyWidth: 8,
  },
  {
    id: "sign",
    name: "Sign function (-1, 0, 1)",
    category: "select",
    description: "(x>0) - (x<0) computed with shifts and or.",
    target: parseProgram(`
      neg  r1, r0
      shri r1, r1, #31
      sari r0, r0, #31
      or   r0, r0, r1
    `),
    live: liveGpr([0]),
    inputGpr: [0],
    inputFpr: [],
    maxLength: 4,
    pool: INT_POOL,
    verifyWidth: 8,
  },
  {
    id: "smax",
    name: "Signed maximum",
    category: "select",
    description: "cmp + cmov signed maximum of r0 and r1 into r2; optimum is a single smax.",
    target: parseProgram(`
      mov   r2, r0
      cmp   r0, r1
      cmovl r2, r1
    `),
    live: liveGpr([2]),
    inputGpr: [0, 1],
    inputFpr: [],
    maxLength: 3,
    pool: INT_POOL,
    knownOptimum: "smax r2, r0, r1",
    verifyWidth: 8,
  },
  {
    id: "avg_no_overflow",
    name: "Unsigned average without overflow",
    category: "arithmetic",
    description: "(a & b) + ((a ^ b) >> 1) — overflow-free unsigned midpoint.",
    target: parseProgram(`
      and  r2, r0, r1
      xor  r3, r0, r1
      shri r3, r3, #1
      add  r2, r2, r3
    `),
    live: liveGpr([2]),
    inputGpr: [0, 1],
    inputFpr: [],
    maxLength: 4,
    pool: INT_POOL,
    verifyWidth: 8,
  },
  {
    id: "mul10",
    name: "Multiply by ten",
    category: "arithmetic",
    description: "Shift/add expansion of x*10; optimum uses two LEA-style ops.",
    target: parseProgram(`
      shli r1, r0, #3
      shli r2, r0, #1
      add  r0, r1, r2
    `),
    live: liveGpr([0]),
    inputGpr: [0],
    inputFpr: [],
    maxLength: 3,
    pool: INT_POOL,
    knownOptimum: "lea r1, r0, r0, #2 ; add r0, r1, r1",
    verifyWidth: 8,
  },
  {
    id: "round_pow2",
    name: "Round up to power of two",
    category: "bitwise",
    description: "Bit-smearing round-up-to-power-of-two; a clz-based rewrite is far shorter.",
    target: parseProgram(`
      addi r0, r0, #-1
      shri r1, r0, #1
      or   r0, r0, r1
      shri r1, r0, #2
      or   r0, r0, r1
      shri r1, r0, #4
      or   r0, r0, r1
      shri r1, r0, #8
      or   r0, r0, r1
      shri r1, r0, #16
      or   r0, r0, r1
      addi r0, r0, #1
    `),
    live: liveGpr([0]),
    inputGpr: [0],
    inputFpr: [],
    maxLength: 12,
    pool: INT_POOL,
    verifyWidth: 8,
  },
  {
    id: "fp_horner",
    name: "Polynomial evaluation (a·x² + b·x + c)",
    category: "float",
    description:
      "Naive polynomial evaluation with separate multiplies and adds; the FMA rewrite is shorter and more accurate — ULP tolerance governs acceptance.",
    target: parseProgram(`
      fmul f4, f0, f0
      fmul f4, f4, f1
      fmul f5, f0, f2
      fadd f4, f4, f5
      fadd f4, f4, f3
    `),
    live: liveFpr([4]),
    inputGpr: [],
    inputFpr: [0, 1, 2, 3],
    maxLength: 5,
    pool: FLOAT_POOL,
    knownOptimum: "fma f4, f1, f0, f2 ; fma f4, f4, f0, f3",
    verifyWidth: 8,
  },
];

export function getTask(id: string): Task {
  const t = TASKS.find((x) => x.id === id);
  if (!t) throw new Error(`unknown task: ${id}`);
  return t;
}
