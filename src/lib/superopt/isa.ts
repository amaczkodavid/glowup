/**
 * Modular ISA abstraction.
 *
 * A single virtual core opcode table carries semantics-independent metadata
 * (operand roles, commutativity, latency/throughput, port usage). Concrete
 * backends (x86-64 SysV, AArch64 AAPCS64) provide printing/encoding templates
 * and per-ISA cost overrides. New ISAs are registered by implementing the
 * `IsaBackend` interface and calling `registerIsa`.
 */

import type { Instruction, OpClass, OperandRole } from "./types";

export interface OpSpec {
  op: string;
  cls: OpClass;
  roles: OperandRole[];
  /** True when swapping rs1/rs2 preserves semantics (used by the proposal kernel). */
  commutative: boolean;
  /** Writes the integer condition flags. */
  writesFlags: boolean;
  /** Reads the integer condition flags. */
  readsFlags: boolean;
  /** Destination is also an input (read-modify-write), e.g. predicated moves. */
  rmw: boolean;
  /** Legal immediate range, inclusive. */
  immMin: number;
  immMax: number;
  /** Baseline latency in cycles on the virtual core. */
  latency: number;
  /** Reciprocal throughput in cycles. */
  rthroughput: number;
  /** Number of micro-ops. */
  uops: number;
  /** Execution ports the op may issue on. */
  ports: number[];
  /** May raise a runtime fault. */
  faulting: boolean;
}

const S = (
  op: string,
  cls: OpClass,
  roles: OperandRole[],
  o: Partial<OpSpec> = {},
): OpSpec => ({
  op,
  cls,
  roles,
  commutative: false,
  writesFlags: false,
  readsFlags: false,
  rmw: false,
  immMin: -128,
  immMax: 127,
  latency: 1,
  rthroughput: 0.25,
  uops: 1,
  ports: [0, 1, 5, 6],
  faulting: false,
  ...o,
});

export const OPCODES: OpSpec[] = [
  S("nop", "nop", [], { latency: 0, rthroughput: 0.2, uops: 1, ports: [0, 1, 5, 6] }),
  S("mov", "move", ["rd", "rs1"]),
  S("movi", "move", ["rd", "imm"], { immMin: -2147483648, immMax: 2147483647 }),
  S("add", "arith", ["rd", "rs1", "rs2"], { commutative: true, writesFlags: true }),
  S("sub", "arith", ["rd", "rs1", "rs2"], { writesFlags: true }),
  S("addi", "arith", ["rd", "rs1", "imm"], { writesFlags: true }),
  S("mul", "arith", ["rd", "rs1", "rs2"], {
    commutative: true,
    latency: 3,
    rthroughput: 1,
    ports: [1],
  }),
  S("muli", "arith", ["rd", "rs1", "imm"], { latency: 3, rthroughput: 1, ports: [1], immMin: -2147483648, immMax: 2147483647 }),
  S("mulhu", "arith", ["rd", "rs1", "rs2"], {
    commutative: true,
    latency: 4,
    rthroughput: 1,
    ports: [1],
  }),
  S("divu", "arith", ["rd", "rs1", "rs2"], {
    latency: 26,
    rthroughput: 8,
    ports: [0],
    faulting: true,
  }),
  S("neg", "arith", ["rd", "rs1"], { writesFlags: true }),
  S("and", "logic", ["rd", "rs1", "rs2"], { commutative: true, writesFlags: true }),
  S("or", "logic", ["rd", "rs1", "rs2"], { commutative: true, writesFlags: true }),
  S("xor", "logic", ["rd", "rs1", "rs2"], { commutative: true, writesFlags: true }),
  S("andn", "logic", ["rd", "rs1", "rs2"], { latency: 1, ports: [1, 5] }),
  S("andi", "logic", ["rd", "rs1", "imm"], { writesFlags: true, immMin: -2147483648, immMax: 2147483647 }),
  S("ori", "logic", ["rd", "rs1", "imm"], { writesFlags: true, immMin: -2147483648, immMax: 2147483647 }),
  S("xori", "logic", ["rd", "rs1", "imm"], { writesFlags: true, immMin: -2147483648, immMax: 2147483647 }),
  S("not", "logic", ["rd", "rs1"]),
  S("shl", "shift", ["rd", "rs1", "rs2"], { latency: 2, rthroughput: 0.5, ports: [0, 6] }),
  S("shr", "shift", ["rd", "rs1", "rs2"], { latency: 2, rthroughput: 0.5, ports: [0, 6] }),
  S("sar", "shift", ["rd", "rs1", "rs2"], { latency: 2, rthroughput: 0.5, ports: [0, 6] }),
  S("shli", "shift", ["rd", "rs1", "imm"], { immMin: 0, immMax: 31, ports: [0, 6] }),
  S("shri", "shift", ["rd", "rs1", "imm"], { immMin: 0, immMax: 31, ports: [0, 6] }),
  S("sari", "shift", ["rd", "rs1", "imm"], { immMin: 0, immMax: 31, ports: [0, 6] }),
  S("rol", "shift", ["rd", "rs1", "imm"], { immMin: 0, immMax: 31, ports: [0, 6] }),
  S("popcnt", "bit", ["rd", "rs1"], { latency: 3, rthroughput: 1, ports: [1] }),
  S("clz", "bit", ["rd", "rs1"], { latency: 3, rthroughput: 1, ports: [1] }),
  S("ctz", "bit", ["rd", "rs1"], { latency: 3, rthroughput: 1, ports: [1] }),
  S("bswap", "bit", ["rd", "rs1"], { latency: 1, ports: [1, 5] }),
  S("blsi", "bit", ["rd", "rs1"], { latency: 1, ports: [1, 5] }),
  S("blsr", "bit", ["rd", "rs1"], { latency: 1, ports: [1, 5] }),
  S("blsmsk", "bit", ["rd", "rs1"], { latency: 1, ports: [1, 5] }),
  S("lea", "arith", ["rd", "rs1", "rs2", "imm"], { immMin: 0, immMax: 3, ports: [1, 5] }),
  S("cmp", "compare", ["rs1", "rs2"], { writesFlags: true }),
  S("cmpi", "compare", ["rs1", "imm"], { writesFlags: true }),
  S("setl", "compare", ["rd"], { readsFlags: true, ports: [0, 6] }),
  S("setb", "compare", ["rd"], { readsFlags: true, ports: [0, 6] }),
  S("sete", "compare", ["rd"], { readsFlags: true, ports: [0, 6] }),
  S("cmovl", "select", ["rd", "rs1"], { readsFlags: true, rmw: true, ports: [0, 6] }),
  S("cmove", "select", ["rd", "rs1"], { readsFlags: true, rmw: true, ports: [0, 6] }),
  S("cmovb", "select", ["rd", "rs1"], { readsFlags: true, rmw: true, ports: [0, 6] }),
  S("smin", "select", ["rd", "rs1", "rs2"], { commutative: true, ports: [0, 1, 5] }),
  S("smax", "select", ["rd", "rs1", "rs2"], { commutative: true, ports: [0, 1, 5] }),
  S("umin", "select", ["rd", "rs1", "rs2"], { commutative: true, ports: [0, 1, 5] }),
  S("umax", "select", ["rd", "rs1", "rs2"], { commutative: true, ports: [0, 1, 5] }),
  S("abs", "select", ["rd", "rs1"], { ports: [0, 1, 5] }),
  S("load", "memory", ["rd", "rs1", "imm"], {
    latency: 5,
    rthroughput: 0.5,
    ports: [2, 3],
    faulting: true,
    immMin: -64,
    immMax: 64,
  }),
  S("store", "memory", ["rs1", "rs2", "imm"], {
    latency: 4,
    rthroughput: 1,
    ports: [4],
    faulting: true,
    immMin: -64,
    immMax: 64,
  }),
  S("fmov", "float", ["rd", "rs1"], { latency: 1, ports: [0, 5] }),
  S("fadd", "float", ["rd", "rs1", "rs2"], {
    commutative: true,
    latency: 4,
    rthroughput: 0.5,
    ports: [0, 1],
  }),
  S("fsub", "float", ["rd", "rs1", "rs2"], { latency: 4, rthroughput: 0.5, ports: [0, 1] }),
  S("fmul", "float", ["rd", "rs1", "rs2"], {
    commutative: true,
    latency: 4,
    rthroughput: 0.5,
    ports: [0, 1],
  }),
  S("fdiv", "float", ["rd", "rs1", "rs2"], { latency: 14, rthroughput: 4, ports: [0] }),
  S("fsqrt", "float", ["rd", "rs1"], { latency: 15, rthroughput: 4, ports: [0] }),
  S("fma", "float", ["rd", "rs1", "rs2", "rs3"], { latency: 4, rthroughput: 0.5, ports: [0, 1] }),
  S("fabs", "float", ["rd", "rs1"], { latency: 1, ports: [0, 5] }),
  S("fneg", "float", ["rd", "rs1"], { latency: 1, ports: [0, 5] }),
  S("fmin", "float", ["rd", "rs1", "rs2"], { latency: 3, ports: [0, 1] }),
  S("fmax", "float", ["rd", "rs1", "rs2"], { latency: 3, ports: [0, 1] }),
  S("vadd", "vector", ["vd", "vs1", "vs2"], {
    commutative: true,
    latency: 4,
    rthroughput: 0.5,
    uops: 1,
    ports: [0, 1],
  }),
  S("vmul", "vector", ["vd", "vs1", "vs2"], {
    commutative: true,
    latency: 4,
    rthroughput: 0.5,
    ports: [0, 1],
  }),
  S("vfma", "vector", ["vd", "vs1", "vs2", "vs3"], { latency: 4, rthroughput: 0.5, ports: [0, 1] }),
  S("vbroadcast", "float", ["vd", "rs1"], { latency: 3, ports: [5] }),
  S("vhadd", "float", ["rd", "vs1"], { latency: 7, rthroughput: 2, ports: [0, 1, 5] }),
];

export const OPSPEC: Record<string, OpSpec> = Object.fromEntries(
  OPCODES.map((o) => [o.op, o]),
);

export function spec(op: string): OpSpec {
  const s = OPSPEC[op];
  if (!s) throw new Error(`unknown opcode: ${op}`);
  return s;
}

/** Opcodes grouped by class, used for class-preserving opcode mutation. */
export const OPCODES_BY_CLASS: Record<string, string[]> = (() => {
  const m: Record<string, string[]> = {};
  for (const o of OPCODES) (m[o.cls] ||= []).push(o.op);
  return m;
})();

/** Opcodes grouped by identical operand signature (safe drop-in mutation). */
export const OPCODES_BY_SIGNATURE: Record<string, string[]> = (() => {
  const m: Record<string, string[]> = {};
  for (const o of OPCODES) (m[o.roles.join(",")] ||= []).push(o.op);
  return m;
})();

export function signatureOf(op: string): string {
  return spec(op).roles.join(",");
}

/* --------------------------------------------------------------------- */
/* Backends                                                               */
/* --------------------------------------------------------------------- */

export interface IsaBackend {
  name: string;
  /** Architectural register names (index-aligned with the virtual core). */
  gprNames: string[];
  fprNames: string[];
  vregNames: string[];
  /** Opcodes not natively available; the search will avoid them. */
  unsupported: Set<string>;
  /** Per-opcode latency overrides. */
  latency: Record<string, number>;
  /** Issue width used by the throughput model. */
  issueWidth: number;
  /** Render one instruction in native syntax. */
  print(ins: Instruction): string;
  /** Approximate machine-code size in bytes (used for I-cache pressure). */
  encodedSize(ins: Instruction): number;
}

function gp(names: string[], i: number): string {
  return names[i] ?? `r${i}`;
}

const X86_GPR = ["rax", "rbx", "rcx", "rdx", "rsi", "rdi", "r8", "r9", "r10", "r11", "r12", "r13"];
const X86_FPR = Array.from({ length: 16 }, (_, i) => `xmm${i}`);
const X86_VREG = Array.from({ length: 16 }, (_, i) => `ymm${i}`);

export const X86_64: IsaBackend = {
  name: "x86-64",
  gprNames: X86_GPR,
  fprNames: X86_FPR,
  vregNames: X86_VREG,
  unsupported: new Set<string>(),
  latency: { mul: 3, mulhu: 4, divu: 36, popcnt: 3, clz: 3, ctz: 3, load: 5, fma: 4 },
  issueWidth: 4,
  print(ins: Instruction): string {
    const r = (i: number) => gp(X86_GPR, i);
    const f = (i: number) => gp(X86_FPR, i);
    const v = (i: number) => gp(X86_VREG, i);
    const { op, rd, rs1, rs2, rs3, imm } = ins;
    switch (op) {
      case "nop":
        return "nop";
      case "mov":
        return `movq %${r(rs1)}, %${r(rd)}`;
      case "movi":
        return `movq $${imm}, %${r(rd)}`;
      case "add":
        return `leaq (%${r(rs1)},%${r(rs2)}), %${r(rd)}`;
      case "sub":
        return `movq %${r(rs1)}, %${r(rd)}\n  subq %${r(rs2)}, %${r(rd)}`;
      case "addi":
        return `leaq ${imm}(%${r(rs1)}), %${r(rd)}`;
      case "mul":
        return `movq %${r(rs1)}, %${r(rd)}\n  imulq %${r(rs2)}, %${r(rd)}`;
      case "muli":
        return `imulq $${imm}, %${r(rs1)}, %${r(rd)}`;
      case "mulhu":
        return `movq %${r(rs1)}, %rax\n  mulq %${r(rs2)}\n  movq %rdx, %${r(rd)}`;
      case "divu":
        return `xorl %edx, %edx\n  movq %${r(rs1)}, %rax\n  divq %${r(rs2)}\n  movq %rax, %${r(rd)}`;
      case "neg":
        return `movq %${r(rs1)}, %${r(rd)}\n  negq %${r(rd)}`;
      case "and":
        return `movq %${r(rs1)}, %${r(rd)}\n  andq %${r(rs2)}, %${r(rd)}`;
      case "or":
        return `movq %${r(rs1)}, %${r(rd)}\n  orq %${r(rs2)}, %${r(rd)}`;
      case "xor":
        return `movq %${r(rs1)}, %${r(rd)}\n  xorq %${r(rs2)}, %${r(rd)}`;
      case "andn":
        return `andnq %${r(rs2)}, %${r(rs1)}, %${r(rd)}`;
      case "andi":
        return `movq %${r(rs1)}, %${r(rd)}\n  andq $${imm}, %${r(rd)}`;
      case "ori":
        return `movq %${r(rs1)}, %${r(rd)}\n  orq $${imm}, %${r(rd)}`;
      case "xori":
        return `movq %${r(rs1)}, %${r(rd)}\n  xorq $${imm}, %${r(rd)}`;
      case "not":
        return `movq %${r(rs1)}, %${r(rd)}\n  notq %${r(rd)}`;
      case "shl":
        return `movq %${r(rs2)}, %rcx\n  shlxq %rcx, %${r(rs1)}, %${r(rd)}`;
      case "shr":
        return `movq %${r(rs2)}, %rcx\n  shrxq %rcx, %${r(rs1)}, %${r(rd)}`;
      case "sar":
        return `movq %${r(rs2)}, %rcx\n  sarxq %rcx, %${r(rs1)}, %${r(rd)}`;
      case "shli":
        return `shlxq $${imm}, %${r(rs1)}, %${r(rd)}`;
      case "shri":
        return `shrxq $${imm}, %${r(rs1)}, %${r(rd)}`;
      case "sari":
        return `sarxq $${imm}, %${r(rs1)}, %${r(rd)}`;
      case "rol":
        return `movq %${r(rs1)}, %${r(rd)}\n  rolq $${imm}, %${r(rd)}`;
      case "popcnt":
        return `popcntq %${r(rs1)}, %${r(rd)}`;
      case "clz":
        return `lzcntq %${r(rs1)}, %${r(rd)}`;
      case "ctz":
        return `tzcntq %${r(rs1)}, %${r(rd)}`;
      case "bswap":
        return `movq %${r(rs1)}, %${r(rd)}\n  bswapq %${r(rd)}`;
      case "blsi":
        return `blsiq %${r(rs1)}, %${r(rd)}`;
      case "blsr":
        return `blsrq %${r(rs1)}, %${r(rd)}`;
      case "blsmsk":
        return `blsmskq %${r(rs1)}, %${r(rd)}`;
      case "lea":
        return `leaq (%${r(rs1)},%${r(rs2)},${1 << imm}), %${r(rd)}`;
      case "cmp":
        return `cmpq %${r(rs2)}, %${r(rs1)}`;
      case "cmpi":
        return `cmpq $${imm}, %${r(rs1)}`;
      case "setl":
        return `setl %${r(rd)}b\n  movzbq %${r(rd)}b, %${r(rd)}`;
      case "setb":
        return `setb %${r(rd)}b\n  movzbq %${r(rd)}b, %${r(rd)}`;
      case "sete":
        return `sete %${r(rd)}b\n  movzbq %${r(rd)}b, %${r(rd)}`;
      case "cmovl":
        return `cmovlq %${r(rs1)}, %${r(rd)}`;
      case "cmove":
        return `cmoveq %${r(rs1)}, %${r(rd)}`;
      case "cmovb":
        return `cmovbq %${r(rs1)}, %${r(rd)}`;
      case "smin":
        return `movq %${r(rs1)}, %${r(rd)}\n  cmpq %${r(rs2)}, %${r(rd)}\n  cmovgq %${r(rs2)}, %${r(rd)}`;
      case "smax":
        return `movq %${r(rs1)}, %${r(rd)}\n  cmpq %${r(rs2)}, %${r(rd)}\n  cmovlq %${r(rs2)}, %${r(rd)}`;
      case "umin":
        return `movq %${r(rs1)}, %${r(rd)}\n  cmpq %${r(rs2)}, %${r(rd)}\n  cmovaq %${r(rs2)}, %${r(rd)}`;
      case "umax":
        return `movq %${r(rs1)}, %${r(rd)}\n  cmpq %${r(rs2)}, %${r(rd)}\n  cmovbq %${r(rs2)}, %${r(rd)}`;
      case "abs":
        return `movq %${r(rs1)}, %${r(rd)}\n  negq %${r(rd)}\n  cmovlq %${r(rs1)}, %${r(rd)}`;
      case "load":
        return `movq ${imm}(%${r(rs1)}), %${r(rd)}`;
      case "store":
        return `movq %${r(rs2)}, ${imm}(%${r(rs1)})`;
      case "fmov":
        return `movapd %${f(rs1)}, %${f(rd)}`;
      case "fadd":
        return `vaddsd %${f(rs2)}, %${f(rs1)}, %${f(rd)}`;
      case "fsub":
        return `vsubsd %${f(rs2)}, %${f(rs1)}, %${f(rd)}`;
      case "fmul":
        return `vmulsd %${f(rs2)}, %${f(rs1)}, %${f(rd)}`;
      case "fdiv":
        return `vdivsd %${f(rs2)}, %${f(rs1)}, %${f(rd)}`;
      case "fsqrt":
        return `vsqrtsd %${f(rs1)}, %${f(rs1)}, %${f(rd)}`;
      case "fma":
        return `vfmadd213sd %${f(rs3)}, %${f(rs2)}, %${f(rd)}`;
      case "fabs":
        return `vandpd .LCabs(%rip), %${f(rs1)}, %${f(rd)}`;
      case "fneg":
        return `vxorpd .LCsign(%rip), %${f(rs1)}, %${f(rd)}`;
      case "fmin":
        return `vminsd %${f(rs2)}, %${f(rs1)}, %${f(rd)}`;
      case "fmax":
        return `vmaxsd %${f(rs2)}, %${f(rs1)}, %${f(rd)}`;
      case "vadd":
        return `vaddpd %${v(rs2)}, %${v(rs1)}, %${v(rd)}`;
      case "vmul":
        return `vmulpd %${v(rs2)}, %${v(rs1)}, %${v(rd)}`;
      case "vfma":
        return `vfmadd231pd %${v(rs3)}, %${v(rs2)}, %${v(rd)}`;
      case "vbroadcast":
        return `vbroadcastsd %${r(rs1)}, %${v(rd)}`;
      case "vhadd":
        return `vextractf128 $1, %${v(rs1)}, %xmm15\n  vaddpd %xmm15, %${v(rs1)}, %xmm15\n  vhaddpd %xmm15, %xmm15, %xmm15\n  vmovq %xmm15, %${r(rd)}`;
      default:
        return `# unsupported ${op}`;
    }
  },
  encodedSize(ins: Instruction): number {
    const s = spec(ins.op);
    let bytes = 3;
    if (s.roles.includes("imm")) bytes += Math.abs(ins.imm) > 127 ? 4 : 1;
    if (s.cls === "memory") bytes += 1;
    if (s.cls === "vector" || s.cls === "float") bytes += 1;
    return bytes;
  },
};

const A64_GPR = ["x0", "x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9", "x10", "x11"];
const A64_FPR = Array.from({ length: 16 }, (_, i) => `d${i}`);
const A64_VREG = Array.from({ length: 16 }, (_, i) => `v${i}`);

export const AARCH64: IsaBackend = {
  name: "aarch64",
  gprNames: A64_GPR,
  fprNames: A64_FPR,
  vregNames: A64_VREG,
  unsupported: new Set(["blsi", "blsr", "blsmsk"]),
  latency: { mul: 3, mulhu: 5, divu: 20, popcnt: 4, clz: 1, ctz: 2, load: 4, fma: 4 },
  issueWidth: 4,
  print(ins: Instruction): string {
    const r = (i: number) => gp(A64_GPR, i);
    const f = (i: number) => gp(A64_FPR, i);
    const v = (i: number) => `${gp(A64_VREG, i)}.2d`;
    const { op, rd, rs1, rs2, rs3, imm } = ins;
    switch (op) {
      case "nop":
        return "nop";
      case "mov":
        return `mov ${r(rd)}, ${r(rs1)}`;
      case "movi":
        return `mov ${r(rd)}, #${imm}`;
      case "add":
        return `add ${r(rd)}, ${r(rs1)}, ${r(rs2)}`;
      case "sub":
        return `sub ${r(rd)}, ${r(rs1)}, ${r(rs2)}`;
      case "addi":
        return `add ${r(rd)}, ${r(rs1)}, #${imm}`;
      case "mul":
        return `mul ${r(rd)}, ${r(rs1)}, ${r(rs2)}`;
      case "muli":
        return `mov x16, #${imm}\n  mul ${r(rd)}, ${r(rs1)}, x16`;
      case "mulhu":
        return `umulh ${r(rd)}, ${r(rs1)}, ${r(rs2)}`;
      case "divu":
        return `udiv ${r(rd)}, ${r(rs1)}, ${r(rs2)}`;
      case "neg":
        return `neg ${r(rd)}, ${r(rs1)}`;
      case "and":
        return `and ${r(rd)}, ${r(rs1)}, ${r(rs2)}`;
      case "or":
        return `orr ${r(rd)}, ${r(rs1)}, ${r(rs2)}`;
      case "xor":
        return `eor ${r(rd)}, ${r(rs1)}, ${r(rs2)}`;
      case "andn":
        return `bic ${r(rd)}, ${r(rs2)}, ${r(rs1)}`;
      case "andi":
        return `and ${r(rd)}, ${r(rs1)}, #${imm}`;
      case "ori":
        return `orr ${r(rd)}, ${r(rs1)}, #${imm}`;
      case "xori":
        return `eor ${r(rd)}, ${r(rs1)}, #${imm}`;
      case "not":
        return `mvn ${r(rd)}, ${r(rs1)}`;
      case "shl":
        return `lsl ${r(rd)}, ${r(rs1)}, ${r(rs2)}`;
      case "shr":
        return `lsr ${r(rd)}, ${r(rs1)}, ${r(rs2)}`;
      case "sar":
        return `asr ${r(rd)}, ${r(rs1)}, ${r(rs2)}`;
      case "shli":
        return `lsl ${r(rd)}, ${r(rs1)}, #${imm}`;
      case "shri":
        return `lsr ${r(rd)}, ${r(rs1)}, #${imm}`;
      case "sari":
        return `asr ${r(rd)}, ${r(rs1)}, #${imm}`;
      case "rol":
        return `ror ${r(rd)}, ${r(rs1)}, #${(32 - imm) & 31}`;
      case "popcnt":
        return `fmov d31, ${r(rs1)}\n  cnt v31.8b, v31.8b\n  addv b31, v31.8b\n  fmov ${r(rd)}, d31`;
      case "clz":
        return `clz ${r(rd)}, ${r(rs1)}`;
      case "ctz":
        return `rbit ${r(rd)}, ${r(rs1)}\n  clz ${r(rd)}, ${r(rd)}`;
      case "bswap":
        return `rev ${r(rd)}, ${r(rs1)}`;
      case "blsi":
        return `neg x16, ${r(rs1)}\n  and ${r(rd)}, ${r(rs1)}, x16`;
      case "blsr":
        return `sub x16, ${r(rs1)}, #1\n  and ${r(rd)}, ${r(rs1)}, x16`;
      case "blsmsk":
        return `sub x16, ${r(rs1)}, #1\n  eor ${r(rd)}, ${r(rs1)}, x16`;
      case "lea":
        return `add ${r(rd)}, ${r(rs1)}, ${r(rs2)}, lsl #${imm}`;
      case "cmp":
        return `cmp ${r(rs1)}, ${r(rs2)}`;
      case "cmpi":
        return `cmp ${r(rs1)}, #${imm}`;
      case "setl":
        return `cset ${r(rd)}, lt`;
      case "setb":
        return `cset ${r(rd)}, lo`;
      case "sete":
        return `cset ${r(rd)}, eq`;
      case "cmovl":
        return `csel ${r(rd)}, ${r(rs1)}, ${r(rd)}, lt`;
      case "cmove":
        return `csel ${r(rd)}, ${r(rs1)}, ${r(rd)}, eq`;
      case "cmovb":
        return `csel ${r(rd)}, ${r(rs1)}, ${r(rd)}, lo`;
      case "smin":
        return `cmp ${r(rs1)}, ${r(rs2)}\n  csel ${r(rd)}, ${r(rs1)}, ${r(rs2)}, lt`;
      case "smax":
        return `cmp ${r(rs1)}, ${r(rs2)}\n  csel ${r(rd)}, ${r(rs1)}, ${r(rs2)}, gt`;
      case "umin":
        return `cmp ${r(rs1)}, ${r(rs2)}\n  csel ${r(rd)}, ${r(rs1)}, ${r(rs2)}, lo`;
      case "umax":
        return `cmp ${r(rs1)}, ${r(rs2)}\n  csel ${r(rd)}, ${r(rs1)}, ${r(rs2)}, hi`;
      case "abs":
        return `cmp ${r(rs1)}, #0\n  cneg ${r(rd)}, ${r(rs1)}, mi`;
      case "load":
        return `ldr ${r(rd)}, [${r(rs1)}, #${imm}]`;
      case "store":
        return `str ${r(rs2)}, [${r(rs1)}, #${imm}]`;
      case "fmov":
        return `fmov ${f(rd)}, ${f(rs1)}`;
      case "fadd":
        return `fadd ${f(rd)}, ${f(rs1)}, ${f(rs2)}`;
      case "fsub":
        return `fsub ${f(rd)}, ${f(rs1)}, ${f(rs2)}`;
      case "fmul":
        return `fmul ${f(rd)}, ${f(rs1)}, ${f(rs2)}`;
      case "fdiv":
        return `fdiv ${f(rd)}, ${f(rs1)}, ${f(rs2)}`;
      case "fsqrt":
        return `fsqrt ${f(rd)}, ${f(rs1)}`;
      case "fma":
        return `fmadd ${f(rd)}, ${f(rs1)}, ${f(rs2)}, ${f(rs3)}`;
      case "fabs":
        return `fabs ${f(rd)}, ${f(rs1)}`;
      case "fneg":
        return `fneg ${f(rd)}, ${f(rs1)}`;
      case "fmin":
        return `fmin ${f(rd)}, ${f(rs1)}, ${f(rs2)}`;
      case "fmax":
        return `fmax ${f(rd)}, ${f(rs1)}, ${f(rs2)}`;
      case "vadd":
        return `fadd ${v(rd)}, ${v(rs1)}, ${v(rs2)}`;
      case "vmul":
        return `fmul ${v(rd)}, ${v(rs1)}, ${v(rs2)}`;
      case "vfma":
        return `fmla ${v(rd)}, ${v(rs2)}, ${v(rs3)}`;
      case "vbroadcast":
        return `dup ${v(rd)}, ${r(rs1)}`;
      case "vhadd":
        return `faddp ${gp(A64_VREG, rs1)}.2d, ${v(rs1)}, ${v(rs1)}\n  fmov ${r(rd)}, d${rs1}`;
      default:
        return `// unsupported ${op}`;
    }
  },
  encodedSize(): number {
    return 4;
  },
};

const REGISTRY = new Map<string, IsaBackend>([
  [X86_64.name, X86_64],
  [AARCH64.name, AARCH64],
]);

export function registerIsa(backend: IsaBackend): void {
  REGISTRY.set(backend.name, backend);
}

export function getIsa(name: string): IsaBackend {
  const b = REGISTRY.get(name);
  if (!b) throw new Error(`unknown ISA backend: ${name}`);
  return b;
}

export function listIsas(): string[] {
  return [...REGISTRY.keys()];
}

/** Pretty-print a whole program in native syntax with a header comment. */
export function disassemble(prog: Instruction[], isaName = "x86-64"): string {
  const isa = getIsa(isaName);
  const lines: string[] = [`  # ${isa.name} projection of the virtual core program`];
  for (const ins of prog) lines.push(`  ${isa.print(ins)}`);
  return lines.join("\n");
}

/** Render an instruction in the virtual-core textual form (round-trippable). */
export function formatInstruction(ins: Instruction): string {
  const s = spec(ins.op);
  const parts = s.roles.map((role) => {
    switch (role) {
      case "rd":
        return s.cls === "float" ? `f${ins.rd}` : `r${ins.rd}`;
      case "rs1":
        return s.cls === "float" ? `f${ins.rs1}` : `r${ins.rs1}`;
      case "rs2":
        return s.cls === "float" ? `f${ins.rs2}` : `r${ins.rs2}`;
      case "rs3":
        return s.cls === "float" ? `f${ins.rs3}` : `r${ins.rs3}`;
      case "vd":
        return `v${ins.rd}`;
      case "vs1":
        return `v${ins.rs1}`;
      case "vs2":
        return `v${ins.rs2}`;
      case "vs3":
        return `v${ins.rs3}`;
      case "imm":
        return `#${ins.imm}`;
    }
  });
  return `${ins.op}${parts.length ? " " + parts.join(", ") : ""}`;
}

export function formatProgram(prog: Instruction[]): string {
  return prog.map((i) => formatInstruction(i)).join("\n");
}

export function parseProgram(text: string): Instruction[] {
  const out: Instruction[] = [];
  for (const raw of text.split("\n")) {
    const line = raw.split(";")[0].split("//")[0].trim();
    if (!line) continue;
    const [op, ...rest] = line.split(/\s+/);
    const args = rest.join("").split(",").filter(Boolean);
    const s = spec(op);
    const ins: Instruction = { op, rd: -1, rs1: -1, rs2: -1, rs3: -1, imm: 0 };
    s.roles.forEach((role, i) => {
      const tok = args[i] ?? "0";
      const num = Number.parseInt(tok.replace(/[^-0-9]/g, ""), 10) || 0;
      if (role === "rd" || role === "vd") ins.rd = num;
      else if (role === "rs1" || role === "vs1") ins.rs1 = num;
      else if (role === "rs2" || role === "vs2") ins.rs2 = num;
      else if (role === "rs3" || role === "vs3") ins.rs3 = num;
      else ins.imm = num;
    });
    out.push(ins);
  }
  return out;
}

/** Validate operand ranges and register indices for a given machine. */
export function validate(
  ins: Instruction,
  gprCount: number,
  fprCount: number,
  vregCount: number,
): string | null {
  const s = OPSPEC[ins.op];
  if (!s) return `unknown opcode ${ins.op}`;
  for (const role of s.roles) {
    if (role === "imm") {
      if (ins.imm < s.immMin || ins.imm > s.immMax) return `immediate out of range for ${ins.op}`;
      continue;
    }
    const value =
      role === "rd" || role === "vd"
        ? ins.rd
        : role === "rs1" || role === "vs1"
          ? ins.rs1
          : role === "rs2" || role === "vs2"
            ? ins.rs2
            : ins.rs3;
    const cap = registerFileCap(s, role, gprCount, fprCount, vregCount);
    if (value < 0 || value >= cap) return `register ${role}=${value} out of range for ${ins.op}`;
  }
  return null;
}

/** Size of the register file addressed by `role` for opcode spec `s`. */
export function registerFileCap(
  s: OpSpec,
  role: OperandRole,
  gprCount: number,
  fprCount: number,
  vregCount: number,
): number {
  if (role === "vd" || role === "vs1" || role === "vs2" || role === "vs3") return vregCount;
  return s.cls === "float" ? fprCount : gprCount;
}
