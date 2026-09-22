/**
 * Concrete CPU state model and interpreter for the virtual core.
 *
 * The datapath width is configurable (8 bits for bounded verification, 32 bits
 * for execution and testing). Memory accesses are bounds- and alignment
 * checked; every fault is recorded in the machine state rather than thrown so
 * that the cost function can penalise it.
 */

import { spec } from "./isa";
import type { CpuState, Instruction, MachineConfig, Program, TestCase } from "./types";
import { DEFAULT_MACHINE } from "./types";

export function maskOf(width: number): number {
  return width >= 32 ? 0xffffffff >>> 0 : ((1 << width) - 1) >>> 0;
}

export function toUnsigned(v: number, width: number): number {
  return (v & maskOf(width)) >>> 0;
}

export function toSigned(v: number, width: number): number {
  const u = toUnsigned(v, width);
  const signBit = 1 << (width - 1);
  if (width >= 32) return u | 0;
  return u >= signBit ? u - (1 << width) : u;
}

export function createState(cfg: MachineConfig = DEFAULT_MACHINE): CpuState {
  return {
    gpr: new Uint32Array(cfg.gprCount),
    fpr: new Float64Array(cfg.fprCount),
    vreg: new Float64Array(cfg.vregCount * 4),
    mem: new Uint8Array(cfg.memBytes),
    flags: { zf: false, sf: false, cf: false, of: false },
    fault: "none",
    faultDetail: "",
    retired: 0,
  };
}

export function cloneState(s: CpuState): CpuState {
  return {
    gpr: Uint32Array.from(s.gpr),
    fpr: Float64Array.from(s.fpr),
    vreg: Float64Array.from(s.vreg),
    mem: Uint8Array.from(s.mem),
    flags: { ...s.flags },
    fault: s.fault,
    faultDetail: s.faultDetail,
    retired: s.retired,
  };
}

export function stateFromTest(tc: TestCase, cfg: MachineConfig = DEFAULT_MACHINE): CpuState {
  const st = createState(cfg);
  for (let i = 0; i < cfg.gprCount; i++) st.gpr[i] = toUnsigned(tc.gpr[i] ?? 0, cfg.width);
  for (let i = 0; i < cfg.fprCount; i++) st.fpr[i] = tc.fpr[i] ?? 0;
  for (let i = 0; i < cfg.vregCount * 4; i++) st.vreg[i] = tc.vreg[i] ?? 0;
  for (const [addr, byte] of tc.mem) {
    if (addr >= 0 && addr < cfg.memBytes) st.mem[addr] = byte & 0xff;
  }
  return st;
}

function popcount32(v: number): number {
  let x = v >>> 0;
  x = x - ((x >>> 1) & 0x55555555);
  x = (x & 0x33333333) + ((x >>> 2) & 0x33333333);
  x = (x + (x >>> 4)) & 0x0f0f0f0f;
  return (Math.imul(x, 0x01010101) >>> 24) & 0xff;
}

function mulhu(a: number, b: number): number {
  const aLo = a & 0xffff;
  const aHi = a >>> 16;
  const bLo = b & 0xffff;
  const bHi = b >>> 16;
  const lolo = aLo * bLo;
  const hilo = aHi * bLo;
  const lohi = aLo * bHi;
  const hihi = aHi * bHi;
  const cross = (lolo >>> 16) + (hilo & 0xffff) + lohi;
  return (hihi + (hilo >>> 16) + (cross >>> 16)) >>> 0;
}

function reverseBytes(v: number, width: number): number {
  const bytes = Math.max(1, Math.floor(width / 8));
  let out = 0;
  for (let i = 0; i < bytes; i++) {
    out = ((out << 8) | ((v >>> (i * 8)) & 0xff)) >>> 0;
  }
  return toUnsigned(out, width);
}

/**
 * Execute `prog` on state `st` in place. Faults stop execution but leave the
 * state observable so live-out comparison can still be performed.
 */
export function execute(prog: Program, st: CpuState, cfg: MachineConfig = DEFAULT_MACHINE): CpuState {
  const W = cfg.width;
  const M = maskOf(W);
  const signBit = W >= 32 ? 0x80000000 : 1 << (W - 1);
  const g = st.gpr;
  const f = st.fpr;
  const v = st.vreg;
  const u = (x: number) => (x & M) >>> 0;
  const sx = (x: number) => toSigned(x, W);

  let fuel = cfg.fuel;
  for (let pc = 0; pc < prog.length; pc++) {
    if (--fuel <= 0) {
      st.fault = "timeout";
      st.faultDetail = `fuel exhausted at pc=${pc}`;
      return st;
    }
    const ins = prog[pc];
    st.retired++;
    const a = ins.rs1 >= 0 ? g[ins.rs1] : 0;
    const b = ins.rs2 >= 0 ? g[ins.rs2] : 0;
    switch (ins.op) {
      case "nop":
        break;
      case "mov":
        g[ins.rd] = u(a);
        break;
      case "movi":
        g[ins.rd] = u(ins.imm);
        break;
      case "add": {
        const r = u(a + b);
        setArithFlags(st, r, a, b, false, W, signBit);
        g[ins.rd] = r;
        break;
      }
      case "addi": {
        const imm = u(ins.imm);
        const r = u(a + imm);
        setArithFlags(st, r, a, imm, false, W, signBit);
        g[ins.rd] = r;
        break;
      }
      case "sub": {
        const r = u(a - b);
        setArithFlags(st, r, a, b, true, W, signBit);
        g[ins.rd] = r;
        break;
      }
      case "neg": {
        const r = u(0 - a);
        setArithFlags(st, r, 0, a, true, W, signBit);
        g[ins.rd] = r;
        break;
      }
      case "mul":
        g[ins.rd] = u(Math.imul(a | 0, b | 0));
        break;
      case "muli":
        g[ins.rd] = u(Math.imul(a | 0, ins.imm | 0));
        break;
      case "mulhu":
        g[ins.rd] =
          W >= 32
            ? mulhu(a >>> 0, b >>> 0) >>> 0
            : u(Math.floor((u(a) * u(b)) / (1 << W)));
        break;
      case "divu": {
        if (u(b) === 0) {
          st.fault = "divide-by-zero";
          st.faultDetail = `divu by zero at pc=${pc}`;
          return st;
        }
        g[ins.rd] = u(Math.floor(u(a) / u(b)));
        break;
      }
      case "and":
        g[ins.rd] = u(a & b);
        setLogicFlags(st, g[ins.rd], signBit);
        break;
      case "or":
        g[ins.rd] = u(a | b);
        setLogicFlags(st, g[ins.rd], signBit);
        break;
      case "xor":
        g[ins.rd] = u(a ^ b);
        setLogicFlags(st, g[ins.rd], signBit);
        break;
      case "andn":
        g[ins.rd] = u(~a & b);
        break;
      case "andi":
        g[ins.rd] = u(a & u(ins.imm));
        setLogicFlags(st, g[ins.rd], signBit);
        break;
      case "ori":
        g[ins.rd] = u(a | u(ins.imm));
        setLogicFlags(st, g[ins.rd], signBit);
        break;
      case "xori":
        g[ins.rd] = u(a ^ u(ins.imm));
        setLogicFlags(st, g[ins.rd], signBit);
        break;
      case "not":
        g[ins.rd] = u(~a);
        break;
      case "shl":
        g[ins.rd] = u(a << (u(b) % W));
        break;
      case "shr":
        g[ins.rd] = u(u(a) >>> (u(b) % W));
        break;
      case "sar":
        g[ins.rd] = u(sx(a) >> (u(b) % W));
        break;
      case "shli":
        g[ins.rd] = u(a << (ins.imm % W));
        break;
      case "shri":
        g[ins.rd] = u(u(a) >>> (ins.imm % W));
        break;
      case "sari":
        g[ins.rd] = u(sx(a) >> (ins.imm % W));
        break;
      case "rol": {
        const k = ins.imm % W;
        g[ins.rd] = u((u(a) << k) | (u(a) >>> (W - k || W)));
        break;
      }
      case "popcnt":
        g[ins.rd] = popcount32(u(a));
        break;
      case "clz": {
        const x = u(a);
        if (x === 0) g[ins.rd] = W;
        else {
          let n = 0;
          for (let i = W - 1; i >= 0; i--, n++) if ((x >>> i) & 1) break;
          g[ins.rd] = n;
        }
        break;
      }
      case "ctz": {
        const x = u(a);
        if (x === 0) g[ins.rd] = W;
        else {
          let n = 0;
          while (((x >>> n) & 1) === 0) n++;
          g[ins.rd] = n;
        }
        break;
      }
      case "bswap":
        g[ins.rd] = reverseBytes(u(a), W);
        break;
      case "blsi":
        g[ins.rd] = u(a & u(0 - a));
        break;
      case "blsr":
        g[ins.rd] = u(a & u(a - 1));
        break;
      case "blsmsk":
        g[ins.rd] = u(a ^ u(a - 1));
        break;
      case "lea":
        g[ins.rd] = u(a + (b << ins.imm));
        break;
      case "cmp": {
        const r = u(a - b);
        setArithFlags(st, r, a, b, true, W, signBit);
        break;
      }
      case "cmpi": {
        const imm = u(ins.imm);
        const r = u(a - imm);
        setArithFlags(st, r, a, imm, true, W, signBit);
        break;
      }
      case "setl":
        g[ins.rd] = st.flags.sf !== st.flags.of ? 1 : 0;
        break;
      case "setb":
        g[ins.rd] = st.flags.cf ? 1 : 0;
        break;
      case "sete":
        g[ins.rd] = st.flags.zf ? 1 : 0;
        break;
      case "cmovl":
        if (st.flags.sf !== st.flags.of) g[ins.rd] = u(a);
        break;
      case "cmove":
        if (st.flags.zf) g[ins.rd] = u(a);
        break;
      case "cmovb":
        if (st.flags.cf) g[ins.rd] = u(a);
        break;
      case "smin":
        g[ins.rd] = u(Math.min(sx(a), sx(b)));
        break;
      case "smax":
        g[ins.rd] = u(Math.max(sx(a), sx(b)));
        break;
      case "umin":
        g[ins.rd] = u(Math.min(u(a), u(b)));
        break;
      case "umax":
        g[ins.rd] = u(Math.max(u(a), u(b)));
        break;
      case "abs":
        g[ins.rd] = u(Math.abs(sx(a)));
        break;
      case "load": {
        const addr = (u(a) + ins.imm) | 0;
        if (addr < 0 || addr + 4 > cfg.memBytes) {
          st.fault = "segfault";
          st.faultDetail = `load out of bounds addr=${addr} at pc=${pc}`;
          return st;
        }
        if (addr % cfg.alignment !== 0) {
          st.fault = "unaligned";
          st.faultDetail = `unaligned load addr=${addr} at pc=${pc}`;
          return st;
        }
        const m = st.mem;
        g[ins.rd] = u(
          (m[addr] | (m[addr + 1] << 8) | (m[addr + 2] << 16) | (m[addr + 3] << 24)) >>> 0,
        );
        break;
      }
      case "store": {
        const addr = (u(a) + ins.imm) | 0;
        if (addr < 0 || addr + 4 > cfg.memBytes) {
          st.fault = "segfault";
          st.faultDetail = `store out of bounds addr=${addr} at pc=${pc}`;
          return st;
        }
        if (addr % cfg.alignment !== 0) {
          st.fault = "unaligned";
          st.faultDetail = `unaligned store addr=${addr} at pc=${pc}`;
          return st;
        }
        const val = u(b);
        st.mem[addr] = val & 0xff;
        st.mem[addr + 1] = (val >>> 8) & 0xff;
        st.mem[addr + 2] = (val >>> 16) & 0xff;
        st.mem[addr + 3] = (val >>> 24) & 0xff;
        break;
      }
      case "fmov":
        f[ins.rd] = f[ins.rs1];
        break;
      case "fadd":
        f[ins.rd] = f[ins.rs1] + f[ins.rs2];
        break;
      case "fsub":
        f[ins.rd] = f[ins.rs1] - f[ins.rs2];
        break;
      case "fmul":
        f[ins.rd] = f[ins.rs1] * f[ins.rs2];
        break;
      case "fdiv":
        f[ins.rd] = f[ins.rs1] / f[ins.rs2];
        break;
      case "fsqrt":
        f[ins.rd] = Math.sqrt(f[ins.rs1]);
        break;
      case "fma":
        f[ins.rd] = f[ins.rs1] * f[ins.rs2] + f[ins.rs3];
        break;
      case "fabs":
        f[ins.rd] = Math.abs(f[ins.rs1]);
        break;
      case "fneg":
        f[ins.rd] = -f[ins.rs1];
        break;
      case "fmin":
        f[ins.rd] = Math.min(f[ins.rs1], f[ins.rs2]);
        break;
      case "fmax":
        f[ins.rd] = Math.max(f[ins.rs1], f[ins.rs2]);
        break;
      case "vadd":
        for (let l = 0; l < 4; l++) v[ins.rd * 4 + l] = v[ins.rs1 * 4 + l] + v[ins.rs2 * 4 + l];
        break;
      case "vmul":
        for (let l = 0; l < 4; l++) v[ins.rd * 4 + l] = v[ins.rs1 * 4 + l] * v[ins.rs2 * 4 + l];
        break;
      case "vfma":
        for (let l = 0; l < 4; l++)
          v[ins.rd * 4 + l] = v[ins.rs1 * 4 + l] * v[ins.rs2 * 4 + l] + v[ins.rs3 * 4 + l];
        break;
      case "vbroadcast":
        for (let l = 0; l < 4; l++) v[ins.rd * 4 + l] = f[ins.rs1];
        break;
      case "vhadd":
        f[ins.rd] =
          v[ins.rs1 * 4] + v[ins.rs1 * 4 + 1] + v[ins.rs1 * 4 + 2] + v[ins.rs1 * 4 + 3];
        break;
      default:
        st.fault = "trap";
        st.faultDetail = `unimplemented opcode ${ins.op} at pc=${pc}`;
        return st;
    }
  }
  return st;
}

function setArithFlags(
  st: CpuState,
  r: number,
  a: number,
  b: number,
  isSub: boolean,
  width: number,
  signBit: number,
): void {
  st.flags.zf = r === 0;
  st.flags.sf = (r & signBit) !== 0;
  const ua = toUnsigned(a, width);
  const ub = toUnsigned(b, width);
  if (isSub) {
    st.flags.cf = ua < ub;
    const sa = (ua & signBit) !== 0;
    const sb = (ub & signBit) !== 0;
    const sr = (r & signBit) !== 0;
    st.flags.of = sa !== sb && sr !== sa;
  } else {
    st.flags.cf = ua + ub > toUnsigned(-1, width);
    const sa = (ua & signBit) !== 0;
    const sb = (ub & signBit) !== 0;
    const sr = (r & signBit) !== 0;
    st.flags.of = sa === sb && sr !== sa;
  }
}

function setLogicFlags(st: CpuState, r: number, signBit: number): void {
  st.flags.zf = r === 0;
  st.flags.sf = (r & signBit) !== 0;
  st.flags.cf = false;
  st.flags.of = false;
}

/* --------------------------------------------------------------------- */
/* Dependence / liveness analysis                                          */
/* --------------------------------------------------------------------- */

export interface InstructionEffects {
  readsGpr: number[];
  writesGpr: number[];
  readsFpr: number[];
  writesFpr: number[];
  readsVreg: number[];
  writesVreg: number[];
  readsFlags: boolean;
  writesFlags: boolean;
  readsMem: boolean;
  writesMem: boolean;
}

export function effects(ins: Instruction): InstructionEffects {
  const s = spec(ins.op);
  const e: InstructionEffects = {
    readsGpr: [],
    writesGpr: [],
    readsFpr: [],
    writesFpr: [],
    readsVreg: [],
    writesVreg: [],
    readsFlags: s.readsFlags,
    writesFlags: s.writesFlags,
    readsMem: ins.op === "load",
    writesMem: ins.op === "store",
  };
  const isF = s.cls === "float";
  for (const role of s.roles) {
    switch (role) {
      case "rd":
        if (s.rmw) (isF ? e.readsFpr : e.readsGpr).push(ins.rd);
        (isF ? e.writesFpr : e.writesGpr).push(ins.rd);
        break;
      case "rs1":
        (isF ? e.readsFpr : e.readsGpr).push(ins.rs1);
        break;
      case "rs2":
        (isF ? e.readsFpr : e.readsGpr).push(ins.rs2);
        break;
      case "rs3":
        (isF ? e.readsFpr : e.readsGpr).push(ins.rs3);
        break;
      case "vd":
        if (ins.op === "vfma") e.readsVreg.push(ins.rd);
        e.writesVreg.push(ins.rd);
        break;
      case "vs1":
        e.readsVreg.push(ins.rs1);
        break;
      case "vs2":
        e.readsVreg.push(ins.rs2);
        break;
      case "vs3":
        e.readsVreg.push(ins.rs3);
        break;
      case "imm":
        break;
    }
  }
  if (ins.op === "vhadd") {
    e.writesFpr.push(ins.rd);
  }
  if (ins.op === "vbroadcast") {
    e.readsFpr.push(ins.rs1);
  }
  if (ins.op === "store") {
    // `store rs1, rs2, imm`: both operands are read, nothing is written to regs.
    e.writesGpr.length = 0;
  }
  return e;
}

/** True when instructions at index i and j may be reordered safely. */
export function commutesInOrder(x: Instruction, y: Instruction): boolean {
  const a = effects(x);
  const b = effects(y);
  const overlap = (p: number[], q: number[]) => p.some((v) => q.includes(v));
  if (overlap(a.writesGpr, b.readsGpr) || overlap(a.writesGpr, b.writesGpr)) return false;
  if (overlap(b.writesGpr, a.readsGpr)) return false;
  if (overlap(a.writesFpr, b.readsFpr) || overlap(a.writesFpr, b.writesFpr)) return false;
  if (overlap(b.writesFpr, a.readsFpr)) return false;
  if (overlap(a.writesVreg, b.readsVreg) || overlap(a.writesVreg, b.writesVreg)) return false;
  if (overlap(b.writesVreg, a.readsVreg)) return false;
  if ((a.writesFlags && (b.readsFlags || b.writesFlags)) || (b.writesFlags && a.readsFlags))
    return false;
  if ((a.writesMem && (b.readsMem || b.writesMem)) || (b.writesMem && a.readsMem)) return false;
  return true;
}

/** Remove instructions whose results cannot influence the live-out set. */
export function deadCodeEliminate(prog: Program, liveGpr: number[], liveFpr: number[]): Program {
  const keep = new Array<boolean>(prog.length).fill(false);
  const liveG = new Set(liveGpr);
  const liveF = new Set(liveFpr);
  let liveFlags = false;
  let liveMem = true;
  for (let i = prog.length - 1; i >= 0; i--) {
    const e = effects(prog[i]);
    const producesLive =
      e.writesGpr.some((r) => liveG.has(r)) ||
      e.writesFpr.some((r) => liveF.has(r)) ||
      e.writesVreg.length > 0 ||
      (e.writesFlags && liveFlags) ||
      (e.writesMem && liveMem);
    if (!producesLive) continue;
    keep[i] = true;
    for (const r of e.writesGpr) if (!e.readsGpr.includes(r)) liveG.delete(r);
    for (const r of e.writesFpr) if (!e.readsFpr.includes(r)) liveF.delete(r);
    for (const r of e.readsGpr) liveG.add(r);
    for (const r of e.readsFpr) liveF.add(r);
    if (e.writesFlags) liveFlags = false;
    if (e.readsFlags) liveFlags = true;
  }
  // Vector defs feeding kept instructions are conservatively retained above.
  return prog.filter((_, i) => keep[i]);
}
