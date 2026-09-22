/**
 * Symbolic execution + SMT (QF_BV) verification.
 *
 * Programs are symbolically executed over bit-vector circuits: each
 * architectural register becomes an array of Boolean literals, every opcode is
 * expanded into gates (ripple-carry adders, shift/add multipliers, barrel
 * shifters, priority encoders, ...) that are Tseitin-encoded into CNF on the
 * fly. Equivalence of a rewrite R and a target T is decided by solving the
 * miter  OR_over_live_outs (R_out XOR T_out)  with the bundled DPLL engine
 * (unit propagation via two-watched literals, chronological backtracking,
 * decision/conflict budgets). SAT yields a counterexample that is injected
 * into the dynamic test database; UNSAT proves equivalence for the modelled
 * semantics at the verification width.
 *
 * Opcodes with no bit-precise encoding here (floating point, memory traffic,
 * unsigned division) fall back to exhaustive bounded enumeration on the
 * concrete interpreter.
 */

import { spec } from "./isa";
import { execute, stateFromTest, toUnsigned } from "./machine";
import { makeCounterexample } from "./testcases";
import type { LiveOut, MachineConfig, Program, TestCase, VerificationResult } from "./types";
import { DEFAULT_MACHINE } from "./types";

export type Lit = number;
export type BV = Lit[]; // little-endian bit array

export class Cnf {
  vars = 0;
  clauses: Lit[][] = [];
  readonly TRUE: Lit;
  readonly FALSE: Lit;
  private andCache = new Map<string, Lit>();

  constructor() {
    this.TRUE = this.newVar();
    this.FALSE = -this.TRUE;
    this.clauses.push([this.TRUE]);
  }
  newVar(): Lit {
    this.vars += 1;
    return this.vars;
  }
  add(clause: Lit[]): void {
    this.clauses.push(clause);
  }
  and(a: Lit, b: Lit): Lit {
    if (a === this.FALSE || b === this.FALSE) return this.FALSE;
    if (a === this.TRUE) return b;
    if (b === this.TRUE) return a;
    if (a === b) return a;
    if (a === -b) return this.FALSE;
    const key = a < b ? `${a},${b}` : `${b},${a}`;
    const hit = this.andCache.get(key);
    if (hit !== undefined) return hit;
    const o = this.newVar();
    this.add([-o, a]);
    this.add([-o, b]);
    this.add([o, -a, -b]);
    this.andCache.set(key, o);
    return o;
  }
  or(a: Lit, b: Lit): Lit {
    return -this.and(-a, -b);
  }
  xor(a: Lit, b: Lit): Lit {
    if (a === this.TRUE) return -b;
    if (a === this.FALSE) return b;
    if (b === this.TRUE) return -a;
    if (b === this.FALSE) return a;
    if (a === b) return this.FALSE;
    if (a === -b) return this.TRUE;
    const o = this.newVar();
    this.add([-o, a, b]);
    this.add([-o, -a, -b]);
    this.add([o, -a, b]);
    this.add([o, a, -b]);
    return o;
  }
  mux(s: Lit, t: Lit, e: Lit): Lit {
    if (s === this.TRUE) return t;
    if (s === this.FALSE) return e;
    if (t === e) return t;
    return this.or(this.and(s, t), this.and(-s, e));
  }
  orAll(lits: Lit[]): Lit {
    let acc = this.FALSE;
    for (const l of lits) acc = this.or(acc, l);
    return acc;
  }
  andAll(lits: Lit[]): Lit {
    let acc = this.TRUE;
    for (const l of lits) acc = this.and(acc, l);
    return acc;
  }
}

export class BvBuilder {
  constructor(
    readonly cnf: Cnf,
    readonly width: number,
  ) {}
  constant(value: number): BV {
    const out: BV = [];
    for (let i = 0; i < this.width; i++) out.push(((value >>> i) & 1) === 1 ? this.cnf.TRUE : this.cnf.FALSE);
    return out;
  }
  variable(): BV {
    const out: BV = [];
    for (let i = 0; i < this.width; i++) out.push(this.cnf.newVar());
    return out;
  }
  not(a: BV): BV {
    return a.map((l) => -l);
  }
  and(a: BV, b: BV): BV {
    return a.map((l, i) => this.cnf.and(l, b[i]));
  }
  or(a: BV, b: BV): BV {
    return a.map((l, i) => this.cnf.or(l, b[i]));
  }
  xor(a: BV, b: BV): BV {
    return a.map((l, i) => this.cnf.xor(l, b[i]));
  }
  fullAdd(a: Lit, b: Lit, c: Lit): { s: Lit; c: Lit } {
    const ab = this.cnf.xor(a, b);
    const s = this.cnf.xor(ab, c);
    const carry = this.cnf.or(this.cnf.and(a, b), this.cnf.and(ab, c));
    return { s, c: carry };
  }
  addWithCarry(a: BV, b: BV, cin: Lit): { sum: BV; carryOut: Lit; carryIntoMsb: Lit } {
    let c = cin;
    let carryIntoMsb = this.cnf.FALSE;
    const sum: BV = [];
    for (let i = 0; i < this.width; i++) {
      if (i === this.width - 1) carryIntoMsb = c;
      const r = this.fullAdd(a[i], b[i], c);
      sum.push(r.s);
      c = r.c;
    }
    return { sum, carryOut: c, carryIntoMsb };
  }
  add(a: BV, b: BV): BV {
    return this.addWithCarry(a, b, this.cnf.FALSE).sum;
  }
  sub(a: BV, b: BV): BV {
    return this.addWithCarry(a, this.not(b), this.cnf.TRUE).sum;
  }
  neg(a: BV): BV {
    return this.sub(this.constant(0), a);
  }
  mul(a: BV, b: BV): BV {
    let acc = this.constant(0);
    for (let i = 0; i < this.width; i++) {
      const shifted: BV = [];
      for (let j = 0; j < this.width; j++) {
        shifted.push(j >= i ? this.cnf.and(a[j - i], b[i]) : this.cnf.FALSE);
      }
      acc = this.add(acc, shifted);
    }
    return acc;
  }
  /** High half of the unsigned 2W-bit product. */
  mulHighUnsigned(a: BV, b: BV): BV {
    const W = this.width;
    const wide = new BvBuilder(this.cnf, 2 * W);
    const az = [...a, ...new Array<Lit>(W).fill(this.cnf.FALSE)];
    const bz = [...b, ...new Array<Lit>(W).fill(this.cnf.FALSE)];
    const prod = wide.mul(az, bz);
    return prod.slice(W, 2 * W);
  }
  shlConst(a: BV, k: number): BV {
    const s = ((k % this.width) + this.width) % this.width;
    const out: BV = [];
    for (let i = 0; i < this.width; i++) out.push(i >= s ? a[i - s] : this.cnf.FALSE);
    return out;
  }
  shrConst(a: BV, k: number): BV {
    const s = ((k % this.width) + this.width) % this.width;
    const out: BV = [];
    for (let i = 0; i < this.width; i++) out.push(i + s < this.width ? a[i + s] : this.cnf.FALSE);
    return out;
  }
  sarConst(a: BV, k: number): BV {
    const s = ((k % this.width) + this.width) % this.width;
    const msb = a[this.width - 1];
    const out: BV = [];
    for (let i = 0; i < this.width; i++) out.push(i + s < this.width ? a[i + s] : msb);
    return out;
  }
  rolConst(a: BV, k: number): BV {
    const s = ((k % this.width) + this.width) % this.width;
    const out: BV = [];
    for (let i = 0; i < this.width; i++) out.push(a[(i - s + this.width) % this.width]);
    return out;
  }
  private barrel(a: BV, b: BV, kind: "shl" | "shr" | "sar"): BV {
    const stages = Math.ceil(Math.log2(this.width));
    let cur = a;
    for (let s = 0; s < stages; s++) {
      const amount = 1 << s;
      const shifted =
        kind === "shl" ? this.shlConst(cur, amount) : kind === "shr" ? this.shrConst(cur, amount) : this.sarConst(cur, amount);
      const sel = b[s] ?? this.cnf.FALSE;
      cur = cur.map((l, i) => this.cnf.mux(sel, shifted[i], l));
    }
    return cur;
  }
  shl(a: BV, b: BV): BV {
    return this.barrel(a, b, "shl");
  }
  shr(a: BV, b: BV): BV {
    return this.barrel(a, b, "shr");
  }
  sar(a: BV, b: BV): BV {
    return this.barrel(a, b, "sar");
  }
  eq(a: BV, b: BV): Lit {
    return this.cnf.andAll(a.map((l, i) => -this.cnf.xor(l, b[i])));
  }
  isZero(a: BV): Lit {
    return -this.cnf.orAll(a);
  }
  ult(a: BV, b: BV): Lit {
    // a <u b  <=>  borrow out of (a - b)
    return -this.addWithCarry(a, this.not(b), this.cnf.TRUE).carryOut;
  }
  slt(a: BV, b: BV): Lit {
    const r = this.addWithCarry(a, this.not(b), this.cnf.TRUE);
    const sr = r.sum[this.width - 1];
    const of = this.cnf.xor(r.carryOut, r.carryIntoMsb);
    return this.cnf.xor(sr, of);
  }
  ite(s: Lit, t: BV, e: BV): BV {
    return t.map((l, i) => this.cnf.mux(s, l, e[i]));
  }
  popcount(a: BV): BV {
    let acc = this.constant(0);
    for (const bit of a) {
      const oneHot = this.constant(0).slice();
      oneHot[0] = bit;
      acc = this.add(acc, oneHot);
    }
    return acc;
  }
  /** Count leading zeros: W when the input is zero. */
  clz(a: BV): BV {
    let result = this.constant(this.width);
    let found = this.cnf.FALSE;
    for (let i = this.width - 1; i >= 0; i--) {
      const take = this.cnf.and(-found, a[i]);
      result = this.ite(take, this.constant(this.width - 1 - i), result);
      found = this.cnf.or(found, a[i]);
    }
    return result;
  }
  ctz(a: BV): BV {
    let result = this.constant(this.width);
    let found = this.cnf.FALSE;
    for (let i = 0; i < this.width; i++) {
      const take = this.cnf.and(-found, a[i]);
      result = this.ite(take, this.constant(i), result);
      found = this.cnf.or(found, a[i]);
    }
    return result;
  }
  bswap(a: BV): BV {
    if (this.width % 8 !== 0) return a.slice();
    const bytes = this.width / 8;
    const out: BV = new Array(this.width);
    for (let byte = 0; byte < bytes; byte++) {
      for (let bit = 0; bit < 8; bit++) {
        out[(bytes - 1 - byte) * 8 + bit] = a[byte * 8 + bit];
      }
    }
    return out;
  }
}

export class UnsupportedSymbolicOp extends Error {
  constructor(op: string) {
    super(`no bit-precise encoding for opcode '${op}'`);
  }
}

interface SymState {
  gpr: BV[];
  zf: Lit;
  sf: Lit;
  cf: Lit;
  of: Lit;
}

export function symbolicExecute(
  prog: Program,
  init: BV[],
  bv: BvBuilder,
): SymState {
  const cnf = bv.cnf;
  const st: SymState = {
    gpr: init.map((x) => x.slice()),
    zf: cnf.FALSE,
    sf: cnf.FALSE,
    cf: cnf.FALSE,
    of: cnf.FALSE,
  };
  const W = bv.width;
  const setArith = (sum: BV, carryOut: Lit, carryIntoMsb: Lit, isSub: boolean) => {
    st.zf = bv.isZero(sum);
    st.sf = sum[W - 1];
    st.cf = isSub ? -carryOut : carryOut;
    st.of = cnf.xor(carryOut, carryIntoMsb);
  };
  const setLogic = (r: BV) => {
    st.zf = bv.isZero(r);
    st.sf = r[W - 1];
    st.cf = cnf.FALSE;
    st.of = cnf.FALSE;
  };

  for (const ins of prog) {
    const a = ins.rs1 >= 0 ? st.gpr[ins.rs1] : bv.constant(0);
    const b = ins.rs2 >= 0 ? st.gpr[ins.rs2] : bv.constant(0);
    switch (ins.op) {
      case "nop":
        break;
      case "mov":
        st.gpr[ins.rd] = a.slice();
        break;
      case "movi":
        st.gpr[ins.rd] = bv.constant(toUnsigned(ins.imm, W));
        break;
      case "add": {
        const r = bv.addWithCarry(a, b, cnf.FALSE);
        setArith(r.sum, r.carryOut, r.carryIntoMsb, false);
        st.gpr[ins.rd] = r.sum;
        break;
      }
      case "addi": {
        const imm = bv.constant(toUnsigned(ins.imm, W));
        const r = bv.addWithCarry(a, imm, cnf.FALSE);
        setArith(r.sum, r.carryOut, r.carryIntoMsb, false);
        st.gpr[ins.rd] = r.sum;
        break;
      }
      case "sub": {
        const r = bv.addWithCarry(a, bv.not(b), cnf.TRUE);
        setArith(r.sum, r.carryOut, r.carryIntoMsb, true);
        st.gpr[ins.rd] = r.sum;
        break;
      }
      case "neg": {
        const r = bv.addWithCarry(bv.constant(0), bv.not(a), cnf.TRUE);
        setArith(r.sum, r.carryOut, r.carryIntoMsb, true);
        st.gpr[ins.rd] = r.sum;
        break;
      }
      case "mul":
        st.gpr[ins.rd] = bv.mul(a, b);
        break;
      case "muli":
        st.gpr[ins.rd] = bv.mul(a, bv.constant(toUnsigned(ins.imm, W)));
        break;
      case "mulhu":
        st.gpr[ins.rd] = bv.mulHighUnsigned(a, b);
        break;
      case "and":
        st.gpr[ins.rd] = bv.and(a, b);
        setLogic(st.gpr[ins.rd]);
        break;
      case "or":
        st.gpr[ins.rd] = bv.or(a, b);
        setLogic(st.gpr[ins.rd]);
        break;
      case "xor":
        st.gpr[ins.rd] = bv.xor(a, b);
        setLogic(st.gpr[ins.rd]);
        break;
      case "andn":
        st.gpr[ins.rd] = bv.and(bv.not(a), b);
        break;
      case "andi":
        st.gpr[ins.rd] = bv.and(a, bv.constant(toUnsigned(ins.imm, W)));
        setLogic(st.gpr[ins.rd]);
        break;
      case "ori":
        st.gpr[ins.rd] = bv.or(a, bv.constant(toUnsigned(ins.imm, W)));
        setLogic(st.gpr[ins.rd]);
        break;
      case "xori":
        st.gpr[ins.rd] = bv.xor(a, bv.constant(toUnsigned(ins.imm, W)));
        setLogic(st.gpr[ins.rd]);
        break;
      case "not":
        st.gpr[ins.rd] = bv.not(a);
        break;
      case "shl":
        st.gpr[ins.rd] = bv.shl(a, b);
        break;
      case "shr":
        st.gpr[ins.rd] = bv.shr(a, b);
        break;
      case "sar":
        st.gpr[ins.rd] = bv.sar(a, b);
        break;
      case "shli":
        st.gpr[ins.rd] = bv.shlConst(a, ins.imm);
        break;
      case "shri":
        st.gpr[ins.rd] = bv.shrConst(a, ins.imm);
        break;
      case "sari":
        st.gpr[ins.rd] = bv.sarConst(a, ins.imm);
        break;
      case "rol":
        st.gpr[ins.rd] = bv.rolConst(a, ins.imm);
        break;
      case "popcnt":
        st.gpr[ins.rd] = bv.popcount(a);
        break;
      case "clz":
        st.gpr[ins.rd] = bv.clz(a);
        break;
      case "ctz":
        st.gpr[ins.rd] = bv.ctz(a);
        break;
      case "bswap":
        st.gpr[ins.rd] = bv.bswap(a);
        break;
      case "blsi":
        st.gpr[ins.rd] = bv.and(a, bv.neg(a));
        break;
      case "blsr":
        st.gpr[ins.rd] = bv.and(a, bv.sub(a, bv.constant(1)));
        break;
      case "blsmsk":
        st.gpr[ins.rd] = bv.xor(a, bv.sub(a, bv.constant(1)));
        break;
      case "lea":
        st.gpr[ins.rd] = bv.add(a, bv.shlConst(b, ins.imm));
        break;
      case "cmp": {
        const r = bv.addWithCarry(a, bv.not(b), cnf.TRUE);
        setArith(r.sum, r.carryOut, r.carryIntoMsb, true);
        break;
      }
      case "cmpi": {
        const imm = bv.constant(toUnsigned(ins.imm, W));
        const r = bv.addWithCarry(a, bv.not(imm), cnf.TRUE);
        setArith(r.sum, r.carryOut, r.carryIntoMsb, true);
        break;
      }
      case "setl":
        st.gpr[ins.rd] = bv.ite(cnf.xor(st.sf, st.of), bv.constant(1), bv.constant(0));
        break;
      case "setb":
        st.gpr[ins.rd] = bv.ite(st.cf, bv.constant(1), bv.constant(0));
        break;
      case "sete":
        st.gpr[ins.rd] = bv.ite(st.zf, bv.constant(1), bv.constant(0));
        break;
      case "cmovl":
        st.gpr[ins.rd] = bv.ite(cnf.xor(st.sf, st.of), a, st.gpr[ins.rd]);
        break;
      case "cmove":
        st.gpr[ins.rd] = bv.ite(st.zf, a, st.gpr[ins.rd]);
        break;
      case "cmovb":
        st.gpr[ins.rd] = bv.ite(st.cf, a, st.gpr[ins.rd]);
        break;
      case "smin":
        st.gpr[ins.rd] = bv.ite(bv.slt(a, b), a, b);
        break;
      case "smax":
        st.gpr[ins.rd] = bv.ite(bv.slt(a, b), b, a);
        break;
      case "umin":
        st.gpr[ins.rd] = bv.ite(bv.ult(a, b), a, b);
        break;
      case "umax":
        st.gpr[ins.rd] = bv.ite(bv.ult(a, b), b, a);
        break;
      case "abs":
        st.gpr[ins.rd] = bv.ite(a[W - 1], bv.neg(a), a);
        break;
      default:
        throw new UnsupportedSymbolicOp(ins.op);
    }
  }
  return st;
}

/* --------------------------------------------------------------------- */
/* DPLL SAT solver (two-watched literals, chronological backtracking)      */
/* --------------------------------------------------------------------- */

export interface SolveResult {
  sat: boolean;
  model?: Int8Array; // index by variable, 1 = true, -1 = false, 0 = unassigned
  decisions: number;
  propagations: number;
  conflicts: number;
  exhausted: boolean;
}

export class DpllSolver {
  private watches: number[][]; // literal index -> clause indices
  private assign: Int8Array;
  private trail: number[] = [];
  private trailLim: number[] = [];
  private reasonDecision: boolean[] = [];
  decisions = 0;
  propagations = 0;
  conflicts = 0;

  constructor(
    private cnf: Cnf,
    private decisionOrder: number[],
    private budget = { decisions: 4_000_000, conflicts: 2_000_000, timeMs: 8000 },
  ) {
    this.assign = new Int8Array(cnf.vars + 1);
    this.watches = new Array((cnf.vars + 1) * 2).fill(null).map(() => []);
    for (let ci = 0; ci < cnf.clauses.length; ci++) {
      const c = cnf.clauses[ci];
      if (c.length === 0) continue;
      this.watches[this.litIndex(c[0])].push(ci);
      if (c.length > 1) this.watches[this.litIndex(c[1])].push(ci);
    }
  }
  private litIndex(l: number): number {
    return l > 0 ? l * 2 : -l * 2 + 1;
  }
  private value(l: number): number {
    const v = this.assign[Math.abs(l)];
    if (v === 0) return 0;
    return l > 0 ? v : -v;
  }
  private enqueue(l: number, decision: boolean): boolean {
    const v = this.value(l);
    if (v === 1) return true;
    if (v === -1) return false;
    this.assign[Math.abs(l)] = l > 0 ? 1 : -1;
    this.trail.push(l);
    this.reasonDecision.push(decision);
    return true;
  }
  private propagate(): boolean {
    let head = this.propHead;
    while (head < this.trail.length) {
      const l = this.trail[head++];
      const falseLit = -l;
      const list = this.watches[this.litIndex(falseLit)];
      const keep: number[] = [];
      let conflict = false;
      for (let k = 0; k < list.length; k++) {
        const ci = list[k];
        if (conflict) {
          keep.push(ci);
          continue;
        }
        const c = this.cnf.clauses[ci];
        if (c.length === 1) {
          keep.push(ci);
          if (this.value(c[0]) === -1) conflict = true;
          else if (!this.enqueue(c[0], false)) conflict = true;
          continue;
        }
        // Ensure the falsified literal is at position 1.
        if (c[0] === falseLit) {
          const t = c[0];
          c[0] = c[1];
          c[1] = t;
        }
        if (this.value(c[0]) === 1) {
          keep.push(ci);
          continue;
        }
        let moved = false;
        for (let i = 2; i < c.length; i++) {
          if (this.value(c[i]) !== -1) {
            const t = c[1];
            c[1] = c[i];
            c[i] = t;
            this.watches[this.litIndex(c[1])].push(ci);
            moved = true;
            break;
          }
        }
        if (moved) continue;
        keep.push(ci);
        this.propagations++;
        if (!this.enqueue(c[0], false)) {
          conflict = true;
        }
      }
      this.watches[this.litIndex(falseLit)] = keep;
      if (conflict) {
        this.propHead = head;
        return false;
      }
    }
    this.propHead = head;
    return true;
  }
  private propHead = 0;

  private backtrack(level: number): void {
    while (this.trailLim.length > level) {
      const limit = this.trailLim.pop() as number;
      while (this.trail.length > limit) {
        const l = this.trail.pop() as number;
        this.reasonDecision.pop();
        this.assign[Math.abs(l)] = 0;
      }
    }
    this.propHead = this.trail.length;
  }

  private pickBranch(): number {
    for (const v of this.decisionOrder) {
      if (this.assign[v] === 0) return v;
    }
    for (let v = 1; v <= this.cnf.vars; v++) {
      if (this.assign[v] === 0) return v;
    }
    return 0;
  }

  solve(assumptions: number[] = []): SolveResult {
    const start = Date.now();
    // Level-0 unit clauses.
    for (const c of this.cnf.clauses) {
      if (c.length === 1 && !this.enqueue(c[0], false)) {
        return { sat: false, decisions: 0, propagations: 0, conflicts: 1, exhausted: false };
      }
    }
    for (const a of assumptions) {
      if (!this.enqueue(a, false)) {
        return { sat: false, decisions: 0, propagations: 0, conflicts: 1, exhausted: false };
      }
    }
    if (!this.propagate()) {
      return { sat: false, decisions: 0, propagations: this.propagations, conflicts: 1, exhausted: false };
    }
    const decisionStack: number[] = []; // decision literals, with a "tried other polarity" flag encoded by sign of marker
    const flipped: boolean[] = [];
    for (;;) {
      if (
        this.decisions > this.budget.decisions ||
        this.conflicts > this.budget.conflicts ||
        (this.decisions & 1023) === 0 && Date.now() - start > this.budget.timeMs
      ) {
        return {
          sat: false,
          decisions: this.decisions,
          propagations: this.propagations,
          conflicts: this.conflicts,
          exhausted: true,
        };
      }
      const v = this.pickBranch();
      if (v === 0) {
        return {
          sat: true,
          model: Int8Array.from(this.assign),
          decisions: this.decisions,
          propagations: this.propagations,
          conflicts: this.conflicts,
          exhausted: false,
        };
      }
      this.decisions++;
      this.trailLim.push(this.trail.length);
      decisionStack.push(v);
      flipped.push(false);
      this.enqueue(v, true);
      while (!this.propagate()) {
        this.conflicts++;
        // Chronological backtracking: flip the most recent unflipped decision.
        let resolved = false;
        while (decisionStack.length > 0) {
          const idx = decisionStack.length - 1;
          const dv = decisionStack[idx];
          const wasFlipped = flipped[idx];
          this.backtrack(idx);
          if (!wasFlipped) {
            flipped[idx] = true;
            this.trailLim.push(this.trail.length);
            this.enqueue(-dv, true);
            resolved = true;
            break;
          }
          decisionStack.pop();
          flipped.pop();
        }
        if (!resolved) {
          return {
            sat: false,
            decisions: this.decisions,
            propagations: this.propagations,
            conflicts: this.conflicts,
            exhausted: false,
          };
        }
      }
    }
  }
}

/* --------------------------------------------------------------------- */
/* Equivalence checking                                                    */
/* --------------------------------------------------------------------- */

export interface VerifyOptions {
  width: number;
  inputGpr: number[];
  live: LiveOut;
  cfg: MachineConfig;
  timeMs: number;
}

function bvValue(model: Int8Array, bits: BV, trueLit: number): number {
  let v = 0;
  for (let i = 0; i < bits.length; i++) {
    const l = bits[i];
    let val: number;
    if (l === trueLit) val = 1;
    else if (l === -trueLit) val = 0;
    else {
      const a = model[Math.abs(l)];
      const positive = a === 1;
      val = l > 0 ? (positive ? 1 : 0) : positive ? 0 : 1;
    }
    if (val) v |= 1 << i;
  }
  return v >>> 0;
}

/** Exhaustive concrete enumeration at the reduced verification width. */
export function boundedEnumeration(
  target: Program,
  candidate: Program,
  opts: VerifyOptions,
): VerificationResult {
  const start = Date.now();
  const cfg: MachineConfig = { ...opts.cfg, width: opts.width };
  const inputs = opts.inputGpr.length ? opts.inputGpr : [0];
  const space = Math.pow(2, opts.width * inputs.length);
  const cap = 1 << 20;
  const exhaustive = space <= cap;
  const total = exhaustive ? space : cap;
  let checked = 0;
  for (let k = 0; k < total; k++) {
    const gpr = new Array(cfg.gprCount).fill(0);
    if (exhaustive) {
      let rest = k;
      for (const r of inputs) {
        gpr[r] = rest % (1 << opts.width);
        rest = Math.floor(rest / (1 << opts.width));
      }
    } else {
      for (const r of inputs) gpr[r] = Math.floor(Math.random() * (1 << opts.width));
    }
    const tc: TestCase = {
      id: `enum-${k}`,
      gpr,
      fpr: new Array(cfg.fprCount).fill(0),
      vreg: new Array(cfg.vregCount * 4).fill(0),
      mem: [],
      origin: "counterexample",
      weight: 8,
    };
    const a = execute(target, stateFromTest(tc, cfg), cfg);
    const b = execute(candidate, stateFromTest(tc, cfg), cfg);
    checked++;
    let mismatch = a.fault !== b.fault;
    if (!mismatch) {
      for (const r of opts.live.gpr) {
        if (toUnsigned(a.gpr[r], opts.width) !== toUnsigned(b.gpr[r], opts.width)) {
          mismatch = true;
          break;
        }
      }
    }
    if (!mismatch) {
      for (const r of opts.live.fpr) {
        if (!Object.is(a.fpr[r], b.fpr[r])) {
          mismatch = true;
          break;
        }
      }
    }
    if (mismatch) {
      return {
        status: "counterexample",
        method: "bounded-enumeration",
        width: opts.width,
        decisions: checked,
        propagations: 0,
        conflicts: 1,
        clauses: 0,
        variables: 0,
        elapsedMs: Date.now() - start,
        counterexample: makeCounterexample(
          gpr.map((v) => (v >= 1 << (opts.width - 1) ? v - (1 << opts.width) : v)),
          opts.cfg,
          "enumeration",
        ),
        detail: `mismatch at input [${inputs.map((r) => `r${r}=${gpr[r]}`).join(", ")}]`,
      };
    }
    if ((k & 4095) === 0 && Date.now() - start > opts.timeMs) {
      return {
        status: "budget-exhausted",
        method: "bounded-enumeration",
        width: opts.width,
        decisions: checked,
        propagations: 0,
        conflicts: 0,
        clauses: 0,
        variables: 0,
        elapsedMs: Date.now() - start,
        detail: `time budget exhausted after ${checked} inputs`,
      };
    }
  }
  return {
    status: exhaustive ? "equivalent" : "budget-exhausted",
    method: "bounded-enumeration",
    width: opts.width,
    decisions: checked,
    propagations: 0,
    conflicts: 0,
    clauses: 0,
    variables: 0,
    elapsedMs: Date.now() - start,
    detail: exhaustive
      ? `exhaustively checked ${checked} inputs at width ${opts.width}`
      : `sampled ${checked} of ${space} inputs at width ${opts.width}`,
  };
}

/** Full SMT-style equivalence check via bit-blasting + DPLL. */
export function verifyEquivalence(
  target: Program,
  candidate: Program,
  opts: VerifyOptions,
): VerificationResult {
  const start = Date.now();
  const usesUnsupported = [...target, ...candidate].some((i) => {
    const cls = spec(i.op).cls;
    return cls === "float" || cls === "vector" || cls === "memory" || i.op === "divu";
  });
  if (usesUnsupported || opts.live.fpr.length > 0 || opts.live.vreg.length > 0) {
    return boundedEnumeration(target, candidate, opts);
  }
  const cnf = new Cnf();
  const bv = new BvBuilder(cnf, opts.width);
  const inputs: BV[] = [];
  const decisionOrder: number[] = [];
  for (let r = 0; r < opts.cfg.gprCount; r++) {
    if (opts.inputGpr.includes(r)) {
      const v = bv.variable();
      inputs.push(v);
      for (const l of v) decisionOrder.push(Math.abs(l));
    } else {
      inputs.push(bv.constant(0));
    }
  }
  let stT: SymState;
  let stR: SymState;
  try {
    stT = symbolicExecute(target, inputs, bv);
    stR = symbolicExecute(candidate, inputs, bv);
  } catch (err) {
    if (err instanceof UnsupportedSymbolicOp) return boundedEnumeration(target, candidate, opts);
    throw err;
  }
  const diffs: number[] = [];
  for (const r of opts.live.gpr) {
    diffs.push(-bv.eq(stT.gpr[r], stR.gpr[r]));
  }
  const miter = cnf.orAll(diffs);
  cnf.add([miter]);
  const solver = new DpllSolver(cnf, decisionOrder, {
    decisions: 8_000_000,
    conflicts: 8_000_000,
    timeMs: opts.timeMs,
  });
  const res = solver.solve();
  const elapsedMs = Date.now() - start;
  if (res.exhausted) {
    return {
      status: "budget-exhausted",
      method: "bit-blast-dpll",
      width: opts.width,
      decisions: res.decisions,
      propagations: res.propagations,
      conflicts: res.conflicts,
      clauses: cnf.clauses.length,
      variables: cnf.vars,
      elapsedMs,
      detail: "solver budget exhausted; falling back to bounded enumeration is recommended",
    };
  }
  if (res.sat && res.model) {
    const model = res.model;
    const gpr = new Array(opts.cfg.gprCount).fill(0);
    for (const r of opts.inputGpr) {
      const raw = bvValue(model, inputs[r], cnf.TRUE);
      gpr[r] = raw >= 1 << (opts.width - 1) ? raw - (1 << opts.width) : raw;
    }
    return {
      status: "counterexample",
      method: "bit-blast-dpll",
      width: opts.width,
      decisions: res.decisions,
      propagations: res.propagations,
      conflicts: res.conflicts,
      clauses: cnf.clauses.length,
      variables: cnf.vars,
      elapsedMs,
      counterexample: makeCounterexample(gpr, opts.cfg, "dpll"),
      detail: `SAT: live-out differs on [${opts.inputGpr.map((r) => `r${r}=${gpr[r]}`).join(", ")}]`,
    };
  }
  return {
    status: "equivalent",
    method: "bit-blast-dpll",
    width: opts.width,
    decisions: res.decisions,
    propagations: res.propagations,
    conflicts: res.conflicts,
    clauses: cnf.clauses.length,
    variables: cnf.vars,
    elapsedMs,
    detail: `UNSAT: miter unsatisfiable over ${opts.inputGpr.length} symbolic input(s) at width ${opts.width}`,
  };
}

export const DEFAULT_VERIFY_CFG = DEFAULT_MACHINE;
