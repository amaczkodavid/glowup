/**
 * Test-case generation, ULP metrics, and the dynamic (growing) test database.
 *
 * The database is seeded with random and adversarial inputs and grows with
 * counterexamples produced by the SMT layer. It supports deduplication (by a
 * canonical fingerprint) and prioritisation (counterexamples first, then
 * adversarial, then random), and is persisted through the Postgres corpus.
 */

import type { MachineConfig, TestCase } from "./types";
import { DEFAULT_MACHINE } from "./types";
import { toUnsigned } from "./machine";

/** Deterministic 32-bit PRNG (mulberry32). */
export class Rng {
  private s: number;
  constructor(seed: number) {
    this.s = seed >>> 0 || 0x9e3779b9;
  }
  next(): number {
    this.s = (this.s + 0x6d2b79f5) >>> 0;
    let t = this.s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  }
  int(maxExclusive: number): number {
    return Math.floor(this.next() * maxExclusive);
  }
  pick<T>(items: readonly T[]): T {
    return items[this.int(items.length)];
  }
  bool(p = 0.5): boolean {
    return this.next() < p;
  }
  u32(): number {
    return Math.floor(this.next() * 4294967296) >>> 0;
  }
  gaussian(): number {
    const u = Math.max(this.next(), 1e-12);
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * this.next());
  }
}

const ADVERSARIAL_INTS = [
  0, 1, -1, 2, -2, 3, 7, 8, 15, 16, 31, 32, 63, 64, 127, 128, 255, 256, 65535, 65536,
  0x7fffffff, -0x80000000, 0x55555555, -0x55555556, 0x0f0f0f0f, -0x0f0f0f10,
];

const ADVERSARIAL_FLOATS = [
  0, -0, 1, -1, 0.5, -0.5, 2, 3.14159265358979, 1e-300, 1e300, Number.MIN_VALUE,
  Number.EPSILON, Number.MAX_SAFE_INTEGER, Infinity, -Infinity, NaN,
];

let counter = 0;
function nextId(prefix: string): string {
  counter += 1;
  return `${prefix}-${counter.toString(36)}-${Math.floor(Math.random() * 1e6).toString(36)}`;
}

export function randomTest(rng: Rng, cfg: MachineConfig = DEFAULT_MACHINE): TestCase {
  const gpr: number[] = [];
  for (let i = 0; i < cfg.gprCount; i++) {
    const roll = rng.next();
    if (roll < 0.25) gpr.push(rng.pick(ADVERSARIAL_INTS) | 0);
    else if (roll < 0.5) gpr.push((rng.u32() & 0xff) | 0);
    else gpr.push(rng.u32() | 0);
  }
  const fpr: number[] = [];
  for (let i = 0; i < cfg.fprCount; i++) {
    const roll = rng.next();
    if (roll < 0.2) fpr.push(rng.pick(ADVERSARIAL_FLOATS));
    else fpr.push(rng.gaussian() * Math.pow(10, rng.int(6) - 3));
  }
  const vreg: number[] = [];
  for (let i = 0; i < cfg.vregCount * 4; i++) vreg.push(rng.gaussian());
  const mem: Array<[number, number]> = [];
  for (let addr = 0; addr < Math.min(cfg.memBytes, 64); addr++) mem.push([addr, rng.int(256)]);
  return { id: nextId("rnd"), gpr, fpr, vreg, mem, origin: "random", weight: 1 };
}

export function adversarialTests(cfg: MachineConfig = DEFAULT_MACHINE): TestCase[] {
  const out: TestCase[] = [];
  for (const value of ADVERSARIAL_INTS) {
    const gpr = new Array(cfg.gprCount).fill(0).map((_, i) => (i === 0 ? value : i === 1 ? ~value : value));
    const fpr = new Array(cfg.fprCount).fill(0).map((_, i) => ADVERSARIAL_FLOATS[(i + value) % ADVERSARIAL_FLOATS.length]);
    out.push({
      id: nextId("adv"),
      gpr,
      fpr,
      vreg: new Array(cfg.vregCount * 4).fill(1),
      mem: Array.from({ length: Math.min(cfg.memBytes, 64) }, (_, i): [number, number] => [i, (value + i) & 0xff]),
      origin: "adversarial",
      weight: 3,
    });
  }
  return out;
}

export function fingerprint(tc: TestCase): string {
  return `${tc.gpr.join(",")}|${tc.fpr.map((x) => (Number.isNaN(x) ? "nan" : x)).join(",")}|${tc.mem
    .slice(0, 16)
    .map(([a, b]) => `${a}:${b}`)
    .join(",")}`;
}

/** Dynamic test database with dedup + priority ordering. */
export class TestDatabase {
  private byFingerprint = new Map<string, TestCase>();
  constructor(initial: TestCase[] = []) {
    for (const t of initial) this.add(t);
  }
  add(tc: TestCase): boolean {
    const fp = fingerprint(tc);
    if (this.byFingerprint.has(fp)) return false;
    this.byFingerprint.set(fp, tc);
    return true;
  }
  addAll(tcs: TestCase[]): number {
    let n = 0;
    for (const t of tcs) if (this.add(t)) n++;
    return n;
  }
  get size(): number {
    return this.byFingerprint.size;
  }
  /** Highest-priority-first list (counterexamples > adversarial > random). */
  all(): TestCase[] {
    const rank = (t: TestCase) =>
      t.origin === "counterexample" ? 0 : t.origin === "seed" ? 1 : t.origin === "adversarial" ? 2 : 3;
    return [...this.byFingerprint.values()].sort((a, b) => rank(a) - rank(b) || b.weight - a.weight);
  }
  sample(rng: Rng, k: number): TestCase[] {
    const all = this.all();
    if (all.length <= k) return all;
    // Always keep the highest priority prefix, sample the rest.
    const head = all.slice(0, Math.min(k >> 1, all.length));
    const rest = all.slice(head.length);
    const chosen: TestCase[] = [...head];
    while (chosen.length < k && rest.length > 0) {
      chosen.push(rest.splice(rng.int(rest.length), 1)[0]);
    }
    return chosen;
  }
}

export function makeCounterexample(
  gpr: number[],
  cfg: MachineConfig = DEFAULT_MACHINE,
  detail = "smt",
): TestCase {
  return {
    id: nextId("cex"),
    gpr: Array.from({ length: cfg.gprCount }, (_, i) => gpr[i] ?? 0),
    fpr: new Array(cfg.fprCount).fill(0),
    vreg: new Array(cfg.vregCount * 4).fill(0),
    mem: Array.from({ length: Math.min(cfg.memBytes, 64) }, (_, i): [number, number] => [i, (i * 37) & 0xff]),
    origin: "counterexample",
    weight: 10 + detail.length / 100,
  };
}

/* --------------------------------------------------------------------- */
/* Numeric comparison metrics                                              */
/* --------------------------------------------------------------------- */

const cmpBuf = new ArrayBuffer(8);
const cmpF64 = new Float64Array(cmpBuf);
const cmpU32 = new Uint32Array(cmpBuf);

function orderedParts(x: number): { hi: number; lo: number } {
  cmpF64[0] = x;
  const lo = cmpU32[0] >>> 0;
  const hi = cmpU32[1] >>> 0;
  if (hi & 0x80000000) {
    // Negative: mirror into a monotonically decreasing signed space.
    return { hi: -(hi & 0x7fffffff), lo: -lo };
  }
  return { hi, lo };
}

/** Unit-in-the-last-place distance between two doubles (saturating). */
export function ulpDistance(a: number, b: number): number {
  if (Number.isNaN(a) && Number.isNaN(b)) return 0;
  if (Number.isNaN(a) || Number.isNaN(b)) return 1e12;
  if (a === b) return 0;
  if (!Number.isFinite(a) || !Number.isFinite(b)) return 1e12;
  const pa = orderedParts(a);
  const pb = orderedParts(b);
  const diff = Math.abs((pa.hi - pb.hi) * 4294967296 + (pa.lo - pb.lo));
  return Math.min(diff, 1e12);
}

/** Number of differing bits between two width-bit integers. */
export function bitDistance(a: number, b: number, width = 32): number {
  let x = (toUnsigned(a, width) ^ toUnsigned(b, width)) >>> 0;
  let n = 0;
  while (x) {
    n += x & 1;
    x >>>= 1;
  }
  return n;
}
