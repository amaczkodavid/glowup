/**
 * Core IR types shared by every subsystem of the synthesis stack.
 *
 * The internal assembly model is a three-address, loop-free instruction
 * sequence over a retargetable virtual core. Concrete ISAs (x86-64, AArch64)
 * are projections of this core; see `isa.ts`.
 */

export type OperandRole = "rd" | "rs1" | "rs2" | "rs3" | "imm" | "vd" | "vs1" | "vs2" | "vs3";

export type OpClass =
  | "nop"
  | "move"
  | "arith"
  | "logic"
  | "shift"
  | "bit"
  | "compare"
  | "select"
  | "memory"
  | "float"
  | "vector";

export interface Instruction {
  /** Opcode mnemonic in the virtual core ISA. */
  op: string;
  /** Destination general purpose / float register index (-1 when unused). */
  rd: number;
  /** Source register indices (-1 when unused). */
  rs1: number;
  rs2: number;
  rs3: number;
  /** Immediate operand (signed 32-bit). */
  imm: number;
}

export type Program = Instruction[];

export type FaultKind = "none" | "segfault" | "unaligned" | "divide-by-zero" | "trap" | "timeout";

export interface MachineConfig {
  /** Bit width of the integer datapath (8 for bounded verification, 32 for execution). */
  width: number;
  /** Number of architectural general purpose registers. */
  gprCount: number;
  /** Number of scalar floating point registers. */
  fprCount: number;
  /** Number of 4-lane vector registers. */
  vregCount: number;
  /** Bytes of addressable memory. */
  memBytes: number;
  /** Required alignment (bytes) for word memory accesses. */
  alignment: number;
  /** Hard instruction-count fuel (prevents runaway evaluation). */
  fuel: number;
}

export const DEFAULT_MACHINE: MachineConfig = {
  width: 32,
  gprCount: 8,
  fprCount: 8,
  vregCount: 4,
  memBytes: 256,
  alignment: 4,
  fuel: 4096,
};

export interface CpuState {
  gpr: Uint32Array;
  fpr: Float64Array;
  vreg: Float64Array; // vregCount * 4 lanes
  mem: Uint8Array;
  flags: { zf: boolean; sf: boolean; cf: boolean; of: boolean };
  fault: FaultKind;
  faultDetail: string;
  retired: number;
}

export interface LiveOut {
  gpr: number[];
  fpr: number[];
  vreg: number[];
  mem: Array<{ addr: number; bytes: number }>;
}

export interface TestCase {
  id: string;
  /** Initial general purpose register file. */
  gpr: number[];
  fpr: number[];
  vreg: number[];
  /** Sparse memory image: address -> byte. */
  mem: Array<[number, number]>;
  origin: "random" | "adversarial" | "counterexample" | "seed";
  /** Prioritisation weight, higher = evaluated first / more informative. */
  weight: number;
}

export interface CostWeights {
  /** Correctness weight (we). */
  we: number;
  /** Performance weight (wp). */
  wp: number;
  /** Penalty applied per runtime fault. */
  faultPenalty: number;
  /** Penalty per unmatched live-out bit. */
  bitPenalty: number;
  /** Multiplier applied to aggregated ULP error. */
  ulpPenalty: number;
  /** Weight for max (as opposed to sum) ULP error. */
  ulpMaxPenalty: number;
  /** Static code-size weight inside perf(). */
  sizeWeight: number;
}

export const DEFAULT_WEIGHTS: CostWeights = {
  we: 1.0,
  wp: 1.0,
  faultPenalty: 400,
  bitPenalty: 1.0,
  ulpPenalty: 0.25,
  ulpMaxPenalty: 1.0,
  sizeWeight: 0.35,
};

export interface CostBreakdown {
  total: number;
  eq: number;
  perf: number;
  bitErrors: number;
  ulpSum: number;
  ulpMax: number;
  faults: number;
  correct: boolean;
}

export interface Candidate {
  program: Program;
  cost: CostBreakdown;
  source: "mcmc" | "mcts" | "enumerative" | "symbolic" | "eqsat" | "target";
  discoveredAtMs: number;
  verified: boolean;
  verdict?: VerificationResult;
}

export interface VerificationResult {
  status: "equivalent" | "counterexample" | "budget-exhausted";
  method: "bit-blast-dpll" | "bounded-enumeration" | "trivial";
  width: number;
  decisions: number;
  propagations: number;
  conflicts: number;
  clauses: number;
  variables: number;
  elapsedMs: number;
  counterexample?: TestCase;
  detail: string;
}

export interface SearchEvent {
  t: number;
  engine: string;
  kind: string;
  message: string;
  cost?: number;
}
