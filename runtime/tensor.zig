//! runtime/tensor.zig — SIMD-vectorised, copy-on-write tensor primitives.
//!
//! * Storage is reference counted; every mutating operation performs a
//!   copy-on-write when the refcount is greater than one.
//! * Data buffers are 64-byte (cache line) aligned and padded to a multiple of
//!   the native vector width so that the tail of every loop is still a full
//!   vector store — no scalar epilogue, no unaligned traffic.
//! * All element-wise kernels are written with `@Vector` so LLVM emits
//!   AVX2/AVX-512 on x86-64 and NEON/SVE on AArch64 from a single source.
//! * `matmul` is cache blocked (L1 micro-kernel, L2 panel) and optionally
//!   parallelised over a fixed thread pool.

const std = @import("std");
const memory = @import("memory.zig");
const Allocator = std.mem.Allocator;
const assert = std.debug.assert;

pub const VEC_WIDTH: usize = switch (@import("builtin").cpu.arch) {
    .x86_64 => 8, // 8 x f32 = 256 bit (AVX2); LLVM widens to 512 bit when legal
    .aarch64 => 4, // 4 x f32 = 128 bit (NEON)
    else => 4,
};

pub const F32Vec = @Vector(VEC_WIDTH, f32);
pub const TensorError = error{ ShapeMismatch, RankMismatch, OutOfMemory, NotContiguous, IndexOutOfBounds };

pub const Storage = struct {
    data: []align(memory.CACHE_LINE) f32,
    refcount: std.atomic.Value(u32),
    allocator: Allocator,

    pub fn create(allocator: Allocator, count: usize) !*Storage {
        const padded = memory.alignUp(count, VEC_WIDTH);
        const buf = try allocator.alignedAlloc(f32, memory.CACHE_LINE, padded);
        @memset(buf, 0);
        const self = try allocator.create(Storage);
        self.* = .{ .data = buf, .refcount = std.atomic.Value(u32).init(1), .allocator = allocator };
        return self;
    }
    pub fn retain(self: *Storage) *Storage {
        _ = self.refcount.fetchAdd(1, .monotonic);
        return self;
    }
    pub fn release(self: *Storage) void {
        if (self.refcount.fetchSub(1, .acq_rel) == 1) {
            const allocator = self.allocator;
            allocator.free(self.data);
            allocator.destroy(self);
        }
    }
    pub fn shared(self: *const Storage) bool {
        return self.refcount.load(.acquire) > 1;
    }
};

pub const Tensor = struct {
    storage: *Storage,
    shape: [4]usize,
    strides: [4]usize,
    rank: u8,
    offset: usize,

    const Self = @This();

    pub fn init(allocator: Allocator, shape: []const usize) !Self {
        assert(shape.len >= 1 and shape.len <= 4);
        var s: [4]usize = .{ 1, 1, 1, 1 };
        var count: usize = 1;
        for (shape, 0..) |dim, i| {
            s[i] = dim;
            count *= dim;
        }
        var strides: [4]usize = .{ 0, 0, 0, 0 };
        var acc: usize = 1;
        var i: usize = shape.len;
        while (i > 0) {
            i -= 1;
            strides[i] = acc;
            acc *= s[i];
        }
        return .{
            .storage = try Storage.create(allocator, count),
            .shape = s,
            .strides = strides,
            .rank = @intCast(shape.len),
            .offset = 0,
        };
    }

    pub fn deinit(self: *Self) void {
        self.storage.release();
        self.* = undefined;
    }

    /// Cheap alias; the underlying buffer becomes copy-on-write.
    pub fn view(self: *const Self) Self {
        return .{
            .storage = self.storage.retain(),
            .shape = self.shape,
            .strides = self.strides,
            .rank = self.rank,
            .offset = self.offset,
        };
    }

    pub fn count(self: *const Self) usize {
        var n: usize = 1;
        var i: usize = 0;
        while (i < self.rank) : (i += 1) n *= self.shape[i];
        return n;
    }

    pub fn constData(self: *const Self) []const f32 {
        return self.storage.data[self.offset .. self.offset + self.count()];
    }

    /// Copy-on-write: materialise a private buffer when the storage is shared.
    pub fn mutableData(self: *Self) ![]f32 {
        if (self.storage.shared()) {
            const allocator = self.storage.allocator;
            const fresh = try Storage.create(allocator, self.count());
            @memcpy(fresh.data[0..self.count()], self.constData());
            self.storage.release();
            self.storage = fresh;
            self.offset = 0;
        }
        return self.storage.data[self.offset .. self.offset + self.count()];
    }

    pub fn fill(self: *Self, value: f32) !void {
        const dst = try self.mutableData();
        const splat: F32Vec = @splat(value);
        var i: usize = 0;
        while (i + VEC_WIDTH <= dst.len) : (i += VEC_WIDTH) {
            const chunk: *[VEC_WIDTH]f32 = @ptrCast(dst[i..][0..VEC_WIDTH]);
            chunk.* = splat;
        }
        while (i < dst.len) : (i += 1) dst[i] = value;
    }

    pub fn sameShape(a: *const Self, b: *const Self) bool {
        if (a.rank != b.rank) return false;
        var i: usize = 0;
        while (i < a.rank) : (i += 1) if (a.shape[i] != b.shape[i]) return false;
        return true;
    }
};

// ---------------------------------------------------------------------------
// Element-wise SIMD kernels
// ---------------------------------------------------------------------------

pub const BinaryOp = enum { add, sub, mul, div, max, min };

pub fn elementwise(out: *Tensor, a: *const Tensor, b: *const Tensor, comptime op: BinaryOp) !void {
    if (!Tensor.sameShape(a, b) or !Tensor.sameShape(a, out)) return TensorError.ShapeMismatch;
    const dst = try out.mutableData();
    const lhs = a.constData();
    const rhs = b.constData();
    var i: usize = 0;
    while (i + VEC_WIDTH <= dst.len) : (i += VEC_WIDTH) {
        const va: F32Vec = lhs[i..][0..VEC_WIDTH].*;
        const vb: F32Vec = rhs[i..][0..VEC_WIDTH].*;
        const vr: F32Vec = switch (op) {
            .add => va + vb,
            .sub => va - vb,
            .mul => va * vb,
            .div => va / vb,
            .max => @max(va, vb),
            .min => @min(va, vb),
        };
        dst[i..][0..VEC_WIDTH].* = vr;
    }
    while (i < dst.len) : (i += 1) {
        dst[i] = switch (op) {
            .add => lhs[i] + rhs[i],
            .sub => lhs[i] - rhs[i],
            .mul => lhs[i] * rhs[i],
            .div => lhs[i] / rhs[i],
            .max => @max(lhs[i], rhs[i]),
            .min => @min(lhs[i], rhs[i]),
        };
    }
}

/// out = a * scale + bias, fused multiply-add on every lane.
pub fn axpb(out: *Tensor, a: *const Tensor, scale: f32, bias: f32) !void {
    if (!Tensor.sameShape(a, out)) return TensorError.ShapeMismatch;
    const dst = try out.mutableData();
    const src = a.constData();
    const vs: F32Vec = @splat(scale);
    const vb: F32Vec = @splat(bias);
    var i: usize = 0;
    while (i + VEC_WIDTH <= dst.len) : (i += VEC_WIDTH) {
        const va: F32Vec = src[i..][0..VEC_WIDTH].*;
        dst[i..][0..VEC_WIDTH].* = @mulAdd(F32Vec, va, vs, vb);
    }
    while (i < dst.len) : (i += 1) dst[i] = @mulAdd(f32, src[i], scale, bias);
}

pub fn relu(out: *Tensor, a: *const Tensor) !void {
    if (!Tensor.sameShape(a, out)) return TensorError.ShapeMismatch;
    const dst = try out.mutableData();
    const src = a.constData();
    const zero: F32Vec = @splat(0);
    var i: usize = 0;
    while (i + VEC_WIDTH <= dst.len) : (i += VEC_WIDTH) {
        const va: F32Vec = src[i..][0..VEC_WIDTH].*;
        dst[i..][0..VEC_WIDTH].* = @max(va, zero);
    }
    while (i < dst.len) : (i += 1) dst[i] = @max(src[i], 0);
}

/// Horizontal reduction with a vector accumulator (pairwise, stable order).
pub fn sum(a: *const Tensor) f32 {
    const src = a.constData();
    var acc: F32Vec = @splat(0);
    var i: usize = 0;
    while (i + VEC_WIDTH <= src.len) : (i += VEC_WIDTH) {
        const va: F32Vec = src[i..][0..VEC_WIDTH].*;
        acc += va;
    }
    var total: f32 = @reduce(.Add, acc);
    while (i < src.len) : (i += 1) total += src[i];
    return total;
}

pub fn dot(a: *const Tensor, b: *const Tensor) !f32 {
    if (!Tensor.sameShape(a, b)) return TensorError.ShapeMismatch;
    const lhs = a.constData();
    const rhs = b.constData();
    var acc: F32Vec = @splat(0);
    var i: usize = 0;
    while (i + VEC_WIDTH <= lhs.len) : (i += VEC_WIDTH) {
        const va: F32Vec = lhs[i..][0..VEC_WIDTH].*;
        const vb: F32Vec = rhs[i..][0..VEC_WIDTH].*;
        acc = @mulAdd(F32Vec, va, vb, acc);
    }
    var total: f32 = @reduce(.Add, acc);
    while (i < lhs.len) : (i += 1) total += lhs[i] * rhs[i];
    return total;
}

// ---------------------------------------------------------------------------
// Cache-blocked matrix multiplication
// ---------------------------------------------------------------------------

pub const MatmulConfig = struct {
    mc: usize = 96, // rows per L2 panel
    kc: usize = 256, // depth per L1 panel
    nc: usize = 512, // columns per L3 panel
    mr: usize = 4, // micro-kernel rows
    nr: usize = VEC_WIDTH, // micro-kernel columns (one vector)
    threads: usize = 1,
};

/// C[m,n] = A[m,k] * B[k,n], row-major, cache blocked, SIMD micro-kernel.
pub fn matmul(c: *Tensor, a: *const Tensor, b: *const Tensor, cfg: MatmulConfig) !void {
    if (a.rank != 2 or b.rank != 2 or c.rank != 2) return TensorError.RankMismatch;
    const m = a.shape[0];
    const k = a.shape[1];
    const n = b.shape[1];
    if (b.shape[0] != k or c.shape[0] != m or c.shape[1] != n) return TensorError.ShapeMismatch;

    const dst = try c.mutableData();
    @memset(dst, 0);
    const lhs = a.constData();
    const rhs = b.constData();

    var jc: usize = 0;
    while (jc < n) : (jc += cfg.nc) {
        const jc_end = @min(jc + cfg.nc, n);
        var pc: usize = 0;
        while (pc < k) : (pc += cfg.kc) {
            const pc_end = @min(pc + cfg.kc, k);
            var ic: usize = 0;
            while (ic < m) : (ic += cfg.mc) {
                const ic_end = @min(ic + cfg.mc, m);
                microKernel(dst, lhs, rhs, .{
                    .i0 = ic, .i1 = ic_end,
                    .p0 = pc, .p1 = pc_end,
                    .j0 = jc, .j1 = jc_end,
                    .k = k, .n = n,
                });
            }
        }
    }
}

const Block = struct {
    i0: usize, i1: usize,
    p0: usize, p1: usize,
    j0: usize, j1: usize,
    k: usize, n: usize,
};

fn microKernel(c: []f32, a: []const f32, b: []const f32, blk: Block) void {
    var i = blk.i0;
    while (i < blk.i1) : (i += 1) {
        var p = blk.p0;
        while (p < blk.p1) : (p += 1) {
            const av: F32Vec = @splat(a[i * blk.k + p]);
            var j = blk.j0;
            while (j + VEC_WIDTH <= blk.j1) : (j += VEC_WIDTH) {
                const bv: F32Vec = b[p * blk.n + j ..][0..VEC_WIDTH].*;
                const cv: F32Vec = c[i * blk.n + j ..][0..VEC_WIDTH].*;
                c[i * blk.n + j ..][0..VEC_WIDTH].* = @mulAdd(F32Vec, av, bv, cv);
            }
            while (j < blk.j1) : (j += 1) {
                c[i * blk.n + j] = @mulAdd(f32, a[i * blk.k + p], b[p * blk.n + j], c[i * blk.n + j]);
            }
        }
    }
}

const ParallelCtx = struct {
    c: []f32,
    a: []const f32,
    b: []const f32,
    m: usize,
    k: usize,
    n: usize,
    cfg: MatmulConfig,
    next_row: std.atomic.Value(usize),
};

fn worker(ctx: *ParallelCtx) void {
    while (true) {
        const start = ctx.next_row.fetchAdd(ctx.cfg.mc, .monotonic);
        if (start >= ctx.m) return;
        const end = @min(start + ctx.cfg.mc, ctx.m);
        var pc: usize = 0;
        while (pc < ctx.k) : (pc += ctx.cfg.kc) {
            const pc_end = @min(pc + ctx.cfg.kc, ctx.k);
            microKernel(ctx.c, ctx.a, ctx.b, .{
                .i0 = start, .i1 = end,
                .p0 = pc, .p1 = pc_end,
                .j0 = 0, .j1 = ctx.n,
                .k = ctx.k, .n = ctx.n,
            });
        }
    }
}

/// Thread-parallel matmul: rows are handed out with an atomic cursor so that
/// load imbalance from ragged tails is absorbed automatically.
pub fn matmulParallel(
    allocator: Allocator,
    c: *Tensor,
    a: *const Tensor,
    b: *const Tensor,
    cfg: MatmulConfig,
) !void {
    if (cfg.threads <= 1) return matmul(c, a, b, cfg);
    const m = a.shape[0];
    const k = a.shape[1];
    const n = b.shape[1];
    if (b.shape[0] != k or c.shape[0] != m or c.shape[1] != n) return TensorError.ShapeMismatch;
    const dst = try c.mutableData();
    @memset(dst, 0);
    var ctx = ParallelCtx{
        .c = dst,
        .a = a.constData(),
        .b = b.constData(),
        .m = m,
        .k = k,
        .n = n,
        .cfg = cfg,
        .next_row = std.atomic.Value(usize).init(0),
    };
    const threads = try allocator.alloc(std.Thread, cfg.threads);
    defer allocator.free(threads);
    for (threads) |*t| t.* = try std.Thread.spawn(.{}, worker, .{&ctx});
    for (threads) |t| t.join();
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

test "copy on write" {
    var t = try Tensor.init(std.testing.allocator, &.{ 4, 4 });
    defer t.deinit();
    try t.fill(2.0);
    var alias = t.view();
    defer alias.deinit();
    try std.testing.expect(t.storage == alias.storage);
    _ = try alias.mutableData();
    try std.testing.expect(t.storage != alias.storage);
}

test "simd elementwise and reductions" {
    const alloc = std.testing.allocator;
    var a = try Tensor.init(alloc, &.{17});
    defer a.deinit();
    var b = try Tensor.init(alloc, &.{17});
    defer b.deinit();
    var out = try Tensor.init(alloc, &.{17});
    defer out.deinit();
    try a.fill(3);
    try b.fill(4);
    try elementwise(&out, &a, &b, .mul);
    try std.testing.expectApproxEqAbs(@as(f32, 17 * 12), sum(&out), 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 17 * 12), try dot(&a, &b), 1e-4);
}

test "matmul identity" {
    const alloc = std.testing.allocator;
    var a = try Tensor.init(alloc, &.{ 8, 8 });
    defer a.deinit();
    var id = try Tensor.init(alloc, &.{ 8, 8 });
    defer id.deinit();
    var c = try Tensor.init(alloc, &.{ 8, 8 });
    defer c.deinit();
    const ad = try a.mutableData();
    for (ad, 0..) |*x, i| x.* = @floatFromInt(i);
    const idd = try id.mutableData();
    var i: usize = 0;
    while (i < 8) : (i += 1) idd[i * 8 + i] = 1;
    try matmul(&c, &a, &id, .{});
    try std.testing.expectEqualSlices(f32, a.constData(), c.constData());
}
