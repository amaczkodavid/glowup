//! Synthesised kernel dispatch: the bridge between the optimiser (which emits
//! virtual-core assembly and native x86-64/AArch64 text) and the runtime.
//!
//! Kernels discovered by the superoptimiser are registered here by name; the
//! runtime executes either the portable Zig reference implementation or, when
//! the synthesised machine-code blob is available, the JIT-mapped version.
//! Both paths are validated against each other by the differential tester.

const std = @import("std");
const memory = @import("memory.zig");
const tensor = @import("tensor.zig");

pub const KernelError = error{ UnknownKernel, MapFailed, VerificationFailed, OutOfMemory };

pub const KernelKind = enum { scalar_u32, scalar_pair_u32, tensor_unary, tensor_binary };

pub const ScalarFn = *const fn (u32) callconv(.c) u32;
pub const ScalarPairFn = *const fn (u32, u32) callconv(.c) u32;

pub const Kernel = struct {
    name: []const u8,
    kind: KernelKind,
    /// Portable reference implementation (always available).
    reference: union(enum) {
        scalar: *const fn (u32) u32,
        pair: *const fn (u32, u32) u32,
    },
    /// Optional synthesised machine code for the host ISA.
    code: ?[]align(std.heap.page_size_min) u8 = null,
};

/// Reference implementations of the benchmark kernels the optimiser targets.
pub fn popcount(x: u32) u32 {
    return @popCount(x);
}
pub fn absValue(x: u32) u32 {
    const s: i32 = @bitCast(x);
    return @bitCast(if (s < 0) -s else s);
}
pub fn isolateLowestSetBit(x: u32) u32 {
    return x & (~x +% 1);
}
pub fn clearLowestSetBit(x: u32) u32 {
    return x & (x -% 1);
}
pub fn maskUpToLowestSetBit(x: u32) u32 {
    return x ^ (x -% 1);
}
pub fn roundUpPow2(x: u32) u32 {
    if (x <= 1) return 1;
    return @as(u32, 1) << @intCast(32 - @clz(x - 1));
}
pub fn signum(x: u32) u32 {
    const s: i32 = @bitCast(x);
    const r: i32 = if (s > 0) 1 else if (s < 0) -1 else 0;
    return @bitCast(r);
}
pub fn signedMax(a: u32, b: u32) u32 {
    const x: i32 = @bitCast(a);
    const y: i32 = @bitCast(b);
    return @bitCast(@max(x, y));
}
pub fn unsignedAverage(a: u32, b: u32) u32 {
    return (a & b) +% ((a ^ b) >> 1);
}
pub fn multiplyByTen(x: u32) u32 {
    return x *% 10;
}

pub const REGISTRY = [_]Kernel{
    .{ .name = "popcount", .kind = .scalar_u32, .reference = .{ .scalar = popcount } },
    .{ .name = "abs", .kind = .scalar_u32, .reference = .{ .scalar = absValue } },
    .{ .name = "isolate_lsb", .kind = .scalar_u32, .reference = .{ .scalar = isolateLowestSetBit } },
    .{ .name = "clear_lsb", .kind = .scalar_u32, .reference = .{ .scalar = clearLowestSetBit } },
    .{ .name = "mask_upto_lsb", .kind = .scalar_u32, .reference = .{ .scalar = maskUpToLowestSetBit } },
    .{ .name = "round_pow2", .kind = .scalar_u32, .reference = .{ .scalar = roundUpPow2 } },
    .{ .name = "sign", .kind = .scalar_u32, .reference = .{ .scalar = signum } },
    .{ .name = "mul10", .kind = .scalar_u32, .reference = .{ .scalar = multiplyByTen } },
    .{ .name = "smax", .kind = .scalar_pair_u32, .reference = .{ .pair = signedMax } },
    .{ .name = "avg_no_overflow", .kind = .scalar_pair_u32, .reference = .{ .pair = unsignedAverage } },
};

pub fn lookup(name: []const u8) KernelError!Kernel {
    for (REGISTRY) |k| {
        if (std.mem.eql(u8, k.name, name)) return k;
    }
    return KernelError.UnknownKernel;
}

/// Map a synthesised machine-code blob as executable memory (W^X respected:
/// the pages are written first, then re-protected read+execute).
pub fn mapExecutable(allocator: std.mem.Allocator, code: []const u8) KernelError![]align(std.heap.page_size_min) u8 {
    const page = std.heap.page_size_min;
    const size = memory.alignUp(code.len, page);
    const buf = allocator.alignedAlloc(u8, .fromByteUnits(page), size) catch return KernelError.OutOfMemory;
    @memcpy(buf[0..code.len], code);
    @memset(buf[code.len..], 0xcc); // int3 padding
    std.posix.mprotect(buf[0..size], std.posix.PROT.READ | std.posix.PROT.EXEC) catch {
        allocator.free(buf);
        return KernelError.MapFailed;
    };
    return buf;
}

/// Differential test between the reference kernel and a synthesised variant.
pub fn differentialTest(kernel: Kernel, candidate: ScalarFn, samples: usize, seed: u64) KernelError!void {
    var prng = std.Random.DefaultPrng.init(seed);
    const rand = prng.random();
    var i: usize = 0;
    while (i < samples) : (i += 1) {
        const x: u32 = switch (i % 4) {
            0 => 0,
            1 => 0xffff_ffff,
            2 => @as(u32, 1) << @intCast(i % 32),
            else => rand.int(u32),
        };
        const expect = switch (kernel.reference) {
            .scalar => |f| f(x),
            .pair => |f| f(x, x),
        };
        if (candidate(x) != expect) return KernelError.VerificationFailed;
    }
}

/// Apply a scalar kernel across a tensor with SIMD-friendly unrolling.
pub fn mapScalar(out: *tensor.Tensor, in: *const tensor.Tensor, f: *const fn (u32) u32) !void {
    const dst = try out.mutableData();
    const src = in.constData();
    if (dst.len != src.len) return tensor.TensorError.ShapeMismatch;
    var i: usize = 0;
    while (i + 4 <= dst.len) : (i += 4) {
        dst[i + 0] = @bitCast(f(@bitCast(src[i + 0])));
        dst[i + 1] = @bitCast(f(@bitCast(src[i + 1])));
        dst[i + 2] = @bitCast(f(@bitCast(src[i + 2])));
        dst[i + 3] = @bitCast(f(@bitCast(src[i + 3])));
    }
    while (i < dst.len) : (i += 1) dst[i] = @bitCast(f(@bitCast(src[i])));
}

test "kernel registry reference semantics" {
    try std.testing.expectEqual(@as(u32, 3), popcount(0b1011 & 0b0111));
    try std.testing.expectEqual(@as(u32, 8), isolateLowestSetBit(24));
    try std.testing.expectEqual(@as(u32, 16), clearLowestSetBit(24));
    try std.testing.expectEqual(@as(u32, 15), maskUpToLowestSetBit(8));
    try std.testing.expectEqual(@as(u32, 64), roundUpPow2(33));
    try std.testing.expectEqual(@as(u32, 5), unsignedAverage(4, 6));
    const k = try lookup("abs");
    try std.testing.expectEqual(KernelKind.scalar_u32, k.kind);
}
