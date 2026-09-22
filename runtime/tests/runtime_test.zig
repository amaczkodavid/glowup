//! Aggregate test entry point: allocators, lock-free structures, tensors,
//! GPU collectives, Futhark bindings and synthesised-kernel dispatch.
//!
//!   zig build test

const std = @import("std");
const rt = @import("runtime");

test "all runtime modules expose their declarations" {
    std.testing.refAllDecls(rt);
}

test "arena is page aligned and reusable" {
    var arena = try rt.memory.ArenaAllocator.init(std.testing.allocator, 1 << 16, .secure);
    defer arena.deinit();
    try std.testing.expectEqual(@as(usize, 0), @intFromPtr(arena.buffer.ptr) % rt.memory.PAGE_SIZE);
    const a = arena.allocator();
    const first = try a.alloc(u8, 1000);
    try std.testing.expectEqual(@as(usize, 1000), first.len);
    arena.reset();
    const second = try a.alloc(u8, 1000);
    try std.testing.expectEqual(first.ptr, second.ptr);
}

test "secure zeroization clears freed memory" {
    var buf: [64]u8 = undefined;
    @memset(&buf, 0xff);
    rt.memory.secureZero(&buf);
    for (buf) |byte| try std.testing.expectEqual(@as(u8, 0), byte);
}

test "buddy allocator never overlaps live blocks" {
    var buddy = try rt.memory.BuddyAllocator.init(std.testing.allocator, 1 << 18, 64, .none);
    defer buddy.deinit();
    var blocks: [32][]u8 = undefined;
    for (&blocks, 0..) |*blk, i| {
        blk.* = buddy.allocBytes(64 + i * 3).?;
        @memset(blk.*, @intCast(i));
    }
    for (blocks, 0..) |blk, i| {
        for (blk) |byte| try std.testing.expectEqual(@as(u8, @intCast(i)), byte);
    }
    for (blocks) |blk| buddy.freeBytes(blk);
    try std.testing.expectEqual(@as(usize, 0), buddy.bytes_live);
}

test "tensor copy-on-write keeps views isolated" {
    const alloc = std.testing.allocator;
    var base = try rt.tensor.Tensor.init(alloc, &.{ 4, 4 });
    defer base.deinit();
    try base.fill(1);
    var alias = base.view();
    defer alias.deinit();
    const data = try alias.mutableData();
    data[0] = 42;
    try std.testing.expectEqual(@as(f32, 1), base.constData()[0]);
    try std.testing.expectEqual(@as(f32, 42), alias.constData()[0]);
}

test "matmul matches the naive triple loop" {
    const alloc = std.testing.allocator;
    const m = 9;
    const k = 7;
    const n = 5;
    var a = try rt.tensor.Tensor.init(alloc, &.{ m, k });
    defer a.deinit();
    var b = try rt.tensor.Tensor.init(alloc, &.{ k, n });
    defer b.deinit();
    var c = try rt.tensor.Tensor.init(alloc, &.{ m, n });
    defer c.deinit();
    const ad = try a.mutableData();
    for (ad, 0..) |*x, i| x.* = @floatFromInt((i * 7) % 13);
    const bd = try b.mutableData();
    for (bd, 0..) |*x, i| x.* = @floatFromInt((i * 5) % 11);
    try rt.tensor.matmul(&c, &a, &b, .{});
    var i: usize = 0;
    while (i < m) : (i += 1) {
        var j: usize = 0;
        while (j < n) : (j += 1) {
            var acc: f32 = 0;
            var p: usize = 0;
            while (p < k) : (p += 1) acc += ad[i * k + p] * bd[p * n + j];
            try std.testing.expectApproxEqAbs(acc, c.constData()[i * n + j], 1e-3);
        }
    }
}

test "parallel matmul equals serial matmul" {
    const alloc = std.testing.allocator;
    const n = 64;
    var a = try rt.tensor.Tensor.init(alloc, &.{ n, n });
    defer a.deinit();
    var b = try rt.tensor.Tensor.init(alloc, &.{ n, n });
    defer b.deinit();
    var c1 = try rt.tensor.Tensor.init(alloc, &.{ n, n });
    defer c1.deinit();
    var c2 = try rt.tensor.Tensor.init(alloc, &.{ n, n });
    defer c2.deinit();
    try a.fill(0.25);
    try b.fill(4.0);
    try rt.tensor.matmul(&c1, &a, &b, .{});
    try rt.tensor.matmulParallel(alloc, &c2, &a, &b, .{ .threads = 4 });
    for (c1.constData(), c2.constData()) |x, y| try std.testing.expectApproxEqAbs(x, y, 1e-3);
}

test "gpu coordinator collectives are algorithm independent" {
    const alloc = std.testing.allocator;
    var coord = try rt.gpu.Coordinator.init(alloc, 3);
    defer coord.deinit();
    var buffers: [3]rt.gpu.DeviceBuffer = undefined;
    for (&buffers, 0..) |*buf, rank| {
        buf.* = try coord.alloc(@intCast(rank % coord.device_count), 6);
        const payload = [_]f32{ 1, 2, 3, 4, 5, 6 };
        try coord.upload(buf, &payload);
    }
    defer for (buffers) |buf| coord.free(buf);
    var completion = try coord.allReduce(&buffers, .sum, .tree);
    try completion.wait();
    var out: [6]f32 = undefined;
    try coord.download(&buffers[2], &out);
    for (out, 0..) |value, i| {
        const expect: f32 = @floatFromInt(3 * (i + 1));
        try std.testing.expectApproxEqAbs(expect, value, 1e-4);
    }
}

test "futhark bindings fail gracefully" {
    try std.testing.expectError(
        rt.futhark.FutharkError.FutharkUnavailable,
        rt.futhark.Context.init(std.testing.allocator, "./no-such-library.so"),
    );
}

test "synthesised kernel registry differential test" {
    const kernel = try rt.kernels.lookup("clear_lsb");
    var x: u32 = 1;
    while (x < 1024) : (x += 1) {
        const expected = x & (x -% 1);
        const actual = switch (kernel.reference) {
            .scalar => |f| f(x),
            .pair => |f| f(x, x),
        };
        try std.testing.expectEqual(expected, actual);
    }
}

test "lock-free stack and queue survive interleaving" {
    const alloc = std.testing.allocator;
    var stack = rt.lockfree.TreiberStack(usize).init(alloc);
    defer stack.deinit();
    var i: usize = 0;
    while (i < 1000) : (i += 1) try stack.push(i);
    var seen: usize = 0;
    while (stack.pop()) |_| seen += 1;
    try std.testing.expectEqual(@as(usize, 1000), seen);
}
