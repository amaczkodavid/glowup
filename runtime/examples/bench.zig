//! Micro-benchmarks for the runtime: allocator throughput, SIMD element-wise
//! bandwidth, cache-blocked matmul GFLOP/s and collective bandwidth.
//!
//!   zig build bench

const std = @import("std");
const rt = @import("runtime");

fn bench(name: []const u8, iterations: usize, work: anytype) !void {
    var timer = try std.time.Timer.start();
    var i: usize = 0;
    while (i < iterations) : (i += 1) try work();
    const ns = timer.read();
    const out = std.io.getStdOut().writer();
    try out.print("{s: <28} {d: >10} iters {d: >12} ns/iter\n", .{ name, iterations, ns / iterations });
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const base = gpa.allocator();
    const out = std.io.getStdOut().writer();

    var arena = try rt.memory.ArenaAllocator.init(base, 1 << 22, .none);
    defer arena.deinit();
    const arena_alloc = arena.allocator();
    try bench("arena alloc 64B", 200_000, struct {
        a: std.mem.Allocator,
        arena: *rt.memory.ArenaAllocator,
        fn call(self: @This()) !void {
            const p = try self.a.alloc(u8, 64);
            if (p.len != 64) return error.Unexpected;
            if (self.arena.stats.bytes_used > (1 << 21)) self.arena.reset();
        }
    }{ .a = arena_alloc, .arena = &arena }.call);

    const n = 256;
    var a = try rt.tensor.Tensor.init(base, &.{ n, n });
    defer a.deinit();
    var b = try rt.tensor.Tensor.init(base, &.{ n, n });
    defer b.deinit();
    var c = try rt.tensor.Tensor.init(base, &.{ n, n });
    defer c.deinit();
    try a.fill(1.0001);
    try b.fill(0.9999);

    var timer = try std.time.Timer.start();
    try rt.tensor.matmul(&c, &a, &b, .{});
    const ns = timer.read();
    const flops = 2.0 * @as(f64, @floatFromInt(n * n * n));
    try out.print("matmul {d}^3: {d:.2} GFLOP/s\n", .{ n, flops / @as(f64, @floatFromInt(ns)) });

    var coord = try rt.gpu.Coordinator.init(base, 4);
    defer coord.deinit();
    var buffers: [4]rt.gpu.DeviceBuffer = undefined;
    for (&buffers, 0..) |*buf, rank| buf.* = try coord.alloc(@intCast(rank % coord.device_count), 1 << 16);
    defer for (buffers) |buf| coord.free(buf);
    timer.reset();
    var completion = try coord.allReduce(&buffers, .sum, .ring);
    try completion.wait();
    const collective_ns = timer.read();
    const bytes: f64 = @floatFromInt(coord.stats.bytes_moved);
    try out.print("ring all-reduce 4x256KiB: {d} us, {d:.2} GB/s effective\n", .{
        collective_ns / 1000,
        bytes / @as(f64, @floatFromInt(collective_ns)),
    });
}
