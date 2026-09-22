//! End-to-end demo: allocate with the custom allocators, run SIMD tensor
//! primitives and a cache-blocked matmul, execute synthesised scalar kernels,
//! drive an asynchronous GPU all-reduce, and (when available) call into the
//! Futhark-generated kernels.
//!
//!   zig build demo

const std = @import("std");
const rt = @import("runtime");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const base = gpa.allocator();

    var out = std.io.getStdOut().writer();
    try out.print("== synthesised-kernel runtime demo ==\n", .{});

    // 1. Arena + slab allocation ------------------------------------------------
    var arena = try rt.memory.ArenaAllocator.init(base, 1 << 20, .secure);
    defer arena.deinit();
    const alloc = arena.allocator();
    const scratch = try alloc.alloc(u8, 4096);
    @memset(scratch, 0xa5);
    try out.print("arena: {d} bytes reserved, {d} used, high water {d}\n", .{
        arena.stats.bytes_reserved, arena.stats.bytes_used, arena.stats.high_water,
    });

    var pool = try rt.memory.PoolAllocator(u64).init(base, 1024, .none);
    defer pool.deinit();
    const cell = pool.create().?;
    cell.* = 0xdead_beef;
    pool.destroy(cell);

    // 2. Lock-free work queue ---------------------------------------------------
    var queue = try rt.lockfree.MpmcQueue(u32).init(base, 64);
    defer queue.deinit();
    var i: u32 = 0;
    while (i < 32) : (i += 1) _ = queue.tryPush(i);
    var drained: u32 = 0;
    while (queue.tryPop()) |_| drained += 1;
    try out.print("lock-free queue: drained {d} items\n", .{drained});

    // 3. SIMD tensors + cache-blocked matmul -----------------------------------
    const n = 128;
    var a = try rt.tensor.Tensor.init(base, &.{ n, n });
    defer a.deinit();
    var b = try rt.tensor.Tensor.init(base, &.{ n, n });
    defer b.deinit();
    var c = try rt.tensor.Tensor.init(base, &.{ n, n });
    defer c.deinit();
    try a.fill(1.5);
    try b.fill(2.0);

    var timer = try std.time.Timer.start();
    try rt.tensor.matmul(&c, &a, &b, .{});
    const serial_ns = timer.lap();
    try rt.tensor.matmulParallel(base, &c, &a, &b, .{ .threads = 4 });
    const parallel_ns = timer.lap();
    try out.print(
        "matmul {d}x{d}: serial {d} us, 4-thread {d} us, checksum {d:.1}\n",
        .{ n, n, serial_ns / 1000, parallel_ns / 1000, rt.tensor.sum(&c) },
    );

    // 4. Synthesised scalar kernels --------------------------------------------
    const kernel = try rt.kernels.lookup("isolate_lsb");
    const value: u32 = 0b1011_0000;
    const isolated = switch (kernel.reference) {
        .scalar => |f| f(value),
        .pair => |f| f(value, value),
    };
    try out.print("kernel '{s}': isolate_lsb(0x{x}) = 0x{x}\n", .{ kernel.name, value, isolated });

    // 5. Asynchronous GPU collectives ------------------------------------------
    var coord = try rt.gpu.Coordinator.init(base, 4);
    defer coord.deinit();
    var buffers: [4]rt.gpu.DeviceBuffer = undefined;
    for (&buffers, 0..) |*buf, rank| {
        buf.* = try coord.alloc(@intCast(rank % coord.device_count), 1024);
        const payload = try base.alloc(f32, 1024);
        defer base.free(payload);
        for (payload, 0..) |*p, j| p.* = @floatFromInt(rank * 1024 + j);
        try coord.upload(buf, payload);
    }
    defer for (buffers) |buf| coord.free(buf);
    var completion = try coord.allReduce(&buffers, .sum, .ring);
    try completion.wait();
    var head: [4]f32 = undefined;
    var probe = try base.alloc(f32, 1024);
    defer base.free(probe);
    try coord.download(&buffers[0], probe);
    @memcpy(&head, probe[0..4]);
    try out.print(
        "gpu ({s} backend, {d} devices): all-reduce head = {d:.0} {d:.0} {d:.0} {d:.0}, collectives={d}\n",
        .{ @tagName(coord.backend), coord.device_count, head[0], head[1], head[2], head[3], coord.stats.collectives },
    );

    // 6. Futhark kernels (optional) --------------------------------------------
    if (rt.futhark.openDefault(base)) |ctx_const| {
        var ctx = ctx_const;
        defer ctx.deinit();
        const xs = [_]f32{ 1, 2, 3, 4, 5, 6, 7, 8 };
        const ys = [_]f32{ 8, 7, 6, 5, 4, 3, 2, 1 };
        var xa = try ctx.newArray1d(&xs);
        defer xa.deinit();
        var ya = try ctx.newArray1d(&ys);
        defer ya.deinit();
        const d = try ctx.dot(&xa, &ya);
        try out.print("futhark: dot = {d:.1}\n", .{d});
    } else {
        try out.print("futhark: library not built (run futhark_kernels/build.sh) — skipped\n", .{});
    }

    try out.print("== demo complete ==\n", .{});
}
