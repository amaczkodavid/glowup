//! Lock-free data structures used to feed synthesised kernels.
//!
//!  * `MpmcQueue`  — bounded multi-producer/multi-consumer queue (Vyukov
//!    bounded MPMC ring: per-slot sequence numbers, no CAS loops on the data,
//!    false-sharing avoided by cache-line padding).
//!  * `TreiberStack` — unbounded lock-free stack with an ABA-resistant tagged
//!    head pointer (packed 48-bit pointer + 16-bit tag inside a u64).
//!  * `SpscRing` — wait-free single-producer/single-consumer ring buffer.

const std = @import("std");
const Allocator = std.mem.Allocator;
const CACHE_LINE = 64;

pub fn MpmcQueue(comptime T: type) type {
    return struct {
        const Self = @This();
        const Cell = struct {
            sequence: std.atomic.Value(usize),
            data: T,
        };

        buffer: []Cell,
        mask: usize,
        allocator: Allocator,
        enqueue_pos: std.atomic.Value(usize) align(CACHE_LINE),
        dequeue_pos: std.atomic.Value(usize) align(CACHE_LINE),

        pub fn init(allocator: Allocator, capacity_pow2: usize) !Self {
            std.debug.assert(std.math.isPowerOfTwo(capacity_pow2));
            const buffer = try allocator.alloc(Cell, capacity_pow2);
            for (buffer, 0..) |*cell, i| {
                cell.sequence = std.atomic.Value(usize).init(i);
                cell.data = undefined;
            }
            return .{
                .buffer = buffer,
                .mask = capacity_pow2 - 1,
                .allocator = allocator,
                .enqueue_pos = std.atomic.Value(usize).init(0),
                .dequeue_pos = std.atomic.Value(usize).init(0),
            };
        }

        pub fn deinit(self: *Self) void {
            self.allocator.free(self.buffer);
            self.* = undefined;
        }

        pub fn tryPush(self: *Self, item: T) bool {
            var pos = self.enqueue_pos.load(.monotonic);
            while (true) {
                const cell = &self.buffer[pos & self.mask];
                const seq = cell.sequence.load(.acquire);
                const diff = @as(isize, @intCast(seq)) - @as(isize, @intCast(pos));
                if (diff == 0) {
                    if (self.enqueue_pos.cmpxchgWeak(pos, pos + 1, .monotonic, .monotonic)) |actual| {
                        pos = actual;
                        continue;
                    }
                    cell.data = item;
                    cell.sequence.store(pos + 1, .release);
                    return true;
                } else if (diff < 0) {
                    return false; // full
                } else {
                    pos = self.enqueue_pos.load(.monotonic);
                }
            }
        }

        pub fn tryPop(self: *Self) ?T {
            var pos = self.dequeue_pos.load(.monotonic);
            while (true) {
                const cell = &self.buffer[pos & self.mask];
                const seq = cell.sequence.load(.acquire);
                const diff = @as(isize, @intCast(seq)) - @as(isize, @intCast(pos + 1));
                if (diff == 0) {
                    if (self.dequeue_pos.cmpxchgWeak(pos, pos + 1, .monotonic, .monotonic)) |actual| {
                        pos = actual;
                        continue;
                    }
                    const item = cell.data;
                    cell.sequence.store(pos + self.mask + 1, .release);
                    return item;
                } else if (diff < 0) {
                    return null; // empty
                } else {
                    pos = self.dequeue_pos.load(.monotonic);
                }
            }
        }

        pub fn len(self: *Self) usize {
            const head = self.enqueue_pos.load(.acquire);
            const tail = self.dequeue_pos.load(.acquire);
            return head -% tail;
        }
    };
}

pub fn TreiberStack(comptime T: type) type {
    return struct {
        const Self = @This();
        pub const Node = struct { next: ?*Node, value: T };

        head: std.atomic.Value(u64) align(CACHE_LINE),
        allocator: Allocator,

        const PTR_MASK: u64 = 0x0000_FFFF_FFFF_FFFF;
        const TAG_SHIFT: u6 = 48;

        pub fn init(allocator: Allocator) Self {
            return .{ .head = std.atomic.Value(u64).init(0), .allocator = allocator };
        }

        fn pack(ptr: ?*Node, tag: u16) u64 {
            const raw: u64 = if (ptr) |p| @intFromPtr(p) else 0;
            return (raw & PTR_MASK) | (@as(u64, tag) << TAG_SHIFT);
        }
        fn unpack(value: u64) ?*Node {
            const raw = value & PTR_MASK;
            if (raw == 0) return null;
            return @ptrFromInt(raw);
        }
        fn tagOf(value: u64) u16 {
            return @truncate(value >> TAG_SHIFT);
        }

        pub fn push(self: *Self, value: T) !void {
            const node = try self.allocator.create(Node);
            node.* = .{ .next = null, .value = value };
            while (true) {
                const current = self.head.load(.acquire);
                node.next = unpack(current);
                const next = pack(node, tagOf(current) +% 1);
                if (self.head.cmpxchgWeak(current, next, .release, .acquire) == null) return;
            }
        }

        pub fn pop(self: *Self) ?T {
            while (true) {
                const current = self.head.load(.acquire);
                const node = unpack(current) orelse return null;
                const next = pack(node.next, tagOf(current) +% 1);
                if (self.head.cmpxchgWeak(current, next, .release, .acquire) == null) {
                    const value = node.value;
                    self.allocator.destroy(node);
                    return value;
                }
            }
        }

        pub fn deinit(self: *Self) void {
            while (self.pop()) |_| {}
            self.* = undefined;
        }
    };
}

pub fn SpscRing(comptime T: type) type {
    return struct {
        const Self = @This();
        buffer: []T,
        mask: usize,
        allocator: Allocator,
        head: std.atomic.Value(usize) align(CACHE_LINE),
        tail: std.atomic.Value(usize) align(CACHE_LINE),

        pub fn init(allocator: Allocator, capacity_pow2: usize) !Self {
            std.debug.assert(std.math.isPowerOfTwo(capacity_pow2));
            return .{
                .buffer = try allocator.alloc(T, capacity_pow2),
                .mask = capacity_pow2 - 1,
                .allocator = allocator,
                .head = std.atomic.Value(usize).init(0),
                .tail = std.atomic.Value(usize).init(0),
            };
        }
        pub fn deinit(self: *Self) void {
            self.allocator.free(self.buffer);
            self.* = undefined;
        }
        pub fn push(self: *Self, value: T) bool {
            const head = self.head.load(.monotonic);
            const tail = self.tail.load(.acquire);
            if (head -% tail > self.mask) return false;
            self.buffer[head & self.mask] = value;
            self.head.store(head +% 1, .release);
            return true;
        }
        pub fn pop(self: *Self) ?T {
            const tail = self.tail.load(.monotonic);
            const head = self.head.load(.acquire);
            if (tail == head) return null;
            const value = self.buffer[tail & self.mask];
            self.tail.store(tail +% 1, .release);
            return value;
        }
    };
}

test "mpmc queue single thread" {
    var q = try MpmcQueue(u64).init(std.testing.allocator, 8);
    defer q.deinit();
    var i: u64 = 0;
    while (i < 8) : (i += 1) try std.testing.expect(q.tryPush(i));
    try std.testing.expect(!q.tryPush(99));
    i = 0;
    while (i < 8) : (i += 1) try std.testing.expectEqual(i, q.tryPop().?);
    try std.testing.expect(q.tryPop() == null);
}

test "mpmc queue concurrent" {
    const Ctx = struct {
        q: *MpmcQueue(u64),
        produced: std.atomic.Value(u64),
        consumed: std.atomic.Value(u64),

        fn producer(ctx: *@This()) void {
            var n: u64 = 0;
            while (n < 10_000) {
                if (ctx.q.tryPush(n)) {
                    n += 1;
                    _ = ctx.produced.fetchAdd(1, .monotonic);
                }
            }
        }
        fn consumer(ctx: *@This()) void {
            var seen: u64 = 0;
            while (seen < 10_000) {
                if (ctx.q.tryPop()) |_| {
                    seen += 1;
                    _ = ctx.consumed.fetchAdd(1, .monotonic);
                }
            }
        }
    };
    var q = try MpmcQueue(u64).init(std.testing.allocator, 1024);
    defer q.deinit();
    var ctx = Ctx{
        .q = &q,
        .produced = std.atomic.Value(u64).init(0),
        .consumed = std.atomic.Value(u64).init(0),
    };
    const p = try std.Thread.spawn(.{}, Ctx.producer, .{&ctx});
    const c = try std.Thread.spawn(.{}, Ctx.consumer, .{&ctx});
    p.join();
    c.join();
    try std.testing.expectEqual(@as(u64, 10_000), ctx.consumed.load(.acquire));
}

test "treiber stack" {
    var s = TreiberStack(u32).init(std.testing.allocator);
    defer s.deinit();
    try s.push(1);
    try s.push(2);
    try std.testing.expectEqual(@as(u32, 2), s.pop().?);
    try std.testing.expectEqual(@as(u32, 1), s.pop().?);
    try std.testing.expect(s.pop() == null);
}

test "spsc ring" {
    var r = try SpscRing(u8).init(std.testing.allocator, 4);
    defer r.deinit();
    try std.testing.expect(r.push(7));
    try std.testing.expectEqual(@as(u8, 7), r.pop().?);
}
