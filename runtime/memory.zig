//! Ultra-low-overhead allocators for the synthesised-kernel runtime.
//!
//! Provides four allocators that all expose `std.mem.Allocator`:
//!   * ArenaAllocator  - bump pointer, O(1) alloc, bulk reset, page aligned.
//!   * SlabAllocator   - fixed-size object slabs with an intrusive free list.
//!   * PoolAllocator   - typed wrapper over the slab allocator.
//!   * BuddyAllocator  - power-of-two buddy system with O(log n) split/merge.
//!
//! Every allocator supports optional secure zeroization on free (`.secure`),
//! honours cache-line (64 B) and page (4 KiB) alignment requirements, and is
//! free of undefined behaviour under `-Doptimize=ReleaseSafe`.

const std = @import("std");
const mem = std.mem;
const Allocator = mem.Allocator;
const assert = std.debug.assert;

pub const CACHE_LINE: usize = 64;
pub const PAGE_SIZE: usize = 4096;

pub const Zeroize = enum { none, secure };

/// Volatile memset that the optimiser may not elide.
pub fn secureZero(bytes: []u8) void {
    const p: [*]volatile u8 = @ptrCast(bytes.ptr);
    var i: usize = 0;
    while (i < bytes.len) : (i += 1) p[i] = 0;
    // Compiler barrier: keep the stores.
    asm volatile ("" ::: "memory");
}

pub inline fn alignUp(value: usize, alignment: usize) usize {
    assert(std.math.isPowerOfTwo(alignment));
    return (value + alignment - 1) & ~(alignment - 1);
}

// ---------------------------------------------------------------------------
// Arena
// ---------------------------------------------------------------------------

pub const ArenaStats = struct {
    bytes_reserved: usize = 0,
    bytes_used: usize = 0,
    high_water: usize = 0,
    allocations: usize = 0,
    resets: usize = 0,
};

pub const ArenaAllocator = struct {
    backing: Allocator,
    buffer: []align(PAGE_SIZE) u8,
    offset: usize,
    stats: ArenaStats,
    zeroize: Zeroize,

    const Self = @This();

    pub fn init(backing: Allocator, capacity: usize, zeroize: Zeroize) !Self {
        const rounded = alignUp(capacity, PAGE_SIZE);
        const buf = try backing.alignedAlloc(u8, PAGE_SIZE, rounded);
        return .{
            .backing = backing,
            .buffer = buf,
            .offset = 0,
            .stats = .{ .bytes_reserved = rounded },
            .zeroize = zeroize,
        };
    }

    pub fn deinit(self: *Self) void {
        if (self.zeroize == .secure) secureZero(self.buffer);
        self.backing.free(self.buffer);
        self.* = undefined;
    }

    pub fn reset(self: *Self) void {
        if (self.zeroize == .secure) secureZero(self.buffer[0..self.offset]);
        self.offset = 0;
        self.stats.bytes_used = 0;
        self.stats.resets += 1;
    }

    pub fn allocator(self: *Self) Allocator {
        return .{
            .ptr = self,
            .vtable = &.{
                .alloc = alloc,
                .resize = resize,
                .remap = remap,
                .free = free,
            },
        };
    }

    fn alloc(ctx: *anyopaque, len: usize, alignment: mem.Alignment, ra: usize) ?[*]u8 {
        _ = ra;
        const self: *Self = @ptrCast(@alignCast(ctx));
        const a = alignment.toByteUnits();
        const start = alignUp(@intFromPtr(self.buffer.ptr) + self.offset, a) - @intFromPtr(self.buffer.ptr);
        if (start + len > self.buffer.len) return null;
        self.offset = start + len;
        self.stats.bytes_used = self.offset;
        self.stats.high_water = @max(self.stats.high_water, self.offset);
        self.stats.allocations += 1;
        return self.buffer.ptr + start;
    }

    fn resize(ctx: *anyopaque, buf: []u8, alignment: mem.Alignment, new_len: usize, ra: usize) bool {
        _ = alignment;
        _ = ra;
        const self: *Self = @ptrCast(@alignCast(ctx));
        const end = @intFromPtr(buf.ptr) + buf.len - @intFromPtr(self.buffer.ptr);
        if (end != self.offset) return new_len <= buf.len;
        const start = end - buf.len;
        if (start + new_len > self.buffer.len) return false;
        self.offset = start + new_len;
        self.stats.bytes_used = self.offset;
        self.stats.high_water = @max(self.stats.high_water, self.offset);
        return true;
    }

    fn remap(ctx: *anyopaque, buf: []u8, alignment: mem.Alignment, new_len: usize, ra: usize) ?[*]u8 {
        if (resize(ctx, buf, alignment, new_len, ra)) return buf.ptr;
        return null;
    }

    fn free(ctx: *anyopaque, buf: []u8, alignment: mem.Alignment, ra: usize) void {
        _ = alignment;
        _ = ra;
        const self: *Self = @ptrCast(@alignCast(ctx));
        if (self.zeroize == .secure) secureZero(buf);
        const end = @intFromPtr(buf.ptr) + buf.len - @intFromPtr(self.buffer.ptr);
        if (end == self.offset) self.offset -= buf.len; // LIFO fast path
        self.stats.bytes_used = self.offset;
    }
};

// ---------------------------------------------------------------------------
// Slab
// ---------------------------------------------------------------------------

pub const SlabAllocator = struct {
    backing: Allocator,
    memory: []align(PAGE_SIZE) u8,
    free_list: ?*Node,
    object_size: usize,
    object_align: usize,
    capacity: usize,
    live: usize,
    zeroize: Zeroize,

    const Node = struct { next: ?*Node };
    const Self = @This();

    pub fn init(
        backing: Allocator,
        object_size: usize,
        object_align: usize,
        count: usize,
        zeroize: Zeroize,
    ) !Self {
        const stride = alignUp(@max(object_size, @sizeOf(Node)), @max(object_align, @alignOf(Node)));
        const bytes = alignUp(stride * count, PAGE_SIZE);
        const buf = try backing.alignedAlloc(u8, PAGE_SIZE, bytes);
        var self: Self = .{
            .backing = backing,
            .memory = buf,
            .free_list = null,
            .object_size = stride,
            .object_align = @max(object_align, @alignOf(Node)),
            .capacity = count,
            .live = 0,
            .zeroize = zeroize,
        };
        var i: usize = count;
        while (i > 0) {
            i -= 1;
            const slot: *Node = @ptrCast(@alignCast(buf.ptr + i * stride));
            slot.* = .{ .next = self.free_list };
            self.free_list = slot;
        }
        return self;
    }

    pub fn deinit(self: *Self) void {
        if (self.zeroize == .secure) secureZero(self.memory);
        self.backing.free(self.memory);
        self.* = undefined;
    }

    pub fn acquire(self: *Self) ?[]u8 {
        const node = self.free_list orelse return null;
        self.free_list = node.next;
        self.live += 1;
        const raw: [*]u8 = @ptrCast(node);
        return raw[0..self.object_size];
    }

    pub fn release(self: *Self, slot: []u8) void {
        assert(slot.len == self.object_size);
        if (self.zeroize == .secure) secureZero(slot);
        const node: *Node = @ptrCast(@alignCast(slot.ptr));
        node.* = .{ .next = self.free_list };
        self.free_list = node;
        self.live -= 1;
    }

    pub fn allocator(self: *Self) Allocator {
        return .{
            .ptr = self,
            .vtable = &.{ .alloc = alloc, .resize = resize, .remap = remap, .free = freeFn },
        };
    }

    fn alloc(ctx: *anyopaque, len: usize, alignment: mem.Alignment, ra: usize) ?[*]u8 {
        _ = ra;
        const self: *Self = @ptrCast(@alignCast(ctx));
        if (len > self.object_size) return null;
        if (alignment.toByteUnits() > self.object_align) return null;
        const slot = self.acquire() orelse return null;
        return slot.ptr;
    }

    fn resize(ctx: *anyopaque, buf: []u8, alignment: mem.Alignment, new_len: usize, ra: usize) bool {
        _ = alignment;
        _ = ra;
        const self: *Self = @ptrCast(@alignCast(ctx));
        _ = buf;
        return new_len <= self.object_size;
    }

    fn remap(ctx: *anyopaque, buf: []u8, alignment: mem.Alignment, new_len: usize, ra: usize) ?[*]u8 {
        if (resize(ctx, buf, alignment, new_len, ra)) return buf.ptr;
        return null;
    }

    fn freeFn(ctx: *anyopaque, buf: []u8, alignment: mem.Alignment, ra: usize) void {
        _ = alignment;
        _ = ra;
        const self: *Self = @ptrCast(@alignCast(ctx));
        const slot = buf.ptr[0..self.object_size];
        self.release(slot);
    }
};

/// Typed pool built on top of the slab allocator.
pub fn PoolAllocator(comptime T: type) type {
    return struct {
        slab: SlabAllocator,

        const Self = @This();

        pub fn init(backing: Allocator, count: usize, zeroize: Zeroize) !Self {
            return .{ .slab = try SlabAllocator.init(backing, @sizeOf(T), @alignOf(T), count, zeroize) };
        }
        pub fn deinit(self: *Self) void {
            self.slab.deinit();
        }
        pub fn create(self: *Self) ?*T {
            const slot = self.slab.acquire() orelse return null;
            return @ptrCast(@alignCast(slot.ptr));
        }
        pub fn destroy(self: *Self, item: *T) void {
            const raw: [*]u8 = @ptrCast(item);
            self.slab.release(raw[0..self.slab.object_size]);
        }
        pub fn live(self: *const Self) usize {
            return self.slab.live;
        }
    };
}

// ---------------------------------------------------------------------------
// Buddy
// ---------------------------------------------------------------------------

/// Power-of-two buddy allocator. `order` k manages blocks of `min_block << k`.
pub const BuddyAllocator = struct {
    backing: Allocator,
    memory: []align(PAGE_SIZE) u8,
    free_lists: []?*Block,
    orders: usize,
    min_block: usize,
    zeroize: Zeroize,
    bytes_live: usize,

    const Block = struct { next: ?*Block, order: usize, free: bool };
    const Self = @This();

    pub fn init(backing: Allocator, total_bytes: usize, min_block: usize, zeroize: Zeroize) !Self {
        assert(std.math.isPowerOfTwo(min_block));
        const total = std.math.ceilPowerOfTwoAssert(usize, alignUp(total_bytes, min_block));
        const orders = std.math.log2_int(usize, total / min_block) + 1;
        const buf = try backing.alignedAlloc(u8, PAGE_SIZE, total);
        const lists = try backing.alloc(?*Block, orders);
        @memset(lists, null);
        const root: *Block = @ptrCast(@alignCast(buf.ptr));
        root.* = .{ .next = null, .order = orders - 1, .free = true };
        lists[orders - 1] = root;
        return .{
            .backing = backing,
            .memory = buf,
            .free_lists = lists,
            .orders = orders,
            .min_block = min_block,
            .zeroize = zeroize,
            .bytes_live = 0,
        };
    }

    pub fn deinit(self: *Self) void {
        if (self.zeroize == .secure) secureZero(self.memory);
        self.backing.free(self.free_lists);
        self.backing.free(self.memory);
        self.* = undefined;
    }

    fn orderFor(self: *const Self, len: usize) ?usize {
        var order: usize = 0;
        var size = self.min_block;
        while (size < len + @sizeOf(Block)) : (order += 1) {
            size <<= 1;
            if (order + 1 >= self.orders) return null;
        }
        return order;
    }

    fn blockSize(self: *const Self, order: usize) usize {
        return self.min_block << @intCast(order);
    }

    fn pop(self: *Self, order: usize) ?*Block {
        const head = self.free_lists[order] orelse return null;
        self.free_lists[order] = head.next;
        return head;
    }

    fn push(self: *Self, block: *Block, order: usize) void {
        block.* = .{ .next = self.free_lists[order], .order = order, .free = true };
        self.free_lists[order] = block;
    }

    pub fn allocBytes(self: *Self, len: usize) ?[]u8 {
        const want = self.orderFor(len) orelse return null;
        var order = want;
        while (order < self.orders and self.free_lists[order] == null) order += 1;
        if (order >= self.orders) return null;
        var block = self.pop(order).?;
        while (order > want) {
            order -= 1;
            const half = self.blockSize(order);
            const raw: [*]u8 = @ptrCast(block);
            const buddy: *Block = @ptrCast(@alignCast(raw + half));
            self.push(buddy, order);
        }
        block.* = .{ .next = null, .order = want, .free = false };
        const raw: [*]u8 = @ptrCast(block);
        self.bytes_live += self.blockSize(want);
        return raw[@sizeOf(Block) .. @sizeOf(Block) + len];
    }

    pub fn freeBytes(self: *Self, slice: []u8) void {
        const raw = slice.ptr - @sizeOf(Block);
        var block: *Block = @ptrCast(@alignCast(raw));
        var order = block.order;
        if (self.zeroize == .secure) secureZero(slice);
        self.bytes_live -= self.blockSize(order);
        while (order + 1 < self.orders) {
            const size = self.blockSize(order);
            const base = @intFromPtr(self.memory.ptr);
            const offset = @intFromPtr(block) - base;
            const buddy_off = offset ^ size;
            const buddy: *Block = @ptrCast(@alignCast(self.memory.ptr + buddy_off));
            if (!self.isFree(buddy, order)) break;
            self.remove(buddy, order);
            if (buddy_off < offset) block = buddy;
            order += 1;
        }
        self.push(block, order);
    }

    fn isFree(self: *Self, candidate: *Block, order: usize) bool {
        var cur = self.free_lists[order];
        while (cur) |c| : (cur = c.next) {
            if (c == candidate) return true;
        }
        return false;
    }

    fn remove(self: *Self, target: *Block, order: usize) void {
        var prev: ?*Block = null;
        var cur = self.free_lists[order];
        while (cur) |c| : ({
            prev = c;
            cur = c.next;
        }) {
            if (c == target) {
                if (prev) |p| p.next = c.next else self.free_lists[order] = c.next;
                return;
            }
        }
    }

    pub fn allocator(self: *Self) Allocator {
        return .{
            .ptr = self,
            .vtable = &.{ .alloc = alloc, .resize = resize, .remap = remap, .free = freeFn },
        };
    }

    fn alloc(ctx: *anyopaque, len: usize, alignment: mem.Alignment, ra: usize) ?[*]u8 {
        _ = ra;
        const self: *Self = @ptrCast(@alignCast(ctx));
        const a = alignment.toByteUnits();
        const slice = self.allocBytes(len + a) orelse return null;
        const addr = alignUp(@intFromPtr(slice.ptr), a);
        return @ptrFromInt(addr);
    }

    fn resize(ctx: *anyopaque, buf: []u8, alignment: mem.Alignment, new_len: usize, ra: usize) bool {
        _ = ctx;
        _ = alignment;
        _ = ra;
        return new_len <= buf.len;
    }

    fn remap(ctx: *anyopaque, buf: []u8, alignment: mem.Alignment, new_len: usize, ra: usize) ?[*]u8 {
        if (resize(ctx, buf, alignment, new_len, ra)) return buf.ptr;
        return null;
    }

    fn freeFn(ctx: *anyopaque, buf: []u8, alignment: mem.Alignment, ra: usize) void {
        _ = alignment;
        _ = ra;
        const self: *Self = @ptrCast(@alignCast(ctx));
        if (self.zeroize == .secure) secureZero(buf);
    }
};

test "arena bump allocation and reset" {
    var arena = try ArenaAllocator.init(std.testing.allocator, 1 << 16, .secure);
    defer arena.deinit();
    const a = arena.allocator();
    const first = try a.alloc(u64, 128);
    @memset(first, 7);
    try std.testing.expectEqual(@as(u64, 7), first[17]);
    try std.testing.expect(arena.stats.bytes_used >= 1024);
    arena.reset();
    try std.testing.expectEqual(@as(usize, 0), arena.stats.bytes_used);
}

test "slab acquire/release round trip" {
    var slab = try SlabAllocator.init(std.testing.allocator, 64, 16, 32, .secure);
    defer slab.deinit();
    var slots: [32][]u8 = undefined;
    for (&slots) |*s| s.* = slab.acquire().?;
    try std.testing.expect(slab.acquire() == null);
    for (slots) |s| slab.release(s);
    try std.testing.expectEqual(@as(usize, 0), slab.live);
}

test "typed pool" {
    const Particle = struct { x: f64, y: f64, vx: f64, vy: f64 };
    var pool = try PoolAllocator(Particle).init(std.testing.allocator, 16, .none);
    defer pool.deinit();
    const p = pool.create().?;
    p.* = .{ .x = 1, .y = 2, .vx = 3, .vy = 4 };
    try std.testing.expectEqual(@as(usize, 1), pool.live());
    pool.destroy(p);
    try std.testing.expectEqual(@as(usize, 0), pool.live());
}

test "buddy split and coalesce" {
    var buddy = try BuddyAllocator.init(std.testing.allocator, 1 << 16, 64, .secure);
    defer buddy.deinit();
    const a = buddy.allocBytes(100).?;
    const b = buddy.allocBytes(100).?;
    try std.testing.expect(a.ptr != b.ptr);
    buddy.freeBytes(a);
    buddy.freeBytes(b);
    try std.testing.expectEqual(@as(usize, 0), buddy.bytes_live);
}
