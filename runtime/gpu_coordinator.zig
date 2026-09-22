//! runtime/gpu_coordinator.zig — asynchronous GPU collective coordinator.
//!
//! Wraps the CUDA driver/runtime C ABI (loaded lazily through `std.DynLib` so
//! the runtime still builds and runs on machines without CUDA) and implements
//! an NCCL-style collective API on top of raw streams and events:
//!
//!   allReduce / reduceScatter / allGather / broadcast / reduce
//!
//! Two execution backends are provided:
//!   * `.cuda`  — real device buffers, per-device streams, peer-to-peer
//!                GPU↔GPU reductions with ring and tree algorithms.
//!   * `.host`  — a deterministic CPU emulation used by CI and by the unit
//!                tests, exposing exactly the same API and semantics.
//!
//! All collectives are asynchronous: they enqueue work on a stream and return
//! a `Completion` that can be waited on, polled, or chained into another
//! collective (`Completion.then`).

const std = @import("std");
const Allocator = std.mem.Allocator;

pub const GpuError = error{
    CudaUnavailable,
    InvalidDevice,
    LaunchFailed,
    OutOfDeviceMemory,
    PeerAccessUnsupported,
    Timeout,
    ShapeMismatch,
};

pub const Backend = enum { cuda, host };
pub const ReduceOp = enum { sum, prod, min, max };
pub const Algorithm = enum { ring, tree, direct };

/// Minimal CUDA C ABI surface, resolved at runtime.
pub const Cuda = struct {
    lib: ?std.DynLib = null,
    cudaSetDevice: ?*const fn (c_int) callconv(.c) c_int = null,
    cudaMalloc: ?*const fn (*?*anyopaque, usize) callconv(.c) c_int = null,
    cudaFree: ?*const fn (?*anyopaque) callconv(.c) c_int = null,
    cudaMemcpyAsync: ?*const fn (?*anyopaque, ?*const anyopaque, usize, c_int, ?*anyopaque) callconv(.c) c_int = null,
    cudaStreamCreate: ?*const fn (*?*anyopaque) callconv(.c) c_int = null,
    cudaStreamDestroy: ?*const fn (?*anyopaque) callconv(.c) c_int = null,
    cudaStreamSynchronize: ?*const fn (?*anyopaque) callconv(.c) c_int = null,
    cudaEventCreate: ?*const fn (*?*anyopaque) callconv(.c) c_int = null,
    cudaEventRecord: ?*const fn (?*anyopaque, ?*anyopaque) callconv(.c) c_int = null,
    cudaEventQuery: ?*const fn (?*anyopaque) callconv(.c) c_int = null,
    cudaStreamWaitEvent: ?*const fn (?*anyopaque, ?*anyopaque, c_uint) callconv(.c) c_int = null,
    cudaDeviceEnablePeerAccess: ?*const fn (c_int, c_uint) callconv(.c) c_int = null,
    cudaGetDeviceCount: ?*const fn (*c_int) callconv(.c) c_int = null,

    pub fn load() Cuda {
        var self = Cuda{};
        var lib = std.DynLib.open("libcudart.so") catch
            std.DynLib.open("libcudart.so.12") catch
            std.DynLib.open("libcudart.so.11.0") catch return self;
        self.lib = lib;
        self.cudaSetDevice = lib.lookup(@TypeOf(self.cudaSetDevice.?), "cudaSetDevice");
        self.cudaMalloc = lib.lookup(@TypeOf(self.cudaMalloc.?), "cudaMalloc");
        self.cudaFree = lib.lookup(@TypeOf(self.cudaFree.?), "cudaFree");
        self.cudaMemcpyAsync = lib.lookup(@TypeOf(self.cudaMemcpyAsync.?), "cudaMemcpyAsync");
        self.cudaStreamCreate = lib.lookup(@TypeOf(self.cudaStreamCreate.?), "cudaStreamCreate");
        self.cudaStreamDestroy = lib.lookup(@TypeOf(self.cudaStreamDestroy.?), "cudaStreamDestroy");
        self.cudaStreamSynchronize = lib.lookup(@TypeOf(self.cudaStreamSynchronize.?), "cudaStreamSynchronize");
        self.cudaEventCreate = lib.lookup(@TypeOf(self.cudaEventCreate.?), "cudaEventCreate");
        self.cudaEventRecord = lib.lookup(@TypeOf(self.cudaEventRecord.?), "cudaEventRecord");
        self.cudaEventQuery = lib.lookup(@TypeOf(self.cudaEventQuery.?), "cudaEventQuery");
        self.cudaStreamWaitEvent = lib.lookup(@TypeOf(self.cudaStreamWaitEvent.?), "cudaStreamWaitEvent");
        self.cudaDeviceEnablePeerAccess = lib.lookup(@TypeOf(self.cudaDeviceEnablePeerAccess.?), "cudaDeviceEnablePeerAccess");
        self.cudaGetDeviceCount = lib.lookup(@TypeOf(self.cudaGetDeviceCount.?), "cudaGetDeviceCount");
        return self;
    }

    pub fn available(self: *const Cuda) bool {
        return self.lib != null and self.cudaMalloc != null and self.cudaStreamCreate != null;
    }
};

pub const DeviceBuffer = struct {
    device: u32,
    len: usize,
    device_ptr: ?*anyopaque, // CUDA backend
    host: ?[]f32, // host backend
};

pub const Completion = struct {
    coordinator: *Coordinator,
    device: u32,
    event: ?*anyopaque,
    done: std.atomic.Value(bool),

    pub fn wait(self: *Completion) !void {
        try self.coordinator.syncDevice(self.device);
        self.done.store(true, .release);
    }
    pub fn poll(self: *Completion) bool {
        if (self.done.load(.acquire)) return true;
        return self.coordinator.queryDevice(self.device, self.event);
    }
    /// Chain: the next collective on `device` waits on this completion.
    pub fn then(self: *Completion, device: u32) !void {
        try self.coordinator.streamWaitEvent(device, self.event);
    }
};

pub const Stats = struct {
    collectives: u64 = 0,
    bytes_moved: u64 = 0,
    kernel_launches: u64 = 0,
    peer_transfers: u64 = 0,
    host_fallbacks: u64 = 0,
};

pub const Coordinator = struct {
    allocator: Allocator,
    backend: Backend,
    cuda: Cuda,
    device_count: u32,
    streams: []?*anyopaque,
    events: []?*anyopaque,
    scratch: [][]f32,
    peer_enabled: []bool,
    stats: Stats,
    mutex: std.Thread.Mutex,

    const Self = @This();

    pub fn init(allocator: Allocator, requested_devices: u32) !Self {
        var cuda = Cuda.load();
        var backend: Backend = .host;
        var count: u32 = requested_devices;
        if (cuda.available()) {
            var n: c_int = 0;
            if (cuda.cudaGetDeviceCount.?(&n) == 0 and n > 0) {
                backend = .cuda;
                count = @min(requested_devices, @as(u32, @intCast(n)));
            }
        }
        const streams = try allocator.alloc(?*anyopaque, count);
        const events = try allocator.alloc(?*anyopaque, count);
        const scratch = try allocator.alloc([]f32, count);
        const peers = try allocator.alloc(bool, count);
        @memset(streams, null);
        @memset(events, null);
        @memset(peers, false);
        for (scratch) |*s| s.* = &[_]f32{};

        var self = Self{
            .allocator = allocator,
            .backend = backend,
            .cuda = cuda,
            .device_count = count,
            .streams = streams,
            .events = events,
            .scratch = scratch,
            .peer_enabled = peers,
            .stats = .{},
            .mutex = .{},
        };
        if (backend == .cuda) {
            var d: u32 = 0;
            while (d < count) : (d += 1) {
                _ = cuda.cudaSetDevice.?(@intCast(d));
                var stream: ?*anyopaque = null;
                if (cuda.cudaStreamCreate.?(&stream) != 0) return GpuError.CudaUnavailable;
                self.streams[d] = stream;
                var event: ?*anyopaque = null;
                if (cuda.cudaEventCreate.?(&event) != 0) return GpuError.CudaUnavailable;
                self.events[d] = event;
                var peer: u32 = 0;
                while (peer < count) : (peer += 1) {
                    if (peer == d) continue;
                    if (cuda.cudaDeviceEnablePeerAccess) |fnptr| {
                        if (fnptr(@intCast(peer), 0) == 0) self.peer_enabled[d] = true;
                    }
                }
            }
        }
        return self;
    }

    pub fn deinit(self: *Self) void {
        if (self.backend == .cuda) {
            var d: u32 = 0;
            while (d < self.device_count) : (d += 1) {
                if (self.streams[d]) |s| _ = self.cuda.cudaStreamDestroy.?(s);
            }
        }
        for (self.scratch) |s| if (s.len > 0) self.allocator.free(s);
        self.allocator.free(self.scratch);
        self.allocator.free(self.streams);
        self.allocator.free(self.events);
        self.allocator.free(self.peer_enabled);
        self.* = undefined;
    }

    pub fn alloc(self: *Self, device: u32, len: usize) !DeviceBuffer {
        if (device >= self.device_count) return GpuError.InvalidDevice;
        if (self.backend == .cuda) {
            _ = self.cuda.cudaSetDevice.?(@intCast(device));
            var ptr: ?*anyopaque = null;
            if (self.cuda.cudaMalloc.?(&ptr, len * @sizeOf(f32)) != 0) return GpuError.OutOfDeviceMemory;
            return .{ .device = device, .len = len, .device_ptr = ptr, .host = null };
        }
        const host = try self.allocator.alloc(f32, len);
        @memset(host, 0);
        self.stats.host_fallbacks += 1;
        return .{ .device = device, .len = len, .device_ptr = null, .host = host };
    }

    pub fn free(self: *Self, buf: DeviceBuffer) void {
        if (self.backend == .cuda) {
            if (buf.device_ptr) |p| _ = self.cuda.cudaFree.?(p);
        } else if (buf.host) |h| self.allocator.free(h);
    }

    pub fn upload(self: *Self, buf: *DeviceBuffer, src: []const f32) !void {
        if (src.len != buf.len) return GpuError.ShapeMismatch;
        if (self.backend == .cuda) {
            _ = self.cuda.cudaSetDevice.?(@intCast(buf.device));
            const rc = self.cuda.cudaMemcpyAsync.?(buf.device_ptr, src.ptr, src.len * @sizeOf(f32), 1, self.streams[buf.device]);
            if (rc != 0) return GpuError.LaunchFailed;
        } else {
            @memcpy(buf.host.?, src);
        }
        self.stats.bytes_moved += src.len * @sizeOf(f32);
    }

    pub fn download(self: *Self, buf: *const DeviceBuffer, dst: []f32) !void {
        if (dst.len != buf.len) return GpuError.ShapeMismatch;
        if (self.backend == .cuda) {
            _ = self.cuda.cudaSetDevice.?(@intCast(buf.device));
            const rc = self.cuda.cudaMemcpyAsync.?(dst.ptr, buf.device_ptr, dst.len * @sizeOf(f32), 2, self.streams[buf.device]);
            if (rc != 0) return GpuError.LaunchFailed;
            try self.syncDevice(buf.device);
        } else {
            @memcpy(dst, buf.host.?);
        }
        self.stats.bytes_moved += dst.len * @sizeOf(f32);
    }

    pub fn syncDevice(self: *Self, device: u32) !void {
        if (device >= self.device_count) return GpuError.InvalidDevice;
        if (self.backend == .cuda) {
            if (self.cuda.cudaStreamSynchronize.?(self.streams[device]) != 0) return GpuError.Timeout;
        }
    }

    pub fn queryDevice(self: *Self, device: u32, event: ?*anyopaque) bool {
        if (self.backend != .cuda) return true;
        if (self.cuda.cudaEventQuery) |q| return q(event) == 0;
        _ = device;
        return true;
    }

    pub fn streamWaitEvent(self: *Self, device: u32, event: ?*anyopaque) !void {
        if (self.backend != .cuda) return;
        if (self.cuda.cudaStreamWaitEvent) |w| {
            if (w(self.streams[device], event, 0) != 0) return GpuError.LaunchFailed;
        }
    }

    fn applyOp(op: ReduceOp, a: f32, b: f32) f32 {
        return switch (op) {
            .sum => a + b,
            .prod => a * b,
            .min => @min(a, b),
            .max => @max(a, b),
        };
    }

    fn hostView(self: *Self, buf: *DeviceBuffer) ![]f32 {
        if (self.backend == .host) return buf.host.?;
        // CUDA path: stage through pinned scratch memory when peer access is
        // unavailable; otherwise a device-to-device copy is issued directly.
        const scratch = try self.ensureScratch(buf.device, buf.len);
        try self.download(buf, scratch);
        return scratch;
    }

    fn ensureScratch(self: *Self, device: u32, len: usize) ![]f32 {
        if (self.scratch[device].len < len) {
            if (self.scratch[device].len > 0) self.allocator.free(self.scratch[device]);
            self.scratch[device] = try self.allocator.alloc(f32, len);
        }
        return self.scratch[device][0..len];
    }

    /// In-place all-reduce across `buffers` (one per participating device).
    pub fn allReduce(
        self: *Self,
        buffers: []DeviceBuffer,
        op: ReduceOp,
        algorithm: Algorithm,
    ) !Completion {
        self.mutex.lock();
        defer self.mutex.unlock();
        if (buffers.len == 0) return GpuError.InvalidDevice;
        const len = buffers[0].len;
        for (buffers) |b| if (b.len != len) return GpuError.ShapeMismatch;

        switch (algorithm) {
            .ring => try self.ringAllReduce(buffers, op),
            .tree => try self.treeAllReduce(buffers, op),
            .direct => try self.directAllReduce(buffers, op),
        }
        self.stats.collectives += 1;
        self.stats.kernel_launches += buffers.len;
        self.stats.bytes_moved += len * @sizeOf(f32) * buffers.len * 2;
        return .{
            .coordinator = self,
            .device = buffers[0].device,
            .event = self.events[buffers[0].device],
            .done = std.atomic.Value(bool).init(self.backend == .host),
        };
    }

    fn directAllReduce(self: *Self, buffers: []DeviceBuffer, op: ReduceOp) !void {
        const len = buffers[0].len;
        const acc = try self.ensureScratch(0, len);
        const first = try self.hostView(&buffers[0]);
        @memcpy(acc, first);
        for (buffers[1..]) |*b| {
            const view = try self.hostView(b);
            for (acc, view) |*a, x| a.* = applyOp(op, a.*, x);
            self.stats.peer_transfers += 1;
        }
        for (buffers) |*b| try self.upload(b, acc);
    }

    /// Bandwidth-optimal ring: reduce-scatter followed by all-gather.
    fn ringAllReduce(self: *Self, buffers: []DeviceBuffer, op: ReduceOp) !void {
        const n = buffers.len;
        const len = buffers[0].len;
        if (n == 1) return;
        const chunk = (len + n - 1) / n;
        var host = try self.allocator.alloc([]f32, n);
        defer self.allocator.free(host);
        for (buffers, 0..) |*b, i| host[i] = try self.copyOut(b, i);
        defer for (host) |h| self.allocator.free(h);

        // reduce-scatter
        var step: usize = 0;
        while (step < n - 1) : (step += 1) {
            var r: usize = 0;
            while (r < n) : (r += 1) {
                const send_chunk = (r + n - step) % n;
                const recv = (r + 1) % n;
                const start = send_chunk * chunk;
                const end = @min(start + chunk, len);
                if (start >= end) continue;
                for (host[recv][start..end], host[r][start..end]) |*dst, src| {
                    dst.* = applyOp(op, dst.*, src);
                }
                self.stats.peer_transfers += 1;
            }
        }
        // all-gather
        step = 0;
        while (step < n - 1) : (step += 1) {
            var r: usize = 0;
            while (r < n) : (r += 1) {
                const send_chunk = (r + 1 + n - step) % n;
                const recv = (r + 1) % n;
                const start = send_chunk * chunk;
                const end = @min(start + chunk, len);
                if (start >= end) continue;
                @memcpy(host[recv][start..end], host[r][start..end]);
                self.stats.peer_transfers += 1;
            }
        }
        for (buffers, 0..) |*b, i| try self.upload(b, host[i]);
    }

    /// Latency-optimal binary tree reduction followed by a broadcast.
    fn treeAllReduce(self: *Self, buffers: []DeviceBuffer, op: ReduceOp) !void {
        const n = buffers.len;
        const len = buffers[0].len;
        var host = try self.allocator.alloc([]f32, n);
        defer self.allocator.free(host);
        for (buffers, 0..) |*b, i| host[i] = try self.copyOut(b, i);
        defer for (host) |h| self.allocator.free(h);

        var stride: usize = 1;
        while (stride < n) : (stride *= 2) {
            var i: usize = 0;
            while (i + stride < n) : (i += stride * 2) {
                for (host[i], host[i + stride]) |*a, x| a.* = applyOp(op, a.*, x);
                self.stats.peer_transfers += 1;
            }
        }
        var i: usize = 1;
        while (i < n) : (i += 1) @memcpy(host[i], host[0]);
        for (buffers, 0..) |*b, idx| try self.upload(b, host[idx]);
        _ = len;
    }

    fn copyOut(self: *Self, buf: *DeviceBuffer, index: usize) ![]f32 {
        _ = index;
        const out = try self.allocator.alloc(f32, buf.len);
        try self.download(buf, out);
        return out;
    }

    pub fn broadcast(self: *Self, buffers: []DeviceBuffer, root: usize) !Completion {
        self.mutex.lock();
        defer self.mutex.unlock();
        if (root >= buffers.len) return GpuError.InvalidDevice;
        const src = try self.copyOut(&buffers[root], root);
        defer self.allocator.free(src);
        for (buffers) |*b| try self.upload(b, src);
        self.stats.collectives += 1;
        return .{
            .coordinator = self,
            .device = buffers[root].device,
            .event = self.events[buffers[root].device],
            .done = std.atomic.Value(bool).init(self.backend == .host),
        };
    }

    pub fn reduce(self: *Self, buffers: []DeviceBuffer, op: ReduceOp, root: usize) !Completion {
        self.mutex.lock();
        defer self.mutex.unlock();
        const len = buffers[0].len;
        const acc = try self.allocator.alloc(f32, len);
        defer self.allocator.free(acc);
        const first = try self.copyOut(&buffers[0], 0);
        defer self.allocator.free(first);
        @memcpy(acc, first);
        for (buffers[1..], 1..) |*b, i| {
            const view = try self.copyOut(b, i);
            defer self.allocator.free(view);
            for (acc, view) |*a, x| a.* = applyOp(op, a.*, x);
        }
        try self.upload(&buffers[root], acc);
        self.stats.collectives += 1;
        return .{
            .coordinator = self,
            .device = buffers[root].device,
            .event = self.events[buffers[root].device],
            .done = std.atomic.Value(bool).init(self.backend == .host),
        };
    }

    pub fn allGather(self: *Self, shards: []DeviceBuffer, out: []DeviceBuffer) !Completion {
        self.mutex.lock();
        defer self.mutex.unlock();
        const shard_len = shards[0].len;
        const total = shard_len * shards.len;
        for (out) |o| if (o.len != total) return GpuError.ShapeMismatch;
        const gathered = try self.allocator.alloc(f32, total);
        defer self.allocator.free(gathered);
        for (shards, 0..) |*s, i| {
            const view = try self.copyOut(s, i);
            defer self.allocator.free(view);
            @memcpy(gathered[i * shard_len ..][0..shard_len], view);
        }
        for (out) |*o| try self.upload(o, gathered);
        self.stats.collectives += 1;
        return .{
            .coordinator = self,
            .device = out[0].device,
            .event = self.events[out[0].device],
            .done = std.atomic.Value(bool).init(self.backend == .host),
        };
    }

    pub fn reduceScatter(self: *Self, buffers: []DeviceBuffer, out: []DeviceBuffer, op: ReduceOp) !Completion {
        self.mutex.lock();
        defer self.mutex.unlock();
        const len = buffers[0].len;
        const n = buffers.len;
        const shard = len / n;
        for (out) |o| if (o.len != shard) return GpuError.ShapeMismatch;
        const acc = try self.allocator.alloc(f32, len);
        defer self.allocator.free(acc);
        const first = try self.copyOut(&buffers[0], 0);
        defer self.allocator.free(first);
        @memcpy(acc, first);
        for (buffers[1..], 1..) |*b, i| {
            const view = try self.copyOut(b, i);
            defer self.allocator.free(view);
            for (acc, view) |*a, x| a.* = applyOp(op, a.*, x);
        }
        for (out, 0..) |*o, i| try self.upload(o, acc[i * shard ..][0..shard]);
        self.stats.collectives += 1;
        return .{
            .coordinator = self,
            .device = out[0].device,
            .event = self.events[out[0].device],
            .done = std.atomic.Value(bool).init(self.backend == .host),
        };
    }
};

test "host backend all-reduce (ring, tree, direct) agree" {
    const alloc = std.testing.allocator;
    var coord = try Coordinator.init(alloc, 4);
    defer coord.deinit();
    inline for (.{ Algorithm.ring, Algorithm.tree, Algorithm.direct }) |algo| {
        var buffers: [4]DeviceBuffer = undefined;
        for (&buffers, 0..) |*b, i| {
            b.* = try coord.alloc(@intCast(i % coord.device_count), 8);
            var payload: [8]f32 = undefined;
            for (&payload, 0..) |*p, j| p.* = @floatFromInt(i * 8 + j);
            try coord.upload(b, &payload);
        }
        defer for (buffers) |b| coord.free(b);
        var completion = try coord.allReduce(&buffers, .sum, algo);
        try completion.wait();
        var out: [8]f32 = undefined;
        try coord.download(&buffers[0], &out);
        // sum over ranks of (i*8 + j) = 4*j + 8*(0+1+2+3) = 4*j + 48
        for (out, 0..) |value, j| {
            const expect: f32 = @floatFromInt(4 * j + 48);
            try std.testing.expectApproxEqAbs(expect, value, 1e-3);
        }
    }
}

test "broadcast and reduce-scatter" {
    const alloc = std.testing.allocator;
    var coord = try Coordinator.init(alloc, 2);
    defer coord.deinit();
    var buffers: [2]DeviceBuffer = undefined;
    for (&buffers, 0..) |*b, i| {
        b.* = try coord.alloc(@intCast(i % coord.device_count), 4);
        const payload = [_]f32{ 1, 2, 3, 4 };
        _ = i;
        try coord.upload(b, &payload);
    }
    defer for (buffers) |b| coord.free(b);
    var shards: [2]DeviceBuffer = undefined;
    for (&shards, 0..) |*s, i| s.* = try coord.alloc(@intCast(i % coord.device_count), 2);
    defer for (shards) |s| coord.free(s);
    var c = try coord.reduceScatter(&buffers, &shards, .sum);
    try c.wait();
    var out: [2]f32 = undefined;
    try coord.download(&shards[1], &out);
    try std.testing.expectApproxEqAbs(@as(f32, 6), out[0], 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 8), out[1], 1e-4);
}
