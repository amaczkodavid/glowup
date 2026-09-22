//! runtime/futhark_bindings.zig — Zig bindings over Futhark-generated C ABI.
//!
//! `futhark c|multicore|cuda|opencl --library kernels.fut` emits `kernels.h`
//! and `kernels.c`; this module wraps the generated entry points with
//! RAII-style Zig types:
//!
//!   Context      -> futhark_context / futhark_context_config
//!   Array1D/2D   -> futhark_new_f32_1d / _2d + futhark_values_f32_*
//!   entry points -> futhark_entry_*
//!
//! Symbols are resolved through `std.DynLib` so the runtime links against a
//! shared library (`libfutkernels.so`) built by `futhark_kernels/build.sh`,
//! and degrades to `error.FutharkUnavailable` when it is missing.

const std = @import("std");
const Allocator = std.mem.Allocator;

pub const FutharkError = error{
    FutharkUnavailable,
    ContextCreationFailed,
    EntryPointMissing,
    KernelFailed,
    ShapeMismatch,
    OutOfMemory,
};

const CfgNewFn = *const fn () callconv(.c) ?*anyopaque;
const CfgFreeFn = *const fn (?*anyopaque) callconv(.c) void;
const CtxNewFn = *const fn (?*anyopaque) callconv(.c) ?*anyopaque;
const CtxFreeFn = *const fn (?*anyopaque) callconv(.c) void;
const CtxSyncFn = *const fn (?*anyopaque) callconv(.c) c_int;
const CtxErrFn = *const fn (?*anyopaque) callconv(.c) ?[*:0]u8;
const New1DFn = *const fn (?*anyopaque, [*]const f32, i64) callconv(.c) ?*anyopaque;
const New2DFn = *const fn (?*anyopaque, [*]const f32, i64, i64) callconv(.c) ?*anyopaque;
const Free1DFn = *const fn (?*anyopaque, ?*anyopaque) callconv(.c) c_int;
const Values1DFn = *const fn (?*anyopaque, ?*anyopaque, [*]f32) callconv(.c) c_int;
const Shape1DFn = *const fn (?*anyopaque, ?*anyopaque) callconv(.c) [*]const i64;

/// Entry points exported by `futhark_kernels/kernels.fut`.
const EntryDot = *const fn (?*anyopaque, *f32, ?*anyopaque, ?*anyopaque) callconv(.c) c_int;
const EntrySaxpy = *const fn (?*anyopaque, *?*anyopaque, f32, ?*anyopaque, ?*anyopaque) callconv(.c) c_int;
const EntryMatmul = *const fn (?*anyopaque, *?*anyopaque, ?*anyopaque, ?*anyopaque) callconv(.c) c_int;
const EntryReduce = *const fn (?*anyopaque, *f32, ?*anyopaque) callconv(.c) c_int;
const EntryFft = *const fn (?*anyopaque, *?*anyopaque, *?*anyopaque, ?*anyopaque, ?*anyopaque) callconv(.c) c_int;
const EntryConv = *const fn (?*anyopaque, *?*anyopaque, ?*anyopaque, ?*anyopaque) callconv(.c) c_int;
const EntrySoftmax = *const fn (?*anyopaque, *?*anyopaque, ?*anyopaque) callconv(.c) c_int;
const EntryStencil = *const fn (?*anyopaque, *?*anyopaque, ?*anyopaque, i32) callconv(.c) c_int;

pub const Library = struct {
    lib: std.DynLib,
    cfg_new: CfgNewFn,
    cfg_free: CfgFreeFn,
    ctx_new: CtxNewFn,
    ctx_free: CtxFreeFn,
    ctx_sync: CtxSyncFn,
    ctx_error: CtxErrFn,
    new_f32_1d: New1DFn,
    new_f32_2d: New2DFn,
    free_f32_1d: Free1DFn,
    free_f32_2d: Free1DFn,
    values_f32_1d: Values1DFn,
    values_f32_2d: Values1DFn,
    shape_f32_1d: Shape1DFn,
    shape_f32_2d: Shape1DFn,
    entry_dot: ?EntryDot,
    entry_saxpy: ?EntrySaxpy,
    entry_matmul: ?EntryMatmul,
    entry_reduce_sum: ?EntryReduce,
    entry_fft: ?EntryFft,
    entry_conv1d: ?EntryConv,
    entry_softmax: ?EntrySoftmax,
    entry_stencil: ?EntryStencil,

    pub fn open(path: []const u8) FutharkError!Library {
        var lib = std.DynLib.open(path) catch return FutharkError.FutharkUnavailable;
        return Library{
            .lib = lib,
            .cfg_new = lib.lookup(CfgNewFn, "futhark_context_config_new") orelse return FutharkError.EntryPointMissing,
            .cfg_free = lib.lookup(CfgFreeFn, "futhark_context_config_free") orelse return FutharkError.EntryPointMissing,
            .ctx_new = lib.lookup(CtxNewFn, "futhark_context_new") orelse return FutharkError.EntryPointMissing,
            .ctx_free = lib.lookup(CtxFreeFn, "futhark_context_free") orelse return FutharkError.EntryPointMissing,
            .ctx_sync = lib.lookup(CtxSyncFn, "futhark_context_sync") orelse return FutharkError.EntryPointMissing,
            .ctx_error = lib.lookup(CtxErrFn, "futhark_context_get_error") orelse return FutharkError.EntryPointMissing,
            .new_f32_1d = lib.lookup(New1DFn, "futhark_new_f32_1d") orelse return FutharkError.EntryPointMissing,
            .new_f32_2d = lib.lookup(New2DFn, "futhark_new_f32_2d") orelse return FutharkError.EntryPointMissing,
            .free_f32_1d = lib.lookup(Free1DFn, "futhark_free_f32_1d") orelse return FutharkError.EntryPointMissing,
            .free_f32_2d = lib.lookup(Free1DFn, "futhark_free_f32_2d") orelse return FutharkError.EntryPointMissing,
            .values_f32_1d = lib.lookup(Values1DFn, "futhark_values_f32_1d") orelse return FutharkError.EntryPointMissing,
            .values_f32_2d = lib.lookup(Values1DFn, "futhark_values_f32_2d") orelse return FutharkError.EntryPointMissing,
            .shape_f32_1d = lib.lookup(Shape1DFn, "futhark_shape_f32_1d") orelse return FutharkError.EntryPointMissing,
            .shape_f32_2d = lib.lookup(Shape1DFn, "futhark_shape_f32_2d") orelse return FutharkError.EntryPointMissing,
            .entry_dot = lib.lookup(EntryDot, "futhark_entry_dot"),
            .entry_saxpy = lib.lookup(EntrySaxpy, "futhark_entry_saxpy"),
            .entry_matmul = lib.lookup(EntryMatmul, "futhark_entry_matmul"),
            .entry_reduce_sum = lib.lookup(EntryReduce, "futhark_entry_reduce_sum"),
            .entry_fft = lib.lookup(EntryFft, "futhark_entry_fft"),
            .entry_conv1d = lib.lookup(EntryConv, "futhark_entry_conv1d"),
            .entry_softmax = lib.lookup(EntrySoftmax, "futhark_entry_softmax"),
            .entry_stencil = lib.lookup(EntryStencil, "futhark_entry_stencil"),
        };
    }

    pub fn close(self: *Library) void {
        self.lib.close();
        self.* = undefined;
    }
};

pub const Context = struct {
    lib: Library,
    cfg: ?*anyopaque,
    ctx: ?*anyopaque,
    allocator: Allocator,

    const Self = @This();

    pub fn init(allocator: Allocator, path: []const u8) FutharkError!Self {
        var lib = try Library.open(path);
        const cfg = lib.cfg_new() orelse return FutharkError.ContextCreationFailed;
        const ctx = lib.ctx_new(cfg) orelse {
            lib.cfg_free(cfg);
            return FutharkError.ContextCreationFailed;
        };
        return .{ .lib = lib, .cfg = cfg, .ctx = ctx, .allocator = allocator };
    }

    pub fn deinit(self: *Self) void {
        self.lib.ctx_free(self.ctx);
        self.lib.cfg_free(self.cfg);
        self.lib.close();
        self.* = undefined;
    }

    pub fn sync(self: *Self) FutharkError!void {
        if (self.lib.ctx_sync(self.ctx) != 0) return FutharkError.KernelFailed;
    }

    pub fn lastError(self: *Self) ?[*:0]u8 {
        return self.lib.ctx_error(self.ctx);
    }

    pub fn newArray1d(self: *Self, values: []const f32) FutharkError!Array1D {
        const handle = self.lib.new_f32_1d(self.ctx, values.ptr, @intCast(values.len)) orelse
            return FutharkError.OutOfMemory;
        return .{ .ctx = self, .handle = handle, .len = values.len };
    }

    pub fn newArray2d(self: *Self, values: []const f32, rows: usize, cols: usize) FutharkError!Array2D {
        if (values.len != rows * cols) return FutharkError.ShapeMismatch;
        const handle = self.lib.new_f32_2d(self.ctx, values.ptr, @intCast(rows), @intCast(cols)) orelse
            return FutharkError.OutOfMemory;
        return .{ .ctx = self, .handle = handle, .rows = rows, .cols = cols };
    }

    // ---- entry points ----------------------------------------------------

    pub fn dot(self: *Self, a: *Array1D, b: *Array1D) FutharkError!f32 {
        const entry = self.lib.entry_dot orelse return FutharkError.EntryPointMissing;
        var out: f32 = 0;
        if (entry(self.ctx, &out, a.handle, b.handle) != 0) return FutharkError.KernelFailed;
        try self.sync();
        return out;
    }

    pub fn saxpy(self: *Self, alpha: f32, x: *Array1D, y: *Array1D) FutharkError!Array1D {
        const entry = self.lib.entry_saxpy orelse return FutharkError.EntryPointMissing;
        var handle: ?*anyopaque = null;
        if (entry(self.ctx, &handle, alpha, x.handle, y.handle) != 0) return FutharkError.KernelFailed;
        try self.sync();
        return .{ .ctx = self, .handle = handle.?, .len = x.len };
    }

    pub fn matmul(self: *Self, a: *Array2D, b: *Array2D) FutharkError!Array2D {
        const entry = self.lib.entry_matmul orelse return FutharkError.EntryPointMissing;
        if (a.cols != b.rows) return FutharkError.ShapeMismatch;
        var handle: ?*anyopaque = null;
        if (entry(self.ctx, &handle, a.handle, b.handle) != 0) return FutharkError.KernelFailed;
        try self.sync();
        return .{ .ctx = self, .handle = handle.?, .rows = a.rows, .cols = b.cols };
    }

    pub fn reduceSum(self: *Self, a: *Array1D) FutharkError!f32 {
        const entry = self.lib.entry_reduce_sum orelse return FutharkError.EntryPointMissing;
        var out: f32 = 0;
        if (entry(self.ctx, &out, a.handle) != 0) return FutharkError.KernelFailed;
        try self.sync();
        return out;
    }

    pub fn softmax(self: *Self, a: *Array1D) FutharkError!Array1D {
        const entry = self.lib.entry_softmax orelse return FutharkError.EntryPointMissing;
        var handle: ?*anyopaque = null;
        if (entry(self.ctx, &handle, a.handle) != 0) return FutharkError.KernelFailed;
        try self.sync();
        return .{ .ctx = self, .handle = handle.?, .len = a.len };
    }

    pub fn conv1d(self: *Self, signal: *Array1D, kernel: *Array1D) FutharkError!Array1D {
        const entry = self.lib.entry_conv1d orelse return FutharkError.EntryPointMissing;
        var handle: ?*anyopaque = null;
        if (entry(self.ctx, &handle, signal.handle, kernel.handle) != 0) return FutharkError.KernelFailed;
        try self.sync();
        return .{ .ctx = self, .handle = handle.?, .len = signal.len };
    }

    /// Radix-2 FFT: returns (real, imaginary) spectra.
    pub fn fft(self: *Self, re: *Array1D, im: *Array1D) FutharkError!struct { re: Array1D, im: Array1D } {
        const entry = self.lib.entry_fft orelse return FutharkError.EntryPointMissing;
        var out_re: ?*anyopaque = null;
        var out_im: ?*anyopaque = null;
        if (entry(self.ctx, &out_re, &out_im, re.handle, im.handle) != 0) return FutharkError.KernelFailed;
        try self.sync();
        return .{
            .re = .{ .ctx = self, .handle = out_re.?, .len = re.len },
            .im = .{ .ctx = self, .handle = out_im.?, .len = im.len },
        };
    }

    pub fn stencil(self: *Self, input: *Array1D, iterations: i32) FutharkError!Array1D {
        const entry = self.lib.entry_stencil orelse return FutharkError.EntryPointMissing;
        var handle: ?*anyopaque = null;
        if (entry(self.ctx, &handle, input.handle, iterations) != 0) return FutharkError.KernelFailed;
        try self.sync();
        return .{ .ctx = self, .handle = handle.?, .len = input.len };
    }
};

pub const Array1D = struct {
    ctx: *Context,
    handle: *anyopaque,
    len: usize,

    pub fn deinit(self: *Array1D) void {
        _ = self.ctx.lib.free_f32_1d(self.ctx.ctx, self.handle);
        self.* = undefined;
    }
    pub fn read(self: *Array1D, out: []f32) FutharkError!void {
        if (out.len != self.len) return FutharkError.ShapeMismatch;
        if (self.ctx.lib.values_f32_1d(self.ctx.ctx, self.handle, out.ptr) != 0)
            return FutharkError.KernelFailed;
        try self.ctx.sync();
    }
    pub fn toOwnedSlice(self: *Array1D, allocator: Allocator) ![]f32 {
        const out = try allocator.alloc(f32, self.len);
        try self.read(out);
        return out;
    }
};

pub const Array2D = struct {
    ctx: *Context,
    handle: *anyopaque,
    rows: usize,
    cols: usize,

    pub fn deinit(self: *Array2D) void {
        _ = self.ctx.lib.free_f32_2d(self.ctx.ctx, self.handle);
        self.* = undefined;
    }
    pub fn read(self: *Array2D, out: []f32) FutharkError!void {
        if (out.len != self.rows * self.cols) return FutharkError.ShapeMismatch;
        if (self.ctx.lib.values_f32_2d(self.ctx.ctx, self.handle, out.ptr) != 0)
            return FutharkError.KernelFailed;
        try self.ctx.sync();
    }
};

/// Convenience: open the library shipped by `futhark_kernels/build.sh`,
/// falling back to a pure-Zig implementation when Futhark is not installed.
pub fn openDefault(allocator: Allocator) ?Context {
    const candidates = [_][]const u8{
        "./futhark_kernels/libfutkernels.so",
        "libfutkernels.so",
        "/usr/local/lib/libfutkernels.so",
    };
    for (candidates) |path| {
        if (Context.init(allocator, path)) |ctx| return ctx else |_| continue;
    }
    return null;
}

test "graceful degradation when the Futhark library is absent" {
    const result = Context.init(std.testing.allocator, "./definitely-missing-libfutkernels.so");
    try std.testing.expectError(FutharkError.FutharkUnavailable, result);
}
