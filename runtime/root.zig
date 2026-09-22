//! Public surface of the synthesised-kernel runtime.

pub const memory = @import("memory.zig");
pub const lockfree = @import("lockfree.zig");
pub const tensor = @import("tensor.zig");
pub const gpu = @import("gpu_coordinator.zig");
pub const futhark = @import("futhark_bindings.zig");
pub const kernels = @import("kernels.zig");

test {
    @import("std").testing.refAllDecls(@This());
}
