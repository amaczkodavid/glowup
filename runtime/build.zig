//! Build script for the ultra-low-level runtime.
//!
//!   zig build            -> static library + demo executable
//!   zig build test       -> allocator / tensor / GPU / bindings unit tests
//!   zig build demo       -> run the end-to-end demo
//!   zig build bench      -> run the matmul + collective micro-benchmarks
//!
//! Options:
//!   -Dcuda=true          link against libcudart at load time (default: lazy)
//!   -Dfuthark=true       link the Futhark-generated shared library
//!   -Dtarget=...         cross compile (x86_64-linux-gnu, aarch64-linux-gnu…)

const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});
    const use_cuda = b.option(bool, "cuda", "Link CUDA runtime eagerly") orelse false;
    const use_futhark = b.option(bool, "futhark", "Link Futhark kernels") orelse false;

    const options = b.addOptions();
    options.addOption(bool, "cuda_enabled", use_cuda);
    options.addOption(bool, "futhark_enabled", use_futhark);

    const root = b.addModule("runtime", .{
        .root_source_file = b.path("root.zig"),
        .target = target,
        .optimize = optimize,
    });
    root.addOptions("build_options", options);

    const lib = b.addLibrary(.{
        .name = "synthruntime",
        .root_module = root,
        .linkage = .static,
    });
    if (use_cuda) {
        lib.linkSystemLibrary("cudart");
        lib.linkLibC();
    }
    if (use_futhark) {
        lib.addLibraryPath(b.path("../futhark_kernels"));
        lib.linkSystemLibrary("futkernels");
        lib.linkLibC();
    }
    b.installArtifact(lib);

    const demo_mod = b.createModule(.{
        .root_source_file = b.path("examples/demo.zig"),
        .target = target,
        .optimize = optimize,
    });
    demo_mod.addImport("runtime", root);
    const demo = b.addExecutable(.{ .name = "demo", .root_module = demo_mod });
    b.installArtifact(demo);

    const run_demo = b.addRunArtifact(demo);
    run_demo.step.dependOn(b.getInstallStep());
    b.step("demo", "Run the end-to-end demo").dependOn(&run_demo.step);

    const test_mod = b.createModule(.{
        .root_source_file = b.path("tests/runtime_test.zig"),
        .target = target,
        .optimize = optimize,
    });
    test_mod.addImport("runtime", root);
    const unit_tests = b.addTest(.{ .root_module = test_mod });
    const run_tests = b.addRunArtifact(unit_tests);
    b.step("test", "Run all runtime unit tests").dependOn(&run_tests.step);

    const bench_mod = b.createModule(.{
        .root_source_file = b.path("examples/bench.zig"),
        .target = target,
        .optimize = .ReleaseFast,
    });
    bench_mod.addImport("runtime", root);
    const bench = b.addExecutable(.{ .name = "bench", .root_module = bench_mod });
    const run_bench = b.addRunArtifact(bench);
    b.step("bench", "Run micro-benchmarks").dependOn(&run_bench.step);
}
