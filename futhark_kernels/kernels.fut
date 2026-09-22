-- futhark_kernels/kernels.fut
--
-- Data-parallel kernels compiled to a C ABI shared library and called from
-- the Zig runtime through `runtime/futhark_bindings.zig`.
--
--   futhark c        --library kernels.fut   # portable CPU
--   futhark multicore --library kernels.fut  # threaded CPU
--   futhark cuda     --library kernels.fut   # NVIDIA GPUs
--   futhark opencl   --library kernels.fut   # portable GPUs
--
-- Every entry point is total (no partial functions) and shape-checked by the
-- Futhark type system, which is what allows the Zig side to treat kernel
-- failures as hard errors rather than undefined behaviour.

-- | Inner product of two vectors.
entry dot [n] (xs: [n]f32) (ys: [n]f32) : f32 =
  f32.sum (map2 (*) xs ys)

-- | y <- alpha*x + y
entry saxpy [n] (alpha: f32) (xs: [n]f32) (ys: [n]f32) : [n]f32 =
  map2 (\x y -> alpha * x + y) xs ys

-- | Dense matrix multiplication, C[m][p] = A[m][n] * B[n][p].
entry matmul [m][n][p] (a: [m][n]f32) (b: [n][p]f32) : [m][p]f32 =
  let bt = transpose b
  in map (\row -> map (\col -> f32.sum (map2 (*) row col)) bt) a

-- | Sum reduction with a balanced tree (associative, deterministic).
entry reduce_sum [n] (xs: [n]f32) : f32 =
  reduce (+) 0f32 xs

-- | Numerically stable softmax.
entry softmax [n] (xs: [n]f32) : [n]f32 =
  let mx = f32.maximum xs
  let exps = map (\x -> f32.exp (x - mx)) xs
  let total = f32.sum exps
  in map (/ total) exps

-- | 1-D convolution with zero padding ("same" mode).
entry conv1d [n][k] (signal: [n]f32) (kernel: [k]f32) : [n]f32 =
  let half = k / 2
  in tabulate n (\i ->
       f32.sum (tabulate k (\j ->
         let idx = i + j - half
         in if idx < 0 || idx >= n then 0f32 else signal[idx] * kernel[j])))

-- | Three-point Jacobi stencil, iterated `iters` times.
entry stencil [n] (xs: [n]f32) (iters: i32) : [n]f32 =
  loop acc = xs for _i < i64.i32 iters do
    tabulate n (\i ->
      let l = if i == 0 then acc[i] else acc[i - 1]
      let r = if i == n - 1 then acc[i] else acc[i + 1]
      in (l + acc[i] + r) / 3f32)

-- | Iterative radix-2 Cooley–Tukey FFT over power-of-two inputs.
--   Returns the real and imaginary parts of the spectrum.
def bit_reverse [n] (xs: [n]f32) : [n]f32 =
  let bits = i64.f64 (f64.log2 (f64.i64 n) + 0.5)
  in tabulate n (\i ->
       let rev = loop r = 0i64 for b < bits do
                   (r << 1) | ((i >> b) & 1)
       in xs[rev])

entry fft [n] (re_in: [n]f32) (im_in: [n]f32) : ([n]f32, [n]f32) =
  let re0 = bit_reverse re_in
  let im0 = bit_reverse im_in
  let stages = i64.f64 (f64.log2 (f64.i64 n) + 0.5)
  let (re, im) =
    loop (re, im) = (re0, im0) for s < stages do
      let m = 1i64 << (s + 1)
      let half = m / 2
      let pairs = n / 2
      let idxs = tabulate pairs (\p ->
                   let block = p / half
                   let off = p % half
                   in (block * m + off, block * m + off + half, off))
      let contributions =
        map (\(a, b, off) ->
               let angle = -2f32 * f32.pi * f32.i64 off / f32.i64 m
               let wr = f32.cos angle
               let wi = f32.sin angle
               let tr = wr * re[b] - wi * im[b]
               let ti = wr * im[b] + wi * re[b]
               in ((a, re[a] + tr, im[a] + ti), (b, re[a] - tr, im[a] - ti)))
            idxs
      let flat_idx = flatten (map (\((a, _, _), (b, _, _)) -> [a, b]) contributions)
      let flat_re = flatten (map (\((_, ar, _), (_, br, _)) -> [ar, br]) contributions)
      let flat_im = flatten (map (\((_, _, ai), (_, _, bi)) -> [ai, bi]) contributions)
      in (scatter (copy re) flat_idx flat_re, scatter (copy im) flat_idx flat_im)
  in (re, im)

-- | Batched matrix multiply used by the fused linear+ReLU benchmark.
entry linear_relu [b][m][k] (x: [b][m]f32) (w: [m][k]f32) (bias: [k]f32) : [b][k]f32 =
  let wt = transpose w
  in map (\row ->
            map2 (\col bi -> f32.max 0f32 (f32.sum (map2 (*) row col) + bi)) wt bias)
         x

-- | Population count over a u32 array (mirrors the superoptimiser benchmark).
entry popcount_u32 [n] (xs: [n]u32) : [n]i32 =
  map (\x -> i32.u32 (u32.popc x)) xs

-- ==
-- entry: dot
-- input { [1.0f32, 2.0f32, 3.0f32] [4.0f32, 5.0f32, 6.0f32] }
-- output { 32.0f32 }

-- ==
-- entry: saxpy
-- input { 2.0f32 [1.0f32, 2.0f32] [3.0f32, 4.0f32] }
-- output { [5.0f32, 8.0f32] }

-- ==
-- entry: reduce_sum
-- input { [1.0f32, 2.0f32, 3.0f32, 4.0f32] }
-- output { 10.0f32 }

-- ==
-- entry: conv1d
-- input { [1.0f32, 2.0f32, 3.0f32] [1.0f32, 1.0f32, 1.0f32] }
-- output { [3.0f32, 6.0f32, 5.0f32] }
