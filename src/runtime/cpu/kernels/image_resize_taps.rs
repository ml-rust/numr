//! Per-axis windows, weights and tap sums of `resize` on the CPU.
//!
//! Each rule follows PyTorch's CPU source in `ATen/native/UpSample.h` and
//! `ATen/native/cpu/UpSampleKernel.cpp`, with the multiply-adds fused where
//! PyTorch's x86-64 AVX2 build fuses them:
//!
//! - Linear: `compute_source_index_and_lambda` with
//!   `area_pixel_compute_source_index(cubic = false)`, every step rounded.
//! - Cubic: `HelperInterpCubic::compute_indices_weights` with
//!   `get_cubic_upsample_coefficients` (`A = -0.75`). The source index is
//!   `fma(scale, i + 0.5, -0.5)`. The coefficients round every step.
//! - Antialias: `HelperInterpBase::_compute_indices_min_size_weights_aa`
//!   with `HelperInterpLinear::aa_filter` or `HelperInterpCubic::aa_filter`
//!   (`a = -0.5`, polynomial fused). For F32 the center, the window bounds
//!   and the filter argument run through f64, as the C++ double literals
//!   promote them.
//!
//! The CUDA kernel in `image_resize.cu` and the WGSL kernel in
//! `image_resize.wgsl` compute the same rules per output in-kernel.

use super::image_resize_acc::ResizeAcc;
use crate::ops::common::ResizeMode;

/// `std::max(a, b)`: `b` when `a < b`, else `a`.
#[inline(always)]
fn cmax<A: PartialOrd>(a: A, b: A) -> A {
    if a < b { b } else { a }
}

/// `std::min(a, b)`: `b` when `b < a`, else `a`.
#[inline(always)]
fn cmin<A: PartialOrd>(a: A, b: A) -> A {
    if b < a { b } else { a }
}

/// `cubic_convolution1(x, -0.75)`, every step rounded.
#[inline(always)]
fn cubic1<A: ResizeAcc>(x: A) -> A {
    let a = A::constant(-0.75);
    ((a + A::constant(2.0)) * x - (a + A::constant(3.0))) * x * x + A::constant(1.0)
}

/// `cubic_convolution2(x, -0.75)`, every step rounded.
#[inline(always)]
fn cubic2<A: ResizeAcc>(x: A) -> A {
    let a = A::constant(-0.75);
    ((a * x - A::constant(5.0) * a) * x + A::constant(8.0) * a) * x - A::constant(4.0) * a
}

/// `HelperInterpCubic::aa_filter` with `a = -0.5`, the polynomial fused.
#[inline(always)]
fn aa_cubic<A: ResizeAcc>(x: A) -> A {
    let x = x.abs();
    let one = A::constant(1.0);
    if x < one {
        let t = A::fma(A::constant(1.5), x, A::constant(-2.5));
        return A::fma(t * x, x, one);
    }
    if x < A::constant(2.0) {
        let u = A::fma(A::constant(-0.5), x, A::constant(2.5));
        let v = A::fma(u, x, A::constant(-4.0));
        return A::fma(v, x, A::constant(2.0));
    }
    A::constant(0.0)
}

/// `HelperInterpLinear::aa_filter`.
#[inline(always)]
fn aa_linear<A: ResizeAcc>(x: A) -> A {
    let x = x.abs();
    let one = A::constant(1.0);
    if x < one { one - x } else { A::constant(0.0) }
}

/// Source indices and weights of every output index along one axis.
#[derive(Debug, Clone)]
pub struct AxisTaps<A> {
    /// Taps of output `o` are `offsets[o]..offsets[o + 1]`.
    offsets: Vec<usize>,
    index: Vec<usize>,
    weight: Vec<A>,
}

impl<A: ResizeAcc> AxisTaps<A> {
    /// Taps for `mode` from `in_size` to `out_size`, both at least 1.
    /// [`ResizeMode::Copy`] has no taps.
    pub fn build(mode: ResizeMode, in_size: usize, out_size: usize) -> Self {
        let mut taps = Self {
            offsets: Vec::with_capacity(out_size + 1),
            index: Vec::new(),
            weight: Vec::new(),
        };
        taps.offsets.push(0);
        if mode == ResizeMode::Copy {
            return taps;
        }
        let scale = A::from_i64(in_size as i64) / A::from_i64(out_size as i64);
        for o in 0..out_size {
            match mode {
                ResizeMode::Linear => taps.push_linear(o, in_size, out_size, scale),
                ResizeMode::Cubic => taps.push_cubic(o, in_size, scale),
                ResizeMode::AaLinear => taps.push_aa(o, in_size, scale, 1.0, aa_linear::<A>),
                ResizeMode::AaCubic => taps.push_aa(o, in_size, scale, 2.0, aa_cubic::<A>),
                ResizeMode::Copy => {}
            }
            taps.offsets.push(taps.index.len());
        }
        taps
    }

    /// Source indices and weights of output `o`.
    #[inline(always)]
    pub fn get(&self, o: usize) -> (&[usize], &[A]) {
        let (lo, hi) = (self.offsets[o], self.offsets[o + 1]);
        (&self.index[lo..hi], &self.weight[lo..hi])
    }

    fn push(&mut self, index: i64, weight: A) {
        // Every pushed index is clamped into `[0, in_size)` by its caller.
        self.index.push(index as usize);
        self.weight.push(weight);
    }

    fn push_linear(&mut self, o: usize, in_size: usize, out_size: usize, scale: A) {
        let one = A::constant(1.0);
        let zero = A::constant(0.0);
        let half = A::constant(0.5);
        if in_size == out_size {
            self.push(o as i64, one);
            self.push(o as i64, zero);
            return;
        }
        let last = in_size as i64 - 1;
        let src = scale * (A::from_i64(o as i64) + half) - half;
        let src = if src < zero { zero } else { src };
        let i0 = src.floor_index().min(last);
        let lambda1 = cmin(cmax(src - A::from_i64(i0), zero), one);
        let i1 = i0 + i64::from(i0 < last);
        self.push(i0, one - lambda1);
        self.push(i1, lambda1);
    }

    fn push_cubic(&mut self, o: usize, in_size: usize, scale: A) {
        let one = A::constant(1.0);
        let half = A::constant(0.5);
        let last = in_size as i64 - 1;
        let src = A::fma(scale, A::from_i64(o as i64) + half, -half);
        let i = src.floor_index().min(last);
        let t = cmin(cmax(src - A::from_i64(i), A::constant(0.0)), one);
        let t2 = one - t;
        let coeffs = [cubic2(t + one), cubic1(t), cubic1(t2), cubic2(t2 + one)];
        for (j, w) in coeffs.into_iter().enumerate() {
            self.push((i + j as i64 - 1).clamp(0, last), w);
        }
    }

    fn push_aa(&mut self, o: usize, in_size: usize, scale: A, half_taps: f64, filter: fn(A) -> A) {
        let one = A::constant(1.0);
        let zero = A::constant(0.0);
        let wide = scale >= one;
        let support = if wide {
            A::constant(half_taps) * scale
        } else {
            A::constant(half_taps)
        };
        let max_taps = support.ceil_count() * 2 + 1;
        let invscale = if wide { one / scale } else { one };
        let center = A::aa_center(scale, o);
        let xmin = A::aa_bound(center - support).max(0);
        let xmax = A::aa_bound(center + support).min(in_size as i64);
        let xsize = (xmax - xmin).clamp(0, max_taps as i64);
        let first = self.weight.len();
        let mut total = zero;
        for k in xmin..xmin + xsize {
            let w = filter(A::aa_arg(A::from_i64(k) - center, invscale));
            self.push(k, w);
            total = total + w;
        }
        if total != zero {
            for w in &mut self.weight[first..] {
                *w = *w / total;
            }
        }
    }
}

/// Sum `x(j) * weights[j]` over the taps of one output with the rounding of
/// PyTorch's AVX2 CPU kernel.
///
/// - Plain filters: `fma(x0, w0, x1 * w1)`, then `fma(xj, wj, acc)`.
/// - Antialias: `x0 * w0`, then rounded products added for taps up to
///   `AA_BLOCK * floor((n - 1) / AA_BLOCK)`, then `fma(xj, wj, acc)`.
#[inline(always)]
pub fn sum_taps<A: ResizeAcc, F: Fn(usize) -> A>(antialias: bool, weights: &[A], x: F) -> A {
    let n = weights.len();
    if n == 0 {
        return A::constant(0.0);
    }
    if n == 1 {
        return x(0) * weights[0];
    }
    if antialias {
        let blocked = A::AA_BLOCK * ((n - 1) / A::AA_BLOCK);
        let mut acc = x(0) * weights[0];
        for (j, &w) in weights.iter().enumerate().skip(1) {
            acc = if j <= blocked {
                acc + x(j) * w
            } else {
                A::fma(x(j), w, acc)
            };
        }
        return acc;
    }
    let mut acc = A::fma(x(0), weights[0], x(1) * weights[1]);
    for (j, &w) in weights.iter().enumerate().skip(2) {
        acc = A::fma(x(j), w, acc);
    }
    acc
}

#[cfg(test)]
mod tests {
    use super::*;

    fn weights(mode: ResizeMode, in_size: usize, out_size: usize, o: usize) -> Vec<f32> {
        AxisTaps::<f32>::build(mode, in_size, out_size)
            .get(o)
            .1
            .to_vec()
    }

    #[test]
    fn linear_equal_sizes_use_one_pixel() {
        let t = AxisTaps::<f32>::build(ResizeMode::Linear, 5, 5);
        assert_eq!(t.get(3), (&[3usize, 3][..], &[1.0f32, 0.0][..]));
    }

    #[test]
    fn linear_halving_averages_pairs() {
        let t = AxisTaps::<f32>::build(ResizeMode::Linear, 8, 4);
        assert_eq!(t.get(1), (&[2usize, 3][..], &[0.5f32, 0.5][..]));
    }

    #[test]
    fn cubic_clamps_border_taps() {
        let t = AxisTaps::<f32>::build(ResizeMode::Cubic, 4, 8);
        let (idx, w) = t.get(0);
        assert_eq!(idx, &[0, 0, 0, 1]);
        let sum: f32 = w.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn antialias_weights_are_normalised_and_widen_on_downscale() {
        for mode in [ResizeMode::AaLinear, ResizeMode::AaCubic] {
            let w = weights(mode, 64, 5, 2);
            let sum: f32 = w.iter().sum();
            assert!((sum - 1.0).abs() < 1e-6, "{mode:?} sum {sum}");
            assert!(w.len() > 4, "{mode:?} has {} taps", w.len());
        }
    }

    #[test]
    fn antialias_equal_sizes_reduce_to_identity() {
        for mode in [ResizeMode::AaLinear, ResizeMode::AaCubic] {
            let t = AxisTaps::<f32>::build(mode, 6, 6);
            for o in 0..6 {
                let (idx, w) = t.get(o);
                for (&i, &wt) in idx.iter().zip(w) {
                    assert_eq!(wt, if i == o { 1.0 } else { 0.0 }, "{mode:?} o={o} i={i}");
                }
            }
        }
    }

    #[test]
    fn sums_fuse_where_pytorch_fuses() {
        let x = [0.1f32, 0.7, 0.3, 0.9, 0.2, 0.6];
        let w = [0.3f32, 0.2, 0.1, 0.15, 0.05, 0.2];
        let plain = sum_taps(false, &w[..2], |j| x[j]);
        assert_eq!(plain, x[0].mul_add(w[0], x[1] * w[1]));
        let aa = sum_taps(true, &w, |j| x[j]);
        let mut want = x[0] * w[0];
        for j in 1..=4 {
            want += x[j] * w[j];
        }
        want = x[5].mul_add(w[5], want);
        assert_eq!(aa, want);
    }
}
