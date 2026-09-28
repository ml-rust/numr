//! Shared loop nest for the SIMD `conv1d` kernels.
//!
//! # Why output positions, and not input channels
//!
//! The tensor layout is `(batch, channels, length)`, so two neighbouring input
//! CHANNELS sit `length` elements apart. Vectorising the channel axis therefore
//! cannot use a vector load at all — it has to pack the lanes one scalar at a
//! time (a manual gather of BOTH input and weight), reduce into a single
//! accumulator, and finish with a horizontal sum. It also degenerates to
//! nothing at all for a depthwise convolution, where `c_in_per_group == 1`.
//!
//! Neighbouring OUTPUT POSITIONS, by contrast, read neighbouring input
//! elements. Fixing `(b, g, oc)` and reducing over `(kx, ic)` while vectorising
//! `ox` turns the weight into a scalar broadcast (`set1`) and the input into a
//! contiguous vector load: directly for `stride == 1`, and from per-phase
//! copies of the input for `stride > 1`. Depthwise vectorises fully.
//!
//! # The interior/boundary split
//!
//! The tap index is
//!
//! ```text
//! ix(ox, kx) = ox * stride + kx * dilation - pad_left
//! ```
//!
//! which is monotonically non-decreasing in `kx`. So "every tap of this `ox` is
//! in bounds" needs checking only at the two ends, `kx = 0` and `kx = K - 1`:
//!
//! ```text
//! ix(ox, 0)     >= 0        <=>  ox * stride >= pad_left
//!                           <=>  ox >= ceil(pad_left / stride)              = interior_lo
//! ix(ox, K - 1) <= length-1 <=>  ox * stride <= length - 1 + pad_left - (K-1)*dilation
//!                           <=>  ox <  ceil((length + pad_left - (K-1)*dilation) / stride)
//!                                                                           = interior_hi
//! ```
//!
//! (The second equivalence uses `floor((M - 1) / s) + 1 == ceil(M / s)` for
//! `M >= 1`. When `M = length + pad_left - (K-1)*dilation` is zero or negative —
//! a kernel whose dilated span exceeds the padded input — NO output position is
//! fully interior, so `interior_hi` is clamped to 0. That subtraction is done in
//! `isize` precisely so `usize` cannot wrap here.)
//!
//! `ox` in `[0, interior_lo)` and `[interior_hi, output_length)` runs scalar,
//! keeping the per-tap `0 <= ix < length` check. Those two edges are together at
//! most `(K-1)*dilation / stride + 1` positions wide, so vectorising them would
//! buy nothing. `ox` in `[interior_lo, interior_hi)` runs vectorised with NO
//! per-tap bounds check at all.
//!
//! # Strided input: phase buffers
//!
//! For `stride > 1`, neighbouring output positions read input elements
//! `stride` apart, so a vector of them cannot be one contiguous load. Write the
//! tap offset `d = kx * dilation` as `q * stride + r` with `r < stride`. Then
//!
//! ```text
//! x[ix0 + j * stride + d] = x[ix0 + (j + q) * stride + r] = P_r[j + q]
//! ```
//!
//! where phase `P_r[t] = x[ix0 + t * stride + r]`. Once per `(batch, group)`
//! the driver de-interleaves every input row of the group into its `stride`
//! phases, each `n + (K-1)*dilation / stride` long. The interior kernel then
//! reads tap `(ic, kx)` for output `j` at `row(ic) + tap(kx) + j`, contiguous
//! in `j`, with `tap(kx) = r * phase_len + q`. For `stride == 1` the rows are
//! the input itself and `tap(kx) = kx * dilation`, so one kernel body serves
//! both cases.
//!
//! # Accumulation order
//!
//! Within a lane the reduction is `ic` outer, `kx` inner — the same order as the
//! scalar kernel — and the bias is added once, after the full reduction. The
//! only numerical difference from scalar is FMA contraction of the multiply-add.
//! Phase buffers hold copies of the same input values, and each accumulator
//! owns its own output positions, so neither the phase split nor the number of
//! accumulators changes any output bit relative to a gathered load.

/// Expands the `conv1d` loop nest for one dtype.
///
/// The ISA-specific block receives the fully interior output run for one
/// `(batch, group, out-channel)` triple and must compute all `n` of its outputs:
///
/// - `$op`: `*mut $ty`, output for `ox = interior_lo`, `n` contiguous elements
/// - `$ip`: `*const $ty`, row base; the element for `(j, ic, kx)` is
///   `$ip[ic * $rs + $taps[kx] + j]`
/// - `$wp`: `*const $ty`, weight at `(c_out_idx, 0, 0)`; the element for
///   `(ic, kx)` is `$wp[ic * kernel_size + kx]`
/// - `$n`: number of interior output positions (always `>= 1`)
/// - `$nic`: `c_in_per_group`
/// - `$bv`: the bias value for this output channel (zero when there is no bias)
/// - `$rs`: distance between the rows of consecutive input channels
/// - `$taps`: `*const usize`, `kernel_size` tap offsets within a row
///
/// For `stride == 1`, `$ip` points into the input and `$rs == length`. For
/// `stride > 1` it points into the phase buffers (see the module docs). Every
/// index above is in bounds by construction, so the block must not
/// bounds-check taps. `kernel_size` is read by the block from its own `params`
/// argument (macro hygiene keeps the locals expanded here invisible to it).
macro_rules! conv1d_body {
    (
        $ty:ty,
        $input:expr, $weight:expr, $bias:expr, $output:expr, $params:expr,
        |$op:ident, $ip:ident, $wp:ident, $n:ident, $nic:ident, $bv:ident, $rs:ident, $taps:ident|
        $interior:block
    ) => {{
        let input: *const $ty = $input;
        let weight: *const $ty = $weight;
        let bias: Option<*const $ty> = $bias;
        let output: *mut $ty = $output;

        let crate::ops::conv_common::Conv1dParams {
            batch,
            c_in,
            length,
            c_out,
            kernel_size,
            stride,
            dilation,
            groups,
            pad_left,
            output_length,
            ..
        } = $params;

        let c_in_per_group = c_in / groups;
        let c_out_per_group = c_out / groups;

        // Interior window, derived once for the whole call (see module docs).
        // `stride == 0` is not a valid convolution; treat the whole output as
        // boundary rather than dividing by zero.
        let (interior_lo, interior_hi) = if stride == 0 {
            (0usize, 0usize)
        } else {
            let span = kernel_size.saturating_sub(1) * dilation;
            let reach = (length + pad_left) as isize - span as isize;
            let hi = if reach <= 0 {
                0
            } else {
                (reach as usize).div_ceil(stride).min(output_length)
            };
            (pad_left.div_ceil(stride).min(hi), hi)
        };

        // Phase-buffer geometry, fixed for the whole call (see module docs).
        let n_interior = interior_hi.saturating_sub(interior_lo);
        let phased = stride > 1 && n_interior > 0;
        let span = kernel_size.saturating_sub(1) * dilation;
        let phase_len = if phased {
            n_interior + span / stride
        } else {
            0
        };
        let row_stride = if phased { stride * phase_len } else { length };
        let taps: Vec<usize> = (0..kernel_size)
            .map(|kx| {
                let d = kx * dilation;
                if phased {
                    (d % stride) * phase_len + d / stride
                } else {
                    d
                }
            })
            .collect();
        let mut phases: Vec<$ty> = if phased {
            vec![<$ty>::default(); c_in_per_group * row_stride]
        } else {
            Vec::new()
        };

        for b in 0..batch {
            for g in 0..groups {
                let c_in_start = g * c_in_per_group;
                let c_out_start = g * c_out_per_group;
                let in_base = (b * c_in + c_in_start) * length;

                // De-interleave this group's input rows once for all its output
                // channels. Entries past the input end are never read by a tap;
                // they are zero-filled only so the buffer is fully initialised.
                if phased {
                    let ix0 = interior_lo * stride - pad_left;
                    for ic in 0..c_in_per_group {
                        let src = input.add(in_base + ic * length);
                        let dst = phases.as_mut_ptr().add(ic * row_stride);
                        for r in 0..stride {
                            for t in 0..phase_len {
                                let ix = ix0 + t * stride + r;
                                *dst.add(r * phase_len + t) = if ix < length {
                                    *src.add(ix)
                                } else {
                                    <$ty>::default()
                                };
                            }
                        }
                    }
                }

                for oc in 0..c_out_per_group {
                    let c_out_idx = c_out_start + oc;
                    let w_base = c_out_idx * c_in_per_group * kernel_size;
                    let out_base = (b * c_out + c_out_idx) * output_length;
                    let bias_val = match bias {
                        Some(p) => *p.add(c_out_idx),
                        None => <$ty>::default(),
                    };

                    // Boundary positions: some tap may fall outside the input,
                    // so every tap keeps its own bounds check.
                    for ox in (0..interior_lo).chain(interior_hi..output_length) {
                        let mut sum = <$ty>::default();
                        for ic in 0..c_in_per_group {
                            let x_row = in_base + ic * length;
                            let w_row = w_base + ic * kernel_size;
                            for kx in 0..kernel_size {
                                let ix = (ox * stride) as isize + (kx * dilation) as isize
                                    - pad_left as isize;
                                if ix >= 0 && (ix as usize) < length {
                                    sum +=
                                        *input.add(x_row + ix as usize) * *weight.add(w_row + kx);
                                }
                            }
                        }
                        *output.add(out_base + ox) = sum + bias_val;
                    }

                    if interior_hi > interior_lo {
                        // `interior_lo * stride >= pad_left` by construction, so
                        // this offset is non-negative.
                        let ix0 = interior_lo * stride - pad_left;
                        let $op = output.add(out_base + interior_lo);
                        let $ip: *const $ty = if phased {
                            phases.as_ptr()
                        } else {
                            input.add(in_base + ix0)
                        };
                        let $wp = weight.add(w_base);
                        let $n = n_interior;
                        let $nic = c_in_per_group;
                        let $bv = bias_val;
                        let $rs = row_stride;
                        let $taps: *const usize = taps.as_ptr();
                        $interior
                    }
                }
            }
        }
    }};
}

/// Expands the interior kernel of one ISA and dtype.
///
/// Four accumulators cover `4 * lanes` neighbouring outputs, so four
/// independent FMA chains stay in flight through the `(ic, kx)` reduction. A
/// one-accumulator loop takes the remaining full vectors and a scalar loop the
/// last `n % lanes` outputs. Every output keeps the `ic`-outer, `kx`-inner
/// order whichever loop computes it.
///
/// - `splat`, `load`, `store`, `add`: the ISA's broadcast, unaligned load,
///   unaligned store and vector add
/// - `fma = |acc, x, w| expr`: returns `acc + x * w` with one rounding
/// - the trailing arguments are the `conv1d_body!` block arguments plus
///   `kernel_size`
macro_rules! conv1d_interior {
    (
        $ty:ty, lanes = $lanes:expr, zero = $zero:expr,
        splat = $splat:path, load = $load:path, store = $store:path, add = $add:path,
        fma = |$a:ident, $x:ident, $w:ident| $fma:expr,
        $op:expr, $ip:expr, $wp:expr, $n:expr, $nic:expr, $bv:expr, $rs:expr, $taps:expr,
        $ks:expr
    ) => {{
        let (op, ip, wp, n, nic, bv, rs, taps, ks) =
            ($op, $ip, $wp, $n, $nic, $bv, $rs, $taps, $ks);
        let lanes: usize = $lanes;
        let bias_vec = $splat(bv);
        let mut j = 0usize;

        while j + 4 * lanes <= n {
            let (mut acc0, mut acc1, mut acc2, mut acc3) = ($zero, $zero, $zero, $zero);
            for ic in 0..nic {
                let x_row = ip.add(ic * rs + j);
                let w_row = wp.add(ic * ks);
                for kx in 0..ks {
                    let $w = $splat(*w_row.add(kx));
                    let xb = x_row.add(*taps.add(kx));
                    acc0 = {
                        let ($a, $x) = (acc0, $load(xb));
                        $fma
                    };
                    acc1 = {
                        let ($a, $x) = (acc1, $load(xb.add(lanes)));
                        $fma
                    };
                    acc2 = {
                        let ($a, $x) = (acc2, $load(xb.add(2 * lanes)));
                        $fma
                    };
                    acc3 = {
                        let ($a, $x) = (acc3, $load(xb.add(3 * lanes)));
                        $fma
                    };
                }
            }
            $store(op.add(j), $add(acc0, bias_vec));
            $store(op.add(j + lanes), $add(acc1, bias_vec));
            $store(op.add(j + 2 * lanes), $add(acc2, bias_vec));
            $store(op.add(j + 3 * lanes), $add(acc3, bias_vec));
            j += 4 * lanes;
        }

        while j + lanes <= n {
            let mut acc0 = $zero;
            for ic in 0..nic {
                let x_row = ip.add(ic * rs + j);
                let w_row = wp.add(ic * ks);
                for kx in 0..ks {
                    let $w = $splat(*w_row.add(kx));
                    acc0 = {
                        let ($a, $x) = (acc0, $load(x_row.add(*taps.add(kx))));
                        $fma
                    };
                }
            }
            $store(op.add(j), $add(acc0, bias_vec));
            j += lanes;
        }

        while j < n {
            let mut sum = <$ty>::default();
            for ic in 0..nic {
                let x_row = ip.add(ic * rs + j);
                let w_row = wp.add(ic * ks);
                for kx in 0..ks {
                    sum += *x_row.add(*taps.add(kx)) * *w_row.add(kx);
                }
            }
            *op.add(j) = sum + bv;
            j += 1;
        }
    }};
}

pub(super) use conv1d_body;
pub(super) use conv1d_interior;

#[cfg(test)]
mod tests {
    use super::super::dispatch::conv1d_f32;
    use crate::dtype::DType;
    use crate::ops::PaddingMode;
    use crate::ops::conv_common::validate_conv1d;
    use crate::runtime::cpu::kernels::simd::conv::scalar::conv1d_scalar_f32;

    /// Checks one strided shape two ways.
    ///
    /// - Bits: every output equals the `ic`-outer, `kx`-inner reduction either
    ///   fused (`mul_add`, the vector lanes) or unfused (`+=`, the scalar edges
    ///   and tail), then `+ bias`. That is the arithmetic the gathered-load
    ///   kernel performed, so a phase-buffer or accumulator indexing error fails.
    /// - Tolerance: within `1e-5` relative (plus `1e-6` absolute for
    ///   cancellation) of the scalar reference kernel.
    #[allow(clippy::too_many_arguments)]
    fn check(
        c_in: usize,
        length: usize,
        c_out_per_group: usize,
        kernel: usize,
        stride: usize,
        dilation: usize,
        groups: usize,
        pad: usize,
    ) {
        let batch = 2;
        let c_out = c_out_per_group * groups;
        let c_in_per_group = c_in / groups;
        let input: Vec<f32> = (0..batch * c_in * length)
            .map(|x| (((x * 37) % 61) as f32) * 0.031 - 0.93)
            .collect();
        let weight: Vec<f32> = (0..c_out * c_in_per_group * kernel)
            .map(|x| (((x * 23) % 47) as f32) * 0.037 - 0.85)
            .collect();
        let bias: Vec<f32> = (0..c_out).map(|x| x as f32 * 0.5 + 1.0).collect();
        let params = validate_conv1d(
            &[batch, c_in, length],
            &[c_out, c_in_per_group, kernel],
            Some(&[c_out][..]),
            stride,
            PaddingMode::Custom(pad, pad, 0, 0),
            dilation,
            groups,
            DType::F32,
            DType::F32,
            Some(DType::F32),
        )
        .expect("valid test shapes");
        let ol = params.output_length;
        let total = batch * c_out * ol;

        let mut got = vec![0.0f32; total];
        let mut scalar = vec![0.0f32; total];
        unsafe {
            conv1d_f32(
                input.as_ptr(),
                weight.as_ptr(),
                Some(bias.as_ptr()),
                got.as_mut_ptr(),
                params,
            );
            conv1d_scalar_f32(
                input.as_ptr(),
                weight.as_ptr(),
                Some(bias.as_ptr()),
                scalar.as_mut_ptr(),
                params,
            );
        }

        let label =
            format!("c_in={c_in} len={length} k={kernel} s={stride} d={dilation} g={groups}");
        for b in 0..batch {
            for oc in 0..c_out {
                let g = oc / c_out_per_group;
                for ox in 0..ol {
                    let (mut fused, mut plain) = (0.0f32, 0.0f32);
                    for ic in 0..c_in_per_group {
                        let x_row = (b * c_in + g * c_in_per_group + ic) * length;
                        let w_row = (oc * c_in_per_group + ic) * kernel;
                        for kx in 0..kernel {
                            let ix = (ox * stride + kx * dilation) as isize - pad as isize;
                            if ix >= 0 && (ix as usize) < length {
                                let (x, w) = (input[x_row + ix as usize], weight[w_row + kx]);
                                fused = x.mul_add(w, fused);
                                plain += x * w;
                            }
                        }
                    }
                    let (fused, plain) = (fused + bias[oc], plain + bias[oc]);
                    let i = (b * c_out + oc) * ol + ox;
                    assert!(
                        got[i].to_bits() == fused.to_bits() || got[i].to_bits() == plain.to_bits(),
                        "{label}: order changed at {i}: got {} fused {fused} plain {plain}",
                        got[i]
                    );
                    let diff = (got[i] - scalar[i]).abs();
                    assert!(
                        diff <= 1e-6 + 1e-5 * scalar[i].abs(),
                        "{label}: {i}: got {} scalar {}",
                        got[i],
                        scalar[i]
                    );
                }
            }
        }
    }

    #[test]
    fn test_phase_buffers_stride2_dense() {
        check(4, 400, 3, 3, 2, 1, 1, 1);
    }

    #[test]
    fn test_phase_buffers_stride2_depthwise() {
        check(6, 400, 1, 4, 2, 1, 6, 1);
    }

    #[test]
    fn test_phase_buffers_stride3_dilated() {
        // Dilation 2 with stride 3 spreads taps over every phase.
        check(3, 300, 2, 5, 3, 2, 1, 4);
    }

    #[test]
    fn test_phase_buffers_stride4_grouped() {
        check(8, 520, 2, 7, 4, 1, 2, 3);
    }

    #[test]
    fn test_stride1_four_accumulators() {
        // Long enough for the 4-accumulator loop at every ISA width, with a tail.
        check(2, 301, 2, 3, 1, 1, 1, 1);
        check(4, 263, 1, 5, 1, 3, 4, 6);
    }
}
