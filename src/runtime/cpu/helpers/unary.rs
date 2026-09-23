//! Unary operation helpers for CPU tensors

use super::super::{CpuClient, CpuRuntime};
use crate::dispatch_dtype;
use crate::error::Result;
use crate::ops::{Kernel, UnaryOp};
use crate::runtime::{ensure_contiguous, row_layout};
use crate::tensor::Tensor;

/// Helper for unary operations (neg, abs, sqrt, exp, log, sin, cos, etc.)
pub fn unary_op_impl(
    client: &CpuClient,
    op: UnaryOp,
    a: &Tensor<CpuRuntime>,
    op_name: &'static str,
) -> Result<Tensor<CpuRuntime>> {
    let dtype = a.dtype();
    let a_contig = ensure_contiguous(a)?;
    let out = Tensor::<CpuRuntime>::empty(a.shape(), dtype, &client.device)?;

    let total = a.numel();
    let (rows, row_len) = row_layout(a.shape(), total);
    let a_ptr = a_contig.ptr();
    let out_ptr = out.ptr();

    dispatch_dtype!(dtype, T => {
        let a_ptr = a_ptr as *const T;
        let out_ptr = out_ptr as *mut T;
        unsafe {
            for row in 0..rows {
                let off = row * row_len;
                <CpuClient as Kernel<CpuRuntime>>::unary_op::<T>(
                    client, op,
                    a_ptr.add(off),
                    out_ptr.add(off),
                    row_len,
                );
            }
        }
    }, op_name);

    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::UnaryOps;
    use crate::runtime::Runtime;
    use crate::runtime::cpu::CpuDevice;

    /// Positive and negative values, not aligned to any particular SIMD
    /// lane width or period.
    fn row_data(w: usize, seed: i32) -> Vec<f32> {
        (0..w)
            .map(|i| ((i as i32 + seed) % 13 - 6) as f32 * 0.37)
            .collect()
    }

    /// Row `r` of a `[B, 1, W]` batched `exp` call must be bitwise identical
    /// to that row computed alone as `[1, 1, W]`. `W` in {8, 16, 17}
    /// straddles `SIMD_THRESHOLD` (32) once batched by `B` in {1, 2, 3, 5},
    /// and 17 is not a multiple of any SIMD lane width, so the scalar tail
    /// of the SIMD polynomial path is exercised too.
    #[test]
    fn test_exp_row_invariant() {
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);

        for &w in &[8usize, 16, 17] {
            for &b in &[1usize, 2, 3, 5] {
                let mut data = Vec::with_capacity(b * w);
                for r in 0..b {
                    data.extend(row_data(w, r as i32));
                }
                let batched = Tensor::<CpuRuntime>::from_slice(&data, &[b, 1, w], &device).unwrap();
                let batched_out: Vec<f32> = client.exp(&batched).unwrap().to_vec();

                for r in 0..b {
                    let row = &data[r * w..(r + 1) * w];
                    let solo = Tensor::<CpuRuntime>::from_slice(row, &[1, 1, w], &device).unwrap();
                    let solo_out: Vec<f32> = client.exp(&solo).unwrap().to_vec();
                    let batched_row = &batched_out[r * w..(r + 1) * w];

                    for i in 0..w {
                        assert_eq!(
                            batched_row[i].to_bits(),
                            solo_out[i].to_bits(),
                            "w={w} b={b} row={r} idx={i}: batched={} solo={}",
                            batched_row[i],
                            solo_out[i],
                        );
                    }
                }
            }
        }
    }
}
