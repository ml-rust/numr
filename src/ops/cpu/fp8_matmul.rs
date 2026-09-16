//! CPU implementation of FP8 matrix multiplication operations.
//!
//! Fused kernel: reads FP8, converts to F32 inline during accumulation,
//! applies scaling, and writes output in the target dtype. No intermediate
//! tensor allocations.

use crate::dtype::{DType, FP8E4M3, FP8E5M2};
use crate::error::{Error, Result};
use crate::ops::Fp8MatmulOps;
use crate::ops::matmul::{matmul_dims_and_batches, matmul_output_shape};
use crate::runtime::cpu::{CpuClient, CpuRuntime};
use crate::tensor::Tensor;

/// Validate FP8 matmul arguments.
fn validate_fp8_matmul(
    a: &Tensor<CpuRuntime>,
    b: &Tensor<CpuRuntime>,
    expected_a_dtype: DType,
    expected_b_dtype: DType,
    out_dtype: DType,
) -> Result<Fp8Geometry> {
    if a.dtype() != expected_a_dtype {
        return Err(Error::DTypeMismatch {
            lhs: a.dtype(),
            rhs: expected_a_dtype,
        });
    }
    if b.dtype() != expected_b_dtype {
        return Err(Error::DTypeMismatch {
            lhs: b.dtype(),
            rhs: expected_b_dtype,
        });
    }
    match out_dtype {
        DType::F32 | DType::F16 | DType::BF16 => {}
        _ => {
            return Err(Error::UnsupportedDType {
                dtype: out_dtype,
                op: "fp8_matmul output",
            });
        }
    }
    let a_shape = a.shape();
    let b_shape = b.shape();
    // Rank-1 operands are refused: the FP8 kernels take matrices only.
    if a_shape.len() < 2 || b_shape.len() < 2 {
        return Err(Error::ShapeMismatch {
            expected: a_shape.to_vec(),
            got: b_shape.to_vec(),
        });
    }
    let out_shape = matmul_output_shape(a_shape, b_shape).ok_or(Error::ShapeMismatch {
        expected: a_shape.to_vec(),
        got: b_shape.to_vec(),
    })?;

    // Geometry and per-operand batch indices come from the shared matmul rule.
    // `matmul_output_shape` already checked `k` against `b`.
    Ok(Fp8Geometry::new(a_shape, b_shape, out_shape))
}

/// Output shape, kernel geometry and per-operand batch indices of one call.
struct Fp8Geometry {
    out_shape: Vec<usize>,
    m: usize,
    k: usize,
    n: usize,
    a_batch_idx: Vec<usize>,
    b_batch_idx: Vec<usize>,
}

impl Fp8Geometry {
    fn new(a_shape: &[usize], b_shape: &[usize], out_shape: Vec<usize>) -> Self {
        let (m, k, n, _, a_batch_idx, b_batch_idx) =
            matmul_dims_and_batches(a_shape, b_shape, &out_shape);
        Self {
            out_shape,
            m,
            k,
            n,
            a_batch_idx,
            b_batch_idx,
        }
    }
}

/// Fused FP8 matmul kernel: converts FP8→F32 inline during multiply-accumulate,
/// applies combined scale, writes output directly in target dtype.
///
/// `convert_a` and `convert_b` are FP8→f32 conversion functions.
fn fused_fp8_matmul_kernel(
    a_ptr: *const u8,
    b_ptr: *const u8,
    out_ptr: u64,
    convert_a: fn(u8) -> f32,
    convert_b: fn(u8) -> f32,
    combined_scale: f32,
    out_dtype: DType,
    geom: &Fp8Geometry,
) {
    let (m, k, n) = (geom.m, geom.k, geom.n);
    let out_batch_stride = m * n;

    // Batch dims broadcast per dimension, so each output batch reads its own
    // source batch per operand rather than a wrapping batch count.
    for (batch, (&a_idx, &b_idx)) in geom.a_batch_idx.iter().zip(&geom.b_batch_idx).enumerate() {
        let a_base = unsafe { a_ptr.add(a_idx * m * k) };
        let b_base = unsafe { b_ptr.add(b_idx * k * n) };

        for i in 0..m {
            for j in 0..n {
                let mut acc: f32 = 0.0;
                for p in 0..k {
                    let a_val = convert_a(unsafe { *a_base.add(i * k + p) });
                    let b_val = convert_b(unsafe { *b_base.add(p * n + j) });
                    acc += a_val * b_val;
                }
                acc *= combined_scale;

                let out_idx = batch * out_batch_stride + i * n + j;
                match out_dtype {
                    DType::F32 => unsafe {
                        let ptr = out_ptr as *mut f32;
                        *ptr.add(out_idx) = acc;
                    },
                    #[cfg(feature = "f16")]
                    DType::F16 => unsafe {
                        let ptr = out_ptr as *mut half::f16;
                        *ptr.add(out_idx) = half::f16::from_f32(acc);
                    },
                    #[cfg(feature = "f16")]
                    DType::BF16 => unsafe {
                        let ptr = out_ptr as *mut half::bf16;
                        *ptr.add(out_idx) = half::bf16::from_f32(acc);
                    },
                    _ => {} // validated above
                }
            }
        }
    }
}

impl Fp8MatmulOps<CpuRuntime> for CpuClient {
    fn fp8_matmul(
        &self,
        a: &Tensor<CpuRuntime>,
        b: &Tensor<CpuRuntime>,
        scale_a: f32,
        scale_b: f32,
        out_dtype: DType,
    ) -> Result<Tensor<CpuRuntime>> {
        let geom = validate_fp8_matmul(a, b, DType::FP8E4M3, DType::FP8E4M3, out_dtype)?;

        let a_contig = crate::runtime::cpu::helpers::ensure_contiguous(a)?;
        let b_contig = crate::runtime::cpu::helpers::ensure_contiguous(b)?;
        let out = Tensor::<CpuRuntime>::empty(&geom.out_shape, out_dtype, &self.device)?;

        fused_fp8_matmul_kernel(
            a_contig.ptr() as *const u8,
            b_contig.ptr() as *const u8,
            out.ptr(),
            |byte| FP8E4M3::from_bits(byte).to_f32(),
            |byte| FP8E4M3::from_bits(byte).to_f32(),
            scale_a * scale_b,
            out_dtype,
            &geom,
        );

        Ok(out)
    }

    fn fp8_matmul_e5m2(
        &self,
        a: &Tensor<CpuRuntime>,
        b: &Tensor<CpuRuntime>,
        scale_a: f32,
        scale_b: f32,
        out_dtype: DType,
    ) -> Result<Tensor<CpuRuntime>> {
        let geom = validate_fp8_matmul(a, b, DType::FP8E5M2, DType::FP8E4M3, out_dtype)?;

        let a_contig = crate::runtime::cpu::helpers::ensure_contiguous(a)?;
        let b_contig = crate::runtime::cpu::helpers::ensure_contiguous(b)?;
        let out = Tensor::<CpuRuntime>::empty(&geom.out_shape, out_dtype, &self.device)?;

        fused_fp8_matmul_kernel(
            a_contig.ptr() as *const u8,
            b_contig.ptr() as *const u8,
            out.ptr(),
            |byte| FP8E5M2::from_bits(byte).to_f32(),
            |byte| FP8E4M3::from_bits(byte).to_f32(),
            scale_a * scale_b,
            out_dtype,
            &geom,
        );

        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::TypeConversionOps;
    use crate::runtime::cpu::CpuDevice;

    fn fp8(client: &CpuClient, data: &[f32], shape: &[usize]) -> Tensor<CpuRuntime> {
        let t = Tensor::<CpuRuntime>::from_slice(data, shape, &client.device).expect("f32 tensor");
        client.cast(&t, DType::FP8E4M3).expect("cast to FP8")
    }

    /// The FP8 kernels take matrices only: a rank-1 operand on either side is
    /// refused, in both entry points.
    #[test]
    fn test_fp8_matmul_rejects_rank1_operands() {
        let client = CpuClient::new(CpuDevice::new());
        let mat = fp8(&client, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let vec3 = fp8(&client, &[1.0, 2.0, 3.0], &[3]);
        let vec2 = fp8(&client, &[1.0, 2.0], &[2]);

        assert!(
            client
                .fp8_matmul(&mat, &vec3, 1.0, 1.0, DType::F32)
                .is_err()
        );
        assert!(
            client
                .fp8_matmul(&vec2, &mat, 1.0, 1.0, DType::F32)
                .is_err()
        );
        let a5 = client.cast(&mat, DType::FP8E5M2).expect("cast to FP8E5M2");
        assert!(
            client
                .fp8_matmul_e5m2(&a5, &vec3, 1.0, 1.0, DType::F32)
                .is_err()
        );
    }

    /// A batched `a` against an unbatched `b`: every batch reads the one `b`,
    /// rather than striding past its end.
    #[test]
    fn test_fp8_matmul_broadcasts_unbatched_b() {
        let client = CpuClient::new(CpuDevice::new());
        let a_data: Vec<f32> = (0..12).map(|i| (i % 5) as f32 - 2.0).collect();
        let b_data = [1.0f32, -1.0, 2.0, 0.5, -2.0, 1.0];
        let a = fp8(&client, &a_data, &[2, 2, 3]);
        let b = fp8(&client, &b_data, &[3, 2]);

        let out = client
            .fp8_matmul(&a, &b, 1.0, 1.0, DType::F32)
            .expect("batched fp8_matmul");
        assert_eq!(out.shape(), &[2, 2, 2]);
        let got = out.to_vec::<f32>();

        for batch in 0..2 {
            let a_slice = fp8(&client, &a_data[batch * 6..(batch + 1) * 6], &[2, 3]);
            let want = client
                .fp8_matmul(&a_slice, &b, 1.0, 1.0, DType::F32)
                .expect("single fp8_matmul")
                .to_vec::<f32>();
            assert_eq!(&got[batch * 4..(batch + 1) * 4], &want[..], "batch {batch}");
        }
    }
}
