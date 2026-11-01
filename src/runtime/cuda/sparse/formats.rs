//! Format selection for the CUDA high-level sparse operations.
//!
//! Each high-level operation picks the format its kernel reads. These
//! helpers convert on the GPU through the client's own conversion kernels,
//! so no index or value data leaves the device.

use super::super::{CudaClient, CudaRuntime};
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::runtime::cuda::kernels::{SparseFormat, SparseMergeOp};
use crate::sparse::{CooData, CscData, CsrData, SparseTensor};
use crate::tensor::Tensor;

impl CudaClient {
    /// Return `a` as CSR, converting COO or CSC on the GPU.
    pub(super) fn gpu_csr(&self, a: &SparseTensor<CudaRuntime>) -> Result<CsrData<CudaRuntime>> {
        let (row_ptrs, col_indices, values) = match a {
            SparseTensor::Csr(data) => return Ok(data.clone()),
            SparseTensor::Coo(data) => crate::dispatch_dtype!(data.values.dtype(), T => {
                self.coo_to_csr_impl::<T>(&data.row_indices, &data.col_indices, &data.values, data.shape)
            }, "to CSR (CUDA)")?,
            SparseTensor::Csc(data) => crate::dispatch_dtype!(data.values.dtype(), T => {
                self.csc_to_csr_impl::<T>(&data.col_ptrs, &data.row_indices, &data.values, data.shape)
            }, "to CSR (CUDA)")?,
        };
        CsrData::new(row_ptrs, col_indices, values, a.shape())
    }

    /// Return `a` as CSC, converting COO or CSR on the GPU.
    pub(super) fn gpu_csc(&self, a: &SparseTensor<CudaRuntime>) -> Result<CscData<CudaRuntime>> {
        let (col_ptrs, row_indices, values) = match a {
            SparseTensor::Csc(data) => return Ok(data.clone()),
            SparseTensor::Coo(data) => crate::dispatch_dtype!(data.values.dtype(), T => {
                self.coo_to_csc_impl::<T>(&data.row_indices, &data.col_indices, &data.values, data.shape)
            }, "to CSC (CUDA)")?,
            SparseTensor::Csr(data) => crate::dispatch_dtype!(data.values.dtype(), T => {
                self.csr_to_csc_impl::<T>(&data.row_ptrs, &data.col_indices, &data.values, data.shape)
            }, "to CSC (CUDA)")?,
        };
        CscData::new(col_ptrs, row_indices, values, a.shape())
    }

    /// Element-wise `a op b` for sparse operands of any format. The result
    /// is CSR, as on the CPU backend.
    pub(super) fn sparse_merge_any(
        &self,
        a: &SparseTensor<CudaRuntime>,
        b: &SparseTensor<CudaRuntime>,
        op: SparseMergeOp,
    ) -> Result<SparseTensor<CudaRuntime>> {
        if a.shape() != b.shape() {
            return Err(Error::ShapeMismatch {
                expected: a.shape().to_vec(),
                got: b.shape().to_vec(),
            });
        }
        if a.dtype() != b.dtype() {
            return Err(Error::DTypeMismatch {
                lhs: a.dtype(),
                rhs: b.dtype(),
            });
        }
        let a = self.gpu_csr(a)?;
        let b = self.gpu_csr(b)?;
        let (row_ptrs, col_indices, values) = crate::dispatch_dtype!(a.values.dtype(), T => {
            self.compressed_merge::<T>(
                SparseFormat::Csr,
                op,
                (&a.row_ptrs, &a.col_indices, &a.values),
                (&b.row_ptrs, &b.col_indices, &b.values),
                a.shape[0],
            )
        }, "sparse merge (CUDA)")?;
        Ok(SparseTensor::Csr(CsrData::new(
            row_ptrs,
            col_indices,
            values,
            a.shape,
        )?))
    }

    /// Return `a` with `f` applied to its stored values. The pattern stays.
    pub(super) fn map_sparse_values(
        &self,
        a: &SparseTensor<CudaRuntime>,
        f: impl FnOnce(&Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>>,
    ) -> Result<SparseTensor<CudaRuntime>> {
        Ok(match a {
            SparseTensor::Csr(d) => SparseTensor::Csr(CsrData::new(
                d.row_ptrs.clone(),
                d.col_indices.clone(),
                f(&d.values)?,
                d.shape,
            )?),
            SparseTensor::Csc(d) => SparseTensor::Csc(CscData::new(
                d.col_ptrs.clone(),
                d.row_indices.clone(),
                f(&d.values)?,
                d.shape,
            )?),
            SparseTensor::Coo(d) => {
                let mut coo = CooData::new(
                    d.row_indices.clone(),
                    d.col_indices.clone(),
                    f(&d.values)?,
                    d.shape,
                )?;
                // SAFETY: the indices are unchanged, so the order flag holds.
                unsafe { coo.set_sorted(d.sorted) };
                SparseTensor::Coo(coo)
            }
        })
    }
}

/// Smallest value of `dtype` whose `f64` value is `>= threshold`, as `f64`.
///
/// The dense keep test `|v| >= threshold` runs in the value dtype on the GPU.
/// Rounding the host threshold up keeps exactly the entries the CPU keeps,
/// which compares in `f64`. A NaN threshold stays NaN and keeps no entry.
pub(super) fn threshold_at_least(dtype: DType, threshold: f64) -> f64 {
    if threshold.is_nan() || threshold <= 0.0 {
        return threshold;
    }
    match dtype {
        DType::F32 => {
            let t = threshold as f32;
            f64::from(if f64::from(t) < threshold {
                t.next_up()
            } else {
                t
            })
        }
        #[cfg(feature = "f16")]
        DType::F16 => {
            let t = half::f16::from_f64(threshold);
            let up = if t.to_f64() < threshold {
                half::f16::from_bits(t.to_bits() + 1)
            } else {
                t
            };
            up.to_f64()
        }
        #[cfg(feature = "f16")]
        DType::BF16 => {
            let t = half::bf16::from_f64(threshold);
            let up = if t.to_f64() < threshold {
                half::bf16::from_bits(t.to_bits() + 1)
            } else {
                t
            };
            up.to_f64()
        }
        _ => threshold,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn thresholds_round_up_in_the_value_dtype() {
        let t = threshold_at_least(DType::F32, 0.1);
        assert!(t >= 0.1);
        assert!(f64::from((t as f32).next_down()) < 0.1);
        #[cfg(feature = "f16")]
        {
            let h = threshold_at_least(DType::F16, 0.1);
            assert!(h >= 0.1);
            assert_eq!(half::f16::from_f64(h).to_f64(), h);
        }
        assert_eq!(threshold_at_least(DType::F64, 0.1), 0.1);
        assert_eq!(threshold_at_least(DType::F32, -2.0), -2.0);
        assert!(threshold_at_least(DType::F32, f64::NAN).is_nan());
    }
}
