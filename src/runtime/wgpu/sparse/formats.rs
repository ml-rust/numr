//! Format selection for the WebGPU high-level sparse operations.
//!
//! Each high-level operation picks the format its kernel reads. These
//! helpers convert on the GPU through the client's own conversion kernels,
//! so no index or value data leaves the device.

use super::super::shaders::{CompressedFormat, SparseMergeOp};
use super::super::{WgpuClient, WgpuRuntime};
use crate::error::{Error, Result};
use crate::sparse::{CooData, CscData, CsrData, SparseTensor};

impl WgpuClient {
    /// Return `a` as CSR, converting COO or CSC on the GPU.
    pub(super) fn gpu_csr(&self, a: &SparseTensor<WgpuRuntime>) -> Result<CsrData<WgpuRuntime>> {
        let (row_ptrs, col_indices, values) = match a {
            SparseTensor::Csr(data) => return Ok(data.clone()),
            SparseTensor::Coo(data) => crate::dispatch_dtype!(data.values.dtype(), T => {
                self.coo_to_csr_impl::<T>(&data.row_indices, &data.col_indices, &data.values, data.shape)
            }, "to CSR (WebGPU)")?,
            SparseTensor::Csc(data) => crate::dispatch_dtype!(data.values.dtype(), T => {
                self.csc_to_csr_impl::<T>(&data.col_ptrs, &data.row_indices, &data.values, data.shape)
            }, "to CSR (WebGPU)")?,
        };
        CsrData::new(row_ptrs, col_indices, values, a.shape())
    }

    /// Return `a` as CSC, converting COO or CSR on the GPU.
    pub(super) fn gpu_csc(&self, a: &SparseTensor<WgpuRuntime>) -> Result<CscData<WgpuRuntime>> {
        let (col_ptrs, row_indices, values) = match a {
            SparseTensor::Csc(data) => return Ok(data.clone()),
            SparseTensor::Coo(data) => crate::dispatch_dtype!(data.values.dtype(), T => {
                self.coo_to_csc_impl::<T>(&data.row_indices, &data.col_indices, &data.values, data.shape)
            }, "to CSC (WebGPU)")?,
            SparseTensor::Csr(data) => crate::dispatch_dtype!(data.values.dtype(), T => {
                self.csr_to_csc_impl::<T>(&data.row_ptrs, &data.col_indices, &data.values, data.shape)
            }, "to CSC (WebGPU)")?,
        };
        CscData::new(col_ptrs, row_indices, values, a.shape())
    }

    /// Element-wise `a op b` for sparse operands of any format. The result
    /// is CSR, as on the CPU backend.
    pub(super) fn sparse_merge_any(
        &self,
        a: &SparseTensor<WgpuRuntime>,
        b: &SparseTensor<WgpuRuntime>,
        op: SparseMergeOp,
    ) -> Result<SparseTensor<WgpuRuntime>> {
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
                CompressedFormat::Csr,
                op,
                (&a.row_ptrs, &a.col_indices, &a.values),
                (&b.row_ptrs, &b.col_indices, &b.values),
                a.shape[0],
            )
        }, "sparse merge (WebGPU)")?;
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
        a: &SparseTensor<WgpuRuntime>,
        f: impl FnOnce(
            &crate::tensor::Tensor<WgpuRuntime>,
        ) -> Result<crate::tensor::Tensor<WgpuRuntime>>,
    ) -> Result<SparseTensor<WgpuRuntime>> {
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
