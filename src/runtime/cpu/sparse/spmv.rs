//! CSR sparse matrix-vector and matrix-matrix multiplication (`spmv_csr`, `spmm_csr`).

use super::{CpuClient, CpuRuntime};
use crate::algorithm::sparse::validate_sparse_operands;
use crate::dtype::Element;
use crate::error::{Error, Result};
use crate::tensor::Tensor;

/// `y = A @ x` for a CSR matrix `A` and dense vector `x`.
pub(super) fn spmv_csr_impl<T: Element>(
    _client: &CpuClient,
    row_ptrs: &Tensor<CpuRuntime>,
    col_indices: &Tensor<CpuRuntime>,
    values: &Tensor<CpuRuntime>,
    x: &Tensor<CpuRuntime>,
    shape: [usize; 2],
) -> Result<Tensor<CpuRuntime>> {
    let [nrows, ncols] = shape;
    let device = values.device();

    validate_sparse_operands::<T>(
        &[row_ptrs.dtype(), col_indices.dtype()],
        &[values.dtype(), x.dtype()],
    )?;

    if x.numel() != ncols {
        return Err(Error::ShapeMismatch {
            expected: vec![ncols],
            got: vec![x.numel()],
        });
    }

    let row_ptrs_data: Vec<i64> = row_ptrs.to_vec();
    let col_indices_data: Vec<i64> = col_indices.to_vec();
    let values_data: Vec<T> = values.to_vec();
    let x_data: Vec<T> = x.to_vec();

    let mut y_data: Vec<T> = Vec::with_capacity(nrows);
    for row in 0..nrows {
        let start = row_ptrs_data[row] as usize;
        let end = row_ptrs_data[row + 1] as usize;
        let mut sum: f64 = 0.0;
        for j in start..end {
            let col = col_indices_data[j] as usize;
            sum += values_data[j].to_f64() * x_data[col].to_f64();
        }
        y_data.push(T::from_f64(sum));
    }

    Tensor::from_slice(&y_data, &[nrows], device)
}

/// `C = A @ B` for a CSR matrix `A` and dense matrix `B`.
pub(super) fn spmm_csr_impl<T: Element>(
    _client: &CpuClient,
    row_ptrs: &Tensor<CpuRuntime>,
    col_indices: &Tensor<CpuRuntime>,
    values: &Tensor<CpuRuntime>,
    b: &Tensor<CpuRuntime>,
    shape: [usize; 2],
) -> Result<Tensor<CpuRuntime>> {
    let [m, k] = shape;
    let device = values.device();

    validate_sparse_operands::<T>(
        &[row_ptrs.dtype(), col_indices.dtype()],
        &[values.dtype(), b.dtype()],
    )?;

    if b.ndim() != 2 {
        return Err(Error::Internal(format!(
            "Expected 2D tensor for SpMM, got {}D",
            b.ndim()
        )));
    }

    let b_shape = b.shape();
    let b_k = b_shape[0];
    let n = b_shape[1];

    if b_k != k {
        return Err(Error::ShapeMismatch {
            expected: vec![k],
            got: vec![b_k],
        });
    }

    let row_ptrs_data: Vec<i64> = row_ptrs.to_vec();
    let col_indices_data: Vec<i64> = col_indices.to_vec();
    let a_values: Vec<T> = values.to_vec();
    let b_data: Vec<T> = b.to_vec();

    let mut c_data: Vec<T> = vec![T::zero(); m * n];
    for row in 0..m {
        let start = row_ptrs_data[row] as usize;
        let end = row_ptrs_data[row + 1] as usize;
        for j in start..end {
            let col = col_indices_data[j] as usize;
            let a_val = a_values[j].to_f64();
            for col_b in 0..n {
                let c_idx = row * n + col_b;
                let current = c_data[c_idx].to_f64();
                c_data[c_idx] = T::from_f64(current + a_val * b_data[col * n + col_b].to_f64());
            }
        }
    }

    Tensor::from_slice(&c_data, &[m, n], device)
}

#[cfg(test)]
mod tests {
    use crate::dtype::DType;
    use crate::error::Error;
    use crate::runtime::Runtime;
    use crate::runtime::cpu::{CpuClient, CpuRuntime};
    use crate::sparse::{SparseOps, SparseTensor};
    use crate::tensor::Tensor;

    #[test]
    fn test_spmv_csr_basic() {
        let device = <CpuRuntime as Runtime>::Device::default();
        let client = CpuClient::new(device.clone());

        // Matrix:
        // [1, 0, 2]
        // [0, 0, 3]
        // [4, 5, 0]
        let row_ptrs = Tensor::from_slice(&[0i64, 2, 3, 5], &[4], &device).unwrap();
        let col_indices = Tensor::from_slice(&[0i64, 2, 2, 0, 1], &[5], &device).unwrap();
        let values = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0], &[5], &device).unwrap();

        // x = [1, 2, 3]
        let x = Tensor::from_slice(&[1.0f32, 2.0, 3.0], &[3], &device).unwrap();

        // y = A * x
        // y[0] = 1*1 + 2*3 = 7
        // y[1] = 3*3 = 9
        // y[2] = 4*1 + 5*2 = 14
        let y = client
            .spmv_csr::<f32>(&row_ptrs, &col_indices, &values, &x, [3, 3])
            .unwrap();

        assert_eq!(y.shape(), &[3]);
        let y_data: Vec<f32> = y.to_vec();
        assert_eq!(y_data, vec![7.0, 9.0, 14.0]);
    }

    type Device = <CpuRuntime as Runtime>::Device;

    /// CSR parts of the f32 matrix `[[1, 0], [0, 2]]`.
    fn f32_csr_parts(
        device: &Device,
    ) -> (Tensor<CpuRuntime>, Tensor<CpuRuntime>, Tensor<CpuRuntime>) {
        let row_ptrs = Tensor::from_slice(&[0i64, 1, 2], &[3], device).unwrap();
        let col_indices = Tensor::from_slice(&[0i64, 1], &[2], device).unwrap();
        let values = Tensor::from_slice(&[1.0f32, 2.0], &[2], device).unwrap();
        (row_ptrs, col_indices, values)
    }

    fn assert_dtype_mismatch<T: std::fmt::Debug>(
        result: crate::error::Result<T>,
        lhs: DType,
        rhs: DType,
    ) {
        match result {
            Err(Error::DTypeMismatch { lhs: l, rhs: r }) => {
                assert_eq!((l, r), (lhs, rhs));
            }
            other => panic!("expected DTypeMismatch {{ {lhs:?}, {rhs:?} }}, got {other:?}"),
        }
    }

    #[test]
    fn test_spmv_csr_mixed_dtype_returns_error() {
        let device = Device::default();
        let client = CpuClient::new(device.clone());
        let (row_ptrs, col_indices, values) = f32_csr_parts(&device);
        let x = Tensor::from_slice(&[1.0f64, 2.0], &[2], &device).unwrap();

        let result = client.spmv_csr::<f32>(&row_ptrs, &col_indices, &values, &x, [2, 2]);
        assert_dtype_mismatch(result, DType::F32, DType::F64);
    }

    #[test]
    fn test_spmv_csr_type_parameter_mismatch_returns_error() {
        let device = Device::default();
        let client = CpuClient::new(device.clone());
        let (row_ptrs, col_indices, values) = f32_csr_parts(&device);
        let x = Tensor::from_slice(&[1.0f32, 2.0], &[2], &device).unwrap();

        let result = client.spmv_csr::<f64>(&row_ptrs, &col_indices, &values, &x, [2, 2]);
        assert_dtype_mismatch(result, DType::F64, DType::F32);
    }

    #[test]
    fn test_spmv_csr_non_i64_indices_return_error() {
        let device = Device::default();
        let client = CpuClient::new(device.clone());
        let (_, col_indices, values) = f32_csr_parts(&device);
        let row_ptrs = Tensor::from_slice(&[0i32, 1, 2], &[3], &device).unwrap();
        let x = Tensor::from_slice(&[1.0f32, 2.0], &[2], &device).unwrap();

        let result = client.spmv_csr::<f32>(&row_ptrs, &col_indices, &values, &x, [2, 2]);
        assert_dtype_mismatch(result, DType::I64, DType::I32);
    }

    #[test]
    fn test_spmv_mixed_dtype_returns_error() {
        let device = Device::default();
        let client = CpuClient::new(device.clone());
        let a = SparseTensor::<CpuRuntime>::from_csr_slices(
            &[0i64, 1, 2],
            &[0i64, 1],
            &[1.0f32, 2.0],
            [2, 2],
            &device,
        )
        .unwrap();
        let x = Tensor::from_slice(&[1.0f64, 2.0], &[2], &device).unwrap();

        assert_dtype_mismatch(client.spmv(&a, &x), DType::F32, DType::F64);
    }

    #[test]
    fn test_spmm_csr_mixed_dtype_returns_error() {
        let device = Device::default();
        let client = CpuClient::new(device.clone());
        let (row_ptrs, col_indices, values) = f32_csr_parts(&device);
        let b = Tensor::from_slice(&[1.0f64, 2.0, 3.0, 4.0], &[2, 2], &device).unwrap();

        let result = client.spmm_csr::<f32>(&row_ptrs, &col_indices, &values, &b, [2, 2]);
        assert_dtype_mismatch(result, DType::F32, DType::F64);
    }

    #[test]
    fn test_spmm_mixed_dtype_returns_error() {
        let device = Device::default();
        let client = CpuClient::new(device.clone());
        let a = SparseTensor::<CpuRuntime>::from_coo_slices(
            &[0i64, 1],
            &[0i64, 1],
            &[1.0f32, 2.0],
            [2, 2],
            &device,
        )
        .unwrap();
        let b = Tensor::from_slice(&[1.0f64, 2.0, 3.0, 4.0], &[2, 2], &device).unwrap();

        assert_dtype_mismatch(client.spmm(&a, &b), DType::F32, DType::F64);
    }

    #[test]
    fn test_dsmm_mixed_dtype_returns_error() {
        let device = Device::default();
        let client = CpuClient::new(device.clone());
        let a = Tensor::from_slice(&[1.0f64, 2.0, 3.0, 4.0], &[2, 2], &device).unwrap();
        let b = SparseTensor::<CpuRuntime>::from_csc_slices(
            &[0i64, 1, 2],
            &[0i64, 1],
            &[1.0f32, 2.0],
            [2, 2],
            &device,
        )
        .unwrap();

        assert_dtype_mismatch(client.dsmm(&a, &b), DType::F64, DType::F32);
    }
}
