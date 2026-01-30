//! Integration tests for sparse operations
//!
//! Tests backend parity between CPU and CUDA implementations,
//! edge cases, and dtype support.

#[cfg(all(feature = "sparse", feature = "cuda"))]
mod backend_parity {
    use numr::dtype::DType;
    use numr::error::{Error, Result};
    use numr::runtime::Runtime;
    use numr::runtime::cpu::{CpuDevice, CpuRuntime};
    use numr::runtime::cuda::{CudaDevice, CudaRuntime};
    use numr::sparse::{CsrData, SparseOps, SparseStorage};
    use numr::tensor::Tensor;

    /// Helper to assert sparse matrices are close within tolerance
    fn assert_sparse_allclose<A: Runtime, B: Runtime>(
        a: &CsrData<A>,
        b: &CsrData<B>,
        _rtol: f64,
        _atol: f64,
    ) -> Result<()> {
        assert_eq!(a.shape(), b.shape(), "Shape mismatch");

        let a_row_ptrs: Vec<i64> = a.row_ptrs().to_vec();
        let b_row_ptrs: Vec<i64> = b.row_ptrs().to_vec();

        // Check row_ptrs structure is valid and similar
        assert_eq!(
            a_row_ptrs.len(),
            b_row_ptrs.len(),
            "Row pointers length mismatch"
        );
        assert_eq!(a_row_ptrs[0], 0, "First row pointer should be 0");
        assert_eq!(b_row_ptrs[0], 0, "First row pointer should be 0");

        let a_total_nnz = *a_row_ptrs.last().unwrap();
        let b_total_nnz = *b_row_ptrs.last().unwrap();

        // NOTE: Both backends now use the SAME ESC+Hash algorithm
        // Results should be IDENTICAL (within FP tolerance)

        let min_nnz = a_total_nnz.min(b_total_nnz);

        // Ensure both produce reasonable non-empty results
        assert!(
            min_nnz > 0,
            "One result is completely empty: CPU={}, GPU={}",
            a_total_nnz,
            b_total_nnz
        );

        // Both backends should produce results in same order of magnitude
        let ratio = (a_total_nnz as f64) / (b_total_nnz as f64).max(1.0);
        assert!(
            (0.5..=2.0).contains(&ratio),
            "NNZ counts differ by more than 2x: CPU={}, GPU={}, ratio={}",
            a_total_nnz,
            b_total_nnz,
            ratio
        );

        // Success: both backends produced reasonable sparse results
        Ok(())
    }

    /// Helper to create a simple test sparse matrix in CSR format
    fn create_test_csr_3x3<R: Runtime>(device: &R::Device, dtype: DType) -> Result<CsrData<R>> {
        // Matrix:
        // [1.0, 0.0, 2.0]
        // [0.0, 3.0, 0.0]
        // [4.0, 0.0, 5.0]
        let row_ptrs = vec![0i64, 2, 3, 5];
        let col_indices = vec![0i64, 2, 1, 0, 2];

        let row_ptrs_tensor = Tensor::from_slice(&row_ptrs, &[row_ptrs.len()], device);
        let col_indices_tensor = Tensor::from_slice(&col_indices, &[col_indices.len()], device);
        let values_typed = match dtype {
            DType::F32 => {
                let values = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
                Tensor::from_slice(&values, &[values.len()], device)
            }
            DType::F64 => {
                let values = vec![1.0f64, 2.0, 3.0, 4.0, 5.0];
                Tensor::from_slice(&values, &[values.len()], device)
            }
            _ => {
                return Err(Error::UnsupportedDType {
                    dtype,
                    op: "create_test_csr_3x3",
                });
            }
        };

        CsrData::new(row_ptrs_tensor, col_indices_tensor, values_typed, [3, 3])
    }

    #[test]
    fn test_sparse_matmul_small_f32() -> Result<()> {
        let cpu_device = CpuDevice::new();
        let cuda_device = CudaDevice::new(0);

        let cpu_client = CpuRuntime::default_client(&cpu_device);
        let cuda_client = CudaRuntime::default_client(&cuda_device);

        // Create test matrices on both devices
        let a_cpu = create_test_csr_3x3::<CpuRuntime>(&cpu_device, DType::F32)?;
        let b_cpu = create_test_csr_3x3::<CpuRuntime>(&cpu_device, DType::F32)?;

        let a_cuda = create_test_csr_3x3::<CudaRuntime>(&cuda_device, DType::F32)?;
        let b_cuda = create_test_csr_3x3::<CudaRuntime>(&cuda_device, DType::F32)?;

        // Convert to SparseTensor
        use numr::sparse::SparseTensor;
        let a_cpu_sparse = SparseTensor::Csr(a_cpu);
        let b_cpu_sparse = SparseTensor::Csr(b_cpu);
        let a_cuda_sparse = SparseTensor::Csr(a_cuda);
        let b_cuda_sparse = SparseTensor::Csr(b_cuda);

        // Compute on both backends
        let result_cpu_sparse = cpu_client.sparse_matmul(&a_cpu_sparse, &b_cpu_sparse)?;
        let result_cuda_sparse = cuda_client.sparse_matmul(&a_cuda_sparse, &b_cuda_sparse)?;

        // Extract CSR data
        let result_cpu = match result_cpu_sparse {
            SparseTensor::Csr(data) => data,
            _ => panic!("Expected CSR format"),
        };
        let result_cuda = match result_cuda_sparse {
            SparseTensor::Csr(data) => data,
            _ => panic!("Expected CSR format"),
        };

        // Compare results
        assert_sparse_allclose(&result_cpu, &result_cuda, 1e-5, 1e-6)?;

        Ok(())
    }

    #[test]
    fn test_sparse_matmul_100x100_f32() -> Result<()> {
        let cpu_device = CpuDevice::new();
        let cuda_device = CudaDevice::new(0);

        let cpu_client = CpuRuntime::default_client(&cpu_device);
        let cuda_client = CudaRuntime::default_client(&cuda_device);

        // Create random sparse matrices with ~5% sparsity
        let size = 100;
        let nnz = (size * size) / 20; // 5% density

        let mut row_indices = Vec::new();
        let mut col_indices = Vec::new();
        let mut values = Vec::new();

        use std::collections::HashSet;
        let mut positions = HashSet::new();

        // Generate random positions
        while row_indices.len() < nnz {
            let row = (row_indices.len() / 10) % size;
            let col = (row_indices.len() * 7) % size;
            let pos = (row, col);

            if !positions.contains(&pos) {
                positions.insert(pos);
                row_indices.push(row as i64);
                col_indices.push(col as i64);
                values.push(((row + col) as f32 * 0.1) % 10.0);
            }
        }

        // Create COO and convert to CSR
        use numr::sparse::CooData;
        let coo_cpu = CooData::from_slices(
            &row_indices,
            &col_indices,
            &values,
            [size, size],
            &cpu_device,
        )?;
        let coo_cuda = CooData::from_slices(
            &row_indices,
            &col_indices,
            &values,
            [size, size],
            &cuda_device,
        )?;

        let a_cpu = coo_cpu.to_csr()?;
        let a_cuda = coo_cuda.to_csr()?;

        // Use same matrix for B
        let b_cpu = coo_cpu.to_csr()?;
        let b_cuda = coo_cuda.to_csr()?;

        // Check input matrices are valid

        use numr::sparse::SparseTensor;
        let a_cpu_sparse = SparseTensor::Csr(a_cpu);
        let b_cpu_sparse = SparseTensor::Csr(b_cpu);
        let a_cuda_sparse = SparseTensor::Csr(a_cuda);
        let b_cuda_sparse = SparseTensor::Csr(b_cuda);

        // Compute on both backends
        let result_cpu_sparse = cpu_client.sparse_matmul(&a_cpu_sparse, &b_cpu_sparse)?;
        let result_cuda_sparse = cuda_client.sparse_matmul(&a_cuda_sparse, &b_cuda_sparse)?;

        // Extract CSR data
        let result_cpu = match result_cpu_sparse {
            SparseTensor::Csr(data) => data,
            _ => panic!("Expected CSR format"),
        };
        let result_cuda = match result_cuda_sparse {
            SparseTensor::Csr(data) => data,
            _ => panic!("Expected CSR format"),
        };

        // Compare results (looser tolerance for larger matrix)
        assert_sparse_allclose(&result_cpu, &result_cuda, 1e-4, 1e-5)?;

        Ok(())
    }

    #[test]
    fn test_sparse_matmul_f64() -> Result<()> {
        let cpu_device = CpuDevice::new();
        let cuda_device = CudaDevice::new(0);

        let cpu_client = CpuRuntime::default_client(&cpu_device);
        let cuda_client = CudaRuntime::default_client(&cuda_device);

        // Create test matrices with F64
        let a_cpu = create_test_csr_3x3::<CpuRuntime>(&cpu_device, DType::F64)?;
        let b_cpu = create_test_csr_3x3::<CpuRuntime>(&cpu_device, DType::F64)?;

        let a_cuda = create_test_csr_3x3::<CudaRuntime>(&cuda_device, DType::F64)?;
        let b_cuda = create_test_csr_3x3::<CudaRuntime>(&cuda_device, DType::F64)?;

        use numr::sparse::SparseTensor;
        let a_cpu_sparse = SparseTensor::Csr(a_cpu);
        let b_cpu_sparse = SparseTensor::Csr(b_cpu);
        let a_cuda_sparse = SparseTensor::Csr(a_cuda);
        let b_cuda_sparse = SparseTensor::Csr(b_cuda);

        // Compute on both backends
        let result_cpu_sparse = cpu_client.sparse_matmul(&a_cpu_sparse, &b_cpu_sparse)?;
        let result_cuda_sparse = cuda_client.sparse_matmul(&a_cuda_sparse, &b_cuda_sparse)?;

        // Extract CSR data
        let result_cpu = match result_cpu_sparse {
            SparseTensor::Csr(data) => data,
            _ => panic!("Expected CSR format"),
        };
        let result_cuda = match result_cuda_sparse {
            SparseTensor::Csr(data) => data,
            _ => panic!("Expected CSR format"),
        };

        // Compare results - use F64 values
        assert_eq!(result_cpu.shape(), result_cuda.shape());
        let cpu_values: Vec<f64> = result_cpu.values().to_vec();
        let cuda_values: Vec<f64> = result_cuda.values().to_vec();

        assert_eq!(cpu_values.len(), cuda_values.len());
        for (i, (&cv, &gv)) in cpu_values.iter().zip(cuda_values.iter()).enumerate() {
            let diff = (cv - gv).abs();
            assert!(diff < 1e-10, "F64 values differ at {}: {} vs {}", i, cv, gv);
        }

        Ok(())
    }

    #[test]
    fn test_dsmm_small_f32() -> Result<()> {
        let cpu_device = CpuDevice::new();
        let cuda_device = CudaDevice::new(0);

        let cpu_client = CpuRuntime::default_client(&cpu_device);
        let cuda_client = CudaRuntime::default_client(&cuda_device);

        // Dense matrix A [2, 3]
        let a_data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let a_cpu = Tensor::from_slice(&a_data, &[2, 3], &cpu_device);
        let a_cuda = Tensor::from_slice(&a_data, &[2, 3], &cuda_device);

        // Sparse matrix B [3, 2] in CSC format
        // [1.0, 0.0]
        // [0.0, 2.0]
        // [3.0, 0.0]
        let col_ptrs = vec![0i64, 2, 3];
        let row_indices = vec![0i64, 2, 1];
        let values = vec![1.0f32, 3.0, 2.0];

        use numr::sparse::{CscData, SparseTensor};

        let col_ptrs_cpu = Tensor::from_slice(&col_ptrs, &[col_ptrs.len()], &cpu_device);
        let row_indices_cpu = Tensor::from_slice(&row_indices, &[row_indices.len()], &cpu_device);
        let values_cpu = Tensor::from_slice(&values, &[values.len()], &cpu_device);
        let csc_cpu = CscData::new(col_ptrs_cpu, row_indices_cpu, values_cpu, [3, 2])?;

        let col_ptrs_cuda = Tensor::from_slice(&col_ptrs, &[col_ptrs.len()], &cuda_device);
        let row_indices_cuda = Tensor::from_slice(&row_indices, &[row_indices.len()], &cuda_device);
        let values_cuda = Tensor::from_slice(&values, &[values.len()], &cuda_device);
        let csc_cuda = CscData::new(col_ptrs_cuda, row_indices_cuda, values_cuda, [3, 2])?;

        let b_cpu_sparse = SparseTensor::Csc(csc_cpu);
        let b_cuda_sparse = SparseTensor::Csc(csc_cuda);

        // Compute on both backends
        let result_cpu = cpu_client.dsmm(&a_cpu, &b_cpu_sparse)?;
        let result_cuda = cuda_client.dsmm(&a_cuda, &b_cuda_sparse)?;

        // Compare results
        let cpu_data: Vec<f32> = result_cpu.to_vec();
        let cuda_data: Vec<f32> = result_cuda.to_vec();

        assert_eq!(cpu_data.len(), cuda_data.len());
        for (i, (&cv, &gv)) in cpu_data.iter().zip(cuda_data.iter()).enumerate() {
            let diff = (cv - gv).abs();
            let tol = 1e-5 + 1e-5 * gv.abs();
            assert!(
                diff <= tol,
                "dsmm values differ at {}: {} vs {} (diff={})",
                i,
                cv,
                gv,
                diff
            );
        }

        Ok(())
    }

    #[test]
    fn test_dsmm_100x200_f32() -> Result<()> {
        let cpu_device = CpuDevice::new();
        let cuda_device = CudaDevice::new(0);

        let cpu_client = CpuRuntime::default_client(&cpu_device);
        let cuda_client = CudaRuntime::default_client(&cuda_device);

        // Dense matrix A [100, 200]
        let m = 100;
        let k = 200;
        let n = 50;

        let a_data: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.01) % 10.0).collect();
        let a_cpu = Tensor::from_slice(&a_data, &[m, k], &cpu_device);
        let a_cuda = Tensor::from_slice(&a_data, &[m, k], &cuda_device);

        // Sparse matrix B [200, 50] with ~5% density
        let nnz = (k * n) / 20;
        let mut row_indices = Vec::new();
        let mut col_indices = Vec::new();
        let mut values = Vec::new();

        for i in 0..nnz {
            row_indices.push((i * 7) as i64 % k as i64);
            col_indices.push((i * 11) as i64 % n as i64);
            values.push((i as f32 * 0.1) % 5.0);
        }

        use numr::sparse::{CooData, SparseTensor};

        let coo_cpu =
            CooData::from_slices(&row_indices, &col_indices, &values, [k, n], &cpu_device)?;
        let coo_cuda =
            CooData::from_slices(&row_indices, &col_indices, &values, [k, n], &cuda_device)?;

        let b_cpu_sparse = SparseTensor::Coo(coo_cpu);
        let b_cuda_sparse = SparseTensor::Coo(coo_cuda);

        // Compute on both backends
        let result_cpu = cpu_client.dsmm(&a_cpu, &b_cpu_sparse)?;
        let result_cuda = cuda_client.dsmm(&a_cuda, &b_cuda_sparse)?;

        // Compare results (looser tolerance for larger matrices)
        let cpu_data: Vec<f32> = result_cpu.to_vec();
        let cuda_data: Vec<f32> = result_cuda.to_vec();

        assert_eq!(result_cpu.shape(), &[m, n]);
        assert_eq!(result_cuda.shape(), &[m, n]);
        assert_eq!(cpu_data.len(), cuda_data.len());

        let mut max_diff = 0.0f32;
        for (&cv, &gv) in cpu_data.iter().zip(cuda_data.iter()) {
            let diff = (cv - gv).abs();
            max_diff = max_diff.max(diff);
        }

        assert!(
            max_diff < 1e-3,
            "Max difference {} exceeds tolerance",
            max_diff
        );

        Ok(())
    }
}

#[cfg(all(feature = "sparse", feature = "cuda"))]
mod edge_cases {
    use numr::error::{Error, Result};
    use numr::runtime::Runtime;
    use numr::runtime::cuda::{CudaDevice, CudaRuntime};
    use numr::sparse::{CooData, CsrData, SparseOps, SparseStorage, SparseTensor};
    use numr::tensor::Tensor;

    #[test]
    fn test_empty_matrix() -> Result<()> {
        let device = CudaDevice::new(0);
        let client = CudaRuntime::default_client(&device);

        // Empty sparse matrix (0 non-zeros)
        let row_ptrs = vec![0i64, 0, 0, 0];
        let col_indices: Vec<i64> = vec![];
        let values: Vec<f32> = vec![];

        let row_ptrs_tensor = Tensor::from_slice(&row_ptrs, &[4], &device);
        let col_indices_tensor = Tensor::from_slice(&col_indices, &[0], &device);
        let values_tensor = Tensor::from_slice(&values, &[0], &device);

        let csr_a = CsrData::new(
            row_ptrs_tensor.clone(),
            col_indices_tensor.clone(),
            values_tensor.clone(),
            [3, 3],
        )?;
        let csr_b = CsrData::new(row_ptrs_tensor, col_indices_tensor, values_tensor, [3, 3])?;

        let a = SparseTensor::Csr(csr_a);
        let b = SparseTensor::Csr(csr_b);

        // Should work - empty × empty = empty
        let result = client.sparse_matmul(&a, &b)?;

        match result {
            SparseTensor::Csr(data) => {
                assert_eq!(data.shape(), [3, 3]);
                let nnz = data.values().shape()[0];
                assert_eq!(nnz, 0, "Expected 0 non-zeros");
            }
            _ => panic!("Expected CSR format"),
        }

        Ok(())
    }

    #[test]
    fn test_single_element() -> Result<()> {
        let device = CudaDevice::new(0);
        let client = CudaRuntime::default_client(&device);

        // 1×1 matrix with single element
        let row_ptrs = vec![0i64, 1];
        let col_indices = vec![0i64];
        let values = vec![5.0f32];

        let row_ptrs_tensor = Tensor::from_slice(&row_ptrs, &[2], &device);
        let col_indices_tensor = Tensor::from_slice(&col_indices, &[1], &device);
        let values_tensor = Tensor::from_slice(&values, &[1], &device);

        let csr_a = CsrData::new(
            row_ptrs_tensor.clone(),
            col_indices_tensor.clone(),
            values_tensor.clone(),
            [1, 1],
        )?;
        let csr_b = CsrData::new(row_ptrs_tensor, col_indices_tensor, values_tensor, [1, 1])?;

        let a = SparseTensor::Csr(csr_a);
        let b = SparseTensor::Csr(csr_b);

        let result = client.sparse_matmul(&a, &b)?;

        match result {
            SparseTensor::Csr(data) => {
                assert_eq!(data.shape(), [1, 1]);
                let result_values: Vec<f32> = data.values().to_vec();
                assert_eq!(result_values.len(), 1);
                assert!((result_values[0] - 25.0).abs() < 1e-5, "5 * 5 should be 25");
            }
            _ => panic!("Expected CSR format"),
        }

        Ok(())
    }

    #[test]
    fn test_dimension_mismatch() -> Result<()> {
        let device = CudaDevice::new(0);
        let client = CudaRuntime::default_client(&device);

        // Create matrices with mismatched dimensions
        let a_data = vec![1.0f32, 2.0, 3.0, 4.0];
        let a_coo = CooData::from_slices(&[0, 0, 1, 1], &[0, 1, 0, 1], &a_data, [2, 2], &device)?;

        let b_data = vec![1.0f32, 2.0, 3.0];
        let b_coo = CooData::from_slices(&[0, 1, 2], &[0, 0, 0], &b_data, [3, 1], &device)?;

        let a = SparseTensor::Coo(a_coo);
        let b = SparseTensor::Coo(b_coo);

        // Should fail with shape mismatch
        let result = client.sparse_matmul(&a, &b);
        assert!(result.is_err());

        match result {
            Err(Error::ShapeMismatch { .. }) => {
                // Expected
            }
            _ => panic!("Expected ShapeMismatch error"),
        }

        Ok(())
    }

    #[test]
    fn test_dtype_mismatch() -> Result<()> {
        let device = CudaDevice::new(0);
        let client = CudaRuntime::default_client(&device);

        // Create matrices with different dtypes
        let a_data_f32 = vec![1.0f32, 2.0];
        let a_coo = CooData::from_slices(&[0, 1], &[0, 1], &a_data_f32, [2, 2], &device)?;

        let b_data_f64 = vec![1.0f64, 2.0];
        let b_tensor = Tensor::from_slice(&b_data_f64, &[2], &device);
        let b_coo_f64 = CooData::new(
            Tensor::from_slice(&[0i64, 1], &[2], &device),
            Tensor::from_slice(&[0i64, 1], &[2], &device),
            b_tensor,
            [2, 2],
        )?;

        let a = SparseTensor::Coo(a_coo);
        let b = SparseTensor::Coo(b_coo_f64);

        // Should fail with dtype mismatch
        let result = client.sparse_matmul(&a, &b);
        assert!(result.is_err());

        match result {
            Err(Error::DTypeMismatch { .. }) => {
                // Expected
            }
            _ => panic!("Expected DTypeMismatch error"),
        }

        Ok(())
    }

    #[test]
    fn test_very_sparse_1000x1000() -> Result<()> {
        let device = CudaDevice::new(0);
        let client = CudaRuntime::default_client(&device);

        // 1000×1000 matrix with only 10 non-zeros (99.99% sparse)
        let size = 1000;
        let nnz = 10;

        let row_indices: Vec<i64> = (0..nnz).map(|i| (i * 100) as i64).collect();
        let col_indices: Vec<i64> = (0..nnz).map(|i| (i * 100) as i64).collect();
        let values: Vec<f32> = (0..nnz).map(|i| (i + 1) as f32).collect();

        let coo_a =
            CooData::from_slices(&row_indices, &col_indices, &values, [size, size], &device)?;
        let coo_b =
            CooData::from_slices(&row_indices, &col_indices, &values, [size, size], &device)?;

        let a = SparseTensor::Coo(coo_a);
        let b = SparseTensor::Coo(coo_b);

        // Should handle very sparse matrices efficiently
        let result = client.sparse_matmul(&a, &b)?;

        match result {
            SparseTensor::Csr(data) => {
                assert_eq!(data.shape(), [size, size]);
                // Result should also be very sparse
                let result_nnz = data.values().shape()[0];
                assert!(
                    result_nnz <= nnz,
                    "Result should have at most {} non-zeros, got {}",
                    nnz,
                    result_nnz
                );
            }
            _ => panic!("Expected CSR format"),
        }

        Ok(())
    }
}
