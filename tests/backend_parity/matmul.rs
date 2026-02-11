// Backend parity tests for MatmulOps trait
//
// Dtype-parameterized: each test runs for all supported dtypes (F32, F64, F16, BF16, FP8).
// Tensors are created in f64 then cast to target dtype via tensor_from_f64().
// Comparison reads back in native dtype - no unnecessary f64 conversion.

use numr::dtype::DType;
use numr::ops::MatmulOps;
use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime, ParallelismConfig};
use numr::tensor::Tensor;

use crate::backend_parity::dtype_helpers::tensor_from_f64;
#[cfg(feature = "cuda")]
use crate::backend_parity::helpers::with_cuda_backend;
#[cfg(feature = "wgpu")]
use crate::backend_parity::helpers::with_wgpu_backend;
use crate::common::{
    assert_tensor_allclose, create_cpu_client, is_dtype_supported, supported_dtypes,
};

// ============================================================================
// Test Utilities
// ============================================================================

struct MatmulTest {
    a: Vec<f64>,
    a_shape: Vec<usize>,
    b: Vec<f64>,
    b_shape: Vec<usize>,
}

impl MatmulTest {
    fn new(a: Vec<f64>, a_shape: Vec<usize>, b: Vec<f64>, b_shape: Vec<usize>) -> Self {
        MatmulTest {
            a,
            a_shape,
            b,
            b_shape,
        }
    }
}

fn test_matmul_parity(test_cases: &[MatmulTest], dtype: DType) {
    // CPU baseline
    let (cpu_client, cpu_device) = create_cpu_client();

    let cpu_results: Vec<Tensor<CpuRuntime>> = test_cases
        .iter()
        .map(|tc| {
            let a = tensor_from_f64(&tc.a, &tc.a_shape, dtype, &cpu_device, &cpu_client)
                .unwrap_or_else(|e| panic!("CPU tensor_from_f64 failed for {dtype:?}: {e}"));
            let b = tensor_from_f64(&tc.b, &tc.b_shape, dtype, &cpu_device, &cpu_client)
                .unwrap_or_else(|e| panic!("CPU tensor_from_f64 failed for {dtype:?}: {e}"));
            cpu_client
                .matmul(&a, &b)
                .unwrap_or_else(|e| panic!("CPU matmul failed for {dtype:?}: {e}"))
        })
        .collect();

    // CUDA parity
    #[cfg(feature = "cuda")]
    if is_dtype_supported("cuda", dtype) {
        with_cuda_backend(|cuda_client, cuda_device| {
            for (idx, tc) in test_cases.iter().enumerate() {
                let a = tensor_from_f64(&tc.a, &tc.a_shape, dtype, &cuda_device, &cuda_client)
                    .unwrap_or_else(|e| panic!("CUDA tensor_from_f64 failed for {dtype:?}: {e}"));
                let b = tensor_from_f64(&tc.b, &tc.b_shape, dtype, &cuda_device, &cuda_client)
                    .unwrap_or_else(|e| panic!("CUDA tensor_from_f64 failed for {dtype:?}: {e}"));

                let result = cuda_client
                    .matmul(&a, &b)
                    .unwrap_or_else(|e| panic!("CUDA matmul failed for {dtype:?}: {e}"));

                assert_tensor_allclose(
                    &result,
                    &cpu_results[idx],
                    dtype,
                    &format!("matmul CUDA vs CPU [{dtype:?}] case {idx}"),
                );
            }
        });
    }

    // WebGPU parity
    #[cfg(feature = "wgpu")]
    if is_dtype_supported("wgpu", dtype) {
        with_wgpu_backend(|wgpu_client, wgpu_device| {
            for (idx, tc) in test_cases.iter().enumerate() {
                let a = tensor_from_f64(&tc.a, &tc.a_shape, dtype, &wgpu_device, &wgpu_client)
                    .unwrap_or_else(|e| panic!("WebGPU tensor_from_f64 failed for {dtype:?}: {e}"));
                let b = tensor_from_f64(&tc.b, &tc.b_shape, dtype, &wgpu_device, &wgpu_client)
                    .unwrap_or_else(|e| panic!("WebGPU tensor_from_f64 failed for {dtype:?}: {e}"));

                let result = wgpu_client
                    .matmul(&a, &b)
                    .unwrap_or_else(|e| panic!("WebGPU matmul failed for {dtype:?}: {e}"));

                assert_tensor_allclose(
                    &result,
                    &cpu_results[idx],
                    dtype,
                    &format!("matmul WebGPU vs CPU [{dtype:?}] case {idx}"),
                );
            }
        });
    }
}

// ============================================================================
// Matmul Parity Tests
// ============================================================================

macro_rules! matmul_case {
    ($name:ident, $cases:expr) => {
        #[test]
        fn $name() {
            for dtype in supported_dtypes("cpu") {
                test_matmul_parity($cases, dtype);
            }
        }
    };
}

matmul_case!(
    test_matmul_2d_parity,
    &[MatmulTest::new(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        vec![2, 3],
        vec![1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0],
        vec![3, 4],
    )]
);

matmul_case!(
    test_matmul_square_parity,
    &[MatmulTest::new(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        vec![3, 3],
        vec![9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
        vec![3, 3],
    )]
);

matmul_case!(
    test_matmul_batched_parity,
    &[MatmulTest::new(
        vec![
            // Batch 0: 3x4
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, // Batch 1: 3x4
            2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0,
        ],
        vec![2, 3, 4],
        vec![
            // Batch 0: 4x2
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, // Batch 1: 4x2
            2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0,
        ],
        vec![2, 4, 2],
    )]
);

matmul_case!(
    test_matmul_vector_parity,
    &[MatmulTest::new(
        vec![1.0, 2.0, 3.0, 4.0],
        vec![1, 4],
        vec![5.0, 6.0, 7.0, 8.0],
        vec![4, 1],
    )]
);

#[test]
fn test_cpu_matmul_parallelism_config_matches_default() {
    let device = CpuDevice::new();
    let default_client = CpuClient::new(device.clone());
    let configured_client =
        default_client.with_parallelism(ParallelismConfig::new(Some(1), Some(1024)));

    // Batched case to exercise batch parallelism path.
    let a_shape = [6, 24, 16];
    let b_shape = [6, 16, 12];
    let a_numel: usize = a_shape.iter().product();
    let b_numel: usize = b_shape.iter().product();
    let a_data: Vec<f32> = (0..a_numel)
        .map(|i| (i as f32 * 0.013).sin() + (i as f32 * 0.007).cos())
        .collect();
    let b_data: Vec<f32> = (0..b_numel)
        .map(|i| (i as f32 * 0.011).cos() - (i as f32 * 0.005).sin())
        .collect();

    let a = Tensor::<CpuRuntime>::from_slice(&a_data, &a_shape, &device);
    let b = Tensor::<CpuRuntime>::from_slice(&b_data, &b_shape, &device);

    let base: Vec<f32> = default_client.matmul(&a, &b).unwrap().to_vec();
    let cfg: Vec<f32> = configured_client.matmul(&a, &b).unwrap().to_vec();

    // Compare with tight tolerance for f32
    assert_eq!(base.len(), cfg.len(), "result length mismatch");
    for (i, (b_val, c_val)) in base.iter().zip(cfg.iter()).enumerate() {
        assert!(
            (b_val - c_val).abs() <= 1e-5,
            "element {} differs: {} vs {} (diff={})",
            i,
            b_val,
            c_val,
            (b_val - c_val).abs()
        );
    }
}
