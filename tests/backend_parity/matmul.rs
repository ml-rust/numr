// Backend parity tests for MatmulOps trait
//
// Tests verify that MatmulOps operations produce identical results across
// CPU, CUDA, and WebGPU backends.

use numr::ops::MatmulOps;
use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime, ParallelismConfig};
use numr::tensor::Tensor;

#[cfg(any(feature = "cuda", feature = "wgpu"))]
use crate::backend_parity::helpers::assert_case_parity_f32;
use crate::backend_parity::helpers::assert_parity_f32;
#[cfg(feature = "cuda")]
use crate::backend_parity::helpers::with_cuda_backend;
#[cfg(feature = "wgpu")]
use crate::backend_parity::helpers::with_wgpu_backend;
use crate::common::create_cpu_client;

// ============================================================================
// Test Utilities
// ============================================================================

struct MatmulTest {
    a: Vec<f32>,
    a_shape: Vec<usize>,
    b: Vec<f32>,
    b_shape: Vec<usize>,
}

impl MatmulTest {
    fn new(a: Vec<f32>, a_shape: Vec<usize>, b: Vec<f32>, b_shape: Vec<usize>) -> Self {
        MatmulTest {
            a,
            a_shape,
            b,
            b_shape,
        }
    }
}

fn test_matmul_parity(test_cases: Vec<MatmulTest>) {
    // CPU baseline
    let cpu_results: Vec<Vec<f32>> = test_cases
        .iter()
        .map(|tc| {
            let (client, device) = create_cpu_client();
            let a = Tensor::from_slice(&tc.a, &tc.a_shape, &device);
            let b = Tensor::from_slice(&tc.b, &tc.b_shape, &device);
            client
                .matmul(&a, &b)
                .expect("CPU matmul failed")
                .to_vec::<f32>()
        })
        .collect();

    // CUDA parity
    #[cfg(feature = "cuda")]
    with_cuda_backend(|cuda_client, cuda_device| {
        for (idx, tc) in test_cases.iter().enumerate() {
            let a = Tensor::from_slice(&tc.a, &tc.a_shape, &cuda_device);
            let b = Tensor::from_slice(&tc.b, &tc.b_shape, &cuda_device);
            let result = cuda_client
                .matmul(&a, &b)
                .expect("CUDA matmul failed")
                .to_vec::<f32>();
            assert_case_parity_f32(&cpu_results, idx, &result, "matmul", "cuda");
        }
    });

    // WebGPU parity
    #[cfg(feature = "wgpu")]
    with_wgpu_backend(|wgpu_client, wgpu_device| {
        for (idx, tc) in test_cases.iter().enumerate() {
            let a = Tensor::from_slice(&tc.a, &tc.a_shape, &wgpu_device);
            let b = Tensor::from_slice(&tc.b, &tc.b_shape, &wgpu_device);
            let result = wgpu_client
                .matmul(&a, &b)
                .expect("WebGPU matmul failed")
                .to_vec::<f32>();
            assert_case_parity_f32(&cpu_results, idx, &result, "matmul", "wgpu");
        }
    });
}

// ============================================================================
// Matmul Parity Tests
// ============================================================================

#[test]
fn test_matmul_2d_parity() {
    // Simple 2x3 @ 3x4 -> 2x4
    let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let b = vec![1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0];

    test_matmul_parity(vec![MatmulTest::new(a, vec![2, 3], b, vec![3, 4])]);
}

#[test]
fn test_matmul_square_parity() {
    // 3x3 @ 3x3 -> 3x3
    let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
    let b = vec![9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0];

    test_matmul_parity(vec![MatmulTest::new(a, vec![3, 3], b, vec![3, 3])]);
}

#[test]
fn test_matmul_batched_parity() {
    // Batched: 2x3x4 @ 2x4x2 -> 2x3x2
    let a = vec![
        // Batch 0: 3x4
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, // Batch 1: 3x4
        2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0,
    ];
    let b = vec![
        // Batch 0: 4x2
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, // Batch 1: 4x2
        2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0,
    ];

    test_matmul_parity(vec![MatmulTest::new(a, vec![2, 3, 4], b, vec![2, 4, 2])]);
}

#[test]
fn test_matmul_vector_parity() {
    // 1x4 @ 4x1 -> 1x1 (dot product as matmul)
    let a = vec![1.0, 2.0, 3.0, 4.0];
    let b = vec![5.0, 6.0, 7.0, 8.0];

    test_matmul_parity(vec![MatmulTest::new(a, vec![1, 4], b, vec![4, 1])]);
}

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
    assert_parity_f32(&base, &cfg, "cpu_matmul_parallelism_config");
}
