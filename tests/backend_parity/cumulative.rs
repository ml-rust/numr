// Backend parity tests for CumulativeOps trait
//
// Tests verify that all CumulativeOps operations produce identical results across
// CPU, CUDA, and WebGPU backends.

use numr::ops::CumulativeOps;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

#[cfg(any(feature = "cuda", feature = "wgpu"))]
use crate::backend_parity::helpers::assert_case_parity_f32;
#[cfg(feature = "cuda")]
use crate::backend_parity::helpers::with_cuda_backend;
#[cfg(feature = "wgpu")]
use crate::backend_parity::helpers::with_wgpu_backend;
use crate::common::create_cpu_client;

// ============================================================================
// Test Utilities
// ============================================================================

struct CumulativeTest {
    data: Vec<f32>,
    shape: Vec<usize>,
    dim: isize,
}

impl CumulativeTest {
    fn new(data: Vec<f32>, shape: Vec<usize>, dim: isize) -> Self {
        CumulativeTest { data, shape, dim }
    }
}

fn apply_cumulative_op<R: Runtime>(
    client: &impl CumulativeOps<R>,
    op: &str,
    tensor: &Tensor<R>,
    dim: isize,
) -> numr::error::Result<Tensor<R>> {
    match op {
        "cumsum" => client.cumsum(tensor, dim),
        "cumprod" => client.cumprod(tensor, dim),
        "logsumexp" => {
            // Convert negative dim to positive index
            let ndim = tensor.shape().len() as isize;
            let dim_usize = if dim < 0 {
                (ndim + dim) as usize
            } else {
                dim as usize
            };
            client.logsumexp(tensor, &[dim_usize], false)
        }
        _ => panic!("Unknown cumulative op: {}", op),
    }
}

fn test_cumulative_parity(op: &str, test_cases: Vec<CumulativeTest>) {
    // CPU baseline
    let cpu_results: Vec<Vec<f32>> = test_cases
        .iter()
        .map(|tc| {
            let (client, device) = create_cpu_client();
            let tensor = Tensor::from_slice(&tc.data, &tc.shape, &device);
            apply_cumulative_op(&client, op, &tensor, tc.dim)
                .expect("CPU operation failed")
                .to_vec::<f32>()
        })
        .collect();

    // CUDA parity
    #[cfg(feature = "cuda")]
    with_cuda_backend(|cuda_client, cuda_device| {
        for (idx, tc) in test_cases.iter().enumerate() {
            let tensor = Tensor::from_slice(&tc.data, &tc.shape, &cuda_device);
            let result = apply_cumulative_op(&cuda_client, op, &tensor, tc.dim)
                .expect("CUDA operation failed")
                .to_vec::<f32>();
            assert_case_parity_f32(&cpu_results, idx, &result, op, "cuda");
        }
    });

    // WebGPU parity
    #[cfg(feature = "wgpu")]
    with_wgpu_backend(|wgpu_client, wgpu_device| {
        for (idx, tc) in test_cases.iter().enumerate() {
            let tensor = Tensor::from_slice(&tc.data, &tc.shape, &wgpu_device);
            let result = apply_cumulative_op(&wgpu_client, op, &tensor, tc.dim)
                .expect("WebGPU operation failed")
                .to_vec::<f32>();
            assert_case_parity_f32(&cpu_results, idx, &result, op, "wgpu");
        }
    });
}

// ============================================================================
// Cumulative Operation Parity Tests
// ============================================================================

#[test]
fn test_cumsum_parity() {
    test_cumulative_parity(
        "cumsum",
        vec![
            // 1D cumsum
            CumulativeTest::new(vec![1.0, 2.0, 3.0, 4.0], vec![4], 0),
            // 2D cumsum along rows
            CumulativeTest::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3], 0),
            // 2D cumsum along columns
            CumulativeTest::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3], 1),
            // 3D cumsum
            CumulativeTest::new(
                vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
                vec![2, 2, 2],
                1,
            ),
        ],
    );
}

#[test]
fn test_cumprod_parity() {
    test_cumulative_parity(
        "cumprod",
        vec![
            // 1D cumprod
            CumulativeTest::new(vec![1.0, 2.0, 3.0, 4.0], vec![4], 0),
            // 2D cumprod along rows
            CumulativeTest::new(vec![2.0, 3.0, 4.0, 5.0, 6.0, 7.0], vec![2, 3], 0),
            // 2D cumprod along columns
            CumulativeTest::new(vec![2.0, 3.0, 4.0, 5.0, 6.0, 7.0], vec![2, 3], 1),
        ],
    );
}

#[test]
fn test_logsumexp_parity() {
    test_cumulative_parity(
        "logsumexp",
        vec![
            // 1D logsumexp
            CumulativeTest::new(vec![1.0, 2.0, 3.0, 4.0], vec![4], 0),
            // 2D logsumexp along rows
            CumulativeTest::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3], 0),
            // 2D logsumexp along columns
            CumulativeTest::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3], 1),
        ],
    );
}
