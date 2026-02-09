// Backend parity tests for CompareOps trait
//
// Tests verify that all CompareOps operations produce identical results across
// CPU, CUDA, and WebGPU backends.

use numr::ops::CompareOps;
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

struct CompareTest {
    a: Vec<f32>,
    a_shape: Vec<usize>,
    b: Vec<f32>,
    b_shape: Vec<usize>,
}

impl CompareTest {
    fn new(a: Vec<f32>, a_shape: Vec<usize>, b: Vec<f32>, b_shape: Vec<usize>) -> Self {
        CompareTest {
            a,
            a_shape,
            b,
            b_shape,
        }
    }
}

fn apply_compare_op<R: Runtime>(
    client: &impl CompareOps<R>,
    op: &str,
    a: &Tensor<R>,
    b: &Tensor<R>,
) -> numr::error::Result<Tensor<R>> {
    match op {
        "eq" => client.eq(a, b),
        "ne" => client.ne(a, b),
        "lt" => client.lt(a, b),
        "le" => client.le(a, b),
        "gt" => client.gt(a, b),
        "ge" => client.ge(a, b),
        _ => panic!("Unknown compare op: {}", op),
    }
}

fn test_compare_parity(op: &str, test_cases: Vec<CompareTest>) {
    // CPU baseline
    let cpu_results: Vec<Vec<f32>> = test_cases
        .iter()
        .map(|tc| {
            let (client, device) = create_cpu_client();
            let a = Tensor::from_slice(&tc.a, &tc.a_shape, &device);
            let b = Tensor::from_slice(&tc.b, &tc.b_shape, &device);
            apply_compare_op(&client, op, &a, &b)
                .expect("CPU operation failed")
                .to_vec::<f32>()
        })
        .collect();

    // CUDA parity
    #[cfg(feature = "cuda")]
    with_cuda_backend(|cuda_client, cuda_device| {
        for (idx, tc) in test_cases.iter().enumerate() {
            let a = Tensor::from_slice(&tc.a, &tc.a_shape, &cuda_device);
            let b = Tensor::from_slice(&tc.b, &tc.b_shape, &cuda_device);
            let result = apply_compare_op(&cuda_client, op, &a, &b)
                .expect("CUDA operation failed")
                .to_vec::<f32>();
            assert_case_parity_f32(&cpu_results, idx, &result, op, "cuda");
        }
    });

    // WebGPU parity
    #[cfg(feature = "wgpu")]
    with_wgpu_backend(|wgpu_client, wgpu_device| {
        for (idx, tc) in test_cases.iter().enumerate() {
            let a = Tensor::from_slice(&tc.a, &tc.a_shape, &wgpu_device);
            let b = Tensor::from_slice(&tc.b, &tc.b_shape, &wgpu_device);
            let result = apply_compare_op(&wgpu_client, op, &a, &b)
                .expect("WebGPU operation failed")
                .to_vec::<f32>();
            assert_case_parity_f32(&cpu_results, idx, &result, op, "wgpu");
        }
    });
}

// ============================================================================
// Compare Operation Parity Tests
// ============================================================================

#[test]
fn test_eq_parity() {
    test_compare_parity(
        "eq",
        vec![
            CompareTest::new(
                vec![1.0, 2.0, 3.0, 4.0],
                vec![4],
                vec![1.0, 2.0, 0.0, 4.0],
                vec![4],
            ),
            CompareTest::new(
                vec![5.0, 5.0, 5.0, 5.0],
                vec![2, 2],
                vec![5.0, 5.0, 5.0, 5.0],
                vec![2, 2],
            ),
        ],
    );
}

#[test]
fn test_ne_parity() {
    test_compare_parity(
        "ne",
        vec![
            CompareTest::new(
                vec![1.0, 2.0, 3.0, 4.0],
                vec![4],
                vec![1.0, 2.0, 0.0, 4.0],
                vec![4],
            ),
            CompareTest::new(
                vec![5.0, 6.0, 7.0, 8.0],
                vec![2, 2],
                vec![5.0, 0.0, 7.0, 0.0],
                vec![2, 2],
            ),
        ],
    );
}

#[test]
fn test_lt_parity() {
    test_compare_parity(
        "lt",
        vec![
            CompareTest::new(
                vec![1.0, 2.0, 3.0, 4.0],
                vec![4],
                vec![2.0, 2.0, 2.0, 5.0],
                vec![4],
            ),
            CompareTest::new(
                vec![1.0, 5.0, 3.0, 7.0],
                vec![2, 2],
                vec![2.0, 4.0, 3.0, 8.0],
                vec![2, 2],
            ),
        ],
    );
}

#[test]
fn test_le_parity() {
    test_compare_parity(
        "le",
        vec![
            CompareTest::new(
                vec![1.0, 2.0, 3.0, 4.0],
                vec![4],
                vec![2.0, 2.0, 2.0, 5.0],
                vec![4],
            ),
            CompareTest::new(
                vec![1.0, 5.0, 3.0, 7.0],
                vec![2, 2],
                vec![2.0, 4.0, 3.0, 8.0],
                vec![2, 2],
            ),
        ],
    );
}

#[test]
fn test_gt_parity() {
    test_compare_parity(
        "gt",
        vec![
            CompareTest::new(
                vec![3.0, 2.0, 1.0, 5.0],
                vec![4],
                vec![2.0, 2.0, 2.0, 4.0],
                vec![4],
            ),
            CompareTest::new(
                vec![5.0, 3.0, 4.0, 2.0],
                vec![2, 2],
                vec![2.0, 4.0, 3.0, 1.0],
                vec![2, 2],
            ),
        ],
    );
}

#[test]
fn test_ge_parity() {
    test_compare_parity(
        "ge",
        vec![
            CompareTest::new(
                vec![3.0, 2.0, 1.0, 5.0],
                vec![4],
                vec![2.0, 2.0, 2.0, 4.0],
                vec![4],
            ),
            CompareTest::new(
                vec![5.0, 3.0, 4.0, 2.0],
                vec![2, 2],
                vec![2.0, 4.0, 3.0, 1.0],
                vec![2, 2],
            ),
        ],
    );
}
