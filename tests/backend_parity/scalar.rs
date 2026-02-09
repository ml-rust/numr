// Backend parity tests for ScalarOps trait
//
// Tests verify that all ScalarOps operations produce identical results across
// CPU, CUDA, and WebGPU backends.

use numr::ops::ScalarOps;
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

struct ScalarTest {
    data: Vec<f32>,
    shape: Vec<usize>,
    scalar: f64,
}

impl ScalarTest {
    fn new(data: Vec<f32>, shape: Vec<usize>, scalar: f64) -> Self {
        ScalarTest {
            data,
            shape,
            scalar,
        }
    }
}

fn apply_scalar_op<R: Runtime>(
    client: &impl ScalarOps<R>,
    op: &str,
    tensor: &Tensor<R>,
    scalar: f64,
) -> numr::error::Result<Tensor<R>> {
    match op {
        "add_scalar" => client.add_scalar(tensor, scalar),
        "sub_scalar" => client.sub_scalar(tensor, scalar),
        "mul_scalar" => client.mul_scalar(tensor, scalar),
        "div_scalar" => client.div_scalar(tensor, scalar),
        "pow_scalar" => client.pow_scalar(tensor, scalar),
        "rsub_scalar" => client.rsub_scalar(tensor, scalar),
        _ => panic!("Unknown scalar op: {}", op),
    }
}

fn test_scalar_parity(op: &str, test_cases: Vec<ScalarTest>) {
    // CPU baseline
    let cpu_results: Vec<Vec<f32>> = test_cases
        .iter()
        .map(|tc| {
            let (client, device) = create_cpu_client();
            let tensor = Tensor::from_slice(&tc.data, &tc.shape, &device);
            apply_scalar_op(&client, op, &tensor, tc.scalar)
                .expect("CPU operation failed")
                .to_vec::<f32>()
        })
        .collect();

    // CUDA parity
    #[cfg(feature = "cuda")]
    with_cuda_backend(|cuda_client, cuda_device| {
        for (idx, tc) in test_cases.iter().enumerate() {
            let tensor = Tensor::from_slice(&tc.data, &tc.shape, &cuda_device);
            let result = apply_scalar_op(&cuda_client, op, &tensor, tc.scalar)
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
            let result = apply_scalar_op(&wgpu_client, op, &tensor, tc.scalar)
                .expect("WebGPU operation failed")
                .to_vec::<f32>();
            assert_case_parity_f32(&cpu_results, idx, &result, op, "wgpu");
        }
    });
}

// ============================================================================
// Scalar Operation Parity Tests
// ============================================================================

#[test]
fn test_add_scalar_parity() {
    test_scalar_parity(
        "add_scalar",
        vec![
            ScalarTest::new(vec![1.0, 2.0, 3.0, 4.0], vec![4], 5.0),
            ScalarTest::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], -2.5),
            ScalarTest::new(vec![0.5, 1.5, 2.5, 3.5], vec![2, 2], 10.0),
        ],
    );
}

#[test]
fn test_sub_scalar_parity() {
    test_scalar_parity(
        "sub_scalar",
        vec![
            ScalarTest::new(vec![5.0, 6.0, 7.0, 8.0], vec![4], 2.0),
            ScalarTest::new(vec![10.0, 20.0, 30.0, 40.0], vec![2, 2], 5.0),
            ScalarTest::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], 0.5),
        ],
    );
}

#[test]
fn test_mul_scalar_parity() {
    test_scalar_parity(
        "mul_scalar",
        vec![
            ScalarTest::new(vec![1.0, 2.0, 3.0, 4.0], vec![4], 2.0),
            ScalarTest::new(vec![2.0, 4.0, 6.0, 8.0], vec![2, 2], 0.5),
            ScalarTest::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], -3.0),
        ],
    );
}

#[test]
fn test_div_scalar_parity() {
    test_scalar_parity(
        "div_scalar",
        vec![
            ScalarTest::new(vec![10.0, 20.0, 30.0, 40.0], vec![4], 2.0),
            ScalarTest::new(vec![100.0, 200.0, 300.0, 400.0], vec![2, 2], 10.0),
            ScalarTest::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], 4.0),
        ],
    );
}

#[test]
fn test_pow_scalar_parity() {
    test_scalar_parity(
        "pow_scalar",
        vec![
            ScalarTest::new(vec![2.0, 3.0, 4.0, 5.0], vec![4], 2.0),
            ScalarTest::new(vec![2.0, 3.0, 4.0, 5.0], vec![2, 2], 3.0),
            ScalarTest::new(vec![4.0, 9.0, 16.0, 25.0], vec![2, 2], 0.5),
        ],
    );
}

#[test]
fn test_rsub_scalar_parity() {
    test_scalar_parity(
        "rsub_scalar",
        vec![
            ScalarTest::new(vec![1.0, 2.0, 3.0, 4.0], vec![4], 10.0),
            ScalarTest::new(vec![2.0, 3.0, 4.0, 5.0], vec![2, 2], 20.0),
            ScalarTest::new(vec![0.5, 1.5, 2.5, 3.5], vec![2, 2], 5.0),
        ],
    );
}
