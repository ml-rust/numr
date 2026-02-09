#![allow(clippy::approx_constant, clippy::excessive_precision)]
// Backend parity tests for UnaryOps trait
//
// Tests verify that all UnaryOps operations produce identical results across
// CPU, CUDA, and WebGPU backends.

use numr::ops::UnaryOps;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

#[cfg(any(feature = "cuda", feature = "wgpu"))]
use crate::backend_parity::helpers::assert_case_parity_f32;
use crate::backend_parity::helpers::assert_parity_u32;
#[cfg(feature = "cuda")]
use crate::backend_parity::helpers::with_cuda_backend;
#[cfg(feature = "wgpu")]
use crate::backend_parity::helpers::with_wgpu_backend;
use crate::common::create_cpu_client;

// ============================================================================
// Test Utilities
// ============================================================================

/// Test data helper: creates input data and shapes for testing
#[derive(Clone)]
struct TestInput {
    data: Vec<f32>,
    shape: Vec<usize>,
}

impl TestInput {
    fn new(data: Vec<f32>, shape: Vec<usize>) -> Self {
        TestInput { data, shape }
    }
}

/// Apply unary operation to tensor
fn apply_unary_op<R: Runtime>(
    client: &impl UnaryOps<R>,
    op: &str,
    x: &Tensor<R>,
) -> numr::error::Result<Tensor<R>> {
    match op {
        "neg" => client.neg(x),
        "abs" => client.abs(x),
        "sign" => client.sign(x),
        "sqrt" => client.sqrt(x),
        "rsqrt" => client.rsqrt(x),
        "square" => client.square(x),
        "cbrt" => client.cbrt(x),
        "recip" => client.recip(x),
        "exp" => client.exp(x),
        "exp2" => client.exp2(x),
        "expm1" => client.expm1(x),
        "log" => client.log(x),
        "log2" => client.log2(x),
        "log10" => client.log10(x),
        "log1p" => client.log1p(x),
        "sin" => client.sin(x),
        "cos" => client.cos(x),
        "tan" => client.tan(x),
        "asin" => client.asin(x),
        "acos" => client.acos(x),
        "atan" => client.atan(x),
        "sinh" => client.sinh(x),
        "cosh" => client.cosh(x),
        "tanh" => client.tanh(x),
        "asinh" => client.asinh(x),
        "acosh" => client.acosh(x),
        "atanh" => client.atanh(x),
        "floor" => client.floor(x),
        "ceil" => client.ceil(x),
        "round" => client.round(x),
        "trunc" => client.trunc(x),
        "isnan" => client.isnan(x),
        "isinf" => client.isinf(x),
        _ => panic!("Unknown unary op: {}", op),
    }
}

/// Helper to test parity for a unary operation
fn test_unary_parity_impl(op: &str, test_inputs: Vec<TestInput>) {
    // CPU baseline (always runs)
    let cpu_results: Vec<Vec<f32>> = test_inputs
        .iter()
        .map(|input| {
            let (client, device) = create_cpu_client();
            let tensor = Tensor::from_slice(&input.data, &input.shape, &device);
            apply_unary_op(&client, op, &tensor)
                .expect("CPU operation failed")
                .to_vec::<f32>()
        })
        .collect();

    // CUDA parity test (if available)
    #[cfg(feature = "cuda")]
    with_cuda_backend(|cuda_client, cuda_device| {
        for (idx, input) in test_inputs.iter().enumerate() {
            let tensor = Tensor::from_slice(&input.data, &input.shape, &cuda_device);
            let cuda_result = apply_unary_op(&cuda_client, op, &tensor)
                .expect("CUDA operation failed")
                .to_vec::<f32>();
            assert_case_parity_f32(&cpu_results, idx, &cuda_result, op, "cuda");
        }
    });

    // WebGPU parity test (if available)
    #[cfg(feature = "wgpu")]
    with_wgpu_backend(|wgpu_client, wgpu_device| {
        for (idx, input) in test_inputs.iter().enumerate() {
            let tensor = Tensor::from_slice(&input.data, &input.shape, &wgpu_device);
            let wgpu_result = apply_unary_op(&wgpu_client, op, &tensor)
                .expect("WebGPU operation failed")
                .to_vec::<f32>();
            assert_case_parity_f32(&cpu_results, idx, &wgpu_result, op, "wgpu");
        }
    });
}

// ============================================================================
// Unary Operation Parity Tests
// ============================================================================

#[test]
fn test_neg_parity() {
    test_unary_parity_impl(
        "neg",
        vec![
            TestInput::new(vec![1.0, -2.0, 3.0, -4.0], vec![4]),
            TestInput::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]),
        ],
    );
}

#[test]
fn test_abs_parity() {
    test_unary_parity_impl(
        "abs",
        vec![
            TestInput::new(vec![1.0, -2.0, 3.0, -4.0], vec![4]),
            TestInput::new(vec![-1.0, -2.0, -3.0, -4.0], vec![2, 2]),
        ],
    );
}

#[test]
fn test_sign_parity() {
    test_unary_parity_impl(
        "sign",
        vec![
            TestInput::new(vec![1.0, -2.0, 0.0, -4.0], vec![4]),
            TestInput::new(vec![-5.0, 0.0, 5.0, 0.0], vec![2, 2]),
        ],
    );
}

#[test]
fn test_sqrt_parity() {
    test_unary_parity_impl(
        "sqrt",
        vec![
            TestInput::new(vec![1.0, 4.0, 9.0, 16.0], vec![4]),
            TestInput::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]),
        ],
    );
}

#[test]
fn test_rsqrt_parity() {
    test_unary_parity_impl(
        "rsqrt",
        vec![
            TestInput::new(vec![1.0, 4.0, 9.0, 16.0], vec![4]),
            TestInput::new(vec![2.0, 4.0, 8.0, 16.0], vec![2, 2]),
        ],
    );
}

#[test]
fn test_square_parity() {
    test_unary_parity_impl(
        "square",
        vec![
            TestInput::new(vec![1.0, -2.0, 3.0, -4.0], vec![4]),
            TestInput::new(vec![2.0, 3.0, 4.0, 5.0], vec![2, 2]),
        ],
    );
}

#[test]
fn test_cbrt_parity() {
    test_unary_parity_impl(
        "cbrt",
        vec![
            TestInput::new(vec![1.0, 8.0, 27.0, 64.0], vec![4]),
            TestInput::new(vec![-8.0, 0.0, 8.0, 1.0], vec![2, 2]),
        ],
    );
}

#[test]
fn test_recip_parity() {
    test_unary_parity_impl(
        "recip",
        vec![
            TestInput::new(vec![1.0, 2.0, 4.0, 5.0], vec![4]),
            TestInput::new(vec![2.0, 4.0, 5.0, 10.0], vec![2, 2]),
        ],
    );
}

#[test]
fn test_exp_parity() {
    test_unary_parity_impl(
        "exp",
        vec![
            TestInput::new(vec![0.0, 1.0, -1.0, 2.0], vec![4]),
            TestInput::new(vec![0.5, -0.5, 1.0, -1.0], vec![2, 2]),
        ],
    );
}

#[test]
fn test_exp2_parity() {
    test_unary_parity_impl(
        "exp2",
        vec![
            TestInput::new(vec![0.0, 1.0, 2.0, 3.0], vec![4]),
            TestInput::new(vec![-1.0, 0.0, 1.0, 2.0], vec![2, 2]),
        ],
    );
}

#[test]
fn test_expm1_parity() {
    test_unary_parity_impl(
        "expm1",
        vec![
            TestInput::new(vec![0.0, 0.1, -0.1, 0.5], vec![4]),
            TestInput::new(vec![0.0, 0.01, -0.01, 1.0], vec![2, 2]),
        ],
    );
}

#[test]
fn test_log_parity() {
    test_unary_parity_impl(
        "log",
        vec![
            TestInput::new(vec![1.0, 2.0, 4.0, 10.0], vec![4]),
            TestInput::new(vec![1.0, 2.0, 5.0, 10.0], vec![2, 2]),
        ],
    );
}

#[test]
fn test_log2_parity() {
    test_unary_parity_impl(
        "log2",
        vec![
            TestInput::new(vec![1.0, 2.0, 4.0, 8.0], vec![4]),
            TestInput::new(vec![2.0, 4.0, 8.0, 16.0], vec![2, 2]),
        ],
    );
}

#[test]
fn test_log10_parity() {
    test_unary_parity_impl(
        "log10",
        vec![
            TestInput::new(vec![1.0, 10.0, 100.0, 1000.0], vec![4]),
            TestInput::new(vec![10.0, 100.0, 1000.0, 10000.0], vec![2, 2]),
        ],
    );
}

#[test]
fn test_log1p_parity() {
    test_unary_parity_impl(
        "log1p",
        vec![
            TestInput::new(vec![0.0, 0.1, 1.0, 9.0], vec![4]),
            TestInput::new(vec![0.0, 0.01, 1.0, 99.0], vec![2, 2]),
        ],
    );
}

#[test]
fn test_sin_parity() {
    test_unary_parity_impl(
        "sin",
        vec![
            TestInput::new(vec![0.0, 1.57079633, 3.14159265, -1.57079633], vec![4]),
            TestInput::new(vec![0.5, 1.0, -0.5, -1.0], vec![2, 2]),
        ],
    );
}

#[test]
fn test_cos_parity() {
    test_unary_parity_impl(
        "cos",
        vec![
            TestInput::new(vec![0.0, 1.57079633, 3.14159265, -1.57079633], vec![4]),
            TestInput::new(vec![0.5, 1.0, -0.5, -1.0], vec![2, 2]),
        ],
    );
}

#[test]
fn test_tan_parity() {
    test_unary_parity_impl(
        "tan",
        vec![
            TestInput::new(vec![0.0, 0.4, -0.4, 0.785398163], vec![4]),
            TestInput::new(vec![0.1, -0.1, 0.2, -0.2], vec![2, 2]),
        ],
    );
}

#[test]
fn test_asin_parity() {
    test_unary_parity_impl(
        "asin",
        vec![
            TestInput::new(vec![0.0, 0.5, -0.5, 1.0], vec![4]),
            TestInput::new(vec![-1.0, -0.5, 0.5, 1.0], vec![2, 2]),
        ],
    );
}

#[test]
fn test_acos_parity() {
    test_unary_parity_impl(
        "acos",
        vec![
            TestInput::new(vec![0.0, 0.5, -0.5, 1.0], vec![4]),
            TestInput::new(vec![-1.0, -0.5, 0.5, 1.0], vec![2, 2]),
        ],
    );
}

#[test]
fn test_atan_parity() {
    test_unary_parity_impl(
        "atan",
        vec![
            TestInput::new(vec![0.0, 1.0, -1.0, 10.0], vec![4]),
            TestInput::new(vec![-10.0, -1.0, 1.0, 10.0], vec![2, 2]),
        ],
    );
}

#[test]
fn test_sinh_parity() {
    test_unary_parity_impl(
        "sinh",
        vec![
            TestInput::new(vec![0.0, 1.0, -1.0, 2.0], vec![4]),
            TestInput::new(vec![-1.0, -0.5, 0.5, 1.0], vec![2, 2]),
        ],
    );
}

#[test]
fn test_cosh_parity() {
    test_unary_parity_impl(
        "cosh",
        vec![
            TestInput::new(vec![0.0, 1.0, -1.0, 2.0], vec![4]),
            TestInput::new(vec![-1.0, -0.5, 0.5, 1.0], vec![2, 2]),
        ],
    );
}

#[test]
fn test_tanh_parity() {
    test_unary_parity_impl(
        "tanh",
        vec![
            TestInput::new(vec![0.0, 1.0, -1.0, 2.0], vec![4]),
            TestInput::new(vec![-1.0, -0.5, 0.5, 1.0], vec![2, 2]),
        ],
    );
}

#[test]
fn test_asinh_parity() {
    test_unary_parity_impl(
        "asinh",
        vec![
            TestInput::new(vec![0.0, 1.0, -1.0, 10.0], vec![4]),
            TestInput::new(vec![-10.0, -1.0, 1.0, 10.0], vec![2, 2]),
        ],
    );
}

#[test]
fn test_acosh_parity() {
    test_unary_parity_impl(
        "acosh",
        vec![
            TestInput::new(vec![1.0, 2.0, 5.0, 10.0], vec![4]),
            TestInput::new(vec![1.0, 1.5, 2.5, 10.0], vec![2, 2]),
        ],
    );
}

#[test]
fn test_atanh_parity() {
    test_unary_parity_impl(
        "atanh",
        vec![
            TestInput::new(vec![0.0, 0.5, -0.5, 0.9], vec![4]),
            TestInput::new(vec![-0.5, -0.1, 0.1, 0.5], vec![2, 2]),
        ],
    );
}

#[test]
fn test_floor_parity() {
    test_unary_parity_impl(
        "floor",
        vec![
            TestInput::new(vec![1.1, -2.3, 3.9, -4.7], vec![4]),
            TestInput::new(vec![0.5, 1.5, -0.5, -1.5], vec![2, 2]),
        ],
    );
}

#[test]
fn test_ceil_parity() {
    test_unary_parity_impl(
        "ceil",
        vec![
            TestInput::new(vec![1.1, -2.3, 3.9, -4.7], vec![4]),
            TestInput::new(vec![0.5, 1.5, -0.5, -1.5], vec![2, 2]),
        ],
    );
}

#[test]
fn test_round_parity() {
    test_unary_parity_impl(
        "round",
        vec![
            TestInput::new(vec![1.1, -2.3, 3.9, -4.7], vec![4]),
            TestInput::new(vec![0.5, 1.5, -0.5, -1.5], vec![2, 2]),
        ],
    );
}

#[test]
fn test_trunc_parity() {
    test_unary_parity_impl(
        "trunc",
        vec![
            TestInput::new(vec![1.1, -2.3, 3.9, -4.7], vec![4]),
            TestInput::new(vec![0.5, 1.5, -0.5, -1.5], vec![2, 2]),
        ],
    );
}

#[test]
fn test_isnan_parity() {
    let data = vec![0.0, f32::NAN, 1.0, f32::NAN];
    let shape = vec![4];
    let (cpu_client, cpu_device) = create_cpu_client();
    let cpu_tensor = Tensor::from_slice(&data, &shape, &cpu_device);
    let cpu_result = cpu_client.isnan(&cpu_tensor).unwrap().to_vec::<u8>();
    let cpu_result_u32: Vec<u32> = cpu_result.iter().map(|&x| x as u32).collect();

    #[cfg(feature = "cuda")]
    with_cuda_backend(|cuda_client, cuda_device| {
        let cuda_tensor = Tensor::from_slice(&data, &shape, &cuda_device);
        let cuda_result = cuda_client.isnan(&cuda_tensor).unwrap().to_vec::<u8>();
        assert_parity_u32(
            &cpu_result_u32,
            &cuda_result.iter().map(|&x| x as u32).collect::<Vec<_>>(),
            "isnan_cuda",
        );
    });

    #[cfg(feature = "wgpu")]
    with_wgpu_backend(|wgpu_client, wgpu_device| {
        let wgpu_tensor = Tensor::from_slice(&data, &shape, &wgpu_device);
        let wgpu_result = wgpu_client.isnan(&wgpu_tensor).unwrap().to_vec::<u32>();
        assert_parity_u32(&cpu_result_u32, &wgpu_result, "isnan_wgpu");
    });
}

#[test]
fn test_isinf_parity() {
    let data = vec![0.0, f32::INFINITY, 1.0, f32::NEG_INFINITY];
    let shape = vec![4];
    let (cpu_client, cpu_device) = create_cpu_client();
    let cpu_tensor = Tensor::from_slice(&data, &shape, &cpu_device);
    let cpu_result = cpu_client.isinf(&cpu_tensor).unwrap().to_vec::<u8>();
    let cpu_result_u32: Vec<u32> = cpu_result.iter().map(|&x| x as u32).collect();

    #[cfg(feature = "cuda")]
    with_cuda_backend(|cuda_client, cuda_device| {
        let cuda_tensor = Tensor::from_slice(&data, &shape, &cuda_device);
        let cuda_result = cuda_client.isinf(&cuda_tensor).unwrap().to_vec::<u8>();
        assert_parity_u32(
            &cpu_result_u32,
            &cuda_result.iter().map(|&x| x as u32).collect::<Vec<_>>(),
            "isinf_cuda",
        );
    });

    #[cfg(feature = "wgpu")]
    with_wgpu_backend(|wgpu_client, wgpu_device| {
        let wgpu_tensor = Tensor::from_slice(&data, &shape, &wgpu_device);
        let wgpu_result = wgpu_client.isinf(&wgpu_tensor).unwrap().to_vec::<u32>();
        assert_parity_u32(&cpu_result_u32, &wgpu_result, "isinf_wgpu");
    });
}
