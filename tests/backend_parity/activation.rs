// Backend parity tests for fused activation-mul operations (ActivationOps trait)
//
// Tests: silu_mul, gelu_mul, relu_mul, sigmoid_mul (forward)
//        silu_mul_bwd, gelu_mul_bwd, relu_mul_bwd, sigmoid_mul_bwd (backward)
//
// Dtype-parameterized: each test runs for all supported dtypes across all backends.

use numr::dtype::DType;
use numr::ops::ActivationOps;
use numr::runtime::Runtime;
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

#[derive(Clone)]
struct FusedTestCase {
    a: Vec<f64>,
    b: Vec<f64>,
    shape: Vec<usize>,
}

impl FusedTestCase {
    fn new(a: Vec<f64>, b: Vec<f64>, shape: Vec<usize>) -> Self {
        Self { a, b, shape }
    }
}

#[derive(Clone, Copy, Debug)]
enum FusedActivationOp {
    SiluMul,
    GeluMul,
    ReluMul,
    SigmoidMul,
}

fn apply_fused_fwd<R: Runtime>(
    client: &impl ActivationOps<R>,
    op: FusedActivationOp,
    a: &Tensor<R>,
    b: &Tensor<R>,
) -> numr::error::Result<Tensor<R>> {
    match op {
        FusedActivationOp::SiluMul => client.silu_mul(a, b),
        FusedActivationOp::GeluMul => client.gelu_mul(a, b),
        FusedActivationOp::ReluMul => client.relu_mul(a, b),
        FusedActivationOp::SigmoidMul => client.sigmoid_mul(a, b),
    }
}

fn apply_fused_bwd<R: Runtime>(
    client: &impl ActivationOps<R>,
    op: FusedActivationOp,
    grad: &Tensor<R>,
    a: &Tensor<R>,
    b: &Tensor<R>,
) -> numr::error::Result<(Tensor<R>, Tensor<R>)> {
    match op {
        FusedActivationOp::SiluMul => client.silu_mul_bwd(grad, a, b),
        FusedActivationOp::GeluMul => client.gelu_mul_bwd(grad, a, b),
        FusedActivationOp::ReluMul => client.relu_mul_bwd(grad, a, b),
        FusedActivationOp::SigmoidMul => client.sigmoid_mul_bwd(grad, a, b),
    }
}

// ============================================================================
// Forward parity tests
// ============================================================================

fn test_fused_fwd_parity(op: FusedActivationOp, test_cases: &[FusedTestCase], dtype: DType) {
    let (cpu_client, cpu_device) = create_cpu_client();

    let cpu_results: Vec<Tensor<numr::runtime::cpu::CpuRuntime>> = test_cases
        .iter()
        .map(|tc| {
            let a = tensor_from_f64(&tc.a, &tc.shape, dtype, &cpu_device, &cpu_client)
                .unwrap_or_else(|e| panic!("CPU tensor_from_f64 failed for {dtype:?}: {e}"));
            let b = tensor_from_f64(&tc.b, &tc.shape, dtype, &cpu_device, &cpu_client)
                .unwrap_or_else(|e| panic!("CPU tensor_from_f64 failed for {dtype:?}: {e}"));
            apply_fused_fwd(&cpu_client, op, &a, &b)
                .unwrap_or_else(|e| panic!("CPU {op:?} failed for {dtype:?}: {e}"))
        })
        .collect();

    #[cfg(feature = "cuda")]
    if is_dtype_supported("cuda", dtype) {
        with_cuda_backend(|cuda_client, cuda_device| {
            for (idx, tc) in test_cases.iter().enumerate() {
                let a = tensor_from_f64(&tc.a, &tc.shape, dtype, &cuda_device, &cuda_client)
                    .unwrap_or_else(|e| panic!("CUDA tensor_from_f64 failed for {dtype:?}: {e}"));
                let b = tensor_from_f64(&tc.b, &tc.shape, dtype, &cuda_device, &cuda_client)
                    .unwrap_or_else(|e| panic!("CUDA tensor_from_f64 failed for {dtype:?}: {e}"));
                let result = apply_fused_fwd(&cuda_client, op, &a, &b)
                    .unwrap_or_else(|e| panic!("CUDA {op:?} failed for {dtype:?}: {e}"));
                assert_tensor_allclose(
                    &result,
                    &cpu_results[idx],
                    dtype,
                    &format!("{op:?} CUDA vs CPU [{dtype:?}] case {idx}"),
                );
            }
        });
    }

    #[cfg(feature = "wgpu")]
    if is_dtype_supported("wgpu", dtype) {
        with_wgpu_backend(|wgpu_client, wgpu_device| {
            for (idx, tc) in test_cases.iter().enumerate() {
                let a = tensor_from_f64(&tc.a, &tc.shape, dtype, &wgpu_device, &wgpu_client)
                    .unwrap_or_else(|e| panic!("WebGPU tensor_from_f64 failed for {dtype:?}: {e}"));
                let b = tensor_from_f64(&tc.b, &tc.shape, dtype, &wgpu_device, &wgpu_client)
                    .unwrap_or_else(|e| panic!("WebGPU tensor_from_f64 failed for {dtype:?}: {e}"));
                let result = apply_fused_fwd(&wgpu_client, op, &a, &b)
                    .unwrap_or_else(|e| panic!("WebGPU {op:?} failed for {dtype:?}: {e}"));
                assert_tensor_allclose(
                    &result,
                    &cpu_results[idx],
                    dtype,
                    &format!("{op:?} WebGPU vs CPU [{dtype:?}] case {idx}"),
                );
            }
        });
    }
}

// ============================================================================
// Backward parity tests
// ============================================================================

fn test_fused_bwd_parity(op: FusedActivationOp, test_cases: &[FusedTestCase], dtype: DType) {
    let (cpu_client, cpu_device) = create_cpu_client();

    // Use the same data for grad as a simple ones-like pattern
    let cpu_results: Vec<(
        Tensor<numr::runtime::cpu::CpuRuntime>,
        Tensor<numr::runtime::cpu::CpuRuntime>,
    )> = test_cases
        .iter()
        .map(|tc| {
            let a = tensor_from_f64(&tc.a, &tc.shape, dtype, &cpu_device, &cpu_client)
                .unwrap_or_else(|e| panic!("CPU tensor_from_f64 failed for {dtype:?}: {e}"));
            let b = tensor_from_f64(&tc.b, &tc.shape, dtype, &cpu_device, &cpu_client)
                .unwrap_or_else(|e| panic!("CPU tensor_from_f64 failed for {dtype:?}: {e}"));
            // Use ones as grad for simplicity
            let grad_data: Vec<f64> = vec![1.0; tc.a.len()];
            let grad = tensor_from_f64(&grad_data, &tc.shape, dtype, &cpu_device, &cpu_client)
                .unwrap_or_else(|e| panic!("CPU tensor_from_f64 failed for {dtype:?}: {e}"));
            apply_fused_bwd(&cpu_client, op, &grad, &a, &b)
                .unwrap_or_else(|e| panic!("CPU {op:?}_bwd failed for {dtype:?}: {e}"))
        })
        .collect();

    #[cfg(feature = "cuda")]
    if is_dtype_supported("cuda", dtype) {
        with_cuda_backend(|cuda_client, cuda_device| {
            for (idx, tc) in test_cases.iter().enumerate() {
                let a = tensor_from_f64(&tc.a, &tc.shape, dtype, &cuda_device, &cuda_client)
                    .unwrap_or_else(|e| panic!("CUDA tensor_from_f64 failed for {dtype:?}: {e}"));
                let b = tensor_from_f64(&tc.b, &tc.shape, dtype, &cuda_device, &cuda_client)
                    .unwrap_or_else(|e| panic!("CUDA tensor_from_f64 failed for {dtype:?}: {e}"));
                let grad_data: Vec<f64> = vec![1.0; tc.a.len()];
                let grad =
                    tensor_from_f64(&grad_data, &tc.shape, dtype, &cuda_device, &cuda_client)
                        .unwrap_or_else(|e| {
                            panic!("CUDA tensor_from_f64 failed for {dtype:?}: {e}")
                        });
                let (d_a, d_b) = apply_fused_bwd(&cuda_client, op, &grad, &a, &b)
                    .unwrap_or_else(|e| panic!("CUDA {op:?}_bwd failed for {dtype:?}: {e}"));
                assert_tensor_allclose(
                    &d_a,
                    &cpu_results[idx].0,
                    dtype,
                    &format!("{op:?}_bwd d_a CUDA vs CPU [{dtype:?}] case {idx}"),
                );
                assert_tensor_allclose(
                    &d_b,
                    &cpu_results[idx].1,
                    dtype,
                    &format!("{op:?}_bwd d_b CUDA vs CPU [{dtype:?}] case {idx}"),
                );
            }
        });
    }

    #[cfg(feature = "wgpu")]
    if is_dtype_supported("wgpu", dtype) {
        with_wgpu_backend(|wgpu_client, wgpu_device| {
            for (idx, tc) in test_cases.iter().enumerate() {
                let a = tensor_from_f64(&tc.a, &tc.shape, dtype, &wgpu_device, &wgpu_client)
                    .unwrap_or_else(|e| panic!("WebGPU tensor_from_f64 failed for {dtype:?}: {e}"));
                let b = tensor_from_f64(&tc.b, &tc.shape, dtype, &wgpu_device, &wgpu_client)
                    .unwrap_or_else(|e| panic!("WebGPU tensor_from_f64 failed for {dtype:?}: {e}"));
                let grad_data: Vec<f64> = vec![1.0; tc.a.len()];
                let grad =
                    tensor_from_f64(&grad_data, &tc.shape, dtype, &wgpu_device, &wgpu_client)
                        .unwrap_or_else(|e| {
                            panic!("WebGPU tensor_from_f64 failed for {dtype:?}: {e}")
                        });
                let (d_a, d_b) = apply_fused_bwd(&wgpu_client, op, &grad, &a, &b)
                    .unwrap_or_else(|e| panic!("WebGPU {op:?}_bwd failed for {dtype:?}: {e}"));
                assert_tensor_allclose(
                    &d_a,
                    &cpu_results[idx].0,
                    dtype,
                    &format!("{op:?}_bwd d_a WebGPU vs CPU [{dtype:?}] case {idx}"),
                );
                assert_tensor_allclose(
                    &d_b,
                    &cpu_results[idx].1,
                    dtype,
                    &format!("{op:?}_bwd d_b WebGPU vs CPU [{dtype:?}] case {idx}"),
                );
            }
        });
    }
}

// ============================================================================
// Test data
// ============================================================================

fn standard_test_cases() -> Vec<FusedTestCase> {
    vec![
        // Small 1D
        FusedTestCase::new(
            vec![-2.0, -1.0, 0.0, 1.0, 2.0],
            vec![0.5, 1.0, 1.5, 2.0, 0.3],
            vec![5],
        ),
        // 2D matrix
        FusedTestCase::new(
            vec![-1.0, 0.5, 1.5, -0.5, 2.0, -2.0],
            vec![1.0, 2.0, 0.5, 1.5, 0.3, 1.0],
            vec![2, 3],
        ),
        // Values near zero (important for derivative accuracy)
        FusedTestCase::new(
            vec![0.01, -0.01, 0.1, -0.1],
            vec![1.0, 1.0, 1.0, 1.0],
            vec![4],
        ),
        // Larger values (tests saturation behavior)
        FusedTestCase::new(
            vec![5.0, -5.0, 10.0, -10.0],
            vec![0.1, 0.2, 0.3, 0.4],
            vec![4],
        ),
    ]
}

// ============================================================================
// Forward tests
// ============================================================================

#[test]
fn test_silu_mul_parity() {
    let cases = standard_test_cases();
    for dtype in supported_dtypes("cpu") {
        test_fused_fwd_parity(FusedActivationOp::SiluMul, &cases, dtype);
    }
}

#[test]
fn test_gelu_mul_parity() {
    let cases = standard_test_cases();
    for dtype in supported_dtypes("cpu") {
        test_fused_fwd_parity(FusedActivationOp::GeluMul, &cases, dtype);
    }
}

#[test]
fn test_relu_mul_parity() {
    let cases = standard_test_cases();
    for dtype in supported_dtypes("cpu") {
        test_fused_fwd_parity(FusedActivationOp::ReluMul, &cases, dtype);
    }
}

#[test]
fn test_sigmoid_mul_parity() {
    let cases = standard_test_cases();
    for dtype in supported_dtypes("cpu") {
        test_fused_fwd_parity(FusedActivationOp::SigmoidMul, &cases, dtype);
    }
}

// ============================================================================
// Backward tests
// ============================================================================

#[test]
fn test_silu_mul_bwd_parity() {
    let cases = standard_test_cases();
    for dtype in supported_dtypes("cpu") {
        test_fused_bwd_parity(FusedActivationOp::SiluMul, &cases, dtype);
    }
}

#[test]
fn test_gelu_mul_bwd_parity() {
    let cases = standard_test_cases();
    for dtype in supported_dtypes("cpu") {
        test_fused_bwd_parity(FusedActivationOp::GeluMul, &cases, dtype);
    }
}

#[test]
fn test_relu_mul_bwd_parity() {
    let cases = standard_test_cases();
    for dtype in supported_dtypes("cpu") {
        test_fused_bwd_parity(FusedActivationOp::ReluMul, &cases, dtype);
    }
}

#[test]
fn test_sigmoid_mul_bwd_parity() {
    let cases = standard_test_cases();
    for dtype in supported_dtypes("cpu") {
        test_fused_bwd_parity(FusedActivationOp::SigmoidMul, &cases, dtype);
    }
}
