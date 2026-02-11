// Backend parity tests for ScalarOps trait
//
// Dtype-parameterized: each test runs for all supported dtypes across all backends.
// Comparison reads back in native dtype via assert_tensor_allclose.

use numr::dtype::DType;
use numr::ops::ScalarOps;
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
struct ScalarTest {
    data: Vec<f64>,
    shape: Vec<usize>,
    scalar: f64,
}

impl ScalarTest {
    fn new(data: Vec<f64>, shape: Vec<usize>, scalar: f64) -> Self {
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

fn test_scalar_parity(op: &str, test_cases: &[ScalarTest], dtype: DType) {
    let (cpu_client, cpu_device) = create_cpu_client();

    let cpu_results: Vec<Tensor<numr::runtime::cpu::CpuRuntime>> = test_cases
        .iter()
        .map(|tc| {
            let tensor = tensor_from_f64(&tc.data, &tc.shape, dtype, &cpu_device, &cpu_client)
                .unwrap_or_else(|e| panic!("CPU tensor_from_f64 failed for {dtype:?}: {e}"));
            apply_scalar_op(&cpu_client, op, &tensor, tc.scalar)
                .unwrap_or_else(|e| panic!("CPU {op} failed for {dtype:?}: {e}"))
        })
        .collect();

    #[cfg(feature = "cuda")]
    if is_dtype_supported("cuda", dtype) {
        with_cuda_backend(|cuda_client, cuda_device| {
            for (idx, tc) in test_cases.iter().enumerate() {
                let tensor =
                    tensor_from_f64(&tc.data, &tc.shape, dtype, &cuda_device, &cuda_client)
                        .unwrap_or_else(|e| {
                            panic!("CUDA tensor_from_f64 failed for {dtype:?}: {e}")
                        });
                let result = apply_scalar_op(&cuda_client, op, &tensor, tc.scalar)
                    .unwrap_or_else(|e| panic!("CUDA {op} failed for {dtype:?}: {e}"));
                assert_tensor_allclose(
                    &result,
                    &cpu_results[idx],
                    dtype,
                    &format!("{op} CUDA vs CPU [{dtype:?}] case {idx}"),
                );
            }
        });
    }

    #[cfg(feature = "wgpu")]
    if is_dtype_supported("wgpu", dtype) {
        with_wgpu_backend(|wgpu_client, wgpu_device| {
            for (idx, tc) in test_cases.iter().enumerate() {
                let tensor =
                    tensor_from_f64(&tc.data, &tc.shape, dtype, &wgpu_device, &wgpu_client)
                        .unwrap_or_else(|e| {
                            panic!("WebGPU tensor_from_f64 failed for {dtype:?}: {e}")
                        });
                let result = apply_scalar_op(&wgpu_client, op, &tensor, tc.scalar)
                    .unwrap_or_else(|e| panic!("WebGPU {op} failed for {dtype:?}: {e}"));
                assert_tensor_allclose(
                    &result,
                    &cpu_results[idx],
                    dtype,
                    &format!("{op} WebGPU vs CPU [{dtype:?}] case {idx}"),
                );
            }
        });
    }
}

macro_rules! scalar_case {
    ($name:ident, $op:expr, $cases:expr) => {
        #[test]
        fn $name() {
            for dtype in supported_dtypes("cpu") {
                test_scalar_parity($op, $cases, dtype);
            }
        }
    };
}

// ============================================================================
// Scalar Operation Parity Tests
// ============================================================================

scalar_case!(
    test_add_scalar_parity,
    "add_scalar",
    &[
        ScalarTest::new(vec![1.0, 2.0, 3.0, 4.0], vec![4], 5.0),
        ScalarTest::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], -2.5),
        ScalarTest::new(vec![0.5, 1.5, 2.5, 3.5], vec![2, 2], 10.0),
    ]
);

scalar_case!(
    test_sub_scalar_parity,
    "sub_scalar",
    &[
        ScalarTest::new(vec![5.0, 6.0, 7.0, 8.0], vec![4], 2.0),
        ScalarTest::new(vec![10.0, 20.0, 30.0, 40.0], vec![2, 2], 5.0),
        ScalarTest::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], 0.5),
    ]
);

scalar_case!(
    test_mul_scalar_parity,
    "mul_scalar",
    &[
        ScalarTest::new(vec![1.0, 2.0, 3.0, 4.0], vec![4], 2.0),
        ScalarTest::new(vec![2.0, 4.0, 6.0, 8.0], vec![2, 2], 0.5),
        ScalarTest::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], -3.0),
    ]
);

scalar_case!(
    test_div_scalar_parity,
    "div_scalar",
    &[
        ScalarTest::new(vec![10.0, 20.0, 30.0, 40.0], vec![4], 2.0),
        ScalarTest::new(vec![100.0, 200.0, 300.0, 400.0], vec![2, 2], 10.0),
        ScalarTest::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], 4.0),
    ]
);

scalar_case!(
    test_pow_scalar_parity,
    "pow_scalar",
    &[
        ScalarTest::new(vec![2.0, 3.0, 4.0, 5.0], vec![4], 2.0),
        ScalarTest::new(vec![2.0, 3.0, 4.0, 5.0], vec![2, 2], 3.0),
        ScalarTest::new(vec![4.0, 9.0, 16.0, 25.0], vec![2, 2], 0.5),
    ]
);

scalar_case!(
    test_rsub_scalar_parity,
    "rsub_scalar",
    &[
        ScalarTest::new(vec![1.0, 2.0, 3.0, 4.0], vec![4], 10.0),
        ScalarTest::new(vec![2.0, 3.0, 4.0, 5.0], vec![2, 2], 20.0),
        ScalarTest::new(vec![0.5, 1.5, 2.5, 3.5], vec![2, 2], 5.0),
    ]
);
