// Backend parity for activations that take a scalar parameter: leaky_relu and elu.
//
// The parameters are not exactly representable in f32, so a backend that
// narrows an F64 parameter to f32 fails the F64 tolerance.

use numr::dtype::DType;
use numr::ops::ActivationOps;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

use crate::backend_parity::dtype_helpers::tensor_from_f64;
#[cfg(feature = "cuda")]
use crate::backend_parity::helpers::with_cuda_backend;
#[cfg(feature = "wgpu")]
use crate::backend_parity::helpers::with_wgpu_backend;
#[cfg(any(feature = "cuda", feature = "wgpu"))]
use crate::common::is_dtype_supported;
use crate::common::{DTypeDomain, assert_tensor_allclose, create_cpu_client, parity_dtypes};

const X: [f64; 8] = [-2.3, -0.7, -0.1, 0.0, 0.2, 0.9, 1.6, -3.1];
const SHAPE: [usize; 2] = [2, 4];

#[derive(Clone, Copy, Debug)]
enum ScalarActivation {
    LeakyRelu(f64),
    Elu(f64),
}

fn apply<R: Runtime, C: ActivationOps<R>>(
    client: &C,
    a: &Tensor<R>,
    op: ScalarActivation,
) -> numr::error::Result<Tensor<R>> {
    match op {
        ScalarActivation::LeakyRelu(slope) => client.leaky_relu(a, slope),
        ScalarActivation::Elu(alpha) => client.elu(a, alpha),
    }
}

fn parity_for(op: ScalarActivation, dtype: DType) {
    let (cpu_client, cpu_device) = create_cpu_client();
    let a = tensor_from_f64(&X, &SHAPE, dtype, &cpu_device, &cpu_client)
        .unwrap_or_else(|e| panic!("CPU tensor_from_f64 failed for {dtype:?}: {e}"));
    let expected = apply(&cpu_client, &a, op)
        .unwrap_or_else(|e| panic!("CPU {op:?} failed for {dtype:?}: {e}"));

    #[cfg(feature = "cuda")]
    if is_dtype_supported("cuda", dtype) {
        with_cuda_backend(|client, device| {
            let a = tensor_from_f64(&X, &SHAPE, dtype, &device, &client)
                .unwrap_or_else(|e| panic!("CUDA tensor_from_f64 failed for {dtype:?}: {e}"));
            let result = apply(&client, &a, op)
                .unwrap_or_else(|e| panic!("CUDA {op:?} failed for {dtype:?}: {e}"));
            assert_tensor_allclose(
                &result,
                &expected,
                dtype,
                &format!("{op:?} CUDA vs CPU [{dtype:?}]"),
            );
        });
    }

    #[cfg(feature = "wgpu")]
    if is_dtype_supported("wgpu", dtype) {
        with_wgpu_backend(|client, device| {
            let a = tensor_from_f64(&X, &SHAPE, dtype, &device, &client)
                .unwrap_or_else(|e| panic!("WebGPU tensor_from_f64 failed for {dtype:?}: {e}"));
            let result = apply(&client, &a, op)
                .unwrap_or_else(|e| panic!("WebGPU {op:?} failed for {dtype:?}: {e}"));
            assert_tensor_allclose(
                &result,
                &expected,
                dtype,
                &format!("{op:?} WebGPU vs CPU [{dtype:?}]"),
            );
        });
    }

    // Without a GPU backend the CPU result has nothing to compare against.
    let _ = expected;
}

#[test]
fn test_leaky_relu_parity() {
    for dtype in parity_dtypes(DTypeDomain::FloatsOnly, "cpu") {
        parity_for(ScalarActivation::LeakyRelu(0.2), dtype);
    }
}

#[test]
fn test_elu_parity() {
    for dtype in parity_dtypes(DTypeDomain::FloatsOnly, "cpu") {
        parity_for(ScalarActivation::Elu(0.7), dtype);
    }
}
