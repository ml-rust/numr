// Half dtypes accumulate in F32 inside every direct convolution kernel.
//
// Every case here has inputs of 1.0 and weights of 2^-10, and every output
// element gathers 4096 products, so the exact answer is 4.0. A running sum
// kept in the storage dtype cannot reach it: F16 has spacing 2^-9 on [2, 4),
// twice the 2^-10 increment, so every add past 2.0 rounds back to 2.0. BF16
// has spacing 2^-9 already on [0.25, 0.5) and stalls at 0.25. A sum kept in
// F32 lands on 4.0 exactly, so the assertion is one storage ulp at 4.0.
//
// The shapes are chosen so CUDA reaches its direct kernels rather than an
// im2col or GEMM path; each test's doc comment names the kernel and the gate
// that routes to it (`src/ops/cuda/conv.rs`, `src/runtime/cuda/kernels/conv.rs`).
// Without `--features f16` there are no half dtypes and each test checks
// nothing.

use numr::dtype::DType;
use numr::ops::{ConvOps, PaddingMode};
use numr::runtime::Runtime;
use numr::tensor::Tensor;

#[cfg(feature = "cuda")]
use crate::backend_parity::helpers::with_cuda_backend;
#[cfg(feature = "cuda")]
use crate::common::is_dtype_supported;
use crate::common::{DTypeDomain, assert_tensor_allclose_tol, create_cpu_client, parity_dtypes};

/// Weight value: 4096 of them sum to exactly 4.0.
const WEIGHT: f64 = 1.0 / 1024.0;

/// The half dtypes the CPU backend runs; empty without `--features f16`.
fn half_dtypes() -> Vec<DType> {
    parity_dtypes(DTypeDomain::FloatsOnly, "cpu")
        .into_iter()
        .filter(|d| matches!(d, DType::F16 | DType::BF16))
        .collect()
}

/// One storage ulp at 4.0. A stalled sum sits at least 2.0 away.
fn half_atol(dtype: DType) -> f64 {
    match dtype {
        DType::F16 => 4e-3,
        DType::BF16 => 3.2e-2,
        other => panic!("half_atol: {other:?} is not a half dtype"),
    }
}

/// Every element of `result` equals its entry in `expected` within one
/// storage ulp.
fn assert_matches<R: Runtime<DType = DType>>(
    result: &Tensor<R>,
    expected: &[f64],
    dtype: DType,
    label: &str,
) {
    let (cpu_client, cpu_device) = create_cpu_client();
    let expected = crate::backend_parity::dtype_helpers::tensor_from_f64(
        expected,
        result.shape(),
        dtype,
        &cpu_device,
        &cpu_client,
    )
    .unwrap_or_else(|e| panic!("{label}: expected tensor failed: {e}"));
    assert_tensor_allclose_tol(result, &expected, 0.0, half_atol(dtype), label);
}

/// Runs `$body` on CPU and, under `cuda`, on CUDA, for every half dtype, and
/// checks the result against `$expected` (a `Vec<f64>` per dtype).
macro_rules! check_half_backends {
    ($label:expr, $expected:expr, |$client:ident, $device:ident, $dtype:ident| $body:expr) => {{
        for $dtype in half_dtypes() {
            let expected: Vec<f64> = $expected;
            let (cpu_client, cpu_device) = create_cpu_client();
            let out = {
                let $client = &cpu_client;
                let $device = &cpu_device;
                $body
            };
            assert_matches(
                &out,
                &expected,
                $dtype,
                &format!("{} CPU [{:?}]", $label, $dtype),
            );

            #[cfg(feature = "cuda")]
            if is_dtype_supported("cuda", $dtype) {
                with_cuda_backend(|cuda_client, cuda_device| {
                    let out = {
                        let $client = &cuda_client;
                        let $device = &cuda_device;
                        $body
                    };
                    assert_matches(
                        &out,
                        &expected,
                        $dtype,
                        &format!("{} CUDA [{:?}]", $label, $dtype),
                    );
                });
            }
        }
    }};
}

fn ones<R: Runtime<DType = DType>>(shape: &[usize], dtype: DType, device: &R::Device) -> Tensor<R> {
    Tensor::<R>::full_scalar(shape, dtype, 1.0, device)
        .unwrap_or_else(|e| panic!("ones {shape:?} [{dtype:?}]: {e}"))
}

fn weights<R: Runtime<DType = DType>>(
    shape: &[usize],
    dtype: DType,
    device: &R::Device,
) -> Tensor<R> {
    Tensor::<R>::full_scalar(shape, dtype, WEIGHT, device)
        .unwrap_or_else(|e| panic!("weights {shape:?} [{dtype:?}]: {e}"))
}

/// `conv1d`, scalar kernel. `c_in = 1024`, `k = 4`, `c_out = 1`: one output
/// channel per group is below `CONV1D_OC_BLOCK`, so not `conv1d_oc4`, and
/// below `MIN_C_OUT_PER_GROUP`, so not im2col. `output_length = 1` is below
/// `CONV1D_OX_MIN_OUTPUT_LENGTH`, so not `conv1d_ox`. CUDA runs `conv1d_*`.
#[test]
fn conv1d_scalar_kernel_half_sum_reaches_four() {
    check_half_backends!(
        "conv1d_scalar_kernel",
        vec![4.0],
        |client, device, dtype| {
            let x = ones(&[1, 1024, 4], dtype, device);
            let w = weights(&[1, 1024, 4], dtype, device);
            client
                .conv1d(&x, &w, None, 1, PaddingMode::Valid, 1, 1)
                .unwrap_or_else(|e| panic!("conv1d [{dtype:?}]: {e}"))
        }
    );
}

/// `conv1d`, register-blocked kernel. `groups = 2`, `c_in = 2048`, `k = 4`,
/// `c_out = 8`: four output channels per group meets `CONV1D_OC_BLOCK`, and
/// `groups != 1` keeps im2col off. Each group contracts `1024 * 4` taps.
/// CUDA runs `conv1d_oc4_*`.
#[test]
fn conv1d_oc4_kernel_half_sum_reaches_four() {
    check_half_backends!(
        "conv1d_oc4_kernel",
        vec![4.0; 8],
        |client, device, dtype| {
            let x = ones(&[1, 2048, 4], dtype, device);
            let w = weights(&[8, 1024, 4], dtype, device);
            client
                .conv1d(&x, &w, None, 1, PaddingMode::Valid, 1, 2)
                .unwrap_or_else(|e| panic!("conv1d [{dtype:?}]: {e}"))
        }
    );
}

/// `conv1d`, position-blocked kernel. Depthwise (`groups = c_in = c_out =
/// 32`), `k = 4096`, `output_length = 8192`: one output channel per group
/// rules out `conv1d_oc4`, `groups != 1` rules out im2col, and the row is
/// long enough for `conv1d_ox`. The blocked grid launches `32 * 8192 / 4`
/// threads, which clears `CONV1D_OX_MIN_WAVES` on any device up to 256
/// compute units. CUDA runs `conv1d_ox_*`, the stride-1 sliding-window path.
#[test]
fn conv1d_ox_kernel_half_sum_reaches_four() {
    const CHANNELS: usize = 32;
    const KERNEL: usize = 4096;
    const OUT_LEN: usize = 8192;
    check_half_backends!(
        "conv1d_ox_kernel",
        vec![4.0; CHANNELS * OUT_LEN],
        |client, device, dtype| {
            let x = ones(&[1, CHANNELS, OUT_LEN + KERNEL - 1], dtype, device);
            let w = weights(&[CHANNELS, 1, KERNEL], dtype, device);
            client
                .conv1d(&x, &w, None, 1, PaddingMode::Valid, 1, CHANNELS)
                .unwrap_or_else(|e| panic!("conv1d [{dtype:?}]: {e}"))
        }
    );
}

/// `conv2d`, direct kernel. `c_in = 256`, `k = 4x4`, `c_out = 1`: one
/// output channel is below `MIN_C_OUT`, so im2col stays off. CUDA runs
/// `conv2d_*`.
#[test]
fn conv2d_kernel_half_sum_reaches_four() {
    check_half_backends!("conv2d_kernel", vec![4.0], |client, device, dtype| {
        let x = ones(&[1, 256, 4, 4], dtype, device);
        let w = weights(&[1, 256, 4, 4], dtype, device);
        client
            .conv2d(&x, &w, None, (1, 1), PaddingMode::Valid, (1, 1), 1)
            .unwrap_or_else(|e| panic!("conv2d [{dtype:?}]: {e}"))
    });
}

/// `depthwise_conv2d`, flat kernel. One channel, `k = 64x64` over a `64x64`
/// input: `output_w = 1` is below `DEPTHWISE_CONV2D_OX_MIN_OUTPUT_WIDTH`.
/// CUDA runs `depthwise_conv2d_*`.
#[test]
fn depthwise_conv2d_flat_kernel_half_sum_reaches_four() {
    check_half_backends!(
        "depthwise_conv2d_flat_kernel",
        vec![4.0],
        |client, device, dtype| {
            let x = ones(&[1, 1, 64, 64], dtype, device);
            let w = weights(&[1, 1, 64, 64], dtype, device);
            client
                .depthwise_conv2d(&x, &w, None, (1, 1), PaddingMode::Valid, (1, 1))
                .unwrap_or_else(|e| panic!("depthwise_conv2d [{dtype:?}]: {e}"))
        }
    );
}

/// `depthwise_conv2d`, column-blocked kernel. `128` channels, `k = 64x64`,
/// output `64x32`: the row width clears
/// `DEPTHWISE_CONV2D_OX_MIN_OUTPUT_WIDTH` and the blocked grid launches
/// `128 * 64 * 8` threads, which clears `DEPTHWISE_CONV2D_OX_MIN_WAVES` on
/// any device up to 256 compute units. CUDA runs `depthwise_conv2d_ox_*`, the
/// stride-1 sliding-window path.
#[test]
fn depthwise_conv2d_ox_kernel_half_sum_reaches_four() {
    const CHANNELS: usize = 128;
    const KERNEL: usize = 64;
    const OUT_H: usize = 64;
    const OUT_W: usize = 32;
    check_half_backends!(
        "depthwise_conv2d_ox_kernel",
        vec![4.0; CHANNELS * OUT_H * OUT_W],
        |client, device, dtype| {
            let x = ones(
                &[1, CHANNELS, OUT_H + KERNEL - 1, OUT_W + KERNEL - 1],
                dtype,
                device,
            );
            let w = weights(&[CHANNELS, 1, KERNEL, KERNEL], dtype, device);
            client
                .depthwise_conv2d(&x, &w, None, (1, 1), PaddingMode::Valid, (1, 1))
                .unwrap_or_else(|e| panic!("depthwise_conv2d [{dtype:?}]: {e}"))
        }
    );
}

/// `conv_transpose1d`, direct gather kernel. `groups = 2` fails both GEMM
/// gates (`use_conv_transpose1d_gemm_first`, `use_conv_transpose1d_gemm`),
/// so CUDA runs `conv_transpose1d_*`. `c_in = 2048` (1024 per group),
/// `k = 4`, `length = 4`, stride 1: output position `ot` gathers
/// `min(ot + 1, 4, 7 - ot)` taps from 1024 input channels, so the exact
/// output row is `[1, 2, 3, 4, 3, 2, 1]` and the centre gathers 4096
/// products. A storage-dtype sum stalls on the three centre positions.
#[test]
fn conv_transpose1d_kernel_half_sum_reaches_four() {
    let row = [1.0, 2.0, 3.0, 4.0, 3.0, 2.0, 1.0];
    check_half_backends!(
        "conv_transpose1d_kernel",
        row.iter().chain(row.iter()).copied().collect(),
        |client, device, dtype| {
            let x = ones(&[1, 2048, 4], dtype, device);
            let w = weights(&[2048, 1, 4], dtype, device);
            client
                .conv_transpose1d(&x, &w, None, 1, PaddingMode::Valid, 0, 1, 2)
                .unwrap_or_else(|e| panic!("conv_transpose1d [{dtype:?}]: {e}"))
        }
    );
}
