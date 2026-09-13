// Backend parity for the CUDA conv_transpose1d GEMM-first fast path on the
// half dtypes (`src/ops/cuda/conv_transpose1d_gemm_first.rs`,
// `use_conv_transpose1d_gemm_first`).
//
// F16 and BF16 run the product in F32 and round once at the fold. These
// tests cross the gate on a shape that only GEMM-first admits, so the
// rounding at the mixed fold is what they check. The F32 and F64 coverage of
// the same path lives in `conv_transpose1d_multichannel.rs`.

use numr::ops::{ConvOps, PaddingMode};

use crate::backend_parity::conv_transpose1d_gemm::{
    gemm_bias, gemm_input, gemm_weight, max_abs, upsample_dtypes,
};
use crate::backend_parity::dtype_helpers::tensor_from_f64;
use crate::backend_parity::helpers::with_cuda_backend;
use crate::common::{
    assert_tensor_allclose_tol, create_cpu_client, gemm_long_k_tolerance, is_dtype_supported,
};

/// A decoder-like upsampling stage that only GEMM-first admits: `c_in=256`
/// clears its `MIN_C_IN` floor, while the gather-first contraction
/// `c_in*k = 256*4 = 1024` sits below `MIN_CONTRACTION`, so the dispatcher
/// has one GEMM path to choose. Both constants are private to
/// `src/ops/cuda/`, so this relation is stated here rather than asserted; if
/// either moves, move this shape with it.
///
/// The tolerance contracts over `c_in`: that is the sum inside the F32 GEMM,
/// and the fold adds `k` terms already carried in F32.
const INPUT_SHAPE: [usize; 3] = [1, 256, 1024];
const WEIGHT_SHAPE: [usize; 3] = [256, 128, 4];
const STRIDE: usize = 2;

/// The shape's output length under `Valid` padding, no `output_padding`, unit
/// dilation.
fn output_length() -> usize {
    (INPUT_SHAPE[2] - 1) * STRIDE + WEIGHT_SHAPE[2]
}

/// GEMM-first on F16 and BF16 vs the CPU reference.
#[test]
fn conv_transpose1d_gemm_first_half_upsample_parity() {
    let c_in = INPUT_SHAPE[1];
    let c_out = WEIGHT_SHAPE[1];
    let input = gemm_input(INPUT_SHAPE.iter().product());
    let weight = gemm_weight(WEIGHT_SHAPE.iter().product());
    let bias = gemm_bias(c_out);
    let operand_scale = max_abs(&input).max(max_abs(&weight));
    let label = "conv_transpose1d_gemm_first_half_upsample";

    for dtype in upsample_dtypes() {
        if !is_dtype_supported("cuda", dtype) {
            continue;
        }
        let (cpu_client, cpu_device) = create_cpu_client();
        let cpu_in = tensor_from_f64(&input, &INPUT_SHAPE, dtype, &cpu_device, &cpu_client)
            .unwrap_or_else(|e| panic!("CPU input tensor failed for {label} [{dtype:?}]: {e}"));
        let cpu_w = tensor_from_f64(&weight, &WEIGHT_SHAPE, dtype, &cpu_device, &cpu_client)
            .unwrap_or_else(|e| panic!("CPU weight tensor failed for {label} [{dtype:?}]: {e}"));
        let cpu_b = tensor_from_f64(&bias, &[c_out], dtype, &cpu_device, &cpu_client)
            .unwrap_or_else(|e| panic!("CPU bias tensor failed for {label} [{dtype:?}]: {e}"));
        let cpu_result = cpu_client
            .conv_transpose1d(
                &cpu_in,
                &cpu_w,
                Some(&cpu_b),
                STRIDE,
                PaddingMode::Valid,
                0,
                1,
                1,
            )
            .unwrap_or_else(|e| panic!("CPU conv_transpose1d failed for {label} [{dtype:?}]: {e}"));

        with_cuda_backend(|client, device| {
            let x = tensor_from_f64(&input, &INPUT_SHAPE, dtype, &device, &client).unwrap_or_else(
                |e| panic!("CUDA input tensor failed for {label} [{dtype:?}]: {e}"),
            );
            let w = tensor_from_f64(&weight, &WEIGHT_SHAPE, dtype, &device, &client)
                .unwrap_or_else(|e| {
                    panic!("CUDA weight tensor failed for {label} [{dtype:?}]: {e}")
                });
            let b = tensor_from_f64(&bias, &[c_out], dtype, &device, &client)
                .unwrap_or_else(|e| panic!("CUDA bias tensor failed for {label} [{dtype:?}]: {e}"));
            let result = client
                .conv_transpose1d(&x, &w, Some(&b), STRIDE, PaddingMode::Valid, 0, 1, 1)
                .unwrap_or_else(|e| {
                    panic!("CUDA conv_transpose1d failed for {label} [{dtype:?}]: {e}")
                });
            let (rtol, atol) = gemm_long_k_tolerance(dtype, c_in, operand_scale);
            assert_tensor_allclose_tol(
                &result,
                &cpu_result,
                rtol,
                atol,
                &format!("{label} CUDA vs CPU [{dtype:?}]"),
            );
        });
    }
}

/// GEMM-first on F16 and BF16 vs the CUDA direct kernel on the same shape.
///
/// The CPU comparison above proves the value; this one proves the two CUDA
/// paths agree with each other, so a shared CPU/CUDA convention slip in the
/// fold's index map cannot hide behind a matching reference.
#[test]
fn conv_transpose1d_gemm_first_half_upsample_matches_direct_kernel() {
    use numr::runtime::Device;
    use numr::runtime::cuda::CudaRuntime;
    use numr::runtime::cuda::kernels::launch_conv_transpose1d;
    use numr::tensor::Tensor;

    let (batch, c_in, length) = (INPUT_SHAPE[0], INPUT_SHAPE[1], INPUT_SHAPE[2]);
    let (c_out, k) = (WEIGHT_SHAPE[1], WEIGHT_SHAPE[2]);
    let output_length = output_length();

    let input = gemm_input(INPUT_SHAPE.iter().product());
    let weight = gemm_weight(WEIGHT_SHAPE.iter().product());
    let bias = gemm_bias(c_out);
    let operand_scale = max_abs(&input).max(max_abs(&weight));

    for dtype in upsample_dtypes() {
        if !is_dtype_supported("cuda", dtype) {
            continue;
        }
        with_cuda_backend(|client, device| {
            let label = "conv_transpose1d_gemm_first_half_upsample_direct";
            let x = tensor_from_f64(&input, &INPUT_SHAPE, dtype, &device, &client).unwrap_or_else(
                |e| panic!("CUDA input tensor failed for {label} [{dtype:?}]: {e}"),
            );
            let w = tensor_from_f64(&weight, &WEIGHT_SHAPE, dtype, &device, &client)
                .unwrap_or_else(|e| {
                    panic!("CUDA weight tensor failed for {label} [{dtype:?}]: {e}")
                });
            let b = tensor_from_f64(&bias, &[c_out], dtype, &device, &client)
                .unwrap_or_else(|e| panic!("CUDA bias tensor failed for {label} [{dtype:?}]: {e}"));

            let gemm = client
                .conv_transpose1d(&x, &w, Some(&b), STRIDE, PaddingMode::Valid, 0, 1, 1)
                .unwrap_or_else(|e| {
                    panic!("CUDA conv_transpose1d failed for {label} [{dtype:?}]: {e}")
                });

            let direct =
                Tensor::<CudaRuntime>::empty(&[batch, c_out, output_length], dtype, &device)
                    .unwrap_or_else(|e| {
                        panic!("CUDA direct output alloc failed for {label} [{dtype:?}]: {e}")
                    });
            // SAFETY: every pointer is a live device allocation whose shape
            // matches the arguments; the launch runs on the client's stream.
            unsafe {
                launch_conv_transpose1d(
                    client.context(),
                    client.stream(),
                    device.id(),
                    dtype,
                    x.ptr(),
                    w.ptr(),
                    Some(b.ptr()),
                    direct.ptr(),
                    batch,
                    c_in,
                    length,
                    c_out,
                    k,
                    output_length,
                    STRIDE,
                    0,
                    1,
                    1,
                )
            }
            .unwrap_or_else(|e| panic!("CUDA direct kernel failed for {label} [{dtype:?}]: {e}"));

            let (rtol, atol) = gemm_long_k_tolerance(dtype, c_in, operand_scale);
            assert_tensor_allclose_tol(
                &gemm,
                &direct,
                rtol,
                atol,
                &format!("{label} gemm-first vs direct [{dtype:?}]"),
            );
        });
    }
}
