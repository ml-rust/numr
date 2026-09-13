// Backend parity for the CUDA conv_transpose1d gather + GEMM fast path
// (`src/ops/cuda/conv_transpose1d_gemm.rs`, `use_conv_transpose1d_gemm`).
//
// The shapes in `conv_transpose1d_multichannel.rs` all sit far below the
// dispatch floors, so every one of them runs the direct kernel and none of
// them would notice a wrong index map in `col_transpose1d.cu`. These tests
// exist to cross the gate.
//
// The GEMM reassociates the sum the direct kernel accumulates tap-outer /
// channel-inner, so the result is within tolerance rather than bit-identical.
// The tolerance is therefore accumulation-aware (`gemm_long_k_tolerance`),
// not output-relative.

use numr::dtype::DType;
use numr::ops::{ConvOps, PaddingMode};

use crate::backend_parity::dtype_helpers::tensor_from_f64;
#[cfg(feature = "cuda")]
use crate::backend_parity::helpers::with_cuda_backend;
use crate::common::{
    DTypeDomain, assert_tensor_allclose_tol, create_cpu_client, gemm_long_k_tolerance,
    is_dtype_supported, parity_dtypes,
};

/// Deterministic, non-repeating, small-magnitude values. The contraction here
/// is thousands of terms long, so large operands would let the narrow dtypes
/// drift purely from accumulation.
pub fn gemm_input(n: usize) -> Vec<f64> {
    (0..n).map(|i| ((i % 13) as f64 - 6.0) * 0.05).collect()
}

/// Weight values that vary with the flat index, which spans `c_in` first then
/// `c_out` — so a transposed or mis-strided weight permute changes the result.
pub fn gemm_weight(n: usize) -> Vec<f64> {
    (0..n).map(|i| ((i % 9) as f64 - 4.0) * 0.03).collect()
}

pub fn gemm_bias(n: usize) -> Vec<f64> {
    (0..n).map(|i| (i as f64) * 0.01 - 0.02).collect()
}

/// Largest absolute operand value, which drives the accumulation-aware bound.
pub fn max_abs(values: &[f64]) -> f64 {
    values.iter().fold(0.0f64, |acc, v| acc.max(v.abs()))
}

/// Runs `conv_transpose1d` on CPU and CUDA for every float dtype and asserts
/// they agree within an accumulation-aware bound. `weight_shape` is
/// `[c_in, c_out, k]`; this path is `groups == 1` only.
#[allow(clippy::too_many_arguments)]
fn assert_conv_transpose1d_gemm_parity(
    label: &str,
    input: &[f64],
    input_shape: &[usize],
    weight: &[f64],
    weight_shape: &[usize],
    bias: Option<&[f64]>,
    stride: usize,
    padding: PaddingMode,
    output_padding: usize,
    dilation: usize,
) {
    let dtypes = parity_dtypes(DTypeDomain::FloatsOnly, "cpu");
    assert_conv_transpose1d_gemm_parity_for(
        label,
        &dtypes,
        input,
        input_shape,
        weight,
        weight_shape,
        bias,
        stride,
        padding,
        output_padding,
        dilation,
    );
}

/// `assert_conv_transpose1d_gemm_parity` over an explicit dtype list, for a
/// shape whose CPU reference is too costly to repeat across every width.
#[allow(clippy::too_many_arguments)]
fn assert_conv_transpose1d_gemm_parity_for(
    label: &str,
    dtypes: &[DType],
    input: &[f64],
    input_shape: &[usize],
    weight: &[f64],
    weight_shape: &[usize],
    bias: Option<&[f64]>,
    stride: usize,
    padding: PaddingMode,
    output_padding: usize,
    dilation: usize,
) {
    let c_out = weight_shape[1];
    let contraction = weight_shape[0] * weight_shape[2];
    let operand_scale = max_abs(input).max(max_abs(weight));

    for &dtype in dtypes {
        let (cpu_client, cpu_device) = create_cpu_client();
        let cpu_in = tensor_from_f64(input, input_shape, dtype, &cpu_device, &cpu_client)
            .unwrap_or_else(|e| panic!("CPU input tensor failed for {label} [{dtype:?}]: {e}"));
        let cpu_w = tensor_from_f64(weight, weight_shape, dtype, &cpu_device, &cpu_client)
            .unwrap_or_else(|e| panic!("CPU weight tensor failed for {label} [{dtype:?}]: {e}"));
        let cpu_b = bias.map(|b| {
            tensor_from_f64(b, &[c_out], dtype, &cpu_device, &cpu_client)
                .unwrap_or_else(|e| panic!("CPU bias tensor failed for {label} [{dtype:?}]: {e}"))
        });
        let cpu_result = cpu_client
            .conv_transpose1d(
                &cpu_in,
                &cpu_w,
                cpu_b.as_ref(),
                stride,
                padding,
                output_padding,
                dilation,
                1,
            )
            .unwrap_or_else(|e| panic!("CPU conv_transpose1d failed for {label} [{dtype:?}]: {e}"));

        #[cfg(feature = "cuda")]
        if is_dtype_supported("cuda", dtype) {
            with_cuda_backend(|cuda_client, cuda_device| {
                let x = tensor_from_f64(input, input_shape, dtype, &cuda_device, &cuda_client)
                    .unwrap_or_else(|e| {
                        panic!("CUDA input tensor failed for {label} [{dtype:?}]: {e}")
                    });
                let w = tensor_from_f64(weight, weight_shape, dtype, &cuda_device, &cuda_client)
                    .unwrap_or_else(|e| {
                        panic!("CUDA weight tensor failed for {label} [{dtype:?}]: {e}")
                    });
                let b = bias.map(|bd| {
                    tensor_from_f64(bd, &[c_out], dtype, &cuda_device, &cuda_client).unwrap_or_else(
                        |e| panic!("CUDA bias tensor failed for {label} [{dtype:?}]: {e}"),
                    )
                });
                let result = cuda_client
                    .conv_transpose1d(
                        &x,
                        &w,
                        b.as_ref(),
                        stride,
                        padding,
                        output_padding,
                        dilation,
                        1,
                    )
                    .unwrap_or_else(|e| {
                        panic!("CUDA conv_transpose1d failed for {label} [{dtype:?}]: {e}")
                    });
                let (rtol, atol) = gemm_long_k_tolerance(dtype, contraction, operand_scale);
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
}

/// Contraction `c_in*K = 2048*4 = 8192` meets `MIN_CONTRACTION` in
/// `src/ops/cuda/conv_transpose1d_gemm.rs` exactly, `c_out = 4` meets its floor,
/// `groups = 1`: the simplest geometry that reaches the gather + GEMM path.
///
/// If that constant is ever raised, raise these shapes with it. A shape that
/// falls below the floor does not fail — it quietly runs the direct kernel and
/// stops covering this path at all.
#[test]
fn conv_transpose1d_gemm_boundary_parity() {
    let input_shape = [1usize, 2048, 5];
    let weight_shape = [2048usize, 4, 4];
    let input = gemm_input(input_shape.iter().product());
    let weight = gemm_weight(weight_shape.iter().product());
    assert_conv_transpose1d_gemm_parity(
        "conv_transpose1d_gemm_boundary",
        &input,
        &input_shape,
        &weight,
        &weight_shape,
        None,
        1,
        PaddingMode::Valid,
        0,
        1,
    );
}

/// Contraction `1024*8 = 8192`, past every floor, with `stride=2`, `dilation=2`, asymmetric `Custom`
/// padding, `output_padding=1` and a bias, over `batch=2`. This is the case
/// that catches a wrong divisibility or range test in the gather: most taps
/// contribute nothing, and the trailing `output_padding` position must gather
/// nothing at all yet still receive its bias.
#[test]
fn conv_transpose1d_gemm_stride_dilation_output_padding_parity() {
    let input_shape = [2usize, 1024, 6];
    let weight_shape = [1024usize, 8, 8];
    let input = gemm_input(input_shape.iter().product());
    let weight = gemm_weight(weight_shape.iter().product());
    let bias = gemm_bias(weight_shape[1]);
    assert_conv_transpose1d_gemm_parity(
        "conv_transpose1d_gemm_stride_dilation_output_padding",
        &input,
        &input_shape,
        &weight,
        &weight_shape,
        Some(&bias),
        2,
        PaddingMode::Custom(2, 1, 0, 0),
        1,
        2,
    );
}

/// An upsampling shape that only gather-first admits and that crosses its
/// chunk budget: `c_in=16` sits below GEMM-first's `MIN_C_IN`, so the
/// dispatcher cannot prefer that path, while `c_in*k = 16*512 = 8192` meets
/// gather-first's `MIN_CONTRACTION` and `c_out=4` its floor. One output
/// position's column is `c_in*k = 8192` elements, so the full buffer crosses
/// `1 << 26` once the output passes 8192 positions; `L=1024, stride=8` gives
/// `L_out = 8696`, which runs the chunk loop twice with a ragged second
/// chunk. Both gates' constants are private to `src/ops/cuda/`; if either
/// moves, move this shape with it.
///
/// GEMM-first admits every float dtype and wins whenever its buffer is the
/// smaller one, which on a decoder's wide-`c_in` shape it always is. A
/// wide-`c_in` shape therefore never reaches this loop any more, so the
/// coverage lives on this narrow one.
const UPSAMPLE_INPUT_SHAPE: [usize; 3] = [1, 16, 1024];
const UPSAMPLE_WEIGHT_SHAPE: [usize; 3] = [16, 4, 512];
const UPSAMPLE_STRIDE: usize = 8;

/// The half widths, and only those the build carries. The F32/F64 boundary
/// tests above already cross the gather-first gate, and the GEMM-first
/// F32/F64 coverage lives in `conv_transpose1d_multichannel.rs`, so the
/// upsampling tests spend their CPU reference on F16 and BF16. Without the
/// `f16` feature the list is empty and those tests check nothing; run them
/// with `--features cuda,f16`.
pub fn upsample_dtypes() -> Vec<DType> {
    parity_dtypes(DTypeDomain::FloatsOnly, "cpu")
        .into_iter()
        .filter(|dtype| matches!(dtype, DType::F16 | DType::BF16))
        .collect()
}

/// The upsampling shape's output length under `Valid` padding, no
/// `output_padding`, unit dilation.
fn upsample_output_length() -> usize {
    (UPSAMPLE_INPUT_SHAPE[2] - 1) * UPSAMPLE_STRIDE + UPSAMPLE_WEIGHT_SHAPE[2]
}

/// Chunked gather-first vs the CPU reference on the upsampling shape.
#[test]
fn conv_transpose1d_gemm_chunked_upsample_parity() {
    let col_elements = UPSAMPLE_INPUT_SHAPE[0]
        * UPSAMPLE_INPUT_SHAPE[1]
        * UPSAMPLE_WEIGHT_SHAPE[2]
        * upsample_output_length();
    assert!(
        col_elements > (1 << 26),
        "column buffer {col_elements} must exceed the chunk budget so the loop runs twice"
    );
    let input = gemm_input(UPSAMPLE_INPUT_SHAPE.iter().product());
    let weight = gemm_weight(UPSAMPLE_WEIGHT_SHAPE.iter().product());
    let bias = gemm_bias(UPSAMPLE_WEIGHT_SHAPE[1]);
    assert_conv_transpose1d_gemm_parity_for(
        "conv_transpose1d_gemm_chunked_upsample",
        &upsample_dtypes(),
        &input,
        &UPSAMPLE_INPUT_SHAPE,
        &weight,
        &UPSAMPLE_WEIGHT_SHAPE,
        Some(&bias),
        UPSAMPLE_STRIDE,
        PaddingMode::Valid,
        0,
        1,
    );
}

/// Chunked gather-first vs the CUDA direct kernel on the upsampling shape.
///
/// The CPU comparison above proves the value; this one proves the two CUDA
/// paths agree with each other, so a shared CPU/CUDA convention slip in the
/// index map cannot hide behind a matching reference.
#[cfg(feature = "cuda")]
#[test]
fn conv_transpose1d_gemm_chunked_upsample_matches_direct_kernel() {
    use numr::runtime::Device;
    use numr::runtime::cuda::CudaRuntime;
    use numr::runtime::cuda::kernels::launch_conv_transpose1d;
    use numr::tensor::Tensor;

    let (batch, c_in, length) = (
        UPSAMPLE_INPUT_SHAPE[0],
        UPSAMPLE_INPUT_SHAPE[1],
        UPSAMPLE_INPUT_SHAPE[2],
    );
    let (c_out, k) = (UPSAMPLE_WEIGHT_SHAPE[1], UPSAMPLE_WEIGHT_SHAPE[2]);
    let output_length = upsample_output_length();
    let contraction = c_in * k;

    let input = gemm_input(UPSAMPLE_INPUT_SHAPE.iter().product());
    let weight = gemm_weight(UPSAMPLE_WEIGHT_SHAPE.iter().product());
    let bias = gemm_bias(c_out);
    let operand_scale = max_abs(&input).max(max_abs(&weight));

    for dtype in upsample_dtypes() {
        if !is_dtype_supported("cuda", dtype) {
            continue;
        }
        with_cuda_backend(|client, device| {
            let label = "conv_transpose1d_gemm_chunked_upsample_direct";
            let x = tensor_from_f64(&input, &UPSAMPLE_INPUT_SHAPE, dtype, &device, &client)
                .unwrap_or_else(|e| {
                    panic!("CUDA input tensor failed for {label} [{dtype:?}]: {e}")
                });
            let w = tensor_from_f64(&weight, &UPSAMPLE_WEIGHT_SHAPE, dtype, &device, &client)
                .unwrap_or_else(|e| {
                    panic!("CUDA weight tensor failed for {label} [{dtype:?}]: {e}")
                });
            let b = tensor_from_f64(&bias, &[c_out], dtype, &device, &client)
                .unwrap_or_else(|e| panic!("CUDA bias tensor failed for {label} [{dtype:?}]: {e}"));

            let gemm = client
                .conv_transpose1d(
                    &x,
                    &w,
                    Some(&b),
                    UPSAMPLE_STRIDE,
                    PaddingMode::Valid,
                    0,
                    1,
                    1,
                )
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
                    UPSAMPLE_STRIDE,
                    0,
                    1,
                    1,
                )
            }
            .unwrap_or_else(|e| panic!("CUDA direct kernel failed for {label} [{dtype:?}]: {e}"));

            let (rtol, atol) = gemm_long_k_tolerance(dtype, contraction, operand_scale);
            assert_tensor_allclose_tol(
                &gemm,
                &direct,
                rtol,
                atol,
                &format!("{label} gather-first vs direct [{dtype:?}]"),
            );
        });
    }
}
