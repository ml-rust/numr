//! GEMM-first formulation of CUDA conv_transpose1d.
//!
//! `conv_transpose1d_gemm` gathers the input into a `[C_in*K, L_out]` column
//! buffer and contracts afterwards. That buffer scales with the OUTPUT length,
//! `stride` times the input, so an upsampling decoder blows past its cap and
//! falls back to the direct kernel exactly where the GEMM would help most.
//!
//! This module runs the two steps in the other order. The GEMM comes first,
//! on the input length:
//!
//! ```text
//! col[n, l, oc*K + k] = sum over ic of x[n, ic, l] * w[ic, oc, k]
//!                     = (x[n]^T @ w.reshape(C_in, C_out*K))[l, oc*K + k]
//! ```
//!
//! and `col2im_transpose1d.cu` then folds each output position's taps:
//!
//! ```text
//! out[n, oc, ox] = bias[oc] + sum over k of col[n, l(ox, k), oc*K + k]
//! ```
//!
//! The MAC count matches the direct kernel and the gather-first GEMM exactly;
//! only the buffer differs, at `L * C_out * K` elements instead of
//! `L_out * C_in * K`.
//!
//! # No atomics
//!
//! The fold is a gather: every output element sums its own taps in fixed
//! order, so it is written once and the result is deterministic. See the
//! kernel header for the index relation and the accumulation-order note.
//!
//! # Weight layout
//!
//! The weight is `[C_in, C_out, K]`, input channels leading, so
//! `w.reshape([C_in, C_out*K])` is a free view and already the GEMM's right
//! operand. Only the input needs a real transpose, `[N, C_in, L]` to
//! `[N, L, C_in]`, and that copy is `stride * K` times smaller than the
//! gather-first column buffer.
//!
//! # Half dtypes
//!
//! `col` holds per-tap partial sums. Stored in F16 or BF16, every tap would
//! round before the fold adds it, on top of the rounding inside the GEMM. So
//! for F16 and BF16 the path casts the transposed input, the weight and the
//! bias to F32, runs the GEMM in F32, and folds with the mixed kernel that
//! reads an F32 `col` and writes the half output. The result rounds once, at
//! the store. F32 and F64 skip the casts and fold in their own dtype.
//!
//! The CPU reference and the direct CUDA kernel sum in the storage dtype, so
//! on a half dtype they carry a rounding per add. The F32 product sits closer
//! to the exact value than they do, and parity tests bound the gap with the
//! accumulation-aware tolerance over `c_in * K`.

use crate::dtype::DType;
use crate::error::Result;
use crate::ops::conv_transpose_common::ConvTranspose1dParams;
use crate::ops::{MatmulOps, TypeConversionOps};
use crate::runtime::cuda::kernels::launch_col2im_transpose1d;
use crate::runtime::cuda::{CudaClient, CudaRuntime};
use crate::tensor::Tensor;

/// Smallest contraction (`c_in`) routed through the GEMM.
///
/// The GEMM contracts over input channels alone here, so the threshold is on
/// `c_in`, not on `c_in * kernel_size` as in the gather-first path. One tile
/// depth of the 64x64x32 kernel: below it the tiled GEMM runs a single K step
/// and the direct kernel's one-thread-per-output loop is no worse.
const MIN_C_IN: usize = 32;

/// Largest column buffer, in elements. `L * C_out * K` elements live for the
/// duration of one call; the fold reads them once. For F16 and BF16 inputs
/// the buffer is F32, so it costs 4 bytes per element, not 2.
const MAX_COL_ELEMENTS: usize = 1 << 27;

/// Column buffer size of this formulation, `None` on overflow.
pub fn gemm_first_col_elements(params: &ConvTranspose1dParams) -> Option<usize> {
    params
        .batch
        .checked_mul(params.length)
        .and_then(|v| v.checked_mul(params.c_out))
        .and_then(|v| v.checked_mul(params.kernel_size))
}

/// Whether conv_transpose1d takes the GEMM-first path.
///
/// Every float dtype qualifies. F32 and F64 run the GEMM and the fold in
/// their own dtype. F16 and BF16 run the product in F32 and round once at the
/// fold, so they no longer pay a rounding per tap; see the module docs.
///
/// Grouped transposed convolution would need one GEMM per group; the direct
/// kernel loses no work to grouping, so grouped shapes stay on it.
pub fn use_conv_transpose1d_gemm_first(params: &ConvTranspose1dParams, dtype: DType) -> bool {
    if !matches!(dtype, DType::F32 | DType::F64 | DType::F16 | DType::BF16) || params.groups != 1 {
        return false;
    }
    match gemm_first_col_elements(params) {
        Some(n) if n <= MAX_COL_ELEMENTS => {}
        _ => return false,
    }
    params.c_in >= MIN_C_IN
}

/// Run conv_transpose1d as a GEMM over the input length followed by a fold.
///
/// `input`, `weight` and `bias` must already be contiguous, `params` must come
/// from `validate_conv_transpose1d`, and `groups` must be 1.
pub fn conv_transpose1d_gemm_first(
    client: &CudaClient,
    input: &Tensor<CudaRuntime>,
    weight: &Tensor<CudaRuntime>,
    bias: Option<&Tensor<CudaRuntime>>,
    params: &ConvTranspose1dParams,
) -> Result<Tensor<CudaRuntime>> {
    let dtype = input.dtype();
    let row = params.c_out * params.kernel_size;

    // Half inputs run the product in F32 so the fold rounds once, at the
    // store. F32 and F64 keep their own dtype through the GEMM and the fold.
    let col_dtype = match dtype {
        DType::F16 | DType::BF16 => DType::F32,
        _ => dtype,
    };

    // `[N, C_in, L]` -> `[N, L, C_in]`: the GEMM's left operand, one real copy.
    let x_t = input.transpose(1, 2)?.contiguous()?;
    // `[C_in, C_out, K]` -> `[1, C_in, C_out*K]`: a view, broadcast over N.
    let w_gemm = weight.reshape(&[1, params.c_in, row])?;

    // The weight cast runs on every call. The weight belongs to the model and
    // this path holds no cache, so the F32 copy is not reused across calls.
    let (x_t, w_gemm, bias_col) = if col_dtype == dtype {
        (x_t, w_gemm, None)
    } else {
        let bias_col = bias.map(|b| client.cast(b, col_dtype)).transpose()?;
        (
            client.cast(&x_t, col_dtype)?,
            client.cast(&w_gemm, col_dtype)?,
            bias_col,
        )
    };
    let bias_ptr = bias_col.as_ref().or(bias).map(|b| b.ptr());

    let col = client.matmul(&x_t, &w_gemm)?;

    let out = Tensor::<CudaRuntime>::empty(
        &[params.batch, params.c_out, params.output_length],
        dtype,
        &client.device,
    )?;

    unsafe {
        launch_col2im_transpose1d(
            &client.context,
            &client.stream,
            client.device.index,
            col_dtype,
            dtype,
            col.ptr(),
            bias_ptr,
            out.ptr(),
            params.batch,
            params.length,
            params.c_out,
            params.kernel_size,
            params.output_length,
            params.stride,
            params.pad_left,
            params.dilation,
        )?;
    }

    Ok(out)
}
