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
//! for F16 and BF16 the path runs the GEMM through `matmul_wide`, which hands
//! back the tensor-core kernel's F32 accumulator without narrowing it, and
//! folds with the mixed kernel that reads an F32 `col` and writes the half
//! output. The result rounds once, at the store. No operand is cast: the
//! half input and weight feed the GEMM as they are, and only the `[C_out]`
//! bias is cast to F32 for the fold. F32 and F64 fold in their own dtype.
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

/// Largest column buffer, in elements. `L * C_out * K` elements per batch row
/// live for the duration of one chunk; the fold reads them once. For F16 and
/// BF16 inputs the buffer is F32, so it costs 4 bytes per element, not 2. The
/// batch is split into chunks that fit, so this bounds memory without
/// bounding the shapes that qualify.
const MAX_COL_ELEMENTS: usize = 1 << 27;

/// Column buffer size of this formulation for ONE batch row, `None` on
/// overflow.
pub fn gemm_first_row_col_elements(params: &ConvTranspose1dParams) -> Option<usize> {
    params
        .length
        .checked_mul(params.c_out)
        .and_then(|v| v.checked_mul(params.kernel_size))
}

/// Batch rows per column chunk under [`MAX_COL_ELEMENTS`], at least one.
///
/// This bounds the buffer, never the shapes admitted: the gate already decided
/// the path, and it rejects a row whose own column exceeds the budget.
fn max_chunk_rows(params: &ConvTranspose1dParams) -> usize {
    let per_row = gemm_first_row_col_elements(params)
        .unwrap_or(usize::MAX)
        .max(1);
    (MAX_COL_ELEMENTS / per_row).max(1)
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
    // ONE batch row's column must fit the budget. The batch must not enter:
    // a gate that reads it hands the same row to this path at one batch size
    // and to another formulation at the next, and the three do not sum a row
    // in the same order, so a row's bits would depend on how many rows share
    // the launch. The batch is chunked instead, which is a memory bound only.
    match gemm_first_row_col_elements(params) {
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
    conv_transpose1d_gemm_first_chunked(client, input, weight, bias, params, max_chunk_rows(params))
}

/// `conv_transpose1d_gemm_first` with an explicit chunk height in batch rows,
/// so a test can force the multi-chunk path on a small tensor. `chunk_max` is
/// clamped to at least one row.
pub(crate) fn conv_transpose1d_gemm_first_chunked(
    client: &CudaClient,
    input: &Tensor<CudaRuntime>,
    weight: &Tensor<CudaRuntime>,
    bias: Option<&Tensor<CudaRuntime>>,
    params: &ConvTranspose1dParams,
    chunk_max: usize,
) -> Result<Tensor<CudaRuntime>> {
    let dtype = input.dtype();
    let row = params.c_out * params.kernel_size;

    // `[C_in, C_out, K]` -> `[1, C_in, C_out*K]`: a view, broadcast over N.
    let w_gemm = weight.reshape(&[1, params.c_in, row])?;

    let out = Tensor::<CudaRuntime>::empty(
        &[params.batch, params.c_out, params.output_length],
        dtype,
        &client.device,
    )?;

    // The batch runs in chunks of rows so the column buffer stays under
    // `MAX_COL_ELEMENTS`; a single chunk covers the whole batch when it fits.
    // A row's GEMM and its fold read only that row, so the split is
    // bit-neutral and every chunk writes its own slice of the output.
    let chunk_max = chunk_max.max(1);
    // The fold reads the bias in the column dtype. Only the half dtypes cast,
    // and only the `[C_out]` vector. The column dtype is known once the first
    // GEMM has run, so the cast is made there and reused by later chunks.
    let mut bias_col: Option<Tensor<CudaRuntime>> = None;
    let mut start = 0usize;
    while start < params.batch {
        let rows = chunk_max.min(params.batch - start);
        // `[N, C_in, L]` -> `[N, L, C_in]`: the GEMM's left operand, one real
        // copy, held for one chunk.
        let x_t = input
            .narrow(0, start, rows)?
            .transpose(1, 2)?
            .contiguous()?;

        // Half inputs keep the GEMM's F32 accumulator so the fold rounds once,
        // at the store; F32 and F64 write their own dtype. `matmul_wide` is
        // `matmul` for the latter two, so one call covers every dtype this
        // path admits.
        let col = client.matmul_wide(&x_t, &w_gemm)?;
        let col_dtype = col.dtype();

        if col_dtype != dtype && bias_col.is_none() {
            bias_col = bias.map(|b| client.cast(b, col_dtype)).transpose()?;
        }
        let bias_ptr = bias_col.as_ref().or(bias).map(|b| b.ptr());
        let out_chunk = out.narrow(0, start, rows)?;

        unsafe {
            launch_col2im_transpose1d(
                &client.context,
                &client.stream,
                client.device.index,
                col_dtype,
                dtype,
                col.ptr(),
                bias_ptr,
                out_chunk.ptr(),
                rows,
                params.length,
                params.c_out,
                params.kernel_size,
                params.output_length,
                params.stride,
                params.pad_left,
                params.dilation,
            )?;
        }
        start += rows;
    }

    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::RandomOps;
    use crate::ops::conv_transpose_common::validate_conv_transpose1d;
    use crate::runtime::Runtime;
    use crate::runtime::cuda::CudaDevice;

    fn setup() -> Option<CudaClient> {
        if !crate::runtime::cuda::is_cuda_available() {
            return None;
        }
        let device = CudaDevice::new(0);
        Some(CudaRuntime::default_client(&device))
    }

    /// Splitting the batch into row chunks changes which launch a row sits in,
    /// never the K order its GEMM accumulates in or the taps its fold sums, so
    /// a chunked run must reproduce the single-chunk run bit for bit — with a
    /// chunk height that does not divide the batch, so the last chunk is
    /// ragged.
    #[test]
    fn chunked_rows_match_one_chunk_bitwise() {
        let Some(client) = setup() else {
            return;
        };
        let (batch, c_in, c_out, k, len) = (5usize, 6usize, 8usize, 3usize, 11usize);
        let input = client.rand(&[batch, c_in, len], DType::F32).expect("input");
        let weight = client.rand(&[c_in, c_out, k], DType::F32).expect("weight");
        let bias = client.rand(&[c_out], DType::F32).expect("bias");
        let params = validate_conv_transpose1d(
            input.shape(),
            weight.shape(),
            Some(bias.shape()),
            2,
            crate::ops::PaddingMode::Custom(1, 1, 0, 0),
            1,
            1,
            1,
            DType::F32,
            DType::F32,
            Some(DType::F32),
        )
        .expect("params");

        let whole = conv_transpose1d_gemm_first_chunked(
            &client,
            &input,
            &weight,
            Some(&bias),
            &params,
            usize::MAX,
        )
        .expect("whole");
        let chunked =
            conv_transpose1d_gemm_first_chunked(&client, &input, &weight, Some(&bias), &params, 2)
                .expect("chunked");

        assert_eq!(whole.shape(), chunked.shape());
        let a: Vec<f32> = whole.to_vec();
        let b: Vec<f32> = chunked.to_vec();
        for (i, (x, y)) in a.iter().zip(&b).enumerate() {
            assert_eq!(x.to_bits(), y.to_bits(), "element {i}: {x} vs {y}");
        }
    }
}
