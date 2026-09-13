//! Gather + GEMM formulation of CUDA conv_transpose1d.
//!
//! The direct conv_transpose1d kernel gives one thread per output element and
//! re-reads the whole input once per output channel. Gathering the contributing
//! input samples into a column buffer turns the same work into one batched
//! GEMM, which reaches the tuned matmul kernels instead.
//!
//! # Gather, not scatter
//!
//! The textbook column form of transposed convolution is col2im: a scatter-add
//! whose overlapping writes need atomics, which numr does not use. The gather
//! form avoids them. For a fixed output position, each tap is fed by at most
//! one input sample, so every column element is written once by one thread.
//! See `col_transpose1d.cu` for the index relation, which matches
//! `runtime/cpu/kernels/conv_transpose.rs` tap for tap.
//!
//! # Layout
//!
//! The gather writes `col` as `[N, C_in*K, L_out]`, contraction axis first, so
//! the spatial axis stays the innermost (contiguous) one on both the input read
//! and the column write. The weight is `[C_in, C_out, K]` — input channels lead
//! for this op — so it needs a real permute to `[C_out, C_in*K]`, done once
//! outside the kernel. Batched over `N` the GEMM yields `[N, C_out, L_out]`
//! directly, with no final permute.
//!
//! # Chunking
//!
//! The column buffer scales with the OUTPUT length, `stride` times the input,
//! so an upsampling decoder would need a buffer far past any sane cap. The
//! output is therefore split along its length: each chunk gathers only its
//! own positions, runs the GEMM for those columns, and the pieces concatenate.
//! A column belongs to exactly one output position, so a chunk boundary moves
//! no work between columns and the per-element contraction inside `matmul`
//! is unchanged. The bound therefore limits memory, not the shapes admitted.
//!
//! # `output_padding`
//!
//! It lengthens the output only. The extra positions gather no input sample,
//! so their column entries are zero and the GEMM writes zeros (plus bias) with
//! no special case anywhere.

use crate::dtype::DType;
use crate::error::Result;
use crate::ops::conv_transpose_common::ConvTranspose1dParams;
use crate::ops::{BinaryOps, MatmulOps, ShapeOps};
use crate::runtime::cuda::kernels::{col_transpose1d_has_kernel, launch_col_transpose1d};
use crate::runtime::cuda::{CudaClient, CudaRuntime};
use crate::tensor::Tensor;

/// Smallest contraction (`c_in * kernel_size`) routed through the gather + GEMM.
///
/// Below this the column buffer costs more to write than the GEMM saves, and
/// the K loop is too short for the tiled kernels to amortise their prologue.
///
/// Measured, and NOT the same crossover conv1d has — do not sync the two. The
/// gather here reads a strided, stride-and-divisibility-filtered input, and the
/// weight needs a real permute copy before the GEMM, so this path carries more
/// fixed cost than conv1d's im2col and needs a longer contraction to pay it off.
///
/// Swept on a fixed geometry with only the kernel size varying: 2048 and 4096
/// both lose to the direct kernel, 8192 wins, and the win grows sharply beyond.
const MIN_CONTRACTION: usize = 8192;

/// Smallest number of output channels. Below this the GEMM has too few rows to
/// reuse a loaded column tile, which is the whole gain over the direct kernel.
///
/// PROVISIONAL, awaiting measurement.
const MIN_C_OUT: usize = 4;

/// Largest column buffer, in elements. A longer output is split into chunks
/// of at most this many column elements, each gathered and multiplied on its
/// own, so the buffer bounds memory without bounding the shapes that qualify.
const MAX_COL_ELEMENTS: usize = 1 << 26;

/// Whether conv_transpose1d takes the gather + GEMM path instead of the direct
/// kernel.
///
/// The principle matches conv1d's im2col gate: the GEMM wins when it is well
/// shaped — a long contraction and enough output channels to reuse each column
/// tile. Narrow-channel and shallow shapes keep the direct kernel, where the
/// column buffer would cost more than it saves. The output length does not
/// enter: it is chunked, so only one position's column must fit the budget.
pub fn use_conv_transpose1d_gemm(params: &ConvTranspose1dParams, dtype: DType) -> bool {
    if !col_transpose1d_has_kernel(dtype) {
        return false;
    }

    // Grouped transposed convolution splits the GEMM into one small problem per
    // group and would need a per-group weight permute. The direct kernel loses
    // no work to grouping, so grouped shapes stay on it.
    if params.groups != 1 {
        return false;
    }

    if max_chunk_length(params) == 0 {
        return false;
    }

    let contraction = params.c_in * params.kernel_size;
    contraction >= MIN_CONTRACTION && params.c_out >= MIN_C_OUT
}

/// Output positions per column chunk under `MAX_COL_ELEMENTS`; zero when a
/// single position's column already exceeds it.
fn max_chunk_length(params: &ConvTranspose1dParams) -> usize {
    params
        .batch
        .checked_mul(params.c_in)
        .and_then(|v| v.checked_mul(params.kernel_size))
        .map(|per_position| MAX_COL_ELEMENTS / per_position.max(1))
        .unwrap_or(0)
}

/// Run conv_transpose1d as a column gather followed by a batched GEMM.
///
/// `input`, `weight` and `bias` must already be contiguous, `params` must come
/// from `validate_conv_transpose1d`, and `groups` must be 1.
pub fn conv_transpose1d_gemm(
    client: &CudaClient,
    input: &Tensor<CudaRuntime>,
    weight: &Tensor<CudaRuntime>,
    bias: Option<&Tensor<CudaRuntime>>,
    params: &ConvTranspose1dParams,
) -> Result<Tensor<CudaRuntime>> {
    conv_transpose1d_gemm_chunked(
        client,
        input,
        weight,
        bias,
        params,
        max_chunk_length(params),
    )
}

/// `conv_transpose1d_gemm` with an explicit chunk length, so a test can force
/// the multi-chunk path on a small tensor. `chunk_max` is clamped to at least
/// one position.
pub(crate) fn conv_transpose1d_gemm_chunked(
    client: &CudaClient,
    input: &Tensor<CudaRuntime>,
    weight: &Tensor<CudaRuntime>,
    bias: Option<&Tensor<CudaRuntime>>,
    params: &ConvTranspose1dParams,
    chunk_max: usize,
) -> Result<Tensor<CudaRuntime>> {
    let dtype = input.dtype();
    let contraction = params.c_in * params.kernel_size;

    // Column row `ic*K + k` must meet weight element `[ic, oc, k]`, so the
    // weight's leading two axes swap. Unlike conv1d's im2col this is a real
    // copy, because input channels lead in this op's weight layout. Done once,
    // shared by every chunk.
    let weight_gemm =
        weight
            .transpose(0, 1)?
            .contiguous()?
            .reshape(&[1, params.c_out, contraction])?;

    // The output is gathered and multiplied in chunks of positions so the
    // column buffer stays under `MAX_COL_ELEMENTS`; a single chunk covers the
    // whole output when it fits.
    let chunk_max = chunk_max.max(1);
    let mut pieces: Vec<Tensor<CudaRuntime>> = Vec::new();
    let mut start = 0usize;
    while start < params.output_length {
        let len = chunk_max.min(params.output_length - start);
        let col =
            Tensor::<CudaRuntime>::empty(&[params.batch, contraction, len], dtype, &client.device)?;
        unsafe {
            launch_col_transpose1d(
                &client.context,
                &client.stream,
                client.device.index,
                dtype,
                input.ptr(),
                col.ptr(),
                params.batch,
                params.c_in,
                params.length,
                params.kernel_size,
                len,
                params.stride,
                params.pad_left,
                params.dilation,
                start,
            )?;
        }
        pieces.push(client.matmul(&weight_gemm, &col)?);
        start += len;
    }
    let out = if pieces.len() == 1 {
        pieces.remove(0)
    } else {
        let refs: Vec<&Tensor<CudaRuntime>> = pieces.iter().collect();
        client.cat(&refs, 2)?
    };

    // The fused `matmul_bias` adds one value per GEMM COLUMN; here a column is
    // an output position and the bias is per output CHANNEL, which is a GEMM
    // row. The bias is broadcast over the channel axis instead.
    match bias {
        Some(b) => {
            let b_channels = b.reshape(&[1, params.c_out, 1])?;
            client.add(&out, &b_channels)
        }
        None => Ok(out),
    }
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

    /// Splitting the output into column chunks changes which GEMM tile an
    /// output position lands in, never the K order it accumulates in, so a
    /// chunked run must reproduce the single-chunk run bit for bit — with a
    /// chunk length that does not divide the output, so the last chunk is
    /// ragged, and a stride, dilation, padding and output_padding that make
    /// the chunk offset enter the gather's divisibility test.
    #[test]
    fn chunked_columns_match_one_chunk_bitwise() {
        let Some(client) = setup() else {
            return;
        };
        let (batch, c_in, c_out, k, len) = (2usize, 6usize, 8usize, 3usize, 19usize);
        let input = client.rand(&[batch, c_in, len], DType::F32).unwrap();
        let weight = client.rand(&[c_in, c_out, k], DType::F32).unwrap();
        let bias = client.rand(&[c_out], DType::F32).unwrap();
        let params = validate_conv_transpose1d(
            input.shape(),
            weight.shape(),
            Some(bias.shape()),
            3,
            crate::ops::PaddingMode::Custom(2, 1, 0, 0),
            1,
            2,
            1,
            DType::F32,
            DType::F32,
            Some(DType::F32),
        )
        .unwrap();

        let whole = conv_transpose1d_gemm_chunked(
            &client,
            &input,
            &weight,
            Some(&bias),
            &params,
            usize::MAX,
        )
        .unwrap();
        let chunked =
            conv_transpose1d_gemm_chunked(&client, &input, &weight, Some(&bias), &params, 7)
                .unwrap();

        assert_eq!(whole.shape(), chunked.shape());
        assert!(
            params.output_length > 14,
            "output_length {} must span at least three chunks of 7",
            params.output_length
        );
        let a: Vec<f32> = whole.to_vec();
        let b: Vec<f32> = chunked.to_vec();
        assert_eq!(a.len(), params.batch * params.c_out * params.output_length);
        for (i, (x, y)) in a.iter().zip(&b).enumerate() {
            assert_eq!(x.to_bits(), y.to_bits(), "element {i}: {x} vs {y}");
        }
    }
}
