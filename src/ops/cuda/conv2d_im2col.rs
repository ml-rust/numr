//! im2col + GEMM formulation of CUDA conv2d.
//!
//! The direct conv2d kernel gives one thread per output element and re-reads
//! the whole input once per output channel. Gathering the receptive fields
//! into a column buffer turns the same work into one batched GEMM, which
//! reaches the tuned matmul kernels instead.
//!
//! # Layout
//!
//! `im2col2d` writes `col` as `[N, C_in*Kh*Kw, H_out*W_out]`, contraction axis
//! first, matching conv1d's im2col and conv_transpose1d's gather. The
//! flattened spatial axis stays innermost (contiguous) on both the input read
//! and the column write, so writes coalesce. The weight
//! `[C_out, C_in, Kh, Kw]` is already row-major in exactly that order — this
//! path is restricted to `groups == 1`, so there is no group axis to reorder
//! around — and reshapes with NO COPY to `[C_out, C_in*Kh*Kw]`. Batched over
//! `N` the GEMM yields `[N, C_out, H_out*W_out]`, which reshapes to
//! `[N, C_out, H_out, W_out]`. No permute anywhere, unlike conv_transpose1d's
//! gather, which does need one because input channels lead in that op's
//! weight layout.
//!
//! # Chunking
//!
//! The column buffer holds `C_in*Kh*Kw` elements per output position, so a
//! large image needs far more column memory than output memory. The output
//! splits into chunks of batch rows and, when one row's column is too large,
//! windows of flattened output positions. Each chunk gathers its own columns,
//! runs its own GEMM, and lands in its slice of the output. A chunk boundary
//! never changes an output element's bits (see [`chunk_plan`]).

use crate::dtype::DType;
use crate::error::Result;
use crate::ops::conv_common::Conv2dParams;
use crate::ops::{BinaryOps, MatmulOps};
use crate::runtime::cuda::kernels::{im2col2d_has_kernel, launch_im2col2d};
use crate::runtime::cuda::{CudaClient, CudaRuntime};
use crate::tensor::Tensor;

/// Smallest contraction (`c_in * kh * kw`) routed through im2col + GEMM.
///
/// This floor sits below the 1-D paths' floors. Do not sync them. conv2d's
/// GEMM is thousands of columns wide (`h_out * w_out`) where the 1-D paths
/// have hundreds, so it amortizes the column buffer at a shorter contraction.
///
/// A sweep over `c_in` at a fixed spatial size puts the crossover between
/// contraction 9, where the direct kernel wins, and 27, where the GEMM wins.
///
/// A smaller spatial extent gives the GEMM fewer columns and moves this
/// crossover up. Re-sweep before trusting it on small images.
const MIN_CONTRACTION: usize = 27;

/// Smallest number of output channels. Below this the GEMM has too few rows
/// to reuse a loaded column tile, which is the whole gain over the direct
/// kernel. The native implicit-GEMM route of conv2d
/// (`conv_fused_native::conv2d_native`) keeps the same floor.
///
/// PROVISIONAL, awaiting measurement.
pub(super) const MIN_C_OUT: usize = 4;

/// Column buffer budget, in elements. The output splits into chunks whose
/// column buffer fits it. This bounds memory only: the gate never reads it,
/// so every shape the gate admits runs on the GEMM whatever its column size.
///
/// PROVISIONAL, awaiting measurement. Kept at the same cap as the 1-D paths.
const MAX_COL_ELEMENTS: usize = 1 << 26;

/// Largest GEMM result of one chunk, in elements. A chunk's result reaches
/// its output slice through `copy_into`, whose strided kernel indexes in
/// 32 bits. A single chunk is the output itself and needs no copy, so this
/// bound applies only once the output splits.
const MAX_PIECE_ELEMENTS: usize = u32::MAX as usize;

/// Multiple a spatial window's width rounds down to, in output positions.
/// It is the widest GEMM block tile, so a full window fills every block. It
/// is also a multiple of the WMMA 128-bit staging width, so a full window
/// keeps the GEMM's N stride on the vector staging path.
const WINDOW_ALIGN: usize = 128;

/// Chunk shape: `rows` batch rows by `width` flattened output positions. The
/// last chunk along either axis can be shorter.
#[derive(Debug, Clone, Copy)]
struct ChunkPlan {
    rows: usize,
    width: usize,
}

/// Chunk shape whose column buffer holds at most `col_cap` elements.
///
/// The whole output is one chunk when its column fits. Otherwise a chunk
/// also keeps its GEMM result within [`MAX_PIECE_ELEMENTS`]. It takes whole
/// batch rows when one row fits. Else it takes one row and a window of output
/// positions, rounded down to [`WINDOW_ALIGN`] when the budget allows it. A
/// chunk always holds at least one position, so a contraction longer than
/// `col_cap` still runs.
///
/// # Determinism
///
/// The plan changes the GEMM's batch count and N, never its M or K, and an
/// output element's bits depend on neither. The kernels `matmul` reaches for
/// these operands form one element as follows:
/// - F32: every tile of the compile-time family does one FMA per k, k in
///   order. The tile depends on N, the bits do not.
/// - F64, and F16 or BF16 without tensor cores: the generic tiled kernel at
///   the dtype's fixed tile does one accumulate per k, k in order.
/// - F16 and BF16 with tensor cores: the WMMA kernel steps K in 16-wide MMA
///   slices, in ascending order, into one F32 accumulator that starts at +0.
///   The block tile, its BLOCK_K and the stride pad depend on N and the batch
///   count. They change only how many all-zero slices trail the last real
///   slice. An all-zero slice adds +0, and the accumulator is never -0, so
///   the slice leaves it unchanged.
///
/// A batch count of one takes the 2-D launcher and a larger one the batched
/// launcher. Both reach the same kernel family for a dtype. The bias add is
/// elementwise and runs on the assembled output. The bitwise test
/// `chunked_matches_one_chunk_bitwise` checks the claim on the running
/// device for all four dtypes.
fn chunk_plan(params: &Conv2dParams, col_cap: usize) -> ChunkPlan {
    let contraction = (params.c_in * params.kernel_h * params.kernel_w).max(1);
    let spatial = params.output_h * params.output_w;
    let whole_col = params
        .batch
        .checked_mul(spatial)
        .and_then(|v| v.checked_mul(contraction));
    if whole_col.is_some_and(|n| n <= col_cap) {
        return ChunkPlan {
            rows: params.batch,
            width: spatial,
        };
    }

    // Output positions one chunk holds, summed over its rows.
    let positions = (col_cap / contraction)
        .min(MAX_PIECE_ELEMENTS / params.c_out.max(1))
        .max(1);
    if spatial <= positions {
        return ChunkPlan {
            rows: (positions / spatial).min(params.batch).max(1),
            width: spatial,
        };
    }
    let width = if positions >= WINDOW_ALIGN {
        positions - positions % WINDOW_ALIGN
    } else {
        positions
    };
    ChunkPlan { rows: 1, width }
}

/// Whether conv2d takes the im2col + GEMM path instead of the direct kernel.
///
/// conv2d asks this only after the native implicit-GEMM route declines the
/// call (`conv_fused_native::conv2d_native`), so F16 and BF16 shapes that
/// route native never reach it.
///
/// The principle matches conv1d's im2col gate: the GEMM wins when it is well
/// shaped — a long contraction and enough output channels to reuse each
/// column tile. Grouped, narrow-channel and shallow shapes keep the direct
/// kernel, where the column buffer would cost more than it saves.
///
/// Neither the column size nor the batch enters. The column is chunked, so
/// its size bounds memory only. A gate that reads the batch hands the same
/// row to the GEMM at one batch size and to the direct kernel at another.
/// The two do not sum a row in the same order, so the row's bits then depend
/// on how many rows share the launch.
pub fn use_conv2d_im2col(params: &Conv2dParams, dtype: DType) -> bool {
    if !im2col2d_has_kernel(dtype) {
        return false;
    }

    // Grouped convolution splits the GEMM into one small problem per group.
    // The direct kernel loses no work to grouping, so grouped shapes (and
    // depthwise, which is always grouped) stay on it. Matches conv1d's
    // im2col and conv_transpose1d's gather.
    if params.groups != 1 {
        return false;
    }

    let contraction = params.c_in * params.kernel_h * params.kernel_w;
    contraction >= MIN_CONTRACTION && params.c_out >= MIN_C_OUT
}

/// Run conv2d as im2col followed by a batched GEMM.
///
/// `input`, `weight` and `bias` must already be contiguous, `params` must come
/// from `validate_conv2d`, and `groups` must be 1.
pub fn conv2d_im2col(
    client: &CudaClient,
    input: &Tensor<CudaRuntime>,
    weight: &Tensor<CudaRuntime>,
    bias: Option<&Tensor<CudaRuntime>>,
    params: &Conv2dParams,
) -> Result<Tensor<CudaRuntime>> {
    conv2d_im2col_capped(client, input, weight, bias, params, MAX_COL_ELEMENTS)
}

/// `conv2d_im2col` with an explicit column budget in elements, so a test can
/// force every chunk shape on a small tensor.
fn conv2d_im2col_capped(
    client: &CudaClient,
    input: &Tensor<CudaRuntime>,
    weight: &Tensor<CudaRuntime>,
    bias: Option<&Tensor<CudaRuntime>>,
    params: &Conv2dParams,
    col_cap: usize,
) -> Result<Tensor<CudaRuntime>> {
    let contraction = params.c_in * params.kernel_h * params.kernel_w;
    let spatial = params.output_h * params.output_w;

    // The weight is already `[C_out, C_in*Kh*Kw]` row-major, no copy needed.
    let weight_gemm = weight.reshape(&[1, params.c_out, contraction])?;

    let plan = chunk_plan(params, col_cap);
    let out = if plan.rows >= params.batch && plan.width >= spatial {
        // One chunk: the GEMM result is the output, with no copy.
        gemm_chunk(
            client,
            input,
            &weight_gemm,
            params,
            params.batch,
            0,
            spatial,
        )?
    } else {
        // Each chunk's GEMM result is copied once into its output slice, one
        // extra pass over the output in total. Only the output and one
        // chunk's result are live at a time. A window narrower than the
        // output makes the slice strided, and `copy_into` writes it in place.
        let out = Tensor::<CudaRuntime>::empty(
            &[params.batch, params.c_out, spatial],
            input.dtype(),
            &client.device,
        )?;
        let mut row = 0usize;
        while row < params.batch {
            let rows = plan.rows.min(params.batch - row);
            let in_rows = input.narrow(0, row, rows)?;
            let out_rows = out.narrow(0, row, rows)?;
            let mut start = 0usize;
            while start < spatial {
                let len = plan.width.min(spatial - start);
                let piece = gemm_chunk(client, &in_rows, &weight_gemm, params, rows, start, len)?;
                client.copy_into(&out_rows.narrow(2, start, len)?, &piece)?;
                start += len;
            }
            row += rows;
        }
        out
    };
    let out = out.reshape(&[params.batch, params.c_out, params.output_h, params.output_w])?;

    // The fused `matmul_bias` adds one value per GEMM COLUMN; here a column is
    // an output position and the bias is per output CHANNEL, which is a GEMM
    // row. The bias is broadcast over the channel axis instead.
    match bias {
        Some(b) => {
            let b_channels = b.reshape(&[1, params.c_out, 1, 1])?;
            client.add(&out, &b_channels)
        }
        None => Ok(out),
    }
}

/// Gather the columns of flattened output positions `[start, start + len)`
/// for the `rows` batch rows of `input`, and multiply them by `weight_gemm`.
/// Returns `[rows, C_out, len]`.
fn gemm_chunk(
    client: &CudaClient,
    input: &Tensor<CudaRuntime>,
    weight_gemm: &Tensor<CudaRuntime>,
    params: &Conv2dParams,
    rows: usize,
    start: usize,
    len: usize,
) -> Result<Tensor<CudaRuntime>> {
    let contraction = params.c_in * params.kernel_h * params.kernel_w;
    let col =
        Tensor::<CudaRuntime>::empty(&[rows, contraction, len], input.dtype(), &client.device)?;

    // SAFETY: `input` is a contiguous `[rows, C_in, H, W]` view and `col` a
    // fresh `[rows, C_in*Kh*Kw, len]` allocation, the sizes the launch reads
    // and writes. The launcher rejects a window past `H_out*W_out`.
    unsafe {
        launch_im2col2d(
            &client.context,
            &client.stream,
            client.device.index,
            input.dtype(),
            input.ptr(),
            col.ptr(),
            rows,
            params.c_in,
            params.height,
            params.width,
            params.kernel_h,
            params.kernel_w,
            params.output_h,
            params.output_w,
            params.stride_h,
            params.stride_w,
            params.pad_top,
            params.pad_left,
            params.dilation_h,
            params.dilation_w,
            start,
            len,
        )?;
    }

    client.matmul(weight_gemm, &col)
}

#[cfg(test)]
#[path = "conv2d_im2col_tests.rs"]
mod tests;
