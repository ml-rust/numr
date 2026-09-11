//! im2col + GEMM formulation of CUDA conv1d.
//!
//! The direct conv1d kernel re-reads the whole input once per output channel.
//! Gathering the receptive fields into a column buffer turns the same work into
//! one GEMM, which reaches the tuned matmul kernels instead.
//!
//! # Layout
//!
//! `im2col` writes `col` as `[N, C_in*K, L_out]`, contraction axis first. Split
//! row-major that is `[N, groups, (C_in/groups)*K, L_out]`, and the weight
//! `[C_out, C_in/groups, K]` reshapes with no copy to
//! `[1, groups, C_out/groups, (C_in/groups)*K]` because output channels are
//! ordered group-major. Batched over `(N, groups)` the GEMM yields
//! `[N, groups, C_out/groups, L_out]`, which reshapes to `[N, C_out, L_out]`.
//! No transpose and no final permute.

use crate::dtype::DType;
use crate::error::Result;
use crate::ops::conv_common::Conv1dParams;
use crate::ops::{BinaryOps, MatmulOps, ShapeOps};
use crate::runtime::cuda::kernels::{im2col_has_kernel, launch_im2col1d};
use crate::runtime::cuda::{CudaClient, CudaRuntime};
use crate::tensor::Tensor;

/// Smallest contraction length worth routing through the GEMM.
///
/// Below this the column buffer costs more to write than the GEMM saves, and
/// the K loop is too short for the tiled kernels to amortise their prologue.
///
/// **Contraction depth is the discriminator, not output length and not total
/// work.** Two shapes with the same GEMM volume can want opposite kernels: the
/// direct kernel handles many shallow outputs well and struggles when few
/// outputs each carry deep contraction, which is exactly where the GEMM wins.
/// Swept with c_out held at both 32 and 512 to confirm the axis: contraction
/// 512 and 1024 lose, 2048 wins at both widths.
///
/// The previous value of 64 was far below that crossover and admitted shapes
/// the GEMM ran a factor of two SLOWER than the direct kernel.
const MIN_CONTRACTION: usize = 2048;

/// Output length past which a shallower contraction still wins on the GEMM.
///
/// The sweep behind `MIN_CONTRACTION` ran at output lengths up to a few
/// hundred. There the GEMM's N is a few tiles wide and the direct kernel's
/// one-thread-per-output shape is competitive. A long output gives the GEMM a
/// wide N to fill the device with, and the direct kernel's throughput does not
/// grow with length. Measured with contraction 448, 896 and 1792: at this
/// length the GEMM ties or wins, and past it the gap widens with length.
const LONG_OUTPUT_LENGTH: usize = 1024;

/// Smallest contraction routed through the GEMM at a long output. The
/// smallest swept; below it is unmeasured and stays on the direct kernel.
const MIN_CONTRACTION_LONG_OUTPUT: usize = 448;

/// Smallest number of output channels per group. Below this the GEMM has too
/// few rows to reuse a loaded column tile, which is the whole gain over the
/// direct kernel. Depthwise convolution sits at one and always stays direct.
const MIN_C_OUT_PER_GROUP: usize = 4;

/// Largest column buffer, in elements. A longer output is split into chunks
/// of at most this many column elements, each gathered and multiplied on its
/// own, so the buffer bounds memory without bounding the shapes that qualify.
const MAX_COL_ELEMENTS: usize = 1 << 26;

/// Whether conv1d takes the im2col + GEMM path instead of the direct kernel.
///
/// The principle: im2col wins when the resulting GEMM is well shaped — a long
/// contraction, enough rows per group to reuse each column tile, and enough
/// columns to fill a tile. Depthwise, narrow-channel and very short outputs
/// keep the direct kernel, where a batch of tiny GEMMs plus the column buffer
/// would cost more than it saves.
pub fn use_conv1d_im2col(params: &Conv1dParams, dtype: DType) -> bool {
    if !im2col_has_kernel(dtype) || params.groups == 0 {
        return false;
    }

    let c_in_per_group = params.c_in / params.groups;
    let c_out_per_group = params.c_out / params.groups;
    let contraction = c_in_per_group * params.kernel_size;

    // One output position's column must fit the chunk budget; the length is
    // chunked, so it does not enter here.
    if max_chunk_length(params) == 0 {
        return false;
    }

    // Grouped convolution splits the GEMM into one small problem per group, each
    // still only `output_length` wide. That shape is far off the tiled kernel's
    // best case, and the direct kernel — which loses no work to grouping — wins
    // by a wide margin. Measured, not assumed: re-check with `benches/conv.rs`
    // before widening this.
    if params.groups != 1 {
        return false;
    }

    if c_out_per_group < MIN_C_OUT_PER_GROUP {
        return false;
    }
    // Once the contraction clears MIN_CONTRACTION the GEMM won at every length
    // swept, so no length floor applies there. A shallower contraction needs
    // the long output to pay for the column buffer.
    contraction >= MIN_CONTRACTION
        || (contraction >= MIN_CONTRACTION_LONG_OUTPUT
            && params.output_length >= LONG_OUTPUT_LENGTH)
}

/// Output positions per column chunk under `MAX_COL_ELEMENTS`; zero when a
/// single position's column already exceeds it.
fn max_chunk_length(params: &Conv1dParams) -> usize {
    params
        .batch
        .checked_mul(params.c_in)
        .and_then(|v| v.checked_mul(params.kernel_size))
        .map(|per_position| MAX_COL_ELEMENTS / per_position.max(1))
        .unwrap_or(0)
}

/// Whether conv1d is a pointwise convolution the GEMM runs on directly.
///
/// With one tap, unit stride, no padding and one group, `im2col` would copy
/// the input unchanged: `out[n] = W[c_out, c_in] @ x[n]`. The GEMM reads the
/// input in place instead, so no column buffer exists and no size gate
/// applies. `c_out_per_group` stays under the same floor as the im2col path;
/// below it the GEMM has too few rows to reuse a column tile.
pub fn use_conv1d_pointwise_gemm(params: &Conv1dParams) -> bool {
    params.kernel_size == 1
        && params.stride == 1
        && params.pad_left == 0
        && params.pad_right == 0
        && params.groups == 1
        && params.output_length == params.length
        && params.c_out >= MIN_C_OUT_PER_GROUP
}

/// Run a pointwise conv1d as one batched GEMM over the input.
///
/// `input`, `weight` and `bias` must already be contiguous and
/// [`use_conv1d_pointwise_gemm`] must hold for `params`.
pub fn conv1d_pointwise_gemm(
    client: &CudaClient,
    input: &Tensor<CudaRuntime>,
    weight: &Tensor<CudaRuntime>,
    bias: Option<&Tensor<CudaRuntime>>,
    params: &Conv1dParams,
) -> Result<Tensor<CudaRuntime>> {
    // `[C_out, C_in, 1]` -> `[1, C_out, C_in]`, a view broadcast over the batch.
    let weight_gemm = weight.reshape(&[1, params.c_out, params.c_in])?;
    let out = client.matmul(&weight_gemm, input)?;
    add_channel_bias(client, out, bias, params.c_out)
}

/// The fused `matmul_bias` adds one value per GEMM COLUMN; here a column is
/// an output position and the bias is per output CHANNEL, which is a GEMM
/// row. The bias is broadcast over the channel axis instead.
fn add_channel_bias(
    client: &CudaClient,
    out: Tensor<CudaRuntime>,
    bias: Option<&Tensor<CudaRuntime>>,
    c_out: usize,
) -> Result<Tensor<CudaRuntime>> {
    match bias {
        Some(b) => {
            let b_channels = b.reshape(&[1, c_out, 1])?;
            client.add(&out, &b_channels)
        }
        None => Ok(out),
    }
}

/// Run conv1d as im2col followed by a batched GEMM.
///
/// `input`, `weight` and `bias` must already be contiguous, and `params` must
/// come from `validate_conv1d`.
pub fn conv1d_im2col(
    client: &CudaClient,
    input: &Tensor<CudaRuntime>,
    weight: &Tensor<CudaRuntime>,
    bias: Option<&Tensor<CudaRuntime>>,
    params: &Conv1dParams,
) -> Result<Tensor<CudaRuntime>> {
    conv1d_im2col_chunked(
        client,
        input,
        weight,
        bias,
        params,
        max_chunk_length(params),
    )
}

/// `conv1d_im2col` with an explicit chunk length, so a test can force the
/// multi-chunk path on a small tensor. `chunk_max` is clamped to at least one
/// position.
fn conv1d_im2col_chunked(
    client: &CudaClient,
    input: &Tensor<CudaRuntime>,
    weight: &Tensor<CudaRuntime>,
    bias: Option<&Tensor<CudaRuntime>>,
    params: &Conv1dParams,
    chunk_max: usize,
) -> Result<Tensor<CudaRuntime>> {
    let dtype = input.dtype();
    let rows = params.c_in * params.kernel_size;
    let c_in_per_group = params.c_in / params.groups;
    let c_out_per_group = params.c_out / params.groups;
    let contraction = c_in_per_group * params.kernel_size;

    // A stride-preserving view of contiguous data.
    let weight_grouped = weight.reshape(&[1, params.groups, c_out_per_group, contraction])?;

    // The output is gathered and multiplied in chunks of positions so the
    // column buffer stays under `MAX_COL_ELEMENTS`; a single chunk covers the
    // whole output when it fits.
    let chunk_max = chunk_max.max(1);
    let mut pieces: Vec<Tensor<CudaRuntime>> = Vec::new();
    let mut start = 0usize;
    while start < params.output_length {
        let len = chunk_max.min(params.output_length - start);
        let col = Tensor::<CudaRuntime>::empty(&[params.batch, rows, len], dtype, &client.device)?;
        unsafe {
            launch_im2col1d(
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
                params.pad_left, // symmetric padding, same convention as the direct kernel
                params.dilation,
                start,
            )?;
        }
        let col_grouped = col.reshape(&[params.batch, params.groups, contraction, len])?;
        let piece = client.matmul(&weight_grouped, &col_grouped)?;
        pieces.push(piece.reshape(&[params.batch, params.c_out, len])?);
        start += len;
    }
    let out = if pieces.len() == 1 {
        pieces.remove(0)
    } else {
        let refs: Vec<&Tensor<CudaRuntime>> = pieces.iter().collect();
        client.cat(&refs, 2)?
    };

    add_channel_bias(client, out, bias, params.c_out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::RandomOps;
    use crate::ops::conv_common::validate_conv1d;
    use crate::runtime::Runtime;
    use crate::runtime::cuda::CudaDevice;

    fn setup() -> Option<(CudaDevice, CudaClient)> {
        if !crate::runtime::cuda::is_cuda_available() {
            return None;
        }
        let device = CudaDevice::new(0);
        let client = CudaRuntime::default_client(&device);
        Some((device, client))
    }

    /// Splitting the output into column chunks changes which GEMM tile an
    /// output position lands in, never the K order it accumulates in, so a
    /// chunked run must reproduce the single-chunk run bit for bit — with a
    /// chunk length that does not divide the output, so the last chunk is
    /// ragged.
    #[test]
    fn chunked_columns_match_one_chunk_bitwise() {
        let Some((device, client)) = setup() else {
            return;
        };
        let (batch, c_in, c_out, k, len) = (2usize, 6usize, 8usize, 3usize, 53usize);
        let input = client.rand(&[batch, c_in, len], DType::F32).unwrap();
        let weight = client.rand(&[c_out, c_in, k], DType::F32).unwrap();
        let bias = client.rand(&[c_out], DType::F32).unwrap();
        let params = validate_conv1d(
            input.shape(),
            weight.shape(),
            Some(bias.shape()),
            1,
            crate::ops::PaddingMode::Custom(2, 0, 0, 0),
            2,
            1,
            DType::F32,
            DType::F32,
            Some(DType::F32),
        )
        .unwrap();
        let _ = device;

        let whole =
            conv1d_im2col_chunked(&client, &input, &weight, Some(&bias), &params, usize::MAX)
                .unwrap();
        let chunked =
            conv1d_im2col_chunked(&client, &input, &weight, Some(&bias), &params, 7).unwrap();

        assert_eq!(whole.shape(), chunked.shape());
        let a: Vec<f32> = whole.to_vec();
        let b: Vec<f32> = chunked.to_vec();
        assert_eq!(a.len(), params.batch * params.c_out * params.output_length);
        for (i, (x, y)) in a.iter().zip(&b).enumerate() {
            assert_eq!(x.to_bits(), y.to_bits(), "element {i}: {x} vs {y}");
        }
    }
}
