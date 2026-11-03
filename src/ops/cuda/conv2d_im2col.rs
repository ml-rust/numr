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
/// kernel.
///
/// PROVISIONAL, awaiting measurement.
const MIN_C_OUT: usize = 4;

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
mod tests {
    use super::*;
    use crate::ops::conv_common::validate_conv2d;
    use crate::ops::{PaddingMode, RandomOps, TypeConversionOps};
    use crate::runtime::Runtime;
    use crate::runtime::cuda::CudaDevice;

    fn setup() -> Option<CudaClient> {
        if !crate::runtime::cuda::is_cuda_available() {
            return None;
        }
        let device = CudaDevice::new(0);
        Some(CudaRuntime::default_client(&device))
    }

    /// `Conv2dParams` for a `groups == 1` conv2d at `dtype`, with bias.
    fn params_for(
        input_shape: &[usize],
        weight_shape: &[usize],
        padding: PaddingMode,
        dtype: DType,
    ) -> Conv2dParams {
        validate_conv2d(
            input_shape,
            weight_shape,
            Some(&weight_shape[..1]),
            (1, 1),
            padding,
            (1, 1),
            1,
            dtype,
            dtype,
            Some(dtype),
        )
        .expect("params")
    }

    /// Element bits of `t`, widened to `u64`. A half tensor is cast to F32
    /// first: that cast is exact and one-to-one, so equal F32 bits mean equal
    /// half bits.
    fn element_bits(client: &CudaClient, t: &Tensor<CudaRuntime>) -> Vec<u64> {
        match t.dtype() {
            DType::F64 => t.to_vec::<f64>().iter().map(|v| v.to_bits()).collect(),
            DType::F32 => t
                .to_vec::<f32>()
                .iter()
                .map(|v| u64::from(v.to_bits()))
                .collect(),
            _ => client
                .cast(t, DType::F32)
                .expect("cast to F32")
                .to_vec::<f32>()
                .iter()
                .map(|v| u64::from(v.to_bits()))
                .collect(),
        }
    }

    /// The default budget admits the motivating shape, 64 to 64 channels at
    /// 3x3 over 1080x1920, whose one-row column exceeds `MAX_COL_ELEMENTS`.
    /// It splits into aligned windows of one row each.
    #[test]
    fn large_column_takes_gemm_in_windows() {
        let params = params_for(
            &[1, 64, 1080, 1920],
            &[64, 64, 3, 3],
            PaddingMode::Custom(1, 1, 1, 1),
            DType::F32,
        );
        let contraction = 64 * 3 * 3;
        assert!(contraction * 1080 * 1920 > MAX_COL_ELEMENTS);
        assert!(use_conv2d_im2col(&params, DType::F32));

        let plan = chunk_plan(&params, MAX_COL_ELEMENTS);
        assert_eq!(plan.rows, 1);
        assert_eq!(plan.width % WINDOW_ALIGN, 0);
        assert!(plan.width * contraction <= MAX_COL_ELEMENTS);
        assert!(plan.width < 1080 * 1920);
    }

    /// The gate keeps its floors and its group rule: a column far over the
    /// budget does not lift them.
    #[test]
    fn gate_keeps_floors_at_any_column_size() {
        let shallow = params_for(
            &[1, 2, 4000, 4000],
            &[8, 2, 3, 3],
            PaddingMode::Valid,
            DType::F32,
        );
        assert!(!use_conv2d_im2col(&shallow, DType::F32));
        let narrow = params_for(
            &[1, 64, 2000, 2000],
            &[MIN_C_OUT - 1, 64, 3, 3],
            PaddingMode::Valid,
            DType::F32,
        );
        assert!(!use_conv2d_im2col(&narrow, DType::F32));
    }

    /// Each budget below lands on a different chunk shape for the shape the
    /// bitwise test runs: 3 rows of 20x20 output positions, contraction 36.
    #[test]
    fn chunk_plan_shapes() {
        let params = params_for(
            &[3, 4, 20, 20],
            &[80, 4, 3, 3],
            PaddingMode::Custom(1, 1, 1, 1),
            DType::F32,
        );
        let (k, spatial) = (36usize, 400usize);
        let plan = |cap| {
            let p = chunk_plan(&params, cap);
            (p.rows, p.width)
        };
        assert_eq!(plan(usize::MAX), (3, spatial));
        assert_eq!(plan(3 * k * spatial), (3, spatial));
        assert_eq!(plan(2 * k * spatial), (2, spatial));
        assert_eq!(plan(k * 300), (1, 256));
        assert_eq!(plan(k * 70), (1, 70));
        assert_eq!(plan(k * 7), (1, 7));
        assert_eq!(plan(1), (1, 1));
    }

    /// A chunk boundary moves an output element to another GEMM launch, tile
    /// or batch slot, never to another accumulation order (see
    /// [`chunk_plan`]). Every chunked run must reproduce the one-chunk run bit
    /// for bit, at every dtype the path takes. The budgets cover ragged batch
    /// chunks, aligned windows with a ragged tail, and unaligned windows.
    /// `c_out = 80` puts the F32 GEMM on a different tile for windows up to
    /// 64 wide than for wider ones.
    #[test]
    fn chunked_matches_one_chunk_bitwise() {
        let Some(client) = setup() else {
            return;
        };
        let (batch, c_in, c_out, k, h, w) = (3usize, 4usize, 80usize, 3usize, 20usize, 20usize);
        let contraction = c_in * k * k;
        let spatial = h * w;
        let caps = [
            2 * contraction * spatial,
            contraction * 300,
            contraction * 70,
            contraction * 7,
        ];

        let mut dtypes = vec![DType::F32, DType::F64];
        if cfg!(feature = "f16") {
            dtypes.extend([DType::F16, DType::BF16]);
        }
        for dtype in dtypes {
            let input = client.rand(&[batch, c_in, h, w], dtype).expect("input");
            let weight = client.rand(&[c_out, c_in, k, k], dtype).expect("weight");
            let bias = client.rand(&[c_out], dtype).expect("bias");
            let params = params_for(
                input.shape(),
                weight.shape(),
                PaddingMode::Custom(1, 1, 1, 1),
                dtype,
            );

            let whole =
                conv2d_im2col_capped(&client, &input, &weight, Some(&bias), &params, usize::MAX)
                    .expect("whole");
            let whole_bits = element_bits(&client, &whole);
            assert_eq!(whole_bits.len(), batch * c_out * spatial);

            for cap in caps {
                let chunked =
                    conv2d_im2col_capped(&client, &input, &weight, Some(&bias), &params, cap)
                        .expect("chunked");
                assert_eq!(whole.shape(), chunked.shape());
                let chunked_bits = element_bits(&client, &chunked);
                for (i, (x, y)) in whole_bits.iter().zip(&chunked_bits).enumerate() {
                    assert_eq!(
                        x, y,
                        "{dtype:?} cap {cap}: element {i} bits {x:#x} vs {y:#x}"
                    );
                }
            }
        }
    }
}
