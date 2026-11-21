// Backend parity for `conv_transpose1d` at channel widths that take the CUDA
// GEMM-first path (`conv_transpose1d_gemm_first`). Every backend runs these
// shapes; only CUDA changes path at this width.

use numr::ops::PaddingMode;

use crate::backend_parity::conv_transpose1d_multichannel::{
    assert_conv_transpose1d_parity, transpose_bias, transpose_input, transpose_weight,
};

/// `c_in=64` with the decoder's `kernel=2*stride`, `padding=ceil(stride/2)`
/// geometry: wide enough to take the CUDA GEMM-first path
/// (`conv_transpose1d_gemm_first`), whose fold sums the taps of a GEMM product
/// instead of running the direct kernel. All four float dtypes take it here;
/// F16 and BF16 run the product in F32 and round once at the fold, which is
/// the same accumulation the CPU reference performs. Bias and batch both on,
/// so the fold's bias add and batch stride are covered.
#[test]
fn conv_transpose1d_gemm_first_upsample_parity() {
    let input_shape = [2usize, 64, 9];
    let weight_shape = [64usize, 6, 8];
    // `transpose_input` over both batches: `64 * 9` is not a multiple of 13,
    // so the second batch is a shifted pattern, not a copy of the first.
    let input = transpose_input(input_shape.iter().product());
    let weight = transpose_weight(weight_shape.iter().product());
    let bias = transpose_bias(6);
    assert_conv_transpose1d_parity(
        "conv_transpose1d_gemm_first_upsample",
        &input,
        &input_shape,
        &weight,
        &weight_shape,
        Some(&bias),
        4,
        PaddingMode::Custom(2, 2, 0, 0),
        0,
        1,
        1,
    );
}

/// Same path with an odd stride, dilation and output padding, so the fold's
/// divisibility test, negative-numerator guard and lengthened tail all run.
#[test]
fn conv_transpose1d_gemm_first_odd_stride_dilation_parity() {
    let input_shape = [1usize, 40, 7];
    let weight_shape = [40usize, 5, 6];
    let input = transpose_input(input_shape.iter().product());
    let weight = transpose_weight(weight_shape.iter().product());
    assert_conv_transpose1d_parity(
        "conv_transpose1d_gemm_first_odd_stride_dilation",
        &input,
        &input_shape,
        &weight,
        &weight_shape,
        None,
        3,
        PaddingMode::Custom(1, 1, 0, 0),
        2,
        2,
        1,
    );
}
