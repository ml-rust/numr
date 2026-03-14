//! Shared utilities for conv autograd operations.

use crate::ops::PaddingMode;

/// Compute effective padding amounts for a single spatial dimension.
///
/// Returns `(pad_before, pad_after)` for the given kernel size and dilation.
pub(super) fn compute_padding(
    padding: PaddingMode,
    kernel_size: usize,
    dilation: usize,
) -> (usize, usize) {
    match padding {
        PaddingMode::Valid => (0, 0),
        PaddingMode::Same => {
            let effective_k = dilation * (kernel_size - 1) + 1;
            let total = effective_k.saturating_sub(1);
            (total / 2, total - total / 2)
        }
        PaddingMode::Custom(left, right, _, _) => (left, right),
    }
}

/// Compute effective padding amounts for 2D convolution.
///
/// Returns `(pad_top, pad_bottom, pad_left, pad_right)`.
pub(super) fn compute_padding_2d(
    padding: PaddingMode,
    kernel_h: usize,
    kernel_w: usize,
    dilation_h: usize,
    dilation_w: usize,
) -> (usize, usize, usize, usize) {
    match padding {
        PaddingMode::Valid => (0, 0, 0, 0),
        PaddingMode::Same => {
            let (top, bottom) = compute_padding(PaddingMode::Same, kernel_h, dilation_h);
            let (left, right) = compute_padding(PaddingMode::Same, kernel_w, dilation_w);
            (top, bottom, left, right)
        }
        PaddingMode::Custom(top, bottom, left, right) => (top, bottom, left, right),
    }
}
