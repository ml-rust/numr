//! Shared constants and block-shape rules for the convolution launchers.
//!
//! The blocking factors here are duplicated in the kernel sources; the tests
//! at the bottom parse the `.cu` files and check the two still agree.

/// Module name for convolution operations
pub const CONV_MODULE: &str = "conv";

/// Candidate conv1d block widths (threads per block along the output axis),
/// narrowest first. `output_length` is as small as ~26 at some hot shapes, so a
/// fixed 256-wide block would leave most lanes idle; the launcher picks the
/// narrowest width that still covers the row.
///
/// A full warp is the floor for `conv1d_oc4` only, which needs each warp on one
/// output-channel slot so its four stores stay coalesced. The scalar kernel
/// indexes `ox`, `oc` and `batch` independently and carries no such assumption,
/// so it takes [`CONV1D_BLOCK_NARROW`] below a warp instead of leaving most of
/// the warp idle.
pub(super) const CONV1D_BLOCK_CANDIDATES: [u32; 4] = [32, 64, 128, 256];

/// Sub-warp block widths for the scalar kernel, narrowest first. Decode-shaped
/// convolutions run at `output_length` of 1, where a 32-wide block retires 31
/// lanes at the bounds check before they do any work. Threads freed here go to
/// the output-channel axis, which always has work.
pub(super) const CONV1D_BLOCK_NARROW: [u32; 5] = [1, 2, 4, 8, 16];

/// Fallback conv1d block width when `output_length` exceeds every candidate.
pub(super) const CONV1D_BLOCK_MAX: u32 = 256;

/// Target threads per block, shared by conv1d and depthwise_conv2d's
/// row/position-indexed kernels. conv1d pads a narrow row out along the second
/// grid axis (`blockDim.y`, output channels) instead of launching a 32-thread
/// block, because an SM holds at most 16 blocks and 32-thread blocks would cap
/// it at 16 of its 48 warp slots. `depthwise_conv2d_ox` reaches the same count
/// with a flat block: it folds the output-row axis into x, so it needs no
/// second block axis to pad with.
pub(super) const CONV_BLOCK_THREADS: u32 = 128;

/// Output channels each thread of `conv1d_oc4_*` accumulates.
pub(super) const CONV1D_OC_BLOCK: usize = 4;

/// Consecutive output positions each thread of `conv1d_ox_*` accumulates.
/// Must match `CONV1D_OX_BLOCK` in `conv1d_ox.cu`.
pub(super) const CONV1D_OX_BLOCK: usize = 4;

/// Module name for the position-blocked conv1d kernel (`conv1d_ox.cu` compiles
/// to its own fatbin, separate from [`CONV_MODULE`]).
pub(super) const CONV1D_OX_MODULE: &str = "conv1d_ox";

/// Minimum `output_length` before `conv1d_ox` is preferred over the scalar
/// `conv1d` kernel for depthwise/narrow-group shapes (the oc4 kernel is
/// unavailable there). Guarantees the row covers at least one full
/// [`CONV1D_OX_BLOCK`]-wide chunk plus a remainder.
pub(super) const CONV1D_OX_MIN_OUTPUT_LENGTH: usize = 2 * CONV1D_OX_BLOCK;

/// Device waves the position-blocked grid must still reach after blocking.
///
/// `output_length` alone does not gate this kernel. Blocking divides the thread
/// count by [`CONV1D_OX_BLOCK`], so a shape with few channels can have a long
/// row and still be left with too few threads to fill the device: a narrow,
/// low-channel-count shape can regress badly versus the untiled kernel once
/// blocking leaves only a handful of warps to hide memory latency. One wave is
/// `compute_units * CONV_BLOCK_THREADS`; two is the smallest count that
/// rejects that shape while keeping every depthwise case that gains.
///
/// The wave counts the WHOLE launch, batch included, so a row can move from
/// the scalar kernel to this one as other rows join it. That is safe, and only
/// because the two kernels form one output element's sum identically: over
/// `ic` ascending, then `kx` ascending, in the same accumulator width, with
/// bias added last. `tests/cuda_conv_batch_invariance.rs` pins that. Any
/// change to either accumulation order breaks the batch invariance of conv1d
/// and the batch term has to leave this rule.
pub(super) const CONV1D_OX_MIN_WAVES: usize = 2;

/// Consecutive output columns each thread of `depthwise_conv2d_ox_*`
/// accumulates. Must match `DEPTHWISE_CONV2D_OX_BLOCK` in
/// `depthwise_conv2d_ox.cu`.
pub(super) const DEPTHWISE_CONV2D_OX_BLOCK: usize = 4;

/// Module name for the column-blocked depthwise conv2d kernel
/// (`depthwise_conv2d_ox.cu` compiles to its own fatbin, separate from
/// [`CONV_MODULE`]).
pub(super) const DEPTHWISE_CONV2D_OX_MODULE: &str = "depthwise_conv2d_ox";

/// Minimum `output_w` before `depthwise_conv2d_ox` is preferred over the flat
/// kernel. Guarantees the row covers at least one full
/// [`DEPTHWISE_CONV2D_OX_BLOCK`]-wide chunk plus a remainder.
pub(super) const DEPTHWISE_CONV2D_OX_MIN_OUTPUT_WIDTH: usize = 2 * DEPTHWISE_CONV2D_OX_BLOCK;

/// Device waves the column-blocked depthwise grid must still reach after
/// blocking. Blocking divides the thread count by
/// [`DEPTHWISE_CONV2D_OX_BLOCK`], so a wide row is not on its own enough to
/// keep the device fed once channels and rows are few.
///
/// The wave counts the whole launch, batch included, on the same footing as
/// [`CONV1D_OX_MIN_WAVES`]: the flat and the column-blocked kernel both sum
/// `ky` ascending then `kx` ascending with bias last, so which one a row gets
/// moves no bits.
pub(super) const DEPTHWISE_CONV2D_OX_MIN_WAVES: usize = 2;

/// CUDA caps the y and z grid dimensions at 65535 blocks.
pub(super) const CUDA_MAX_GRID_YZ: usize = 65535;

/// Block width (threads along the output-position axis) for the kernels that
/// index position, channel and batch independently, so may go below a warp.
/// `x_extent` is the number of thread slots the axis needs, already divided by
/// the per-thread blocking factor where one applies.
pub(super) fn position_block_width(x_extent: usize) -> u32 {
    CONV1D_BLOCK_NARROW
        .into_iter()
        .find(|&w| x_extent <= w as usize)
        .unwrap_or_else(|| {
            CONV1D_BLOCK_CANDIDATES
                .into_iter()
                .find(|&w| x_extent <= w as usize)
                .unwrap_or(CONV1D_BLOCK_MAX)
        })
}

/// Whether a position-blocked grid of `threads` still reaches `min_waves`
/// waves of the device.
///
/// `compute_units` is 0 on an unknown profile, which makes the test trivially
/// true and leaves the caller's row-extent gate deciding.
pub(super) fn fills_device(threads: usize, compute_units: usize, min_waves: usize) -> bool {
    let wave = compute_units
        .saturating_mul(CONV_BLOCK_THREADS as usize)
        .saturating_mul(min_waves);
    threads >= wave
}

#[cfg(test)]
mod tests {
    use super::{CONV1D_OC_BLOCK, CONV1D_OX_BLOCK, DEPTHWISE_CONV2D_OX_BLOCK};

    /// Blocking factors that appear in BOTH the kernel source and this
    /// launcher. The launcher sizes the grid from them and the kernel decides
    /// how much work a thread does, so a mismatch does not fail to build — it
    /// silently leaves outputs uncomputed. Parse the kernel and check.
    fn kernel_define(source: &str, name: &str) -> usize {
        let needle = format!("#define {name} ");
        let line = source
            .lines()
            .find(|l| l.starts_with(&needle))
            .unwrap_or_else(|| panic!("{name} is not defined in the kernel source"));
        line[needle.len()..]
            .trim()
            .trim_end_matches('u')
            .parse()
            .unwrap_or_else(|e| panic!("{name} is not a plain integer literal: {e}"))
    }

    #[test]
    fn oc_block_matches_the_kernel() {
        let source = include_str!("../conv.cu");
        assert_eq!(
            kernel_define(source, "CONV1D_OC_BLOCK"),
            CONV1D_OC_BLOCK,
            "conv.cu and tuning.rs disagree on the oc4 blocking factor"
        );
    }

    #[test]
    fn ox_block_matches_the_kernel() {
        let source = include_str!("../conv1d_ox.cu");
        assert_eq!(
            kernel_define(source, "CONV1D_OX_BLOCK"),
            CONV1D_OX_BLOCK,
            "conv1d_ox.cu and tuning.rs disagree on the position blocking factor"
        );
    }

    #[test]
    fn depthwise_ox_block_matches_the_kernel() {
        let source = include_str!("../depthwise_conv2d_ox.cu");
        assert_eq!(
            kernel_define(source, "DEPTHWISE_CONV2D_OX_BLOCK"),
            DEPTHWISE_CONV2D_OX_BLOCK,
            "depthwise_conv2d_ox.cu and tuning.rs disagree on the column blocking factor"
        );
    }
}
