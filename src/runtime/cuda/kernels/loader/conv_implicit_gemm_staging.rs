//! Operand staging variant of the implicit-GEMM conv2d.
//!
//! `kernels/conv_implicit_gemm.cu` instantiates three NHWC variants and two
//! NCHW variants per accumulator and tile. [`select_conv_staging`] picks one
//! per launch from the layout, the input view and the kernel window.

use crate::ops::ConvLayout;

use super::conv_implicit_gemm_tile::{ConvTile, ConvWindow};

/// Operand staging variant of one implicit-GEMM conv kernel.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub(super) enum ConvStaging {
    /// NHWC. A block owns a 2D output patch. Each 32-channel chunk of the
    /// input patch plus halo is staged once with 16-byte loads and serves
    /// every tap. K runs channel-chunk-major.
    Halo,
    /// NHWC. 16-byte gathers per tap, K in tap-major order.
    Vector,
    /// NHWC. 16-bit gathers, K in flat `(kh, kw, ci)` order.
    Scalar,
    /// NCHW. The halo body, with the halo read along W in aligned 8-byte
    /// runs of 4 pixels and transposed into the NHWC halo layout.
    NchwHalo,
    /// NCHW. The scalar gather with a channel stride: 16-bit gathers along
    /// W, K in flat `(kh, kw, ci)` order. Takes every NCHW shape.
    NchwGather,
}

impl ConvStaging {
    pub(super) const fn suffix(self) -> &'static str {
        match self {
            Self::Halo => "halo",
            Self::Vector => "vec",
            Self::Scalar => "scalar",
            Self::NchwHalo => "nchw_halo",
            Self::NchwGather => "nchw_gather",
        }
    }

    /// True for the halo body, whose grid counts output patches.
    pub(super) const fn is_halo(self) -> bool {
        matches!(self, Self::Halo | Self::NchwHalo)
    }
}

/// `(rows, columns)` of the input patch under one output patch of `tile`,
/// halo included, or `None` on overflow or an empty kernel:
/// `((ph - 1) * stride_h + (kh - 1) * dil_h + 1, (pw - 1) * stride_w + (kw - 1) * dil_w + 1)`.
fn halo_extent(tile: ConvTile, w: &ConvWindow) -> Option<(usize, usize)> {
    let (ph, pw) = tile.patch();
    let extent = |patch: usize, stride: usize, kernel: usize, dil: usize| {
        (patch - 1)
            .checked_mul(stride)?
            .checked_add(kernel.checked_sub(1)?.checked_mul(dil)?)?
            .checked_add(1)
    };
    let h = extent(ph, w.stride.0, w.kernel.0, w.dilation.0)?;
    let wd = extent(pw, w.stride.1, w.kernel.1, w.dilation.1)?;
    Some((h, wd))
}

/// Input pixels under one output patch of `tile`, halo included.
fn halo_pixels(tile: ConvTile, w: &ConvWindow) -> Option<usize> {
    let (h, wd) = halo_extent(tile, w)?;
    h.checked_mul(wd)
}

/// 4-pixel runs the NCHW halo loader stages per chunk:
/// `halo_h * ceil((mis + halo_w) / 4)`. `mis` is the first halo column's
/// offset from a multiple of 4 input columns. Each patch starts at an
/// output column that is a multiple of 16, so `mis = (-pad_left) mod 4` for
/// every block. Matches `row_runs` in `conv_implicit_gemm_halo_nchw.cuh`.
fn nchw_halo_runs(tile: ConvTile, w: &ConvWindow) -> Option<usize> {
    let (h, wd) = halo_extent(tile, w)?;
    let mis = (4 - w.pad_left % 4) % 4;
    h.checked_mul(wd.checked_add(mis)?.div_ceil(4))
}

/// Pick the staging variant for `tile`.
///
/// `input_strides` are the input's `(n, c, h, w)` element strides. Strides
/// are never negative here; `ops/cuda/conv_fused_native.rs` refuses those
/// views.
///
/// NHWC. The halo and vector variants read 8 channels of one pixel per
/// 16-byte load. That needs `C_in % 8 == 0`, a 16-byte aligned input and
/// weight, and input N/H/W element strides that are multiples of 8.
/// - Halo: the 16-byte conditions hold and the halo of one output patch
///   fits the tile's halo buffer ([`ConvTile::halo_capacity`]). A 3x3
///   stride-1 dilation-1 kernel always fits, and so does a 1x1 stride-1
///   kernel. Large strides, dilations and kernels do not.
/// - Vector: the 16-byte conditions hold and the halo does not fit.
/// - Scalar: every other shape.
///
/// NCHW.
/// - NchwHalo: `C_in % 8 == 0`, an 8-byte aligned input, input N/C/H
///   strides that are multiples of 4, W stride 1, a 16-byte aligned weight,
///   and a halo that fits both the halo buffer and the run table
///   ([`ConvTile::nchw_run_capacity`]). A 3x3 stride-1 dilation-1 kernel
///   always fits. `C_in % 8 == 0` matches the NHWC halo rule: a chunk of
///   fewer channels pads more of each tap's K-tile with zeros than the flat
///   K of the gather.
/// - NchwGather: every other shape.
pub(super) fn select_conv_staging(
    layout: ConvLayout,
    c_in: usize,
    input_ptr: u64,
    input_strides: [i64; 4],
    weight_ptr: u64,
    tile: ConvTile,
    window: &ConvWindow,
) -> ConvStaging {
    let [sn, sc, sh, sw] = input_strides;
    let halo_fits = halo_pixels(tile, window).is_some_and(|px| px <= tile.halo_capacity());
    match layout {
        ConvLayout::Nhwc => {
            let wide = c_in.is_multiple_of(8)
                && input_ptr.is_multiple_of(16)
                && weight_ptr.is_multiple_of(16)
                && [sn, sh, sw].iter().all(|s| s % 8 == 0);
            if !wide {
                ConvStaging::Scalar
            } else if halo_fits {
                ConvStaging::Halo
            } else {
                ConvStaging::Vector
            }
        }
        ConvLayout::Nchw => {
            let runs_fit =
                nchw_halo_runs(tile, window).is_some_and(|r| r <= tile.nchw_run_capacity());
            let halo = c_in.is_multiple_of(8)
                && input_ptr.is_multiple_of(8)
                && weight_ptr.is_multiple_of(16)
                && [sn, sc, sh].iter().all(|s| s % 4 == 0)
                && sw == 1
                && halo_fits
                && runs_fit;
            if halo {
                ConvStaging::NchwHalo
            } else {
                ConvStaging::NchwGather
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::conv_implicit_gemm_tile::test_window as window;
    use super::*;

    /// NHWC `(n, c, h, w)` strides of a 17x23 image with `c` channels per
    /// pixel.
    fn nhwc(c: i64) -> [i64; 4] {
        [17 * 23 * c, 1, 23 * c, c]
    }

    fn pick_nhwc(
        c_in: usize,
        ptr: u64,
        strides: [i64; 4],
        weight: u64,
        w: &ConvWindow,
    ) -> ConvStaging {
        select_conv_staging(
            ConvLayout::Nhwc,
            c_in,
            ptr,
            strides,
            weight,
            ConvTile::Tile128x64,
            w,
        )
    }

    #[test]
    fn staging_needs_aligned_channel_groups() {
        let w3 = window(2, 17, 23, 3, 1, 1);
        assert_eq!(
            pick_nhwc(64, 0x1000, nhwc(64), 0x2000, &w3),
            ConvStaging::Halo
        );
        // C_in not a multiple of 8.
        assert_eq!(
            pick_nhwc(3, 0x1000, nhwc(3), 0x2000, &w3),
            ConvStaging::Scalar
        );
        // Input view starting 3 channels into a buffer.
        assert_eq!(
            pick_nhwc(64, 0x1006, nhwc(64), 0x2000, &w3),
            ConvStaging::Scalar
        );
        // Weight view off 16-byte alignment.
        assert_eq!(
            pick_nhwc(64, 0x1000, nhwc(64), 0x2008, &w3),
            ConvStaging::Scalar
        );
        // Channel slice of an 84-channel buffer: pixel stride 84.
        assert_eq!(
            pick_nhwc(64, 0x1000, nhwc(84), 0x2000, &w3),
            ConvStaging::Scalar
        );
        // A size-1 dimension reports stride 0.
        assert_eq!(
            pick_nhwc(8, 0x1000, [0, 1, 23 * 8, 8], 0x2000, &w3),
            ConvStaging::Halo
        );
    }

    #[test]
    fn halo_staging_needs_the_halo_to_fit() {
        let pick = |tile: ConvTile, w: ConvWindow| {
            select_conv_staging(ConvLayout::Nhwc, 64, 0, nhwc(64), 0, tile, &w)
        };
        for tile in [
            ConvTile::Tile128x64,
            ConvTile::Tile256x32,
            ConvTile::Tile256x64,
        ] {
            // 3x3 stride 1: 10x18 = 180 <= 192 and 10x34 = 340 <= 352.
            assert_eq!(pick(tile, window(1, 9, 9, 3, 1, 1)), ConvStaging::Halo);
            // 1x1 stride 1: the patch itself.
            assert_eq!(pick(tile, window(1, 9, 9, 1, 1, 1)), ConvStaging::Halo);
            // Stride 2 doubles the halo; 5x5 and dilation 2 widen it.
            assert_eq!(pick(tile, window(1, 9, 9, 3, 2, 1)), ConvStaging::Vector);
            assert_eq!(pick(tile, window(1, 9, 9, 5, 1, 1)), ConvStaging::Vector);
            assert_eq!(pick(tile, window(1, 9, 9, 3, 1, 2)), ConvStaging::Vector);
        }
        assert_eq!(
            halo_pixels(ConvTile::Tile128x64, &window(1, 1, 1, 3, 1, 1)),
            Some(180)
        );
        assert_eq!(
            halo_pixels(ConvTile::Tile256x64, &window(1, 1, 1, 3, 1, 1)),
            Some(340)
        );
        assert_eq!(
            halo_pixels(ConvTile::Tile256x32, &window(1, 1, 1, 0, 1, 1)),
            None
        );
    }

    /// NCHW `(n, c, h, w)` strides of a contiguous `c x h x w` image.
    fn nchw(c: i64, h: i64, w: i64) -> [i64; 4] {
        [c * h * w, h * w, w, 1]
    }

    fn pick_nchw(c_in: usize, ptr: u64, strides: [i64; 4], w: &ConvWindow) -> ConvStaging {
        select_conv_staging(
            ConvLayout::Nchw,
            c_in,
            ptr,
            strides,
            0x2000,
            ConvTile::Tile128x64,
            w,
        )
    }

    #[test]
    fn nchw_halo_needs_aligned_runs() {
        let w3 = window(2, 17, 24, 3, 1, 1);
        let s = nchw(64, 17, 24);
        assert_eq!(pick_nchw(64, 0x1000, s, &w3), ConvStaging::NchwHalo);
        // Odd W: the H stride is 23, not a multiple of 4.
        assert_eq!(
            pick_nchw(64, 0x1000, nchw(64, 17, 23), &w3),
            ConvStaging::NchwGather
        );
        // 8-byte alignment is enough; 2-byte alignment is not.
        assert_eq!(pick_nchw(64, 0x1008, s, &w3), ConvStaging::NchwHalo);
        assert_eq!(pick_nchw(64, 0x1002, s, &w3), ConvStaging::NchwGather);
        // C_in not a multiple of 8.
        assert_eq!(
            pick_nchw(3, 0x1000, nchw(3, 17, 24), &w3),
            ConvStaging::NchwGather
        );
        // W stride 2.
        assert_eq!(
            pick_nchw(64, 0x1000, [64 * 17 * 48, 17 * 48, 48, 2], &w3),
            ConvStaging::NchwGather
        );
        // Weight off 16-byte alignment.
        assert_eq!(
            select_conv_staging(
                ConvLayout::Nchw,
                64,
                0x1000,
                s,
                0x2008,
                ConvTile::Tile128x64,
                &w3
            ),
            ConvStaging::NchwGather
        );
        // Stride 2, dilation 2 and 5x5 do not fit the halo.
        for w in [
            window(1, 9, 9, 3, 2, 1),
            window(1, 9, 9, 3, 1, 2),
            window(1, 9, 9, 5, 1, 1),
        ] {
            assert_eq!(pick_nchw(64, 0x1000, s, &w), ConvStaging::NchwGather);
        }
    }

    #[test]
    fn nchw_halo_runs_fit_every_3x3_padding() {
        for pad_left in 0..8 {
            for (tile, runs) in [
                (ConvTile::Tile128x64, [50, 60]),
                (ConvTile::Tile256x32, [90, 100]),
            ] {
                let w = ConvWindow {
                    pad_left,
                    ..window(1, 9, 9, 3, 1, 1)
                };
                let got = nchw_halo_runs(tile, &w).unwrap();
                // An 18- or 34-wide halo needs one more run per row only
                // when mis is 3.
                let mis = (4 - pad_left % 4) % 4;
                assert_eq!(got, runs[usize::from(mis == 3)], "pad_left {pad_left}");
                assert!(got <= tile.nchw_run_capacity());
            }
        }
        // 1x1: one patch row per halo row, 8 rows of 4 or 8 runs.
        assert_eq!(
            nchw_halo_runs(ConvTile::Tile128x64, &window(1, 9, 9, 1, 1, 1)),
            Some(32)
        );
        assert_eq!(
            nchw_halo_runs(ConvTile::Tile256x64, &window(1, 9, 9, 1, 1, 1)),
            Some(64)
        );
    }
}
